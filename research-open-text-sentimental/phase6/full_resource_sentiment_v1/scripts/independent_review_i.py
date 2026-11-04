"""Read-only independent Workstream I audit. Writes derived checks, never text."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from releasetrain_sentiment import score_post

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "results" / "broad_sentiment_2026-10-22-main"
I = ROOT / "results" / "workstream_i_2026-10-22-main"
RAW = ROOT / "restricted" / "broad_release_train" / "2026-10-22-main"
MEASURES = ROOT / "restricted" / "workstream_i_2026-10-22-main" / "prefix_diagnostics.jsonl"
SEED = 20261024
NBOOT = 2000


def rows(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def score(text):
    return float(score_post({"title": text, "description": "", "comments": []})[0]["compound"])


def check(x, y, label, tolerance=1e-8):
    if x is None or y is None:
        assert x is None and y is None, label
    else:
        assert abs(float(x) - float(y)) <= tolerance, f"{label}: {x} != {y}"


def paired_mean_ci(values):
    a = np.asarray(values, float)
    rng = np.random.default_rng(SEED)
    bs = np.empty(NBOOT)
    for k in range(NBOOT):
        bs[k] = a[rng.integers(0, len(a), len(a))].mean()
    return float(a.mean()), [float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]


def strata_check(values, reported, name):
    assert len(values) == reported["n"], name
    check(np.mean(values) if values else None, reported["mean"], name)


def ranks(a):
    a = np.asarray(a,float)
    idx = np.argsort(a,kind="mergesort")
    out = np.empty(len(a),float)
    k=0
    while k<len(a):
        j=k+1
        while j<len(a) and a[idx[j]]==a[idx[k]]:j+=1
        out[idx[k:j]]=(k+1+j)/2
        k=j
    return out


def rho(x,y):
    return float(np.corrcoef(ranks(x),ranks(y))[0,1])


def sample_direct_check(posts, threads, comments, measures):
    fids=sorted((rid for rid,t in threads.items() if t["f_eligible"]),key=lambda rid: hashlib.sha256((rid+'f').encode()).hexdigest())[:10]
    pools=[
        [rid for rid,t in threads.items() if t["body_nonempty"] and t["author_reply_eligible"]],
        [rid for rid,t in threads.items() if not t["body_nonempty"]],
        [rid for rid,t in threads.items() if t["count_gap"]],
        [rid for rid,t in threads.items() if not t["count_gap"]],
        [rid for rid,t in threads.items() if t["created_month"]=="2026-10"],
        [rid for rid,t in threads.items() if t["created_month"]!="2026-10"],
    ]
    chosen=set(fids)
    for pool in pools:
        for rid in sorted(pool,key=lambda x:hashlib.sha256((x+'sample').encode()).hexdigest()):
            if rid not in chosen:chosen.add(rid);break
    remaining=sorted(posts,key=lambda rid:hashlib.sha256((rid+'remaining').encode()).hexdigest())
    for rid in remaining:
        if len(chosen)>=40:break
        chosen.add(rid)
    assert len(chosen)==40 and len(set(fids)&chosen)>=10
    page_groups=defaultdict(list)
    for rid in chosen:page_groups[posts[rid]["capture_page"]].append(rid)
    n_prefix=0;n_role=0;n_time=0
    for page,rids in page_groups.items():
        rawrows=json.loads((RAW/f"page_{page:05d}.json").read_text())["data"]
        for rid in rids:
            raw=rawrows[posts[rid]["capture_row_index"]]
            assert str(raw["redditId"])==rid
            m=measures[rid]
            title=str(raw.get("title") or "")
            body=str(raw.get("author_description") or "")
            post_text=(title+' '+body) if title.strip() and body.strip() else (title if title.strip() else body)
            assert len(title.split())==m["title_words"]
            assert len(body.split())==m["body_words"]
            assert len(post_text)==m["original_post_chars"]
            assert len(post_text.split())==m["original_post_words"]
            if m["k_title_body"] is not None and m["k_title_body"]>=3:
                k=m["k_title_body"]
                check(score(' '.join(title.split()[:k])),m["title_prefix_compound"],rid+' title prefix')
                check(score(' '.join(body.split()[:k])),m["body_prefix_compound"],rid+' body prefix')
                n_prefix+=1
            saved=comments.get(rid,[])
            authors=[c for c in saved if c["is_author"]]
            for c in saved:
                rawc=raw["comments"][c["original_index"]]
                flag=rawc.get("is_submitter")
                explicit=flag is True or str(flag).strip().lower() in {"true","t","yes","1"}
                pa=str(raw.get("author") or "").strip();ca=str(rawc.get("author") or "").strip()
                assert c["is_author"]==(explicit or bool(pa and ca and pa==ca))
                n_role+=1
                assert c["timestamp_valid"] and c["timestamp_epoch_seconds"] is not None
                n_time+=1
            if authors and m["k_post_first_reply"] is not None and m["k_post_first_reply"]>=3:
                first=min(authors,key=lambda c:(c["chronological_index"],c["original_index"]))
                rawc=raw["comments"][first["original_index"]]
                text=str(rawc.get("body") or rawc.get("text") or "")
                k=m["k_post_first_reply"]
                check(score(' '.join(post_text.split()[:k])),m["post_prefix_compound"],rid+' post prefix')
                check(score(' '.join(text.split()[:k])),m["first_reply_prefix_compound"],rid+' reply prefix')
                n_prefix+=1
    return {"raw_posts_checked":40,"f_posts_checked":len(fids),"prefix_scores_checked":n_prefix,"reply_roles_checked":n_role,"timestamps_checked":n_time}


def main():
    manifest=json.loads((I/'manifest.json').read_text())
    assert manifest['package_version']=='0.2.1' and manifest['seed']==SEED
    raw_manifest=json.loads((RAW/'manifest.json').read_text())
    for page in raw_manifest['pages']:
        assert digest(RAW/page['file'])==page['sha256']
    ghash=json.loads((G/'hashes.json').read_text())
    for name,h in ghash.items():
        assert digest(G/name)==h==manifest['g_input_hashes'][name]
    ihash=json.loads((I/'hashes.json').read_text())
    for name,h in ihash.items():
        assert digest(I/name)==h==manifest['output_sha256'][name]
    posts={r['reddit_id']:r for r in rows(G/'post_scores.jsonl')}
    threads={r['reddit_id']:r for r in rows(G/'thread_metrics.jsonl')}
    comments=defaultdict(list)
    comments_all=rows(G/'comment_scores.jsonl')
    for c in comments_all:comments[c['reddit_id']].append(c)
    measures={r['reddit_id']:r for r in rows(MEASURES)}
    assert len(posts)==len(threads)==len(measures)==37596 and len(comments_all)==273353
    assert set(posts)==set(threads)==set(measures)
    assert sum(t['count_gap'] for t in threads.values())==18260
    assert [sum(t[k] for t in threads.values()) for k in ('author_reply_eligible','both_roles_eligible','order_eligible','elapsed_eligible','f_eligible')]==[17282,17035,24318,24313,1864]
    assert sum(p['body_nonempty'] for p in posts.values())==35443
    assert all(not(t['f_eligible'] and (t['reserved_membership'] or t['fresh_membership'])) for t in threads.values())
    local=rows(ROOT/'results'/'normalized_records_compact.jsonl')
    expected={r['reddit_id']:r['labels']['s2_target'] for r in local if 'development_v2' in r['source_memberships'] and not({'reserved_test','fresh_500'}&set(r['source_memberships'])) and r['labels'].get('s2_in_scope') is True and r['reddit_id'] in threads}
    actual={rid:t['s2_target'] for rid,t in threads.items() if t['f_eligible']}
    assert actual==expected and Counter(actual.values())=={1:1628,0:236}
    report={x:json.loads((I/f'I_{x}.json').read_text()) for x in ('A_length','B_length','C_roles','D_order','E_elapsed','F_length')}
    sample=sample_direct_check(posts,threads,comments,measures)
    # A/B full-length strata and paired prefix diagnostics, from independent per-ID joins.
    a=[];ap=[];af=[];abins=defaultdict(list)
    b=[];bp=[];bf=[];bbins=defaultdict(list)
    for rid,m in measures.items():
        p=posts[rid];t=threads[rid]
        if p['body_compound'] is not None:
            diff=p['body_compound']-p['title_compound'];a.append(diff)
            ratio=m['body_words']/m['title_words']
            binname='le_1' if ratio<=1 else '(1,2]' if ratio<=2 else '(2,5]' if ratio<=5 else 'gt_5'
            abins[binname].append(diff)
            if m['k_title_body'] is not None and m['k_title_body']>=3:
                ap.append(m['body_prefix_compound']-m['title_prefix_compound']);af.append(diff)
        if t['author_reply_eligible']:
            diff=t['author_mean']-t['combined_compound'];b.append(diff)
            ratio=m['mean_author_reply_words']/m['original_post_words']
            binname='le_0.25' if ratio<=.25 else '(0.25,0.5]' if ratio<=.5 else '(0.5,1]' if ratio<=1 else 'gt_1'
            bbins[binname].append(diff)
            if m['k_post_first_reply'] is not None and m['k_post_first_reply']>=3:
                first=min((c for c in comments[rid] if c['is_author']),key=lambda c:(c['chronological_index'],c['original_index']))
                bp.append(m['first_reply_prefix_compound']-m['post_prefix_compound'])
                bf.append(first['compound']-t['combined_compound'])
    assert len(a)==35443 and len(b)==17282 and len(ap)==33723 and len(bp)==16357
    for name,vals in abins.items():strata_check(vals,report['A_length']['word_ratio_bins'][name],'A '+name)
    for name,vals in bbins.items():strata_check(vals,report['B_length']['reply_over_original_word_ratio_bins'][name],'B '+name)
    for vals,block,label in ((ap,report['A_length']['prefix_diagnostic']['prefix'],'A prefix'),(af,report['A_length']['prefix_diagnostic']['full_text_on_same_subset'],'A full subset'),(bp,report['B_length']['prefix_diagnostic']['prefix'],'B prefix'),(bf,report['B_length']['prefix_diagnostic']['full_first_reply_on_same_subset'],'B first full')):
        mean,ci=paired_mean_ci(vals);check(mean,block['mean'],label);[check(ci[j],block['mean_ci95'][j],label+' CI') for j in (0,1)]
    # F: independently derive quintiles, fixed-weight standardization, OLS and CIs.
    f=[(rid,t,measures[rid]) for rid,t in threads.items() if t['f_eligible']]
    lengths=np.array([m['original_post_chars'] for _,_,m in f],float)
    cuts=np.unique(np.quantile(lengths,np.linspace(0,1,6)))
    for j,c in enumerate(cuts):check(c,report['F_length']['length_bins']['cuts'][j],'F cut')
    bins=[]
    for rid,t,m in f:
        chars=m['original_post_chars']
        k=next((j-1 for j in range(1,len(cuts)) if chars<=cuts[j]),len(cuts)-2)
        bins.append(k)
    bins=np.array(bins)
    y=np.array([t['combined_compound'] for _,t,_ in f],float)
    target=np.array([t['s2_target'] for _,t,_ in f],int)
    weights=[];diffs=[];bt=[];bo=[]
    for k in range(len(cuts)-1):
        idx=bins==k;tval=y[idx&(target==1)];oval=y[idx&(target==0)]
        row=report['F_length']['length_bins']['rows'][k]
        assert len(tval)==row['n_troubleshooting'] and len(oval)==row['n_other'] and len(tval)>=30 and len(oval)>=10
        d=float(tval.mean()-oval.mean());check(d,row['mean_diff'],'F bin diff')
        w=float(np.mean(idx));check(w,row['weight'],'F bin weight')
        weights.append(w);diffs.append(d)
    standardized=float(np.dot(weights,diffs))
    check(standardized,report['F_length']['pooled_length_adjusted']['mean_diff'],'F adjusted')
    X=np.column_stack([np.ones(len(f)),target,np.log1p(lengths),np.array([int(t['body_nonempty']) for _,t,_ in f]),np.array([int(t['created_month']=='2026-10') for _,t,_ in f]),np.array([int(t['count_gap']) for _,t,_ in f]),np.array([int(t['source_is_reddit']) for _,t,_ in f])])
    beta=np.linalg.lstsq(X,y,rcond=None)[0]
    check(beta[1],report['F_length']['ols_s2_target']['s2_target_coefficient'],'F OLS')
    # Match group-stratified resampling, with fixed original length bins and weights.
    ti=np.flatnonzero(target==1);oi=np.flatnonzero(target==0);rng=np.random.default_rng(SEED)
    std_boot=[]
    for _ in range(NBOOT):
        ts=ti[rng.integers(0,len(ti),len(ti))];os=oi[rng.integers(0,len(oi),len(oi))]
        std_boot.append(sum(weights[k]*(y[ts[bins[ts]==k]].mean()-y[os[bins[os]==k]].mean()) for k in range(len(weights))))
    std_ci=np.percentile(std_boot,[2.5,97.5])
    for j in (0,1):check(std_ci[j],report['F_length']['pooled_length_adjusted']['mean_diff_ci95'][j],'F adjusted CI')
    rng=np.random.default_rng(SEED);ols_boot=[]
    for _ in range(NBOOT):
        idx=np.concatenate((ti[rng.integers(0,len(ti),len(ti))],oi[rng.integers(0,len(oi),len(oi))]))
        ols_boot.append(np.linalg.lstsq(X[idx],y[idx],rcond=None)[0][1])
    ols_ci=np.percentile(ols_boot,[2.5,97.5])
    for j in (0,1):check(ols_ci[j],report['F_length']['ols_s2_target']['s2_target_ci95'][j],'F OLS CI')
    # C/D/E: independently form all declared subgroups and check their N and point estimates.
    crows=[];drows=[];erows=[]
    for rid,t in threads.items():
        gap='count_gap' if t['count_gap'] else 'no_gap';date='recent_oct2026' if t['created_month']=='2026-10' else 'older'
        if t['both_roles_eligible']:
            na=t['n_author_replies_nonempty'];nc=t['n_community_replies_nonempty']
            crows.append((t['community_mean']-t['author_mean'],gap,date,'author_'+('1' if na==1 else 'ge2')+'_community_'+('1' if nc==1 else 'ge2'),.5<=na/nc<=2))
        if t['order_eligible']:
            cc=sorted((v for v in comments[rid] if v['timestamp_valid']),key=lambda v:(v['timestamp_epoch_seconds'],v['original_index']))
            assert len(cc)>=2
            diff=cc[-1]['compound']-cc[0]['compound'];check(diff,t['last_minus_first'],'D thread')
            n=len(cc);nclass='2' if n==2 else '3' if n==3 else '4-7' if n<=7 else 'ge_8'
            drows.append((diff,gap,date,nclass,cc[0]['role']+'->'+cc[-1]['role']))
            if t['elapsed_eligible']:
                hours=(cc[-1]['timestamp_epoch_seconds']-cc[0]['timestamp_epoch_seconds'])/3600
                check(hours,t['elapsed_hours'],'E elapsed')
                dur='lt_1h' if hours<1 else 'from_1_to_24h' if hours<=24 else 'gt_24h'
                erows.append((hours,diff,gap,date,nclass,dur))
    assert len(crows)==17035 and len(drows)==24318 and len(erows)==24313
    cr=report['C_roles'];dr=report['D_order'];er=report['E_elapsed']
    for name,idx in (('count_gap',1),('no_gap',1),('older',2),('recent_oct2026',2)):
        strata_check([r[0] for r in crows if r[idx]==name],cr['count_gap_and_date'][name],'C '+name)
        strata_check([r[0] for r in drows if r[idx]==name],dr['count_gap_and_date'][name],'D '+name)
        group=[r for r in erows if r[idx+1]==name]
        assert len(group)==er['count_gap_and_date'][name]['n']
        check(rho([r[0] for r in group],[r[1] for r in group]),er['count_gap_and_date'][name]['spearman'],'E '+name)
    for name,block in cr['role_count_cells'].items():strata_check([r[0] for r in crows if r[3]==name],block,'C cell '+name)
    strata_check([r[0] for r in crows if r[4]],cr['balanced_role_ratio_0.5_to_2'],'C balanced')
    for name,block in dr['timed_reply_count_bins'].items():strata_check([r[0] for r in drows if r[3]==name],block,'D count '+name)
    for name,block in dr['first_last_role_pairs'].items():strata_check([r[0] for r in drows if r[4]==name],block,'D role '+name)
    for name,block in er['duration_bands'].items():strata_check([r[1] for r in erows if r[5]==name],block,'E duration '+name)
    for name,block in er['timed_reply_count_bins'].items():
        group=[r for r in erows if r[4]==name];assert len(group)==block['n']
        check(rho([r[0] for r in group],[r[1] for r in group]),block['spearman'],'E timed '+name)
    check(rho([r[0] for r in erows],[r[1] for r in erows]),er['overall_spearman']['spearman'],'E overall rho')
    c_mean,c_ci=paired_mean_ci([r[0] for r in crows])
    d_mean,d_ci=paired_mean_ci([r[0] for r in drows])
    for j in (0,1):
        check(c_ci[j],cr['overall']['mean_ci95'][j],'C overall CI')
        check(d_ci[j],dr['overall']['mean_ci95'][j],'D overall CI')
    hours=np.array([r[0] for r in erows],float);change=np.array([r[1] for r in erows],float)
    rng=np.random.default_rng(SEED);rho_boot=[]
    for _ in range(NBOOT):
        idx=rng.integers(0,len(hours),len(hours))
        rho_boot.append(rho(hours[idx],change[idx]))
    e_ci=np.percentile(rho_boot,[2.5,97.5])
    for j in (0,1):check(e_ci[j],er['overall_spearman']['spearman_ci95'][j],'E rho CI')
    assert er['timestamp_coverage']['valid_timestamped_nonempty_comments']==len(comments_all)==er['timestamp_coverage']['all_nonempty_saved_comments']
    check(sum(abs(r[0])>=.1 for r in drows)/len(drows),dr['overall']['share_abs_ge_0.10'],'D abs .1')
    check(sum(abs(r[0])>=.3 for r in drows)/len(drows),dr['overall']['share_abs_ge_0.30'],'D abs .3')
    # Public I output schema contains only aggregate numbers/definitions.
    forbidden={'title','body','text','username','author','url','description','snippet'}
    for name in ihash:
        path=I/name
        vals=rows(path) if path.suffix=='.jsonl' else [json.loads(path.read_text())]
        for obj in vals:
            stack=[obj]
            while stack:
                item=stack.pop()
                if isinstance(item,dict):
                    assert not(set(item)&forbidden), (name,set(item)&forbidden)
                    stack.extend(item.values())
                elif isinstance(item,list):stack.extend(item)
    output={'status':'passed','raw_pages_hashed':len(raw_manifest['pages']),'g_outputs_hashed':len(ghash),'i_outputs_hashed':len(ihash),'restricted_prefix_sha256':digest(MEASURES),'restricted_dir_mode':oct(MEASURES.parent.stat().st_mode & 0o777),'primary_counts':manifest['denominators'],'sample':sample,'A_prefix':paired_mean_ci(ap)[0],'A_full_same_subset':paired_mean_ci(af)[0],'B_prefix':paired_mean_ci(bp)[0],'B_first_full_same_subset':paired_mean_ci(bf)[0],'F_standardized':standardized,'F_standardized_ci95':std_ci.tolist(),'F_ols':float(beta[1]),'F_ols_ci95':ols_ci.tolist(),'C_mean':c_mean,'C_ci95':c_ci,'D_mean':d_mean,'D_ci95':d_ci,'E_rho':rho(hours,change),'E_ci95':e_ci.tolist()}
    dest=ROOT/'results'/'codex_independent_review_i_2026-10-28.json'
    dest.write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
