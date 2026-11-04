from pathlib import Path
import csv,collections,hashlib,json
B=Path(__file__).resolve().parents[1]
P=B/'results/tps_gds_full_ai_dataset_2026-11-02/full_dataset.csv'
O=B/'planning/master_research_2026-11-03/INVENTORY_DIAGNOSTICS.json'
with P.open(newline='') as f: rows=list(csv.DictReader(f))
assert len(rows)==6425 and len({r['reddit_id'] for r in rows})==6425
by_source=collections.Counter((r['final_relevance'],r['final_discussion'],r['decision_source']) for r in rows)
reply={}
for label in ['TPS','GDS','Other','Unclear']:
 q=[r for r in rows if r['final_relevance']=='relevant' and r['final_discussion']==label]
 reply[label]={'n':len(q),'author_ge1':sum(int(r['n_author_replies_nonempty'] or 0)>=1 for r in q),'author_ge4':sum(int(r['n_author_replies_nonempty'] or 0)>=4 for r in q),'community_ge1':sum(int(r['n_community_replies_nonempty'] or 0)>=1 for r in q)}
result={'input':str(P),'input_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'unique_rows':len(rows),'counts_by_relevance_purpose_source':[{'relevance':a,'purpose':b,'source':c,'n':v} for (a,b,c),v in sorted(by_source.items())],'reply_coverage':reply,'interpretation':'Availability and recorded decision provenance only; not validation or population prevalence.'}
O.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(reply,indent=2))
