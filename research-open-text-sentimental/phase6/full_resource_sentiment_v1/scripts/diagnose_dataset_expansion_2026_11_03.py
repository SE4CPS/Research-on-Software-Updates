import csv,json,hashlib,random,collections,pathlib,math,datetime,os
B=pathlib.Path('/Users/manumathewjiss/Documents/Research-on-Software-Updates/research-open-text-sentimental/phase6/full_resource_sentiment_v1')
O=B/'planning/dataset_expansion_2026-11-03';O.mkdir(parents=True,exist_ok=True)
R=B/'results/dataset_forensic_audit_2026-11-03/FULL_RECORD_RECONCILIATION.csv'
S=B/'results/broad_sentiment_2026-10-22-main/post_scores.jsonl'
P=B/'restricted/relevance_audit_sample_2026-10-28/private_id_key.jsonl'
def rows_json(p):
 with p.open() as f:
  for l in f:
   if l.strip():yield json.loads(l)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
scores={str(x['reddit_id']):x for x in rows_json(S)}
with R.open() as f: rows=list(csv.DictReader(f))
old=set()
for x in rows_json(P):
 for k in ('reddit_id','post_id'):
  if x.get(k):old.add(str(x[k]))
excluded=[x for x in rows if x['in_broad_37596']=='1' and x['in_current_6425']=='0']
elig=[x for x in excluded if x['protected_id']=='0']
assert len(excluded)==31691 and len(elig)==30998
for x in elig:
 s=scores[x['reddit_id']]
 x['_period']='october' if str(s.get('created_month','')).startswith('2026-10') else 'older'
 x['_rich']=int(x['n_author_replies'])>=4
 x['_label']=x['broad_relevance_prediction'] if x['broad_prediction_status']=='ok' else 'error'
 x['_exposed']=x['reddit_id'] in old or x['in_phase2_master_3362']=='1'
counts=collections.Counter((x['_label'],x['_period'],x['_rich'],x['_exposed']) for x in elig)
lab=collections.Counter(x['_label'] for x in elig)
period=collections.Counter(x['_period'] for x in elig)
rich=collections.Counter((x['_label'],x['_rich']) for x in elig)
comment=collections.Counter()
for x in elig:
 n=int(x['n_author_replies']);c=int(x['n_comments_nonempty']);t=int(x['n_timed_replies']);
 for name,flag in [('comments_ge1',c>=1),('author_ge1',n>=1),('author_ge2',n>=2),('author_ge4',n>=4),('author_ge8',n>=8),('community_ge1',int(x['n_community_replies'])>=1),('timed_ge2',t>=2),('both_roles',n>=1 and int(x['n_community_replies'])>=1),('title',x['title_available']=='1'),('body',x['body_available']=='1')]:
  if flag:comment[name]+=1
# deterministic fresh probability draw, oversampling rare prediction and rich-reply cells; this is an ID-only manifest, not a review packet.
fresh=[x for x in elig if not x['_exposed']]
assert len(fresh)==30401
by=collections.defaultdict(list)
for x in fresh:by[(x['_label'],x['_period'],x['_rich'])].append(x['reddit_id'])
keys=sorted(by); target=240; mins={k:min(3,len(by[k])) for k in keys}; remain=target-sum(mins.values())
weights={k:math.sqrt(len(by[k])) for k in keys};total=sum(weights.values()); raw={k:remain*weights[k]/total for k in keys};n={k:mins[k]+min(len(by[k])-mins[k],int(raw[k])) for k in keys}
while sum(n.values())<target:
 possible=[k for k in keys if n[k]<len(by[k])]
 k=max(possible,key=lambda k:(raw[k]-int(raw[k]),len(by[k]),k));n[k]+=1;raw[k]-=1
rng=random.Random(20261103); selected=[]
for k in keys:
 ids=sorted(by[k]); chosen=rng.sample(ids,n[k])
 for i in chosen:selected.append({'reddit_id':i,'prediction_stratum':k[0],'period':k[1],'author_replies_ge4':k[2],'stratum_N':len(ids),'stratum_n':n[k],'inclusion_probability':n[k]/len(ids),'inverse_probability_weight':len(ids)/n[k]})
assert len(selected)==240 and len({x['reddit_id'] for x in selected})==240
assert all(rr['protected_id']=='0' and rr['in_current_6425']=='0' for rr in (next(x for x in fresh if x['reddit_id']==y['reddit_id']) for y in selected))
priv=O/'restricted';priv.mkdir(exist_ok=True);os.chmod(priv,0o700)
with (priv/'PILOT_SELECTED_IDS.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(selected[0]));w.writeheader();w.writerows(sorted(selected,key=lambda z:z['reddit_id']))
strata=[{'prediction':k[0],'period':k[1],'author_replies_ge4':k[2],'N':len(by[k]),'n':n[k],'inclusion_probability':n[k]/len(by[k]),'weight':len(by[k])/n[k]} for k in keys]
with (O/'PILOT_STRATA.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(strata[0]));w.writeheader();w.writerows(strata)
nine=['1lrpn7u','1lsuvac','1wrcc8z','1wrfhbj','1wrmi5d','1wrqch6','1wrt20o','1wrxo3u','1wsw8jg']
rr={x['reddit_id']:x for x in rows};nineout=[]
for i in nine:
 x=rr[i];nineout.append({k:x[k] for k in ['reddit_id','in_broad_37596','in_current_6425','protected_id','broad_relevance_prediction','title_available','body_available','n_comments_nonempty','n_author_replies','n_timed_replies']})
with (O/'NINE_OMITTED_CASES.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(nineout[0]));w.writeheader();w.writerows(nineout)
q={'sources':{'reconciliation':str(R),'sha256':sha(R),'scores':str(S),'scores_sha256':sha(S)},'counts':{'broad_excluded':len(excluded),'nonprotected_excluded':len(elig),'fresh_nonprotected_excluded':len(fresh),'prior_exposed_nonprotected_excluded':len(elig)-len(fresh),'prediction':dict(lab),'period':dict(period),'conversation':dict(comment),'rich_by_prediction':{f'{a}:{b}':v for (a,b),v in rich.items()},'strata':strata},'nine':nineout,'pilot':{'seed':20261103,'n':len(selected),'stratum_definition':'prediction x period x author replies >=4; prior-exposed excluded','selected_csv_sha256':sha(priv/'PILOT_SELECTED_IDS.csv')}}
(O/'DIAGNOSTICS.json').write_text(json.dumps(q,indent=2)+'\n')
print(json.dumps(q['counts'],indent=2))
