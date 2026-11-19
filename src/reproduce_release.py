"""Portable reproduction from derived subsystem counts; no R12/R13 access."""
from pathlib import Path
import csv,json,math,itertools
from scoped_sampling import minimax_stratified_plan_w0,miss_probability,unacceptable_count
from scoped_sampling_r15 import eval_adaptive_2stage_r15
ROOT=Path(__file__).resolve().parents[1]
def read(name):return list(csv.DictReader((ROOT/'inputs'/name).open(encoding='utf-8')))
def metrics(ns,ws):
 N,W=sum(ns),sum(ws);w0=unacceptable_count(N,.05);plan,risk=minimax_stratified_plan_w0(ns,w0,.05)
 I=sum(n+(1-miss_probability(nf,wf,n))*(nf-n) for nf,wf,n in zip(ns,ws,plan))
 mixed=sum(n+(1-miss_probability(N,W,n))*(nf-n) for nf,n in zip(ns,plan))
 ad=eval_adaptive_2stage_r15(ns,ws,tuple(0 for _ in ns),w0,.045,.05)
 return dict(N=N,W=W,plan='/'.join(map(str,plan)),risk=risk,actual=I,mixed=mixed,adaptive=ad['inspections'])
def main():
 sim=[]
 for r in read('frozen_subsystem_counts.csv'):
  m=metrics(tuple(map(int,r['ns'].split('/'))),tuple(map(int,r['ws'].split('/'))));sim.append(dict(r,**m))
 output={}
 for name,rr in [('clean27',sim),('real13',[r for r in sim if r['real_holdout']=='1'])]:
  N=sum(r['N'] for r in rr);output[name]={'reviews':len(rr),'N':N,**{k:sum(r[k] for r in rr)/N for k in ('actual','mixed','adaptive')}}
 expected={'clean27':(.44356728768090065,.7640150534067485,.480404528678201),'real13':(.42063477742436794,.8021142941042406,.4701478458001032)}
 for name,v in expected.items():
  for k,e in zip(('actual','mixed','adaptive'),v):assert abs(output[name][k]-e)<1e-10
 empirical=[]
 for r in read('empirical_subsystem_counts.csv'):
  m=metrics(tuple(map(int,r['ns'].split('/'))),tuple(map(int,r['ws'].split('/'))));empirical.append(dict(rulebook=r['rulebook'],checker=r['checker'],**m))
 checks=0
 for ns in ((2,3),(1,2,3)):
  N=sum(ns)
  for W in range(N+1):
   for plan in itertools.product(*(range(x+1) for x in ns)):
    def p0(n,w,m):return math.comb(n-w,m)/math.comb(n,m) if m<=n-w else 0.
    closed=sum(m+(1-p0(N,W,m))*(n-m) for n,m in zip(ns,plan));total=0.
    for bad in itertools.combinations(range(N),W):
     start=0
     for n,m in zip(ns,plan):
      w=sum(start<=j<start+n for j in bad);total+=m+(1-p0(n,w,m))*(n-m);start+=n
    assert abs(total/math.comb(N,W)-closed)<1e-10;checks+=1
 q=.1;reused=sum(math.comb(4,r)*q**r*(1-q)**(4-r) for r in range(2,5));fresh=q**4+sum(math.comb(4,r)*q**(2*r)*(1-q)**(4-r) for r in (2,3));assert reused>.05 and fresh<.05
 output.update(empirical=empirical,enumerated_configurations=checks,fallible_reuse_risk=reused,fresh_second_stage_risk=fresh)
 (ROOT/'outputs').mkdir(exist_ok=True);(ROOT/'outputs/reproduced_results.json').write_text(json.dumps(output,indent=2))
 print('PASS: both cohorts, six authored-property batches, 240 enumerations and fallible-reference counterexample')
if __name__=='__main__':main()
