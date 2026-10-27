"""Additional checks of section A/C: opening streaks against realised outcomes (A9), flow -> outcome stacked.

Sortie : results/deep_C_extra.json. Usage : python -m analysis.deep_extra
"""
import numpy as np, json
from analysis import blockboot as bb
from analysis.deep import Ctx, boot_stat, logit, RNG
from analysis.run import irls, cluster_boot_logit
c=Ctx(); D=c.D; y=c.y
st=D["start_s"]; consec=np.r_[False,np.diff(st)==300]
streak=np.zeros(c.n)
for i in range(3,c.n):
    if consec[i] and consec[i-1] and consec[i-2] and y[i-1]==y[i-2]==y[i-3]: streak[i]=1 if y[i-1]==1 else -1
out={}
for name,msk in c.splits():
    r={}
    for lab,s in (("after_3_up",1),("after_3_down",-1),("other",0)):
        sel=msk&consec&(streak==s)
        pm=np.nanmean(D["pm"][:,10:40],1); P=np.nanmean(D["P"][:,10:40],1)
        r[lab]={"n":int(sel.sum()),"pm_open":boot_stat(pm[sel]),"model_open":boot_stat(P[sel]),"y":boot_stat(y[sel]),"pm_minus_y":boot_stat((pm-y)[sel])}
    # regression: y ~ logit(pm_open) + streak dummies (does price fully account?)
    sel=msk&consec&np.isfinite(np.nanmean(D["pm"][:,10:40],1))
    X=np.column_stack([np.ones(sel.sum()), logit(np.nanmean(D["pm"][:,10:40],1)[sel]), (streak[sel]==1).astype(float),(streak[sel]==-1).astype(float)])
    beta,draws=cluster_boot_logit(X,y[sel],np.nonzero(sel)[0],b=300)
    r["logit_y_on_price_and_streak"]={"beta":beta.tolist(),"lo":np.quantile(draws,.025,0).tolist(),"hi":np.quantile(draws,.975,0).tolist(),"n":int(sel.sum())}
    if name=="test":
        # block bootstraps (UTC day, UTC hour) of the same logit
        for unit in bb.UNITS:
            keys=bb.block_keys(st[sel],unit)
            _,dr=cluster_boot_logit(X,y[sel],keys,b=300,rng=bb.RNG)
            r["logit_y_on_price_and_streak"][unit]={**bb.ci(dr),"n_groups":int(len(np.unique(keys)))}
    out[name]=r
    print(name, {k:(v["n"], round(v["pm_open"]["est"],4), round(v["model_open"]["est"],4), round(v["y"]["est"],4), (round(v["pm_minus_y"]["est"],4),round(v["pm_minus_y"]["lo"],4),round(v["pm_minus_y"]["hi"],4))) for k,v in r.items() if k!="logit_y_on_price_and_streak"})
    print("  logit", np.round(beta,3), np.round(np.quantile(draws,.025,0),3), np.round(np.quantile(draws,.975,0),3))
# flow -> outcome, stacked on the price
flow=D["flow_usd"]/1000
for name,msk in c.splits():
    ts=np.arange(30,271,30); rows=[]
    for t in ts:
        cf=flow[msk][:,t-29:t+1].sum(1); cf=np.sign(cf)*np.log1p(np.abs(cf))
        rows.append(np.column_stack([np.ones(msk.sum()), c.x[msk][:,t], cf, np.nonzero(msk)[0]]))
    A=np.vstack(rows); ok=np.all(np.isfinite(A),1)
    beta,draws=cluster_boot_logit(A[ok][:,:3], np.tile(y[msk],len(ts))[ok], A[ok][:,3].astype(int), b=300)
    out[name]["flow_pooled"]={"beta":beta.tolist(),"lo":np.quantile(draws,.025,0).tolist(),"hi":np.quantile(draws,.975,0).tolist()}
    print(name,"flow pooled", np.round(beta,3), np.round(np.quantile(draws,.025,0),3), np.round(np.quantile(draws,.975,0),3))
json.dump(out,open("results/deep_C_extra.json","w"),indent=1,default=float)
