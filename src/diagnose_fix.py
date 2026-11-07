"""Where do hover_fix rockets sit when attempts time out under sampled actions?"""
import gymnasium as gym, numpy as np, sys
from stable_baselines3 import PPO
for s in [1,2,3]:
    m=PPO.load(f"models/{sys.argv[1] if len(sys.argv)>1 else 'hover_fix'}_seed{s}",device="cpu"); e=gym.make("LunarLander-v3")
    res=[];
    for ep in range(40):
        o,_=e.reset(seed=30000+ep); d=False;t=0;tot=0;tail=[]
        while not d:
            o,r,a,b,_=e.step(int(m.predict(o,deterministic=False)[0])); d=a or b;t+=1;tot+=r
            if t>=800: tail.append((o[1],o[6]+o[7],abs(o[0])))
        res.append((b,tot,t,np.array(tail).mean(0) if tail else (np.nan,)*3))
    to=[x for x in res if x[0]]
    print(f"seed{s}: timeouts {len(to)}/40, frames {np.mean([x[2] for x in res]):.0f}", end=" ")
    if to:
        T=np.array([x[3] for x in to]); print(f"| timeout score {np.mean([x[1] for x in to]):.0f}, height {T[:,0].mean():.2f}, legs {T[:,1].mean():.2f}, |x| {T[:,2].mean():.2f}")
    else: print()
