"""Is the hack rocket actually airborne? Track height and leg contact across attempts."""
import gymnasium as gym, numpy as np
from stable_baselines3 import PPO
for name in ["hover_hack_seed1","hover_hack_seed2","hover_hack_seed3"]:
    m=PPO.load(f"models/{name}",device="cpu"); e=gym.make("LunarLander-v3")
    ys=[];legs=[];steady=[];xs=[]
    for ep in range(20):
        o,_=e.reset(seed=10000+ep); d=False; t=0
        while not d:
            o,r,a,b,_=e.step(int(m.predict(o,deterministic=True)[0])); d=a or b; t+=1
            if t>200: ys.append(o[1]);legs.append(o[6]+o[7]>0);xs.append(abs(o[0]));steady.append(abs(o[4])<.2 and max(abs(o[2]),abs(o[3]))<.2)
    print(name,"height mean %.2f min %.2f max %.2f | any leg down %.0f%% | steady %.0f%% | |x| %.2f"%(np.mean(ys),np.min(ys),np.max(ys),100*np.mean(legs),100*np.mean(steady),np.mean(xs)))
