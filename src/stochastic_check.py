"""Training samples actions randomly from the policy; evaluation takes the top choice.
Compare both, on the plain game, to explain any gap between training and test scores."""
import gymnasium as gym, numpy as np
from stable_baselines3 import PPO
for rule in ["hover_fix","hover_fix_2m"]:
  for det in [True,False]:
    sc=[];fr=[];to=[]
    for s in [1,2,3]:
        m=PPO.load(f"models/{rule}_seed{s}",device="cpu"); e=gym.make("LunarLander-v3")
        for ep in range(50):
            o,_=e.reset(seed=20000+ep); d=False;t=0;tot=0
            while not d:
                o,r,a,b,_=e.step(int(m.predict(o,deterministic=det)[0])); d=a or b;t+=1;tot+=r
            sc.append(tot);fr.append(t);to.append(b)
    print(f"{rule:10} {'top choice' if det else 'sampled':10} score {np.mean(sc):6.1f} frames {np.mean(fr):6.1f} timeout {100*np.mean(to):4.1f}%")
