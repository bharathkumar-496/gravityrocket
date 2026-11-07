"""Four clips, one labelled 1080x1080 MP4 for LinkedIn. Short clips hold their last frame."""
import imageio.v2 as iio, numpy as np
from PIL import Image, ImageDraw, ImageFont
P=[("videos/1_random.gif","1. Untrained","random actions: crash",(224,72,47)),
   ("videos/2_landed.gif","2. Trained","built-in reward: lands",(27,170,125)),
   ("videos/3_hover.gif","3. My bad rule","+1 per steady frame: never lands",(224,72,47)),
   ("videos/4_fixed.gif","4. Fixed","pay for change, charge for time",(27,170,125))]
clips=[[f[...,:3] for f in iio.mimread(p[0],memtest=False)] for p in P]
N=max(map(len,clips))
F=lambda s:ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf",s)
F2=lambda s:ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf",s)
W=540;H=540;vh=360
w=iio.get_writer("videos/rocket_story.mp4",fps=25,codec="libx264",macro_block_size=1,quality=8,pixelformat="yuv420p")
for i in list(range(N))+[N-1]*25:
    c=Image.new("RGB",(1080,1080),(16,18,24)); d=ImageDraw.Draw(c)
    for k,(p,t,s,col) in enumerate(P):
        x,y=(k%2)*W,(k//2)*H
        f=clips[k][min(i,len(clips[k])-1)]
        c.paste(Image.fromarray(f).resize((W-20,vh)),(x+10,y+100))
        d.text((x+14,y+18),t,font=F(32),fill=(255,255,255)); d.text((x+14,y+60),s,font=F2(22),fill=col)
    w.append_data(np.array(c))
w.close()
