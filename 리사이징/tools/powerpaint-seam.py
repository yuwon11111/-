"""Multiscale texture join. Original alpha, outer core and source backup stay exact."""
import numpy as np
from PIL import Image,ImageFilter
def seam_transition(before,context,fill,context_box):
 h,w=fill.shape
 if before.shape!=(h,w,4) or context.shape!=(h,w,3):raise ValueError('Invalid seam image dimensions')
 width=max(8,min(96,round(min(h,w)*.125)))
 reach=Image.fromarray((fill*255).astype('uint8'));distance=np.full((h,w),width+1,np.float32);distance[fill]=0
 for step in range(1,width+1):
  reach=reach.filter(ImageFilter.MaxFilter(3));new=(np.array(reach)>0)&(distance>width);distance[new]=step
 valid=np.zeros((h,w),bool);x0,y0,x1,y1=context_box;valid[max(0,y0):min(h,y1),max(0,x0):min(w,x1)]=True
 if np.any(fill&~valid):raise ValueError('Generated region outside context')
 foreground=before[:,:,3]>0;region=valid&(foreground|fill)&(distance<=width)
 # Texture uses a close transition; progressively coarser light/colour bands
 # use wider transitions, avoiding duplicated crust ridges from plain feathering.
 feather=max(4,min(16,width//2));alpha=np.clip((width-distance)/feather,0,1);alpha[fill]=1;alpha*=valid&(foreground|fill)
 base=before[:,:,:3].astype(np.float32);generated=context.astype(np.float32)
 # Missing RGB must not act as black during Gaussian filtering near the silhouette.
 base[~foreground]=generated[~foreground]
 def blur(a,r):return np.array(Image.fromarray(np.clip(a,0,255).astype('uint8')).filter(ImageFilter.GaussianBlur(r)),dtype=np.float32)
 result=np.zeros_like(base);levels=[0,1,2,4,8,16,32];mask_image=Image.fromarray(np.rint(alpha*255).astype('uint8'))
 for i,r in enumerate(levels):
  a=base if r==0 else blur(base,r);g=generated if r==0 else blur(generated,r)
  if i<len(levels)-1:a=a-blur(base,levels[i+1]);g=g-blur(generated,levels[i+1])
  weight=np.array(mask_image.filter(ImageFilter.GaussianBlur(max(1,r))),dtype=np.float32)/255
  result+=g*weight[:,:,None]+a*(1-weight[:,:,None])
 after=before.copy();after[region,:3]=np.clip(np.rint(result[region]),0,255).astype('uint8');after[fill,3]=255
 if not np.array_equal(after[~region],before[~region]):raise ValueError('Outside seam changed')
 if not np.array_equal(after[:,:,3][~fill],before[:,:,3][~fill]):raise ValueError('Original alpha changed')
 changed=int(np.any(after[foreground]!=before[foreground],axis=1).sum())
 return after,region,{'seamAdjustedPixels':changed,'seamWidth':width,'seamTechnique':'multiscale-texture'}
