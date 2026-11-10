"""Offline SAM2 automatic proposals. Original RGB only; no image synthesis."""
import os,json,sys,time,argparse
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from pathlib import Path
import torch,numpy as np
from collections import deque
from PIL import Image
from transformers import Sam2Model
def main(config_file):
 c=json.loads(Path(config_file).read_text(encoding='utf-8'));out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(2);torch.set_num_interop_threads(1);start=time.monotonic()
 im=Image.open(c['input']).convert('RGBA');w,h=im.size
 if w*h>4000000:raise ValueError('사진이 너무 큽니다.')
 print(json.dumps({'phase':'모델 준비'}),flush=True);model=Sam2Model.from_pretrained(c['model'],local_files_only=True).eval()
 rgb=np.asarray(im.convert('RGB').resize((1024,1024),Image.Resampling.BILINEAR),dtype=np.float32)/255
 rgb=(rgb-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32);pixel=torch.from_numpy(rgb.transpose(2,0,1).copy())[None]
 side=12;points=[[(x+.5)*1024/side,(y+.5)*1024/side] for y in range(side) for x in range(side)];features=None;candidates=[]
 with torch.inference_mode():
  for offset in range(0,len(points),8):
   batch=points[offset:offset+8];p=torch.tensor([[ [point] for point in batch]],dtype=torch.float32);labels=torch.ones((1,len(batch),1),dtype=torch.int64)
   args={'pixel_values':pixel} if features is None else {'image_embeddings':features}
   result=model(**args,input_points=p,input_labels=labels,multimask_output=True);features=result.image_embeddings
   for i in range(len(batch)):
    for k in range(result.pred_masks.shape[2]):
     score=float(result.iou_scores[0,i,k]);logit=result.pred_masks[0,i,k].numpy();mask=logit>0;area=int(mask.sum())
     if score<.87 or area<256*256*.008 or area>256*256*.8:continue
     stability=float((logit>1).sum()/max(1,(logit>-1).sum()))
     if stability<.92:continue
     yy,xx=np.nonzero(mask);edges=int(xx.min()==0)+int(yy.min()==0)+int(xx.max()==255)+int(yy.max()==255)
     if edges>=2:continue
     candidates.append({'score':score,'stability':stability,'area':area,'low':logit.copy(),'mask':mask,'point':batch[i]})
   print(json.dumps({'phase':'물체 후보 분석','progress':min(100,round((offset+len(batch))/len(points)*100))}),flush=True)
 candidates.sort(key=lambda x:x['score'],reverse=True);kept=[]
 for candidate in candidates:
  duplicate=False
  for other in kept:
   intersection=np.logical_and(candidate['mask'],other['mask']).sum();union=np.logical_or(candidate['mask'],other['mask']).sum()
   if intersection/max(1,union)>.68:duplicate=True;break
  if not duplicate:kept.append(candidate)
 # Suppress texture fragments wholly contained in a confident larger object.
 chosen=[]
 for candidate in sorted(kept,key=lambda x:x['area'],reverse=True):
  if any(np.logical_and(candidate['mask'],o['mask']).sum()/max(1,candidate['area'])>.92 for o in chosen):continue
  chosen.append(candidate)
 chosen=chosen[:8];rgba=np.array(im);objects=[]
 for i,item in enumerate(chosen):
  mask=item['mask'];seen=np.zeros_like(mask);minimum=max(4,int(mask.sum()*.005))
  for yy,xx in zip(*np.nonzero(mask)):
   if seen[yy,xx]:continue
   component=[];queue=deque([(yy,xx)]);seen[yy,xx]=True
   while queue:
    y,x=queue.popleft();component.append((y,x))
    for ny,nx in [(y-1,x),(y+1,x),(y,x-1),(y,x+1)]:
     if 0<=ny<256 and 0<=nx<256 and mask[ny,nx] and not seen[ny,nx]:seen[ny,nx]=True;queue.append((ny,nx))
   if len(component)<minimum:
    for y,x in component:item['low'][y,x]=-100
  logit=torch.from_numpy(item['low'])[None,None];mask=torch.nn.functional.interpolate(logit,size=(h,w),mode='bilinear',align_corners=False)[0,0].numpy()>0
  alpha=np.where(mask,rgba[:,:,3],0).astype(np.uint8);image=Image.fromarray(np.dstack([rgba[:,:,:3],alpha]));name=f'object-{i+1}.png';image.save(out/name);ys,xs=np.nonzero(alpha>24)
  objects.append({'file':name,'name':f'물체 {i+1}','score':item['score'],'stability':item['stability'],'bbox':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]})
 report={'objects':objects,'seconds':time.monotonic()-start,'size':[w,h],'model':'SAM2.1 tiny','method':'automatic-grid-proposals','generatedRgb':False}
 (out/'result.json').write_text(json.dumps(report,ensure_ascii=False),encoding='utf-8');print(json.dumps({'phase':'완료','objects':len(objects)}),flush=True)
if __name__=='__main__':main(sys.argv[1])
