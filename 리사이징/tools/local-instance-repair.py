"""Generate only an explicit transparent occlusion mask, using an existing local model."""
from pathlib import Path
import os,sys,json,time
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import numpy as np
from PIL import Image
c=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'));out=Path(c['output']);im=Image.open(c['input']).convert('RGBA');mask=Image.open(c['mask']).convert('L')
if mask.size!=im.size:raise ValueError('생성 영역과 그림 크기가 다릅니다.')
before=np.array(im);allowed=(np.array(mask)>128)&(before[:,:,3]==0);mask=Image.fromarray((allowed*255).astype(np.uint8));box=mask.getbbox()
if not box or allowed.sum()>im.width*im.height*.3:raise ValueError('확실한 가려진 영역을 지정하세요. 전체 그림을 생성하지 않습니다.')
mask.save(out/'generated-mask.png');size=max(512,box[2]-box[0]+128,box[3]-box[1]+128);size=min(size,max(im.size));cx=(box[0]+box[2])//2;cy=(box[1]+box[3])//2;x=max(0,min(im.width-size,cx-size//2));y=max(0,min(im.height-size,cy-size//2));roi=(x,y,x+size,y+size)
matte=c.get('matte','#5f4b3d');condition=Image.new('RGBA',im.size,matte);condition.alpha_composite(im);condition=condition.convert('RGB').crop(roi).resize((256,256));cropmask=mask.crop(roi).resize((256,256))
hint=str(c.get('hint','same original product'))[:300];prompt='one single '+hint+', reconstruct only the missing part, matching material and texture, professional product photography, same lighting'
print(json.dumps({'phase':'로컬 생성 모델 준비'}),flush=True)
import torch
from diffusers import StableDiffusionInpaintPipeline,DPMSolverMultistepScheduler
torch.set_num_threads(4);torch.set_num_interop_threads(1);start=time.monotonic()
pipe=StableDiffusionInpaintPipeline.from_pretrained(c['inpaintModel'],variant='fp16',torch_dtype=torch.float32,use_safetensors=True,local_files_only=True);pipe.scheduler=DPMSolverMultistepScheduler.from_config(pipe.scheduler.config);pipe.disable_attention_slicing();pipe.vae.enable_slicing();pipe.set_progress_bar_config(disable=True)
steps=8
def progress(pipe,i,t,kwargs):print(json.dumps({'phase':'가려진 부분 보완','progress':round((i+1)/steps*100)}),flush=True);return kwargs
response=pipe(prompt=prompt,negative_prompt='text, logo, extra object, duplicate object, butter, plastic, deformed',image=condition,mask_image=cropmask,width=256,height=256,num_inference_steps=steps,guidance_scale=7.5,generator=torch.Generator('cpu').manual_seed(int(c.get('seed',20261101))),callback_on_step_end=progress)
if any(response.nsfw_content_detected or []):raise ValueError('사용할 수 있는 생성 결과를 얻지 못했습니다.')
patch=response.images[0].resize((size,size),Image.Resampling.LANCZOS);generated=Image.new('RGBA',im.size);generated.paste(patch,(x,y));generated.putalpha(mask);result=im.copy();result.alpha_composite(generated);result.save(out/'restored.png');after=np.array(result);protected=before[:,:,3]>0;changed=int(np.any(before[:,:,:3][protected]!=after[:,:,:3][protected],axis=1).sum())
if changed:raise ValueError('보호 영역이 달라져 결과를 적용하지 않았습니다.')
report={'image':'restored.png','mask':'generated-mask.png','objects':[],'protectedRgbChanged':changed,'generatedPixels':int(allowed.sum()),'seconds':time.monotonic()-start,'method':'local-masked-inpainting','prompt':prompt,'warning':'가려진 부분은 추정 생성입니다. 원본 복원이 아닙니다.'};(out/'result.json').write_text(json.dumps(report,ensure_ascii=False),encoding='utf-8');print(json.dumps({'phase':'보완 후보 준비 완료'}),flush=True)
