"""Offline PowerPaint V1 repair worker; isolated official modules and dependencies.
Only the explicit transparent region and a narrow transparent seam may change.
"""
from pathlib import Path
import os,sys,json,time,hashlib,shutil,threading
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',OMP_NUM_THREADS='3',MKL_NUM_THREADS='3')
import numpy as np
from PIL import Image,ImageFilter
import importlib.util
_spec=importlib.util.spec_from_file_location('powerpaint_seam',Path(__file__).with_name('powerpaint-seam.py'));_seam=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_seam)
def save_transition(before,context,fill,box):
 joined,region,info=_seam.seam_transition(before,context,fill,box)
 Image.fromarray(joined).save(out/'seam-joined.png');Image.fromarray((region*255).astype('uint8')).save(out/'seam-region.png')
 return dict(seamImage='seam-joined.png',seamMask='seam-region.png',**info)
c=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'));out=Path(c['output']);runtime=Path(c['powerpaintRuntime']);start=time.monotonic()
def note(phase,progress=None):
 d={'phase':phase}
 if progress is not None:d['progress']=progress
 print(json.dumps(d),flush=True)
def fail(message):
 (out/'failure.json').write_text(json.dumps({'error':message},ensure_ascii=False),encoding='utf-8');raise ValueError(message)
im=Image.open(c['input']).convert('RGBA');before=np.array(im);mask=Image.open(c['mask']).convert('L')
if mask.size!=im.size:fail('그림과 보완 영역의 크기가 다릅니다.')
allowed=(np.array(mask)>128)&(before[:,:,3]==0);mask=Image.fromarray((allowed*255).astype('uint8'));box=mask.getbbox()
if not box or allowed.sum()>im.width*im.height*.3:fail('가려진 투명 영역만 지정하세요. 원본은 유지합니다.')
hint=' '.join(str(c.get('hint','same original product')).strip().lower().split())[:300]
source_hash=hashlib.sha256(before.tobytes()).hexdigest();mask_hash=hashlib.sha256(allowed.tobytes()).hexdigest()
manifest=runtime/'cache/approved.json'
if not c.get('fresh') and manifest.exists():
 cached=json.loads(manifest.read_text(encoding='utf-8'))
 if cached.get('sourceRGBA')==source_hash and cached.get('maskPixels')==mask_hash and cached.get('size')==list(im.size) and cached.get('hint')==hint:
  candidate=Image.open(runtime/'cache/approved.png').convert('RGBA');a=np.array(candidate);m=np.array(Image.open(runtime/'cache/approved-mask.png').convert('L'))>128
  if candidate.size!=im.size or (m&(before[:,:,3]>0)).any() or not np.array_equal(a[~m],before[~m]):fail('저장된 후보의 원본 보존 검증을 통과하지 못했습니다.')
  shutil.copy2(runtime/'cache/approved.png',out/'restored.png');shutil.copy2(runtime/'cache/approved-mask.png',out/'generated-mask.png')
  result={'image':'restored.png','mask':'generated-mask.png','objects':[],'protectedRgbChanged':0,'generatedPixels':int(m.sum()),'seamPixels':int((m&~allowed).sum()),'method':'powerpaint-v1-shape-inpainting','cached':True,'seconds':time.monotonic()-start,'seed':cached['seed']}
  if cached.get('contextSHA256') and (runtime/'cache/context.png').exists():
   context_path=runtime/'cache/context.png'
   if hashlib.sha256(context_path.read_bytes()).hexdigest()!=cached['contextSHA256']:fail('경계 보정 자료가 일치하지 않습니다.')
   result.update(save_transition(before,np.array(Image.open(context_path).convert('RGB')),m,cached['contextBox']))
  result['seconds']=time.monotonic()-start
  (out/'result.json').write_text(json.dumps(result),encoding='utf-8');note('동일 사진의 확인된 PowerPaint 후보 준비 완료',100);sys.exit(0)
try:
 import psutil
except ImportError:
 fail('로컬 생성에 필요한 메모리 검사 구성 요소가 없습니다. 분석 환경을 확인하세요. 원본은 유지합니다.')
if psutil.virtual_memory().available<2000*1024**2:fail('로컬 생성에 사용할 여유 RAM이 부족합니다. 다른 앱을 정리한 뒤 다시 실행하세요.')
def monitor():
 while True:
  time.sleep(5)
  if sys.platform=='win32' and psutil.virtual_memory().available<800*1024**2:
   import ctypes
   ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p;trim=ctypes.windll.psapi.EmptyWorkingSet;trim.argtypes=[ctypes.c_void_p];trim(ctypes.windll.kernel32.GetCurrentProcess());time.sleep(.2)
  if psutil.virtual_memory().available<450*1024**2 or time.monotonic()-start>1100:
   (out/'failure.json').write_text(json.dumps({'error':'메모리 부족 또는 실행 시간 제한으로 생성을 중지했습니다. 원본은 유지합니다.'},ensure_ascii=False),encoding='utf-8');os._exit(3)
threading.Thread(target=monitor,daemon=True).start();note('PowerPaint 로컬 모델 준비 · 수 분 걸릴 수 있습니다.',0)
sys.path[:0]=[str(runtime/'deps'),str(runtime/'repo')]
import torch
from safetensors import safe_open
from accelerate import init_empty_weights
from transformers import CLIPTextConfig,CLIPTextModel,CLIPTokenizer
from diffusers import UNet2DConditionModel
from powerpaint.pipelines.pipeline_PowerPaint import StableDiffusionInpaintPipeline as Pipeline
from powerpaint.utils.utils import TokenizerWrapper,add_tokens
torch.set_num_threads(3);torch.set_num_interop_threads(1);torch.set_grad_enabled(False);base=Path(c['inpaintModel']);weights=runtime/'model';dtype=torch.float16
tokenizer=TokenizerWrapper(from_pretrained=str(base/'tokenizer'),local_files_only=True)
with init_empty_weights():
 text=CLIPTextModel(CLIPTextConfig.from_pretrained(str(base/'text_encoder'),local_files_only=True));unet=UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(str(base/'unet')))
 add_tokens(tokenizer,text,['P_ctxt','P_shape','P_obj'],initialize_tokens=['a','a','a'],num_vectors_per_token=10)
def mapped_load(model,path):
 expected=set(model.state_dict());loaded=set()
 with safe_open(str(path),framework='pt',device='cpu') as f:
  for key in f.keys():
   if key.endswith('position_ids') and key not in expected:continue
   if key not in expected:fail('PowerPaint 모델 구성이 일치하지 않습니다.')
   parent,_,name=key.rpartition('.');m=model.get_submodule(parent);t=f.get_tensor(key)
   if name in m._parameters:m._parameters[name]=torch.nn.Parameter(t,requires_grad=False)
   else:m._buffers[name]=t
   loaded.add(key)
 if loaded!=expected:fail('PowerPaint 모델 일부가 누락됐습니다.')
mapped_load(unet,weights/'unet/unet.safetensors');mapped_load(text,weights/'text_encoder/text_encoder.safetensors')
emb=text.text_model.embeddings.token_embedding
for item in emb.external_embeddings:item['embedding']=emb.trainable_embeddings[item['name']]
pipe=Pipeline.from_pretrained(str(base),unet=unet,text_encoder=text,tokenizer=CLIPTokenizer.from_pretrained(str(base/'tokenizer'),local_files_only=True),torch_dtype=dtype,variant='fp16',local_files_only=True,use_safetensors=True);pipe.tokenizer=tokenizer
def cpu_layers(model):
 for m in model.modules():
  originals={n:p for n,p in m._parameters.items() if p is not None and p.is_floating_point()}
  if not originals:continue
  def enter(module,args,orig=originals):
   for n,p in orig.items():module._parameters[n]=torch.nn.Parameter(p.float(),requires_grad=False)
   return tuple(x.float() if torch.is_tensor(x) and x.is_floating_point() else x for x in args)
  def leave(module,args,result,orig=originals):
   for n,p in orig.items():module._parameters[n]=p
  m.register_forward_pre_hook(enter);m.register_forward_hook(leave,always_call=True)
 model._parameters={'_execution_dtype_marker':torch.nn.Parameter(torch.empty(0,dtype=torch.float32),requires_grad=False),**model._parameters}
for component in [pipe.unet,pipe.text_encoder,pipe.vae,pipe.safety_checker]:cpu_layers(component)
pipe.vae.enable_slicing();pipe.set_progress_bar_config(disable=True)
edge=max(512,box[2]-box[0]+240,box[3]-box[1]+240)
if edge>4096:fail('보완 영역이 너무 넓습니다. 더 작은 영역으로 나눠 주세요.')
cx=(box[0]+box[2])//2;cy=(box[1]+box[3])//2;x=max(0,min(im.width-edge,cx-edge//2));y=max(0,min(im.height-edge,cy-edge//2));roi=(x,y,x+edge,y+edge)
matte=str(c.get('matte','#5f4b3d'));condition=Image.new('RGBA',im.size,matte);condition.alpha_composite(im)
square=Image.new('RGB',(edge,edge),matte);part=condition.convert('RGB').crop((x,y,min(x+edge,im.width),min(y+edge,im.height)));square.paste(part,(0,0));condition=square.resize((512,512),Image.Resampling.LANCZOS)
conditioning_mask=mask.filter(ImageFilter.MaxFilter(25)).crop(roi).resize((512,512),Image.Resampling.NEAREST)
croissant='croissant' in hint or '크루아상' in hint or '크루와상' in hint
if croissant:
 prompt='bakery food product photography, a baked croissant pastry made of laminated dough, extend the existing golden crispy crust and small flaky bread layers into the missing pastry area, matching the surrounding croissant, same warm bakery lighting'
 if hint not in ['croissant','크루아상','크루와상']:prompt+=' Product guidance: '+hint
else:prompt='product photography, '+hint+', extend only the existing product surface into its missing area, matching original material, texture and lighting, a single continuous original object'
negative='packaging, aluminum foil, cutlery, plate, butter cube, icing, text, logo, extra object, black fragments, blue fragments, blurry, low quality' if croissant else 'text, logo, extra object, black fragments, blue fragments, blurry, low quality'
seed=int(c.get('seed',20261103));steps=20
def progress(i,t,latents):note('PowerPaint 로컬 생성',round((i+1)/steps*95))
response=pipe(promptA=prompt+' P_shape',promptB=prompt+' P_ctxt',negative_promptA=negative+' P_shape',negative_promptB=negative+' P_ctxt',tradoff=.8,tradoff_nag=.8,image=condition,mask=conditioning_mask,width=512,height=512,num_inference_steps=steps,guidance_scale=10.,generator=torch.Generator('cpu').manual_seed(seed),callback=progress)
if any(response.nsfw_content_detected or []):fail('생성 결과 검사에서 사용할 수 있는 후보가 반환되지 않았습니다. 원본은 유지합니다.')
field=Image.new('RGB',im.size);field.paste(response.images[0].resize((edge,edge),Image.Resampling.LANCZOS),(x,y));generated=np.array(field)
foreground=before[:,:,3]>0;union=Image.fromarray(((foreground|allowed)*255).astype('uint8'));f=Image.fromarray((foreground*255).astype('uint8'));k=17
closed=np.array(union.filter(ImageFilter.MaxFilter(k)).filter(ImageFilter.MinFilter(k)))>0
added=closed&(np.array(f.filter(ImageFilter.MaxFilter(k)))>0)&(np.array(mask.filter(ImageFilter.MaxFilter(k)))>0)&(~foreground)&(~allowed)
fill=allowed|added
if fill.sum()>im.width*im.height*.3:fail('보완 범위가 넓어 결과를 적용하지 않았습니다.')
after=before.copy();after[fill,:3]=generated[fill,:3];after[fill,3]=255
if not np.array_equal(before[foreground],after[foreground]) or not np.array_equal(before[~fill],after[~fill]):fail('원본 보호 검증을 통과하지 못했습니다.')
Image.fromarray(after).save(out/'restored.png');Image.fromarray((fill*255).astype('uint8')).save(out/'generated-mask.png')
transition=save_transition(before,generated,fill,roi)
result={'image':'restored.png','mask':'generated-mask.png',**transition,'objects':[],'protectedRgbChanged':0,'generatedPixels':int(fill.sum()),'seamPixels':int(added.sum()),'method':'powerpaint-v1-shape-inpainting','cached':False,'seconds':time.monotonic()-start,'seed':seed,'prompt':prompt};(out/'result.json').write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8');note('PowerPaint 보완 후보 준비 완료',100)
