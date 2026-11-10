"""Local-only Diffusers background outpainting and optional photo proposals.

No paid providers. Source/model data stay on this computer except model GETs.
"""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import base64, json, secrets, threading, time, os

ROOT=Path(__file__).resolve().parents[1]
MODEL='stable-diffusion-v1-5/stable-diffusion-inpainting'
REVISION='8a4288a76071f7280aedbdb3253bdb9e9d5d84bb'
os.environ.setdefault('HF_HOME',str(ROOT/'.local-ai/models'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
ALLOWED={'https://coex-image-resize-20261017.shrewd-tapir-9956.chatgpt.site','http://127.0.0.1:4177','http://127.0.0.1:8179','http://localhost:8179'}
jobs={};pool=ThreadPoolExecutor(max_workers=1);pipeline=None

def load_pipeline(job):
 global pipeline
 if pipeline is None:
  local=ROOT/'.local-ai/models/inpainting';ready=(local/'local-model-lock.json').exists()
  job['message']='Loading verified local model.' if ready else 'Preparing local model; first use downloads model files.'
  import torch
  from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
  torch.set_num_threads(max(1,min(4,(os.cpu_count() or 4)//2)))
  pipeline=StableDiffusionInpaintPipeline.from_pretrained(str(local) if ready else MODEL,revision=None if ready else REVISION,variant='fp16',torch_dtype=torch.float32,use_safetensors=True,local_files_only=ready)
  pipeline.scheduler=DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
  pipeline.enable_attention_slicing()
  if hasattr(pipeline.vae,'enable_slicing'):pipeline.vae.enable_slicing()
  pipeline.set_progress_bar_config(disable=True)
 return pipeline

def prepare(source,width,height):
 from PIL import Image,ImageOps
 source=source.convert('RGBA');scale=min(1,width/source.width,height/source.height)
 size=(max(1,round(source.width*scale)),max(1,round(source.height*scale)))
 original=source.resize(size,Image.Resampling.LANCZOS)
 x=(width-size[0])//2;y=(height-size[1])//2
 # Uniform source fit; never stretch or crop the subject.
 canvas=Image.new('RGB',(width,height),source.convert('RGB').resize((1,1)).getpixel((0,0)))
 canvas.paste(original,(x,y),original)
 mask=Image.new('L',(width,height),255);mask.paste(ImageOps.invert(original.getchannel('A')),(x,y))
 return canvas,mask,original,(x,y)

def background_context(canvas,mask,original,offset):
 """Condition on an edge band, not the central logo/product/poster content.

 The original is composited back after diffusion; model pixels inside it are discarded.
 """
 from PIL import ImageDraw
 context=canvas.copy();generation_mask=mask.copy();x,y=offset
 band=max(8,min(32,round(min(original.size)*.05)))
 if original.width>band*2 and original.height>band*2:
  box=(x+band,y+band,x+original.width-band,y+original.height-band)
  corners=[original.convert('RGB').getpixel(p) for p in [(0,0),(original.width-1,0),(0,original.height-1),(original.width-1,original.height-1)]]
  color=tuple(round(sum(c[i] for c in corners)/4) for i in range(3))
  ImageDraw.Draw(context).rectangle(box,fill=color)
  ImageDraw.Draw(generation_mask).rectangle(box,fill=255)
 return context,generation_mask

def simple_background(original,width,height,offset):
 """Continue a verified flat/linear-gradient edge without inventing objects."""
 import numpy as np
 from PIL import Image
 w,h=original.size
 if w<4 or h<4:return None
 coords=[]
 for t in np.linspace(0,1,64):coords.extend([(round(t*(w-1)),0),(round(t*(w-1)),h-1),(0,round(t*(h-1))),(w-1,round(t*(h-1)))])
 values=np.array([original.getpixel(p) for p in coords],dtype=np.float32)
 if np.mean(values[:,3]>250)<.98:return None
 points=np.array([[1,x/max(1,w-1),y/max(1,h-1)] for x,y in coords],dtype=np.float32)
 colors=values[:,:3];keep=np.ones(len(points),dtype=bool)
 for _ in range(2):
  coefficients=np.linalg.lstsq(points[keep],colors[keep],rcond=None)[0]
  error=np.max(np.abs(points@coefficients-colors),axis=1);keep=error<4
  if np.mean(keep)<.96:return None
 if np.percentile(error,95)>3:return None
 result=Image.new('RGB',(width,height));x,y=offset
 xx=(np.arange(width,dtype=np.float32)-x)/max(1,w-1)
 for row in range(0,height,64):
  yy=(np.arange(row,min(height,row+64),dtype=np.float32)-y)/max(1,h-1)
  pixels=coefficients[0]+xx[None,:,None]*coefficients[1]+yy[:,None,None]*coefficients[2]
  result.paste(Image.fromarray(np.clip(np.rint(pixels),0,255).astype('uint8')),(0,row))
 result.paste(original,offset,original)
 return result

def soften_outside(image, original, offset, band=32, iterations=360):
    import numpy as np
    from PIL import Image
    if original.mode == 'RGBA' and original.getchannel('A').getextrema() != (255, 255):
        # Pure transparent padding around one opaque rectangle is a coordinate
        # envelope, not a translucent cutout. Correct the actual photo boundary.
        alpha = original.getchannel('A')
        box = alpha.getbbox()
        if box and alpha.crop(box).getextrema() == (255, 255):
            return soften_outside(image, original.crop(box), (offset[0] + box[0], offset[1] + box[1]), band, iterations)
        return image
    width, height = image.size
    sw, sh = original.size
    x, y = offset
    segments = []

    def rectangle(xa, ya, xb, yb):
        xa = max(0, xa)
        ya = max(0, ya)
        xb = min(width, xb)
        yb = min(height, yb)
        if xb <= xa or yb <= ya:
            return
        yy, xx = np.mgrid[ya:yb, xa:xb]
        segments.append((yy * width + xx).ravel())
    rectangle(x - band, y - band, x, y + sh + band)
    rectangle(x + sw, y - band, x + sw + band, y + sh + band)
    rectangle(x, y - band, x + sw, y)
    rectangle(x, y + sh, x + sw, y + sh + band)
    if not segments:
        return image
    linear = np.sort(np.concatenate(segments))
    ys = linear // width
    xs = linear % width
    n = len(linear)
    dx = np.maximum(0, np.maximum(x - xs, xs - (x + sw - 1)))
    dy = np.maximum(0, np.maximum(y - ys, ys - (y + sh - 1)))
    distance = np.maximum(dx, dy)
    boundary = distance == 1
    active = np.flatnonzero((distance > 1) & (distance < band) & (xs > 0) & (xs < width - 1) & (ys > 0) & (ys < height - 1))
    source = np.asarray(original.convert('RGB'), dtype=np.uint8)
    raw = np.asarray(image.convert('RGB'), dtype=np.uint8)
    field = np.zeros((n + 1, 3), dtype=np.float32)
    sy = np.clip(ys[boundary] - y, 0, sh - 1)
    sx = np.clip(xs[boundary] - x, 0, sw - 1)
    field[np.flatnonzero(boundary)] = source[sy, sx].astype(np.float32) - raw[ys[boundary], xs[boundary]].astype(np.float32)
    neighbours = []
    for delta in [-width, width, -1, 1]:
        wanted = linear[active] + delta
        positions = np.searchsorted(linear, wanted)
        safe = np.minimum(positions, n - 1)
        valid = (positions < n) & (linear[safe] == wanted)
        neighbours.append(np.where(valid, positions, n).astype(np.int32))
    for _ in range(iterations):
        average = (field[neighbours[0]] + field[neighbours[1]] + field[neighbours[2]] + field[neighbours[3]]) * 0.25
        field[active] = average
    result = raw.copy()
    result[ys, xs] = np.clip(np.rint(raw[ys, xs].astype(np.float32) + field[:n]), 0, 255).astype('uint8')
    result[y:y + sh, x:x + sw] = source
    return Image.fromarray(result)


def run(job,data):
 try:
  from PIL import Image
  if job.get('cancelled'):raise InterruptedError('Cancelled')
  job.update(status='running',message='Preparing image',progress=0)
  w,h=data['width'],data['height'];mode=data['mode']
  raw=base64.b64decode(data['source'].split(',',1)[-1],validate=True)
  if len(raw)>20*1024*1024:raise ValueError('Source image exceeds 20 MB')
  source=Image.open(BytesIO(raw))
  if source.width*source.height>24000000:raise ValueError('Source image exceeds 24 million pixels')
  source.load()
  canvas,mask,original,offset=prepare(source,w,h)
  if mode=='replace':mask.paste(255,(0,0,w,h))
  # Small inference canvas on this CPU; original pixels are restored at output resolution.
  factor=min(1,512/max(w,h));iw=max(64,round(w*factor/8)*8);ih=max(64,round(h*factor/8)*8)
  simple=simple_background(original,w,h,offset) if mode=='background' else None
  if mode=='background' and mask.getextrema()==(0,0):
   result=canvas;job.update(message='No empty border to extend; original retained.',generated=False,method='original-no-empty-space')
  elif simple is not None:
   result=simple;job.update(message='Simple background continued without generative AI.',generated=False,method='simple-background-continuation')
  else:
   import torch
   pipe=load_pipeline(job)
   if job.get('cancelled'):raise InterruptedError('Cancelled')
   steps=int(data.get('steps',12));steps=max(6,min(30,steps))
   def progress(pipe,step,t,kwargs):
    if job.get('cancelled'):raise InterruptedError('Cancelled')
    job.update(progress=round((step+1)/steps*100),message=f'Generating {step+1}/{steps}')
    return kwargs
   prompt=str(data.get('prompt',''))[:700]
   prompt=('seamless continuation of the original empty background, matching texture, colors and lighting, simple surrounding space, no frame, no sign, no new foreground subject, '+prompt) if mode=='background' else ('professional product or portrait photography, natural lighting, sharp detail, no text, no logo, '+prompt)
   context_mode=data.get('context','full');conditioning,generation_mask=background_context(canvas,mask,original,offset) if mode=='background' and context_mode=='edge' else (canvas,mask)
   response=pipe(prompt=prompt,negative_prompt='text, letters, logo, watermark, frame, border, poster, sign, distorted, deformed, duplicate subject',image=conditioning.resize((iw,ih)),mask_image=generation_mask.resize((iw,ih)),width=iw,height=ih,num_inference_steps=steps,guidance_scale=7.5,strength=1.0,generator=torch.Generator('cpu').manual_seed(int(data.get('seed',secrets.randbelow(2**31)))),callback_on_step_end=progress)
   if any(response.nsfw_content_detected or []):raise ValueError('The model did not return a usable photo. Try another description.')
   result=response.images[0].resize((w,h),Image.Resampling.LANCZOS)
   if mode=='background' and original.getchannel('A').getextrema()==(255,255):result.paste(original,offset,original)
   if data.get('diagnostics'):
    raw_result=result.copy()
    if mode=='background':raw_result.paste(original,offset,original)
    raw_buffer=BytesIO();raw_result.save(raw_buffer,format='PNG');job['rawImage']='data:image/png;base64,'+base64.b64encode(raw_buffer.getvalue()).decode()
   if mode=='background':
    result=soften_outside(result,original,offset)
    result.paste(original,offset,original) # Opaque original pixels are exact; transparent holes receive background.
   job['safetyChecked']=True
   job['generated']=True
   job['method']=(context_mode+'-context-inpainting') if mode=='background' else 'new-photo-candidate'
  if job.get('cancelled'):raise InterruptedError('Cancelled')
  image=BytesIO();result.save(image,format='PNG')
  job.update(status='complete',progress=100,image='data:image/png;base64,'+base64.b64encode(image.getvalue()).decode(),width=w,height=h,model=MODEL if job.get('generated') else None,preserved=mode=='background',conditioning=(data.get('context','full') if mode=='background' else 'full-generation') if job.get('generated') else None,postprocess='native-sparse-band-harmonic-correction' if job.get('generated') and mode=='background' else None,sourceRect={'x':offset[0],'y':offset[1],'width':original.width,'height':original.height},elapsedSeconds=round(time.time()-job['started'],2))
 except InterruptedError:job.update(status='cancelled',message='Cancelled')
 except Exception as error:job.update(status='error',message=str(error))

class Handler(SimpleHTTPRequestHandler):
 def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(ROOT/'dist'),**kwargs)
 def allowed(self):return not self.headers.get('Origin') or self.headers.get('Origin') in ALLOWED
 def end_headers(self):
  origin=self.headers.get('Origin')
  if origin in ALLOWED:self.send_header('Access-Control-Allow-Origin',origin);self.send_header('Vary','Origin')
  self.send_header('Access-Control-Allow-Private-Network','true')
  super().end_headers()
 def send_json(self,data,status=200):
  body=json.dumps(data).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
 def do_OPTIONS(self):
  if not self.allowed():self.send_json({'error':'Origin not allowed'},403);return
  self.send_response(204);self.send_header('Access-Control-Allow-Methods','GET,POST,OPTIONS');self.send_header('Access-Control-Allow-Headers','Content-Type');self.end_headers()
 def do_GET(self):
  if self.path.startswith('/api/'):
   if not self.allowed():self.send_json({'error':'Origin not allowed'},403);return
   if self.path=='/api/health':self.send_json({'service':'coex-local-photo-ai','model':MODEL,'modelLoaded':pipeline is not None,'paidApi':False,'backend':'cpu'});return
   job=jobs.get(self.path.removeprefix('/api/jobs/'))
   self.send_json(job or {'error':'Unknown job'},200 if job else 404);return
  super().do_GET()
 def do_POST(self):
  if not self.allowed():self.send_json({'error':'Origin not allowed'},403);return
  if self.headers.get('Content-Type','').split(';')[0]!='application/json':self.send_json({'error':'JSON required'},415);return
  try:
   size=int(self.headers.get('Content-Length','0'))
   if size<2 or size>28*1024*1024:raise ValueError('Request size invalid')
   data=json.loads(self.rfile.read(size))
   if self.path.endswith('/cancel'):
    job=jobs.get(self.path.split('/')[-2]);
    if job:job['cancelled']=True
    self.send_json({'cancelled':bool(job)});return
   if self.path!='/api/jobs':self.send_json({'error':'Not found'},404);return
   if data.get('mode') not in ['background','replace']:raise ValueError('Invalid mode')
   if data.get('context','full') not in ['full','edge']:raise ValueError('Invalid context mode')
   if type(data.get('diagnostics',False))!=bool:raise ValueError('Invalid diagnostics option')
   w=data.get('width');h=data.get('height')
   if type(w)!=int or type(h)!=int or not 64<=w<=12000 or not 64<=h<=12000 or w*h>24000000:raise ValueError('Invalid target size')
   if not isinstance(data.get('source'),str):raise ValueError('Source image required')
   if sum(j['status'] in ['queued','running'] for j in jobs.values())>=2:self.send_json({'error':'Local AI is busy'},409);return
   # Retain bounded results in process memory only.
   for key in list(jobs):
    if time.time()-jobs[key]['started']>3600 and jobs[key]['status'] not in ['queued','running']:del jobs[key]
   if len(jobs)>=12:
    for key in list(jobs):
     if jobs[key]['status'] not in ['queued','running']:del jobs[key];break
   key=secrets.token_urlsafe(24);job={'id':key,'status':'queued','progress':0,'started':time.time()};jobs[key]=job;pool.submit(run,job,data);self.send_json({'id':key},202)
  except Exception as error:self.send_json({'error':str(error)},400)

class LocalPhotoServer(ThreadingHTTPServer):
 request_queue_size=64

if __name__=='__main__':
 print('Local photo AI: http://127.0.0.1:8179 (CPU; model loads on first generation)',flush=True)
 LocalPhotoServer(('127.0.0.1',8179),Handler).serve_forever()
