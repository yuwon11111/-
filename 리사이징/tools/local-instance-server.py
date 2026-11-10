"""Local-only instance segmentation sidecar; existing 8179 server stays unchanged."""

from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler

from concurrent.futures import ThreadPoolExecutor

from pathlib import Path

import os,json,base64,secrets,sys,subprocess,time,argparse,threading

def stop_process(process):

 if process.poll() is not None:return

 if sys.platform=='win32':subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

 else:process.kill()

parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--jobs',required=True);parser.add_argument('--inpaint',required=True);parser.add_argument('--port',type=int,default=8191);args=parser.parse_args()

ROOT=Path(args.jobs).resolve();ROOT.mkdir(parents=True,exist_ok=True);JOBS={};PROCESSES={};LOCK=threading.Lock();pool=ThreadPoolExecutor(max_workers=1)

RUNTIME=Path(__file__).resolve().parents[1]/'artifacts/ai/powerpaint-v1'
def powerpaint_ready():return all((RUNTIME/p).is_file() for p in ['model/unet/unet.safetensors','model/text_encoder/text_encoder.safetensors','repo/powerpaint/pipelines/pipeline_PowerPaint.py','deps/diffusers/__init__.py','runtime-manifest.json'])

def run(key,source,repair=None):

 if JOBS[key].get('cancelled'):JOBS[key].update(status='cancelled',phase='중지');return

 job=JOBS[key];folder=ROOT/key;folder.mkdir();(folder/'input.png').write_bytes(source);config={'input':str(folder/'input.png'),'output':str(folder),'model':str(Path(args.model).resolve())};(folder/'config.json').write_text(json.dumps(config),encoding='utf-8');job.update(status='running',phase='개체 분석 준비')

 if repair:

  (folder/'mask.png').write_bytes(repair['mask']);config.update(mask=str(folder/'mask.png'),hint=repair['hint'],matte=repair['matte'],inpaintModel=str(Path(args.inpaint).resolve()),externalConsent=repair.get('externalConsent',False),sourceUnmodifiedVersion=repair.get('sourceUnmodifiedVersion'),powerpaintRuntime=str(RUNTIME),fresh=repair.get('fresh',False),seed=repair.get('seed',20261103),task=repair.get('task','object'));(folder/'config.json').write_text(json.dumps(config),encoding='utf-8')

 try:

  with (folder/'runner.log').open('w',encoding='utf-8') as log:

   runner=('openai-background-repair.py' if repair.get('provider')=='openai' and repair.get('task')=='background' else {'powerpaint':'powerpaint-instance-repair.py','openai':'openai-instance-repair.py'}.get(repair.get('provider'),'local-instance-repair.py')) if repair else 'local-instance-runner.py'

   process=subprocess.Popen([sys.executable,'-X','utf8','-u',str(Path(__file__).with_name(runner)),str(folder/'config.json')],stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))

   PROCESSES[key]=process;watchdog=threading.Timer((1200 if repair.get('provider')=='powerpaint' else 900) if repair else 240,lambda:stop_process(process));watchdog.start()

   if job.get('cancelled'):stop_process(process)

   for line in process.stdout:

    try:message=json.loads(line);job.update({k:message[k] for k in ['phase','progress'] if k in message})

    except (ValueError,TypeError):pass

   code=process.wait(timeout=240);watchdog.cancel();PROCESSES.pop(key,None)

  if job.get('cancelled'):job.update(status='cancelled',phase='분석 중지');return

  if code and repair and repair.get('provider')=='powerpaint':
   failure=json.loads((folder/'failure.json').read_text(encoding='utf-8')) if (folder/'failure.json').exists() else {};raise ValueError(failure.get('error','PowerPaint 후보를 준비하지 못했습니다. 원본은 유지합니다.'))
  if code and repair and repair.get('provider')=='openai':raise ValueError('외부 생성 후보를 얻지 못했습니다. API 설정·사용 내역을 확인하세요. 자동 재시도하지 않습니다. 원본은 유지합니다.')

  if code:raise ValueError('로컬 생성 모델에서 사용할 보완 결과를 얻지 못했습니다. 원본 그림은 유지합니다.' if repair else '로컬 개체 분석을 완료하지 못했습니다. 기록을 확인하세요.')

  result=json.loads((folder/'result.json').read_text(encoding='utf-8'));job.update(status='completed',result=result,phase='개체 후보 준비 완료')

 except Exception as error:job.update(status='failed',error=str(error));(folder/'failure.json').write_text(json.dumps({'error':str(error),'kind':'repair' if repair else 'segmentation'},ensure_ascii=False),encoding='utf-8')

class Handler(BaseHTTPRequestHandler):

 def log_message(self,*a):pass

 def allowed(self):return self.headers.get('Origin') in [None,'http://127.0.0.1:8179','http://localhost:8179']

 def send_headers(self,status,kind='application/json'):

  self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Access-Control-Allow-Origin',self.headers.get('Origin','http://127.0.0.1:8179'));self.send_header('Access-Control-Allow-Private-Network','true');self.send_header('Cache-Control','no-store');self.end_headers()

 def send(self,status,value):self.send_headers(status);self.wfile.write(json.dumps(value,ensure_ascii=False).encode('utf-8'))

 def do_OPTIONS(self):

  if not self.allowed():self.send(403,{'error':'Origin not allowed'});return

  self.send_response(204);self.send_header('Access-Control-Allow-Origin',self.headers.get('Origin'));self.send_header('Access-Control-Allow-Methods','GET, POST, OPTIONS');self.send_header('Access-Control-Allow-Headers','Content-Type');self.send_header('Access-Control-Allow-Private-Network','true');self.end_headers()

 def do_GET(self):

  if not self.allowed():self.send(403,{'error':'Origin not allowed'});return

  if self.path=='/health':self.send(200,{'service':'coex-local-instance','ready':Path(args.model,'model.safetensors').exists(),'inference':'local-cpu','automaticGeneration':False,'restorationReady':Path(args.inpaint,'local-model-lock.json').exists(),'busy':any(j['status'] in ['queued','running'] for j in JOBS.values()),'powerpaintReady':powerpaint_ready(),'openaiReady':os.environ.get('COEX_ENABLE_OPENAI_IMAGES')=='1' and bool(os.environ.get('OPENAI_API_KEY'))});return

  pieces=self.path.strip('/').split('/');key=pieces[1] if len(pieces)>1 and pieces[0]=='jobs' else '';job=JOBS.get(key)

  if not job:self.send(404,{'error':'Job not found'});return

  if len(pieces)==2:self.send(200,job);return

  result=job.get('result',{})

  if len(pieces)==3 and (any(x['file']==pieces[2] for x in result.get('objects',[])) or pieces[2] in [result.get('image'),result.get('mask'),result.get('seamImage'),result.get('seamMask')]):

   file=ROOT/key/pieces[2];self.send_headers(200,'image/png');self.wfile.write(file.read_bytes());return

  self.send(404,{'error':'Not found'})

 def do_POST(self):

  if not self.allowed():self.send(403,{'error':'Origin not allowed'});return

  pieces=self.path.strip('/').split('/')

  if len(pieces)==3 and pieces[0]=='jobs' and pieces[2]=='cancel':

   key=pieces[1];job=JOBS.get(key)

   if not job:self.send(404,{'error':'Job not found'});return

   job['cancelled']=True

   if key in PROCESSES:stop_process(PROCESSES[key])

   self.send(200,{'cancelled':True});return

  if self.path not in ['/jobs','/restore']:self.send(404,{'error':'Not found'});return

  try:

   length=int(self.headers.get('Content-Length','0'))

   if not 0<length<30000000:raise ValueError('사진 크기가 너무 큽니다.')

   data=json.loads(self.rfile.read(length));src=data.get('source','');prefix='data:image/png;base64,'

   if not src.startswith(prefix):raise ValueError('PNG 사진만 지원합니다.')

   source=base64.b64decode(src[len(prefix):],validate=True)

   if source[:8]!=b'\x89PNG\r\n\x1a\n' or len(source)<24:raise ValueError('잘못된 PNG입니다.')

   w=int.from_bytes(source[16:20],'big');h=int.from_bytes(source[20:24],'big')

   if min(w,h)<32 or max(w,h)>12000 or w*h>4000000:raise ValueError('사진은 400만 픽셀 이내로 준비하세요.')

   repair=None

   if self.path=='/restore':

    mask=data.get('mask','')

    if not mask.startswith(prefix):raise ValueError('생성 영역 PNG가 필요합니다.')

    mask=base64.b64decode(mask[len(prefix):],validate=True)

    if mask[:8]!=source[:8] or mask[16:24]!=source[16:24]:raise ValueError('생성 영역 크기가 다릅니다.')

    provider=data.get('provider','local')

    if provider not in ['local','openai','powerpaint']:raise ValueError('지원하지 않는 생성 서비스입니다.')

    if provider=='powerpaint' and not powerpaint_ready():raise ValueError('PowerPaint 로컬 모델이 준비되지 않았습니다.')
    if provider=='openai' and (data.get('externalConsent') is not True or os.environ.get('COEX_ENABLE_OPENAI_IMAGES')!='1' or not os.environ.get('OPENAI_API_KEY')):raise ValueError('외부 이미지 서비스 설정과 이번 전송 동의를 확인하세요.')

    task=data.get('task','object')
    if task not in ['object','background'] or (task=='background' and provider!='openai'):raise ValueError('지원하지 않는 보완 작업입니다.')
    if task=='background' and data.get('sourceUnmodifiedVersion')!='background-source-unmodified-v1':raise ValueError('배경 보호 기능이 업데이트되었습니다. 작업을 저장하고 사이트를 새로고침한 뒤 다시 시도하세요.')
    repair={'sourceUnmodifiedVersion':data.get('sourceUnmodifiedVersion'),'task':task,'provider':provider,'fresh':data.get('fresh') is True,'seed':secrets.randbelow(2147483647) if data.get('fresh') is True else 20261103,'externalConsent':data.get('externalConsent') is True,'mask':mask,'hint':str(data.get('hint','same original product'))[:300],'matte':str(data.get('matte','#5f4b3d'))[:9]}

   with LOCK:

    if any(j['status'] in ['queued','running'] for j in JOBS.values()):self.send(409,{'error':'다른 개체 분석이 진행 중입니다.'});return

    key=secrets.token_urlsafe(18);JOBS[key]={'id':key,'status':'queued','phase':'분석 대기','started':time.time()}

   pool.submit(run,key,source,repair);self.send(202,{'id':key})

  except Exception as e:self.send(400,{'error':str(e)})

ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()

