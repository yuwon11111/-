from pathlib import Path
import sys,urllib.request,json,socket,subprocess,time
root=Path(__file__).resolve().parents[1]
try:
 with urllib.request.urlopen('http://127.0.0.1:8191/health',timeout=2) as r:health=json.load(r)
 if health.get('service')=='coex-local-instance':print('Local instance service already ready');sys.exit(0)
except Exception:pass
with socket.socket() as probe:
 if probe.connect_ex(('127.0.0.1',8191))==0:raise RuntimeError('8191 is owned by another application; no process was stopped.')
model=root/'artifacts/ai/instance-sam2';assert (model/'model.safetensors').exists(),'Local instance model is not installed'
jobs=root/'artifacts/ai/instance-jobs';jobs.mkdir(exist_ok=True)
with (jobs/'server.out.log').open('a',encoding='utf-8') as stdout,(jobs/'server.err.log').open('a',encoding='utf-8') as stderr:
 p=subprocess.Popen([sys.executable,str(root/'tools/local-instance-server.py'),'--model',str(model),'--jobs',str(jobs),'--inpaint',str(root/'.local-ai/models/inpainting')],cwd=root,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
 print('Started own instance service PID '+str(p.pid),flush=True)
for _ in range(20):
 time.sleep(.2)
 try:
  with urllib.request.urlopen('http://127.0.0.1:8191/health',timeout=1) as r:health=json.load(r)
  if health.get('service')=='coex-local-instance':print('Local instance service ready; PID '+str(p.pid));break
 except Exception:pass
else:raise RuntimeError('Local instance service did not become ready; inspect its own logs.')
