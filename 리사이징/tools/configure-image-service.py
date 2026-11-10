"""Local password-entry window. No network API calls and no key in command lines/logs."""
from pathlib import Path
import os,json,urllib.request,subprocess,sys,time,threading

def health():
 try:
  with urllib.request.urlopen('http://127.0.0.1:8191/health',timeout=2) as r:return json.load(r)
 except OSError:return None

def activate(root,key,persist=False):
 import psutil
 key=key.strip()
 if len(key)<20 or any(c.isspace() for c in key):raise ValueError('API 키를 확인하세요.')
 h=health()
 if h and h.get('service')!='coex-local-instance':raise ValueError('8191을 다른 프로그램이 사용 중입니다. 종료하지 않았습니다.')
 if h and h.get('busy') is not False:raise ValueError('개체 분석·생성 작업이 진행 중이거나 상태를 확인할 수 없습니다. 작업이 끝난 뒤 다시 설정하세요.')
 if h:
  listeners=[c for c in psutil.net_connections(kind='tcp') if c.laddr.port==8191 and c.status=='LISTEN']
  if len(listeners)!=1:raise ValueError('보조 서버 소유권을 확인할 수 없습니다.')
  p=psutil.Process(listeners[0].pid)
  if str(root/'tools/local-instance-server.py').lower() not in ' '.join(p.cmdline()).lower():raise ValueError('보조 서버 경로가 다릅니다. 종료하지 않았습니다.')
  if any(any(s in ' '.join(c.cmdline()) for s in ['local-instance-runner.py','local-instance-repair.py','openai-instance-repair.py']) for c in p.children(recursive=True)):raise ValueError('모델 작업이 진행 중입니다. 완료 후 다시 설정하세요.')
  pid=p.pid;parent=p.parent()
  if parent and str(root/'.local-ai/env/Scripts/python.exe').lower()==parent.exe().lower():pid=parent.pid
  subprocess.run(['taskkill','/PID',str(pid),'/T','/F'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 env=os.environ.copy();env['OPENAI_API_KEY']=key;env['COEX_ENABLE_OPENAI_IMAGES']='1'
 try:
  result=subprocess.run([sys.executable,str(root/'tools/start-instance-analysis.py')],cwd=root,env=env,capture_output=True,timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
  if result.returncode or not (health() or {}).get('openaiReady'):raise ValueError('보조 서버를 활성화하지 못했습니다. 8179 사이트는 유지됩니다.')
 except Exception:
  if not health():subprocess.run([sys.executable,str(root/'tools/start-instance-analysis.py')],cwd=root,capture_output=True,timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
  raise
 if persist:
  import winreg
  with winreg.CreateKey(winreg.HKEY_CURRENT_USER,'Environment') as reg:
   winreg.SetValueEx(reg,'OPENAI_API_KEY',0,winreg.REG_SZ,key);winreg.SetValueEx(reg,'COEX_ENABLE_OPENAI_IMAGES',0,winreg.REG_SZ,'1')
 return True

def main():
 import tkinter as tk
 from tkinter import ttk,messagebox
 root=Path(__file__).resolve().parents[1];app=tk.Tk();app.title('빵 보완 · OpenAI API 설정');app.geometry('560x330');app.resizable(False,False)
 ttk.Label(app,text='OpenAI 이미지 편집 API 연결',font=('',15,'bold')).pack(anchor='w',padx=20,pady=(20,10))
 ttk.Label(app,text='API 키는 이 PC에서 입력합니다. 생성 버튼을 누르기 전에는\n사진 전송·유료 API 호출을 실행하지 않습니다.').pack(anchor='w',padx=20)
 value=tk.StringVar();entry=ttk.Entry(app,textvariable=value,show='*',width=65);entry.pack(padx=20,pady=14);entry.focus_set()
 persist=tk.BooleanVar(value=False);ttk.Checkbutton(app,text='다음 실행에도 사용 · 내 Windows 사용자 환경변수에 저장',variable=persist).pack(anchor='w',padx=20)
 status=tk.StringVar(value='저장하지 않으면 이번 보조 서버 실행 동안만 사용합니다.');ttk.Label(app,textvariable=status,wraplength=510).pack(anchor='w',padx=20,pady=12)
 def submit():
  key=value.get();value.set('');button.config(state='disabled');status.set('보조 서버 연결 설정 중… 8179 사이트는 유지합니다.')
  save=persist.get()
  def finish(error):
   button.config(state='normal')
   if error:status.set(error)
   else:status.set('연결 설정 완료. 사이트에서 OpenAI 방식을 선택하고 보완 영역을 확인한 뒤 생성 버튼을 누르세요. 실제 API 권한·잔액은 생성 시 확인됩니다.')
  def worker():
   try:activate(root,key,save);error=None
   except Exception as e:error=str(e) if isinstance(e,ValueError) else '연결 설정을 완료하지 못했습니다. 보조 서버 상태를 확인하세요.'
   app.after(0,lambda:finish(error))
  threading.Thread(target=worker,daemon=True).start()
 button=ttk.Button(app,text='API 연결 설정 · 생성하지 않음',command=submit);button.pack(anchor='w',padx=20)
 if (health() or {}).get('openaiReady'):status.set('이미 연결 설정되어 있습니다. 창을 닫고 사이트에서 생성 버튼을 사용하세요.')
 app.protocol('WM_DELETE_WINDOW',lambda:app.destroy() if str(button['state'])!='disabled' else status.set('설정 작업이 끝난 뒤 창을 닫으세요.'))
 app.mainloop()

if __name__=='__main__':main()
