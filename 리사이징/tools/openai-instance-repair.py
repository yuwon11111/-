"""Optional paid image edits. Requires explicit server enablement and per-request consent.

Only the selected cropped object and its mask leave this PC. The poster never does.
Model output is composited locally into transparent approved pixels only.
"""
from pathlib import Path
import os,sys,json,base64,io,time,urllib.request,urllib.error
from PIL import Image
import numpy as np

def png(im):
 stream=io.BytesIO();im.save(stream,format='PNG');return stream.getvalue()

def repair(c,request=None):
 key=os.environ.get('OPENAI_API_KEY','')
 if os.environ.get('COEX_ENABLE_OPENAI_IMAGES')!='1' or not key or c.get('externalConsent') is not True:
  raise ValueError('외부 이미지 서비스가 활성화되지 않았거나 이번 전송에 동의하지 않았습니다.')
 out=Path(c['output']);im=Image.open(c['input']).convert('RGBA');mask=Image.open(c['mask']).convert('L')
 if mask.size!=im.size:raise ValueError('생성 영역과 그림 크기가 다릅니다.')
 before=np.array(im);allowed=(np.array(mask)>128)&(before[:,:,3]==0)
 if not allowed.any() or allowed.sum()>im.width*im.height*.3:raise ValueError('가려진 부분만 지정하세요.')
 # Preserve aspect ratio in a square input and invert this mapping after generation.
 edge=max(im.size);offset=((edge-im.width)//2,(edge-im.height)//2)
 condition=Image.new('RGBA',(edge,edge),'#5f4b3d');condition.alpha_composite(im,offset)
 api_mask=Image.new('RGBA',(edge,edge),(255,255,255,255));a=np.array(api_mask);x,y=offset;a[y:y+im.height,x:x+im.width,3][allowed]=0;api_mask=Image.fromarray(a)
 source=condition.convert('RGB').resize((1024,1024),Image.Resampling.LANCZOS);api_mask=api_mask.resize((1024,1024),Image.Resampling.NEAREST)
 hint=str(c.get('hint','same original product'))[:300]
 prompt='Complete only the missing occluded part of this single '+hint+'. Match the visible material, color, shape and lighting. Keep the same camera and object position. Do not add objects, text, logos, butter or a new background. The transparent mask specifies the missing region. Return the same framing.'
 payload={'model':'gpt-image-2.5-sunburst','images':[{'image_url':'data:image/png;base64,'+base64.b64encode(png(source)).decode()}],'mask':{'image_url':'data:image/png;base64,'+base64.b64encode(png(api_mask)).decode()},'prompt':prompt,'n':1,'size':'1024x1024','quality':'medium','output_format':'png'}
 print(json.dumps({'phase':'외부 서비스에서 보완 후보 생성 중 · 요청 수락 후 중지해도 요금이 발생할 수 있습니다.'}),flush=True)
 start=time.monotonic();req=urllib.request.Request('https://api.openai.com/v1/images/edits',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
 try:
  with (request or urllib.request.urlopen)(req,timeout=300) as response:body=response.read(40000000);result=json.loads(body)
 except urllib.error.HTTPError as e:
  messages={401:'API 키를 확인하세요.',403:'이미지 모델 사용 권한을 확인하세요.',429:'API 사용 한도 또는 잔액을 확인하세요.'}
  raise ValueError('외부 이미지 서비스 오류 '+str(e.code)+': '+messages.get(e.code,'요청을 완료하지 못했습니다. 자동 재시도하지 않습니다.')) from None
 except (TimeoutError,urllib.error.URLError):raise ValueError('외부 서비스 응답을 받지 못했습니다. 과금 여부를 확인한 뒤 다시 실행하세요. 자동 재시도하지 않습니다.') from None
 raw=base64.b64decode(result['data'][0]['b64_json'],validate=True);patch=Image.open(io.BytesIO(raw));patch.load()
 if patch.size!=(1024,1024):raise ValueError('생성 이미지의 크기가 달라 적용하지 않았습니다.')
 patch=patch.convert('RGBA').resize((edge,edge),Image.Resampling.LANCZOS).crop((x,y,x+im.width,y+im.height));after=before.copy();p=np.array(patch);after[allowed,:3]=p[allowed,:3];after[allowed,3]=255
 protected=before[:,:,3]>0;changed=int(np.any(before[protected]!=after[protected],axis=1).sum())
 if changed:raise ValueError('보호할 원본 픽셀이 달라 결과를 폐기했습니다.')
 Image.fromarray(after).save(out/'restored.png');Image.fromarray((allowed*255).astype('uint8')).save(out/'generated-mask.png')
 report={'image':'restored.png','mask':'generated-mask.png','objects':[],'protectedRgbChanged':changed,'generatedPixels':int(allowed.sum()),'seconds':time.monotonic()-start,'method':'openai-masked-inpainting','model':'gpt-image-2.5-sunburst','warning':'가려진 부분은 추정 생성입니다. 원본 복원이 아닙니다.'}
 (out/'result.json').write_text(json.dumps(report,ensure_ascii=False),encoding='utf-8');return report

if __name__=='__main__':repair(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
