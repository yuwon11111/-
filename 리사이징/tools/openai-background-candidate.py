"""Create one explicitly requested whole-background candidate from a reference image."""
import base64,io,json,os,sys,time,urllib.request,urllib.error
from pathlib import Path
import numpy as np
from PIL import Image

def png(im):
    output=io.BytesIO();im.save(output,format='PNG');return output.getvalue()

def candidate(config,send=None):
    if config.get('sourceUnmodifiedVersion')!='background-source-unmodified-v1':
        raise ValueError('배경 보호 기능이 업데이트되었습니다. 작업을 저장하고 사이트를 새로고침한 뒤 다시 시도하세요.')
    key=os.environ.get('OPENAI_API_KEY','')
    if not key or os.environ.get('COEX_ENABLE_OPENAI_IMAGES')!='1' or config.get('externalConsent') is not True:
        raise ValueError('OpenAI API 설정을 확인하세요.')
    source=Image.open(config['input']).convert('RGBA'); mask=Image.open(config['mask']).convert('L')
    if source.size!=mask.size:raise ValueError('배경 유효 영역의 크기가 다릅니다.')
    before=np.asarray(source); original_alpha=np.asarray(mask); coverage=original_alpha>0
    partial=(before[:,:,3]>0)&(before[:,:,3]<255)
    if np.any(((before[:,:,3]>0)&~coverage)|(partial&(before[:,:,3]!=original_alpha))):
        raise ValueError('분리 배경과 원본 사진의 투명도가 맞지 않습니다. 원본을 확인하세요.')
    eligible=coverage&~partial
    if not eligible.any():raise ValueError('생성할 배경 영역이 없습니다.')
    # Work in the fixed API canvas: allocating max(w,h)**2 makes a thin
    # 12000x32 input consume hundreds of MB despite its modest pixel count.
    w,h=source.size;edge=max(w,h)
    sw=max(1,min(1024,round(w*1024/edge)));sh=max(1,min(1024,round(h*1024/edge)))
    if min(sw,sh)<64:raise ValueError('이미지 비율이 너무 길어 배경 세부가 손실됩니다. 더 넓은 배경 사진을 선택하세요.')
    x=(1024-sw)//2;y=(1024-sh)//2
    ref=Image.new('RGBA',(1024,1024),(110,94,80,255))
    ref.alpha_composite(source.resize((sw,sh),Image.Resampling.LANCZOS),(x,y));ref=ref.convert('RGB')
    prompt=('Create a coherent single empty background for a graphic design, using this image only as a reference for its palette, scene, texture and lighting. '
            'Retain coherent structural background surfaces suggested by the reference, such as a table, wall or pedestal, when present. '
            'Replace gaps, artifacts and any depicted products, food, labels, logos, symbols or lettering with plausible background. '
            'Do not add people, products, food, labels, logos, symbols, text or unrelated props. Keep a natural continuous backdrop. '
            +str(config.get('hint',''))[:300])
    payload={'model':'gpt-image-2.5-sunburst','images':[{'image_url':'data:image/png;base64,'+base64.b64encode(png(ref)).decode()}],
             'prompt':prompt,'size':'1024x1024','quality':'medium','n':1,'output_format':'png'}
    request=urllib.request.Request('https://api.openai.com/v1/images/edits',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
    start=time.monotonic();print(json.dumps({'phase':'OpenAI 배경 후보 생성 중 · 요청 수락 후 중지해도 비용이 발생할 수 있습니다.'},ensure_ascii=True),flush=True)
    try:
        with (send or urllib.request.urlopen)(request,timeout=300) as response:result=json.loads(response.read(40000000))
    except urllib.error.HTTPError as exc:
        messages={401:'API 키를 확인하세요.',403:'이미지 모델 사용 권한을 확인하세요.',429:'API 사용 한도 또는 잔액을 확인하세요.'}
        raise ValueError('OpenAI 오류 '+str(exc.code)+': '+messages.get(exc.code,'요청을 완료하지 못했습니다. 자동 재시도하지 않습니다.')) from None
    except (TimeoutError,urllib.error.URLError):
        raise ValueError('응답을 받지 못했습니다. 사용 내역을 확인한 뒤 다시 실행하세요. 자동 재시도하지 않습니다.') from None
    image=Image.open(io.BytesIO(base64.b64decode(result['data'][0]['b64_json'],validate=True)));image.load()
    if image.size!=(1024,1024):raise ValueError('생성 결과 크기가 맞지 않습니다.')
    patch=np.array(image.convert('RGBA').crop((x,y,x+sw,y+sh)).resize((w,h),Image.Resampling.LANCZOS))
    after=before.copy();after[eligible,:3]=patch[eligible,:3];after[eligible,3]=original_alpha[eligible]
    protected=~eligible
    if np.any(after[protected]!=before[protected]):raise ValueError('보호 영역이 변경되었습니다.')
    if np.any(after[eligible,3]!=original_alpha[eligible]):raise ValueError('원본 투명도가 변경되었습니다.')
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    Image.fromarray(after).save(out/'restored.png');Image.fromarray((eligible*255).astype('uint8')).save(out/'generated-mask.png')
    report={'image':'restored.png','mask':'generated-mask.png','objects':[],'protectedChanged':0,'protectedRgbChanged':0,
            'partialAlphaChanged':0,'method':'openai-background-candidate','generatedPixels':int(eligible.sum()),
            'model':payload['model'],'seconds':time.monotonic()-start,'usage':result.get('usage')}
    (out/'result.json').write_text(json.dumps(report,ensure_ascii=False),encoding='utf8');return report

if __name__=='__main__':candidate(json.loads(Path(sys.argv[1]).read_text(encoding='utf8')))





