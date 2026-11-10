"""Candidate background repair. Called explicitly; API key remains server-side."""
import base64, io, json, os, sys, time, urllib.request, urllib.error
from pathlib import Path
import numpy as np
from PIL import Image

def png(im):
    b=io.BytesIO(); im.save(b,format='PNG'); return b.getvalue()

def build_api_mask(requested, edge, x, y, width, height):
    # Conservative projection: each requested original pixel marks every API
    # cell intersecting its area, preventing thin regions from disappearing.
    size=1024
    padded=np.zeros((edge,edge),dtype=np.uint8)
    padded[y:y+height,x:x+width]=requested.astype(np.uint8)*255
    if edge<=size:
        cells=np.asarray(Image.fromarray(padded).resize((size,size),Image.Resampling.NEAREST))>0
    else:
        yy,xx=np.nonzero(requested);xx=xx+x;yy=yy+y
        x0=(xx*size)//edge;x1=((xx+1)*size-1)//edge
        y0=(yy*size)//edge;y1=((yy+1)*size-1)//edge
        cells=np.zeros((size,size),dtype=bool)
        for cy,cx in ((y0,x0),(y0,x1),(y1,x0),(y1,x1)):cells[cy,cx]=True
    cover=np.asarray(Image.fromarray((cells*255).astype(np.uint8)).resize((edge,edge),Image.Resampling.NEAREST))[y:y+height,x:x+width]>0
    if np.any(requested&~cover):raise ValueError('마스크 축소로 보완 영역이 누락됩니다. 범위를 확인하세요.')
    if cover.sum()>width*height*.8:raise ValueError('마스크를 축소하면 보완 영역이 너무 넓어집니다. 범위를 줄이세요.')
    guide=np.full((size,size,4),255,dtype=np.uint8);guide[:,:,3][cells]=0
    return Image.fromarray(guide), int(cover.sum())

def repair(config, send=None):
    if config.get('sourceUnmodifiedVersion')!='background-source-unmodified-v1':
        raise ValueError('배경 보호 기능이 업데이트되었습니다. 작업을 저장하고 사이트를 새로고침한 뒤 다시 시도하세요.')
    key=os.environ.get('OPENAI_API_KEY','')
    if not key or os.environ.get('COEX_ENABLE_OPENAI_IMAGES')!='1' or config.get('externalConsent') is not True:
        raise ValueError('OpenAI API 설정을 확인하세요.')
    source=Image.open(config['input']).convert('RGBA')
    mask=Image.open(config['mask']).convert('L')
    if mask.size!=source.size: raise ValueError('마스크 크기가 다릅니다.')
    before=np.array(source); raw_requested=np.array(mask)>128
    # Keep the selected background's original partial alpha pixels intact. The
    # frontend must send the unchanged source and a separate mask.
    partial=(before[:,:,3]>0)&(before[:,:,3]<255)
    requested=raw_requested&~partial
    if not requested.any() or requested.sum()>source.width*source.height*.8: raise ValueError('보완할 영역이 없거나 너무 넓습니다.')
    w,h=source.size; edge=max(w,h); x=(edge-w)//2; y=(edge-h)//2
    canvas=Image.new('RGBA',(edge,edge),(110,94,80,255)); canvas.alpha_composite(source,(x,y))
    api_mask,projected_count=build_api_mask(requested,edge,x,y,w,h)
    photo=canvas.convert('RGB').resize((1024,1024),Image.Resampling.LANCZOS)
    payload={'model':'gpt-image-2.5-sunburst','images':[{'image_url':'data:image/png;base64,'+base64.b64encode(png(photo)).decode()}], 'mask':{'image_url':'data:image/png;base64,'+base64.b64encode(png(api_mask)).decode()},'prompt':'Fill the masked region with a natural continuation of the surrounding empty background, perspective, texture and lighting. Do not create objects, food, text or logos. '+str(config.get('hint',''))[:300], 'size':'1024x1024','quality':'medium','n':1,'output_format':'png'}
    request=urllib.request.Request('https://api.openai.com/v1/images/edits',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
    start=time.monotonic()
    print(json.dumps({'phase':'OpenAI 배경 후보 생성 중 · 요청 수락 후 중지해도 비용이 발생할 수 있습니다.'},ensure_ascii=True),flush=True)
    try:
        with (send or urllib.request.urlopen)(request,timeout=300) as response: result=json.loads(response.read(40000000))
    except urllib.error.HTTPError as e:
        messages={401:'API 키를 확인하세요.',403:'이미지 모델 사용 권한을 확인하세요.',429:'API 사용 한도 또는 잔액을 확인하세요.'}
        raise ValueError('OpenAI 오류 '+str(e.code)+': '+messages.get(e.code,'요청을 완료하지 못했습니다. 자동 재시도하지 않습니다.')) from None
    except (TimeoutError,urllib.error.URLError):
        raise ValueError('응답을 받지 못했습니다. 사용 내역을 확인한 뒤 다시 실행하세요. 자동 재시도하지 않습니다.') from None
    img=Image.open(io.BytesIO(base64.b64decode(result['data'][0]['b64_json'],validate=True)));img.load()
    if img.size!=(1024,1024): raise ValueError('생성 결과 크기가 맞지 않습니다.')
    patch=np.array(img.convert('RGBA').resize((edge,edge),Image.Resampling.LANCZOS).crop((x,y,x+w,y+h)))
    after=before.copy();after[requested,:3]=patch[requested,:3];after[requested,3]=255
    protected=~requested
    if np.any(after[protected]!=before[protected]) or np.any(after[partial]!=before[partial]): raise ValueError('보호 영역이 변경되었습니다.')
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    Image.fromarray(after).save(out/'restored.png');Image.fromarray((requested*255).astype('uint8')).save(out/'generated-mask.png')
    report={'image':'restored.png','mask':'generated-mask.png','objects':[],'protectedChanged':0,'protectedRgbChanged':0,'method':'openai-background-inpainting','generatedPixels':int(requested.sum()),'partialAlphaExcludedPixels':int((raw_requested&partial).sum()),'partialAlphaChanged':0,'projectedMaskPixels':projected_count,'model':payload['model'],'seconds':time.monotonic()-start,'usage':result.get('usage')}
    (out/'result.json').write_text(json.dumps(report,ensure_ascii=False),encoding='utf8'); return report

def dispatch(config, send=None):
    # Reserved, explicit internal mode carried through the legacy sidecar's
    # background-only matte field; legacy colour values keep the old path.
    if config.get('matte') == 'whole-v1':
        if config.get('task') != 'background' or config.get('sourceUnmodifiedVersion') != 'background-source-unmodified-v1' or config.get('externalConsent') is not True:
            raise ValueError('전체 배경 요청을 확인할 수 없습니다.')
        import importlib.util
        module_path = Path(__file__).with_name('openai-background-candidate.py')
        spec = importlib.util.spec_from_file_location('coex_whole_background_candidate', module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.candidate(config, send=send)
    return repair(config, send=send)

if __name__ == "__main__":
    dispatch(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
