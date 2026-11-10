"""Offline BiRefNet photo cutout experiment. Does not infer original design layers."""
from pathlib import Path
import argparse,json,sys,time
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'artifacts/ai/python'))
import numpy as np
import onnxruntime as ort
from PIL import Image,ImageOps

def cutout(source,destination):
 model_dir=ROOT/'artifacts/ai/birefnet-lite'
 config=json.loads((model_dir/'preprocessor_config.json').read_text())
 start=time.monotonic()
 options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
 session=ort.InferenceSession(str(model_dir/'model.onnx'),options,providers=['CPUExecutionProvider'])
 image=ImageOps.exif_transpose(Image.open(source)).convert('RGBA')
 size=config.get('size',{'width':1024,'height':1024})
 rgb=image.convert('RGB').resize((size['width'],size['height']),Image.Resampling.BILINEAR)
 values=np.asarray(rgb,dtype=np.float32)/255
 values=(values-np.asarray(config['image_mean'],dtype=np.float32))/np.asarray(config['image_std'],dtype=np.float32)
 tensor=values.transpose(2,0,1)[None].copy()
 result=session.run(None,{session.get_inputs()[0].name:tensor})[-1]
 logits=result.squeeze();mask=1/(1+np.exp(-np.clip(logits,-80,80)))
 alpha=Image.fromarray(np.round(mask*255).astype(np.uint8)).resize(image.size,Image.Resampling.BILINEAR)
 original=np.asarray(image.getchannel('A'),dtype=np.uint16);combined=np.asarray(alpha,dtype=np.uint16)*original//255
 image.putalpha(Image.fromarray(combined.astype(np.uint8)))
 destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True);image.save(destination,'PNG')
 report={'model':'BiRefNet_lite-ONNX','runtime':ort.__version__,'provider':session.get_providers(),'elapsed_seconds':round(time.monotonic()-start,3),'input':str(Path(source).resolve()),'output':str(destination.resolve()),'size':image.size,'model_input':session.get_inputs()[0].shape,'raw_output_range':[float(logits.min()),float(logits.max())],'transparent_pixels':int((combined==0).sum()),'opaque_pixels':int((combined==255).sum()),'network_requests':0,'scope':'photo foreground estimate; not OCR, original-layer recovery, or AI design'}
 destination.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 return report

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('input');parser.add_argument('output');args=parser.parse_args()
 print(json.dumps(cutout(args.input,args.output),ensure_ascii=False))
