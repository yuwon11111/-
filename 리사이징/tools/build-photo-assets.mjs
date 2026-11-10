// Reproducible static transport assets; model values are never quantized or changed.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {gzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),source=path.join(root,'model-source','birefnet-web512'),assets=path.join(root,'dist','vendor','ai');
const manifest=JSON.parse(fs.readFileSync(path.join(source,'model.json'),'utf8')),parts=[],whole=createHash('sha256');let transportBytes=0;
const sha=value=>createHash('sha256').update(value).digest('hex');
for(const part of manifest.chunks){
 if(!/^web512-fp32-\d\d\.bin$/.test(part.path))throw Error('Unexpected model source path');
 const raw=fs.readFileSync(path.join(source,part.path));if(raw.length!==part.bytes||sha(raw)!==part.sha256)throw Error('Model source checksum mismatch: '+part.path);whole.update(raw);
 const packed=gzipSync(raw,{level:9,mtime:0});packed[9]=255;const name=part.path+'.gz';fs.writeFileSync(path.join(assets,name),packed);transportBytes+=packed.length;
 parts.push({path:name,bytes:packed.length,sha256:sha(packed),compression:'gzip',rawBytes:raw.length,rawSha256:part.sha256});
}
if(whole.digest('hex')!==manifest.sha256)throw Error('Complete model checksum mismatch');
fs.writeFileSync(path.join(assets,'model-web512-fp32.json'),JSON.stringify({...manifest,chunks:parts,transport:'gzip-chunks',transportBytes},null,2)+'\n');
const size=folder=>fs.readdirSync(folder,{withFileTypes:true}).reduce((sum,entry)=>sum+(entry.isDirectory()?size(path.join(folder,entry.name)):fs.statSync(path.join(folder,entry.name)).size),0),expandedBytes=size(path.join(root,'dist'));
if(expandedBytes>256*1024*1024-2*1024*1024)throw Error('Static output exceeds the expanded archive budget');
console.log(JSON.stringify({modelBytes:manifest.bytes,transportBytes,staticBytes:expandedBytes,modelSha256:manifest.sha256}));
