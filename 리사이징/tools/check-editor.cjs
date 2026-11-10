const fs=require('node:fs'),path=require('node:path');

const required=[
 'index.html','documents.mjs','pptx-native-elements.mjs','pptx-object-layers.mjs',
 'project-fonts.mjs','layers.mjs','vendor/layers/fabric.mjs',
 'vendor/pptx/pptx-renderer.js','vendor/capture/html-to-image.js',
 'vendor/pdfjs/pdf.min.mjs','vendor/pdfjs/pdf.worker.min.mjs'
];
const existingUrl='http://127.0.0.1:8179/';

function checkEditor(projectRoot){
 const dist=path.join(projectRoot,'dist');
 const files=required.map(name=>{
  const filename=path.join(dist,name);
  try{const stat=fs.statSync(filename);return {file:name,ok:stat.isFile()&&stat.size>0,bytes:stat.size};}
  catch(error){if(error.code==='ENOENT')return {file:name,ok:false,reason:'missing'};return {file:name,ok:false,reason:error.code||error.message};}
 });
 return {scope:'Required local editor/PDF/PPTX files only; no server or model started',ok:files.every(file=>file.ok),files};
}

function readResponse(url,mime,maxBytes,http=require('node:http')){
 return new Promise((resolve,reject)=>{
  let request,deadline;
  const fail=error=>{clearTimeout(deadline);reject(error);};
  request=http.get(url,response=>{
   if(response.statusCode!==200||!mime.test(response.headers['content-type']||'')){
    response.resume();fail(new Error('Unexpected service response: '+url));return;
   }
   const chunks=[];let bytes=0;
   response.on('data',chunk=>{
    bytes+=chunk.length;
    if(bytes>maxBytes){request.destroy(new Error('Unexpected response size: '+url));return;}
    chunks.push(chunk);
   });
   response.on('error',fail);
   response.on('aborted',()=>fail(new Error('Incomplete service response: '+url)));
   response.on('end',()=>{clearTimeout(deadline);resolve(Buffer.concat(chunks));});
  });
  request.on('error',fail);
  deadline=setTimeout(()=>request.destroy(new Error('Connection check timed out: '+url)),3000);
 });
}

async function verifyExistingEditor(projectRoot,http){
 const healthBytes=await readResponse(existingUrl+'api/health',/^application\/json(?:;|$)/i,16384,http);
 let health;try{health=JSON.parse(healthBytes.toString('utf8'));}catch{throw new Error('The existing port did not return valid editor health.');}
 if(health?.service!=='coex-local-photo-ai')throw new Error('Port 8179 is not the expected editor service.');
 const assets=[['index.html',existingUrl,/^text\/html(?:;|$)/i],['documents.mjs',existingUrl+'documents.mjs',/^(?:text|application)\/javascript(?:;|$)/i]];
 await Promise.all(assets.map(async([filename,url,mime])=>{
  const expected=fs.readFileSync(path.join(projectRoot,'dist',filename));
  const actual=await readResponse(url,mime,expected.length,http);
  if(!actual.equals(expected))throw new Error('The running editor does not match this project: '+filename);
 }));
 return {ok:true,url:existingUrl,service:health.service,reusedExistingServer:true,checkedFiles:assets.map(a=>a[0]),serverStarted:false};
}

function openExistingEditor(url,spawn=require('node:child_process').spawn){
 return new Promise((resolve,reject)=>{
  const browser=spawn('explorer.exe',[url],{windowsHide:true,stdio:'ignore'});
  browser.once('error',reject);
  browser.once('spawn',()=>{browser.unref();resolve();});
 });
}

async function main(args=process.argv.slice(2)){
 const projectRoot=path.resolve(__dirname,'..');
 const preflight=checkEditor(projectRoot);
 if(!preflight.ok){console.log(JSON.stringify(preflight,null,2));console.error('Required editor files are unavailable. Restore the listed project files before starting.');process.exitCode=1;return;}
 if(!args.includes('--check-existing')&&!args.includes('--open-existing')){console.log(JSON.stringify(preflight,null,2));return;}
 try{
  const result=await verifyExistingEditor(projectRoot);
  if(args.includes('--open-existing'))await openExistingEditor(result.url);
  console.log(JSON.stringify(result,null,2));
 }catch(error){
  console.error('Could not reuse the existing editor at '+existingUrl+' '+error.message);
  console.error('No server was started, stopped or moved to another port.');
  process.exitCode=1;
 }
}
if(require.main===module)void main();
module.exports={checkEditor,required,verifyExistingEditor,readResponse,openExistingEditor,main};
