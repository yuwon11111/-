const http=require('node:http'),fs=require('node:fs'),path=require('node:path');
const root=path.join(__dirname,'dist');
const types={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.mjs':'text/javascript; charset=utf-8','.css':'text/css','.wasm':'application/wasm','.json':'application/json','.md':'text/plain; charset=utf-8','.bcmap':'application/octet-stream','.ttf':'font/ttf','.pfb':'application/octet-stream'};
const server=http.createServer((req,res)=>{
  let name;try{name=decodeURIComponent(new URL(req.url,'http://localhost').pathname);}catch{res.writeHead(400);res.end();return;}
  const filename=path.resolve(root,'.'+(name==='/'?'/index.html':name));
  if(!filename.startsWith(root+path.sep)){res.writeHead(403);res.end();return;}
  fs.stat(filename,(error,stat)=>{if(error||!stat.isFile()){res.writeHead(404);res.end();return;}
    res.writeHead(200,{'Content-Type':types[path.extname(filename)]||'application/octet-stream','X-Content-Type-Options':'nosniff'});fs.createReadStream(filename).pipe(res);
  });
});
const port=Number(process.env.PORT)||4174;
server.on('error',error=>{console.error(error.code==='EADDRINUSE'?'Port '+port+' is already in use. This editor was not started; existing processes were not changed.':error.message);process.exitCode=1;});
function verifyEditorResponse(url,filename,mime){
  return new Promise((resolve,reject)=>{
    let expected;try{expected=fs.readFileSync(path.join(root,filename));}catch(error){reject(error);return;}
    let request,deadline;
    const fail=error=>{clearTimeout(deadline);reject(error);};
    request=http.get(url,response=>{
      if(response.statusCode!==200||!mime.test(response.headers['content-type']||'')){
        response.resume();fail(new Error('Editor readiness check failed for '+filename));return;
      }
      const chunks=[];let bytes=0;
      response.on('data',chunk=>{
        bytes+=chunk.length;
        if(bytes>expected.length){request.destroy(new Error('Unexpected editor content: '+filename));return;}
        chunks.push(chunk);
      });
      response.on('error',fail);
      response.on('end',()=>{
        clearTimeout(deadline);
        if(!Buffer.concat(chunks).equals(expected)){reject(new Error('Unexpected editor content: '+filename));return;}
        resolve();
      });
    });
    request.on('error',fail);
    deadline=setTimeout(()=>request.destroy(new Error('Editor readiness check timed out: '+filename)),3000);
  });
}
server.listen(port,'127.0.0.1',()=>{
  const url='http://127.0.0.1:'+port+'/';
  console.log('Local: '+url);
  // Opt in only from the user's launcher, after the socket is listening.
  if(process.env.COEX_OPEN_BROWSER==='1'&&process.platform==='win32'){
    Promise.all([
      verifyEditorResponse(url,'index.html',/^text\/html(?:;|$)/i),
      verifyEditorResponse(url+'documents.mjs','documents.mjs',/^(?:text|application)\/javascript(?:;|$)/i)
    ]).then(()=>{
      const browser=require('node:child_process').spawn('explorer.exe',[url],{windowsHide:true,stdio:'ignore'});
      browser.on('error',error=>console.error('Open '+url+' in your browser. '+error.message));
      browser.unref();
    }).catch(error=>console.error('The editor connection could not be verified. '+error.message));
  }
});
