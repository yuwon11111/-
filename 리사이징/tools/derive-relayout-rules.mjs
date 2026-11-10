// Derive per-aspect-ratio layout statistics from hand-labeled multi-size banner sets.
// usage: node tools/derive-relayout-rules.mjs <labels.json> [more labels.json ...] [--out out.json]
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const args=process.argv.slice(2),outAt=args.indexOf('--out');
const output=outAt>=0?args.splice(outAt,2)[1]:path.join(root,'rules','reference-relayout-stats.json');
if(!args.length)throw Error('labels.json path required');
const buckets=[['ultra_wide',3,Infinity,'x'],['wide',1.4,3,'x'],['square',0.85,1.4,'y'],['portrait',0.7,0.85,'y'],['story',0.54,0.7,'y'],['tall',0.45,0.54,'y'],['skyscraper',0,0.45,'y']];
const bucketOf=r=>buckets.find(([,lo,hi])=>r>=lo&&r<hi);
const median=a=>{const s=[...a].sort((p,q)=>p-q),m=s.length>>1;return s.length%2?s[m]:(s[m-1]+s[m])/2};
const r2=v=>Math.round(v*100)/100;
const center=b=>[(b[0]+b[2])/2,(b[1]+b[3])/2];
const overlapY=(a,b)=>Math.min(a[3],b[3])-Math.max(a[1],b[1]);
const groups=Object.fromEntries(buckets.map(([id])=>[id,[]])),all=[];
for(const input of args){
 const labels=JSON.parse(fs.readFileSync(input,'utf8'));
 for(const set of labels.sets)for(const s of set.sizes){const item={...s,set:set.set,bg:s.bg??set.bg??'unknown'};all.push(item);groups[bucketOf(s.w/s.h)[0]].push(item)}
}
const roleStats=items=>{const roles={};for(const s of items)for(const [role,b] of Object.entries(s.el))(roles[role]??=[]).push(b);
 return Object.fromEntries(Object.entries(roles).map(([role,bs])=>[role,{presence:r2(bs.length/items.length),
  box:[0,1,2,3].map(i=>r2(median(bs.map(b=>b[i])))),width:r2(median(bs.map(b=>b[2]-b[0]))),height:r2(median(bs.map(b=>b[3]-b[1]))),
  area:r2(median(bs.map(b=>(b[2]-b[0])*(b[3]-b[1]))))}]))};
const count=(items,key)=>items.reduce((m,s)=>{for(const k of [].concat(key(s)))m[k]=(m[k]||0)+1;return m},{});
const out={sources:args.map(a=>path.relative(root,path.resolve(a)).replaceAll('\\','/')),generatedAt:new Date().toISOString().slice(0,10),samples:all.length,buckets:{},background:{},pairs:[]};
for(const [id,lo,hi,axis] of buckets){
 const items=groups[id];if(!items.length)continue;
 const before={},layouts={stack:0,split:0,row:0,other:0};
 for(const s of items){
  const order=Object.entries(s.el).sort((a,b)=>center(a[1])[axis==='x'?0:1]-center(b[1])[axis==='x'?0:1]).map(([r])=>r);
  for(let i=0;i<order.length;i++)for(let j=i+1;j<order.length;j++)before[order[i]+'<'+order[j]]=(before[order[i]+'<'+order[j]]||0)+1;
  const t=s.el.title,h=s.el.hero;
  if(axis==='x'&&id==='ultra_wide')layouts.row++;
  else if(t&&h&&center(h)[0]-center(t)[0]>0.25&&overlapY(t,h)>0)layouts.split++;
  else if(axis==='y'||!h)layouts.stack++;
  else layouts.other++;
 }
 const roles=roleStats(items),names=Object.keys(roles);
 const wins=r=>names.reduce((n,o)=>n+(before[r+'<'+o]||0)-(before[o+'<'+r]||0),0);
 // two-column banners: horizontal order is meaningless inside the text column, so report columns top-to-bottom
 const mid=(r,i)=>median(items.filter(s=>s.el[r]).map(s=>center(s.el[r])[i]));
 const columns=id==='wide'?{left:names.filter(r=>mid(r,0)<0.5).sort((a,b)=>mid(a,1)-mid(b,1)),right:names.filter(r=>mid(r,0)>=0.5).sort((a,b)=>mid(a,1)-mid(b,1))}:undefined;
 // empty bands at the canvas edges matter for story safe zones
 const edges={top:r2(median(items.map(s=>Math.min(...Object.values(s.el).map(b=>b[1]))))),bottom:r2(median(items.map(s=>1-Math.max(...Object.values(s.el).map(b=>b[3])))))};
 out.buckets[id]={ratio:[lo,hi===Infinity?null:hi],axis,samples:items.length,sizes:[...new Set(items.map(s=>s.w+'x'+s.h))],
  layout:Object.fromEntries(Object.entries(layouts).filter(([,n])=>n)),flags:count(items,s=>s.flags||[]),bg:count(items,s=>s.bg),edges,
  ...(columns?{columns}:{order:names.sort((a,b)=>wins(b)-wins(a))}),roles};
}
for(const [bg,items] of Object.entries(Object.groupBy(all,s=>s.bg)))
 out.background[bg]={samples:items.length,buckets:count(items,s=>bucketOf(s.w/s.h)[0]),textOnPhoto:r2(items.filter(s=>(s.flags||[]).includes('text_on_photo')).length/items.length),roles:roleStats(items)};
// same-design pairs: how each role moves and scales from the shorter to the taller canvas, in canvas-width units
for(const [key,items] of Object.entries(Object.groupBy(all.filter(s=>s.pair),s=>s.set+'/'+s.pair))){
 if(items.length<2)continue;
 const [a,b]=items.sort((p,q)=>q.w/q.h-p.w/p.h),roles={};
 for(const r of Object.keys(a.el).filter(r=>b.el[r])){
  const ha=(a.el[r][3]-a.el[r][1])*a.h/a.w,hb=(b.el[r][3]-b.el[r][1])*b.h/b.w;
  roles[r]={from:a.el[r].map(r2),to:b.el[r].map(r2),scale:r2(hb/ha),centerY:[r2(center(a.el[r])[1]),r2(center(b.el[r])[1])]};
 }
 out.pairs.push({pair:key,from:a.w+'x'+a.h,to:b.w+'x'+b.h,bg:a.bg,roles});
}
fs.writeFileSync(output,JSON.stringify(out,null,1)+'\n');
for(const [id,b] of Object.entries(out.buckets))console.log(id.padEnd(11),String(b.samples).padStart(3),JSON.stringify(b.layout),b.columns?'left['+b.columns.left.join(' > ')+'] right['+b.columns.right.join(' > ')+']':b.order.join(' > '),'| edges',JSON.stringify(b.edges),'| bg',JSON.stringify(b.bg));
for(const p of out.pairs)console.log('pair',p.pair,p.from,'->',p.to,Object.entries(p.roles).map(([r,v])=>r+' x'+v.scale+' y'+v.centerY.join('>')).join(', '));
