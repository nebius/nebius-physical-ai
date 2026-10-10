"""Define the report's visual design and interactive views without synthetic run data."""

HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Marble × Workbench · World laboratory</title><style>__STYLE__</style></head>
<body><header><a class="brand" href="#">n<span>◈</span> Nebius <b>Physical AI</b></a><div class="header-right"><span class="dot"></span> WORLD LABORATORY <span class="version" id="workflow-number">01 / 02</span></div></header>
<main><section class="intro"><div><p class="eyebrow">WORLD LABS MARBLE × NATIVE WORKBENCH</p><h1 id="title"></h1><p class="subtitle" id="subtitle"></p></div><a class="download" href="result.json" download>↓ Run evidence <span>JSON</span></a></section>
<section class="workspace"><div class="stage"><div class="stage-top"><div class="pills"><button class="selected" data-view="frames" id="frames-tab">GPU capture</button><button data-view="world">Explore world</button><button data-view="points" id="points-tab">Spatial cloud</button></div><span id="view-status">RECORDED ON GPU</span></div>
<img id="frame" alt="Actual GPU output frame"><canvas id="viewport"></canvas><div id="loading" hidden></div>
<div class="stage-caption"><span id="scene-name"></span><span id="frame-label"></span></div><div class="axis">Y <i>↟</i><br> <i>↙</i> X <i>↗</i> Z</div></div>
<aside><p class="eyebrow">EXECUTION RECORD</p><div class="device"><span class="chip">▧</span><div><small>NEBIUS GPU</small><strong id="gpu"></strong></div></div>
<div class="stats"><article><small id="primary-label"></small><strong id="primary"></strong></article><article><small>CUDA median / frame</small><strong id="timing"></strong></article><article><small>Captured frames</small><strong id="count"></strong></article><article><small id="coverage-label">Mean coverage</small><strong id="coverage"></strong></article></div>
<div class="chart"><div><small>CUDA FRAME TIME</small><span id="engine"></span></div><svg id="chart" viewBox="0 0 300 75" role="img" aria-label="Measured CUDA time by frame"></svg><div class="chart-footer"><span>Frame 0</span><span id="last-frame"></span></div></div>
<div class="source"><span class="dot"></span><strong id="source-kind"></strong><p id="source-note"></p></div></aside></section>
<section class="transport"><button id="play" aria-label="Play recorded GPU frames">▶</button><span id="time">000 / 000</span><input id="scrub" aria-label="Frame" type="range" min="0" value="0"><button id="reset">Reset view ↗</button><span class="dimension" id="dimension"></span></section>
<section class="filmstrip" id="filmstrip"></section>
<section class="pipeline"><div class="pipeline-title"><p class="eyebrow">NATIVE WORKFLOW</p><h2>From a world to evidence.</h2></div><article><span>01</span><div><h3>Acquire the world</h3><p>Splats + collision geometry.<br>Source attribution and SHA-256.</p></div></article><i>→</i><article><span>02</span><div><h3 id="pipeline-gpu"></h3><p id="pipeline-description"></p></div></article><i>→</i><article><span>03</span><div><h3>Inspect & export</h3><p>Interactive views and raw outputs.<br>Every frame traces to this run.</p></div></article></section>
<section class="details"><div><p class="eyebrow">WHAT YOU ARE SEEING</p><p id="explanation"></p></div><div class="links"><a href="world.spz" download>Gaussian splats ↗</a><a href="collider.glb" download>Collision mesh ↗</a><a href="depth.npz" id="depth-link" download>Raw depth + cameras ↗</a><a href="result.json" download>Camera poses + timings ↗</a></div></section>
<footer><span>NEBIUS PHYSICAL AI / WORLD LABORATORY</span><span id="run-id"></span></footer></main>
<script id="report" type="application/json">__REPORT__</script>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.178.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.178.0/examples/jsm/","@sparkjsdev/spark":"https://sparkjs.dev/releases/spark/0.1.10/spark.module.js"}}</script>
<script type="module">__SCRIPT__</script></body></html>"""

CSS = """
*{box-sizing:border-box} :root{--bg:#0b0e12;--panel:#13181e;--line:#29323b;--muted:#8998a7;--ink:#e7eee9;--accent:#c6fb5f}body{margin:0;background:var(--bg);color:var(--ink);font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:inherit;text-decoration:none}button,input{font:inherit}button{cursor:pointer}header{height:76px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;padding:0 4vw}.brand{font-size:19px;font-weight:650;display:flex;gap:10px;align-items:center}.brand>span{color:var(--accent);font-size:24px}.brand>b{border-left:1px solid var(--line);padding-left:18px;margin-left:8px;font-weight:400;font-size:14px;color:var(--muted)}.header-right{font:10px monospace;letter-spacing:2px;display:flex;align-items:center;gap:10px}.version{color:var(--muted);margin-left:32px}.dot{width:6px;height:6px;background:var(--accent);border-radius:50%;display:inline-block}.eyebrow{font:10px monospace;letter-spacing:2px;color:var(--muted);margin:0 0 18px}main{max-width:1800px;margin:auto;padding:42px 4vw 0}.intro{display:flex;align-items:center;justify-content:space-between;margin-bottom:30px}h1{font-size:clamp(36px,4vw,65px);font-weight:450;letter-spacing:-3px;margin:0 0 12px}.subtitle{color:var(--muted);font-size:15px;margin:0;max-width:800px;line-height:1.6}.download{padding:15px 18px;background:var(--accent);color:#102005;border-radius:5px;font-weight:600;white-space:nowrap}.download span{font:10px monospace;margin-left:14px;opacity:.6}.workspace{display:grid;grid-template-columns:minmax(0,1fr) 310px;border:1px solid var(--line);border-radius:8px;overflow:hidden}.stage{position:relative;min-height:540px;background:#171d1b;overflow:hidden}#frame,#viewport{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}#viewport{display:none}.stage-top{position:absolute;top:18px;left:20px;right:20px;z-index:3;display:flex;align-items:center;justify-content:space-between;gap:12px}.pills{display:flex;gap:4px;background:#0c111dcc;backdrop-filter:blur(20px);border:1px solid #ffffff24;padding:4px;border-radius:7px}.pills button{color:#c4cdc8;border:0;background:none;padding:9px 12px;border-radius:4px;font-size:12px}.pills button.selected{background:var(--accent);color:#122107}.stage-top>span{font:9px monospace;letter-spacing:1px;color:white;text-shadow:0 1px 6px #000}.stage-caption{position:absolute;bottom:0;padding:65px 24px 22px;left:0;right:0;display:flex;justify-content:space-between;gap:20px;background:linear-gradient(transparent,#070d15b8);pointer-events:none;font:10px monospace;letter-spacing:1px}.axis{position:absolute;bottom:60px;right:25px;text-align:center;font:10px monospace;color:#fff8;pointer-events:none}.axis i{font-size:18px;color:var(--accent)}aside{padding:28px 24px;background:var(--panel);border-left:1px solid var(--line)}.device{display:flex;gap:12px;align-items:center;margin-bottom:24px}.chip{color:var(--accent);font-size:30px}.device small{display:block;color:var(--muted);font:9px monospace;letter-spacing:1px;margin-bottom:8px}.device strong{display:block;font-size:15px;line-height:1.4}.stats{display:grid;grid-template-columns:1fr 1fr;gap:24px 12px;padding:20px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}.stats small{font-size:10px;color:var(--muted);display:block;margin-bottom:10px}.stats strong{font:25px monospace;color:var(--accent)}.chart{margin:24px 0}.chart>div{display:flex;justify-content:space-between;gap:10px;font:9px monospace;color:var(--muted)}.chart svg{width:100%;height:75px;margin-top:15px;overflow:visible}.source{padding:16px;background:#1b2427;border-radius:5px}.source strong{font-size:10px;margin-left:8px;font-weight:500}.source p{font-size:11px;color:var(--muted);line-height:1.6;margin-bottom:0}.transport{display:flex;gap:18px;align-items:center;padding:18px 0}.transport button{background:var(--panel);border:1px solid var(--line);border-radius:5px;color:var(--ink);height:36px;padding:0 14px}.transport>span{font:10px monospace;color:var(--muted);white-space:nowrap}.transport input{flex:1;accent-color:var(--accent);min-width:40px}.filmstrip{display:grid;grid-template-columns:repeat(8,1fr);gap:10px;margin:4px 0 36px}.filmstrip button{padding:0;position:relative;overflow:hidden;border:1px solid var(--line);border-radius:4px;background:var(--panel);aspect-ratio:16/9}.filmstrip img{width:100%;height:100%;object-fit:cover;opacity:.72;transition:opacity .2s}.filmstrip button:hover img{opacity:1}.filmstrip span{position:absolute;bottom:7px;left:8px;color:white;font:9px monospace;text-shadow:0 1px 2px black}.pipeline{display:flex;align-items:center;justify-content:space-between;gap:20px;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:30px 0}.pipeline-title{max-width:250px}.pipeline-title .eyebrow{margin-bottom:10px}h2{font-size:21px;letter-spacing:-.6px;font-weight:450;margin:0}.pipeline article{display:flex;align-items:flex-start;gap:12px}.pipeline article>span{font:11px monospace;color:var(--accent);border:1px solid #c6fb5f33;padding:8px;border-radius:50%}.pipeline h3{font-size:12px;font-weight:550;margin:2px 0 10px}.pipeline article p{font-size:11px;color:var(--muted);line-height:1.7;margin:0}.pipeline>i{color:#506151}.details{display:grid;grid-template-columns:1.3fr 1fr;gap:60px;padding:30px 0}.details p:not(.eyebrow){color:var(--muted);line-height:1.8;font-size:12px;margin:0}.details .eyebrow{margin-bottom:10px}.links{display:grid;grid-template-columns:1fr 1fr;gap:10px;align-content:start}.links a{font-size:11px;padding:13px;border:1px solid var(--line);border-radius:4px}.links a:hover{color:var(--accent);border-color:var(--accent)}footer{border-top:1px solid var(--line);padding:22px 0;display:flex;justify-content:space-between;color:#6d7c89;font:9px monospace;letter-spacing:1px}#loading{position:absolute;z-index:4;left:50%;top:50%;transform:translate(-50%,-50%);padding:18px;background:#0b0e12df;border:1px solid var(--line);border-radius:5px;max-width:80%;font-size:13px;line-height:1.7}.scan{--accent:#71e0f5}.scan .stage{background:#071e28}.scan .download,.scan .pills button.selected{color:#062530}@media(max-width:1100px){.workspace{grid-template-columns:1fr 265px}aside{padding:20px 16px}.stage{min-height:470px}.pipeline{flex-wrap:wrap}.pipeline-title{width:100%;max-width:none}.download{margin-left:20px}}@media(max-width:750px){header{height:60px}.header-right,.brand>b{display:none}main{padding-top:25px}.intro{display:block}.download{display:inline-block;margin:20px 0 0}h1{letter-spacing:-1.5px}.workspace{display:flex;flex-direction:column}.stage{min-height:360px}.stage-top{left:10px;right:10px}.stage-top>span{display:none}aside{border-left:0;border-top:1px solid var(--line)}aside>.eyebrow,.chart{display:none}.device{margin-bottom:15px}.stats{grid-template-columns:repeat(4,1fr);gap:10px}.stats strong{font-size:20px}.source{margin-top:16px}.filmstrip{grid-template-columns:repeat(4,1fr);gap:6px}.pipeline>i{display:none}.pipeline article{width:100%}.details{grid-template-columns:1fr;gap:24px}.transport{gap:8px}.dimension,#reset{display:none}footer{font-size:8px;gap:20px}.pills button{font-size:10px;padding:8px}}
"""

JAVASCRIPT = r"""
const report=JSON.parse(document.querySelector('#report').textContent);
const scan=report.kind==='scan';
if(scan) document.body.classList.add('scan');
const el=id=>document.getElementById(id);
const text=(id,value)=>el(id).textContent=value;
text('title',scan?'Make space measurable.':'Step inside a Marble world.');
text('subtitle',scan?'Real collision geometry. Millions of CUDA ray queries. An inspectable spatial dataset.':'A Marble world becomes a reproducible camera dataset, rendered on a Nebius GPU.');
text('workflow-number',scan?'02 / 02':'01 / 02');
text('frames-tab',scan?'Depth scan':'GPU capture');
el('points-tab').hidden=!scan; el('depth-link').hidden=!scan;
text('gpu',report.gpu.name.replace('NVIDIA ','').replace('Blackwell Server Edition',''));
text('primary-label',scan?'CUDA ray queries':'Gaussian primitives');
text('primary',new Intl.NumberFormat('en',{notation:'compact',maximumFractionDigits:1}).format(scan?report.metrics.rays:report.metrics.gaussians));
const timing=[...report.metrics.cuda_frame_ms].sort((a,b)=>a-b);
const middle=Math.floor(timing.length/2);
text('timing',((timing[middle]+timing[Math.floor((timing.length-1)/2)])/2).toFixed(2)+' ms');
text('count',report.frames.toString().padStart(3,'0'));
text('coverage-label',scan?'Ray hit rate':'Alpha > 0.5 pixels');
text('coverage',(100*report.metrics.coverage.reduce((a,b)=>a+b,0)/report.frames).toFixed(1)+'%');
text('engine',report.metrics.engine.toUpperCase());
text('last-frame','Frame '+(report.frames-1));
text('source-kind',report.world.generated_this_run?'Generated in this run':'Real imported Marble world');
text('source-note',report.world.generated_this_run?'World API generation followed by measured Nebius CUDA execution.':'MIT-licensed upstream example. This run executed the GPU consumer; it did not generate the world.');
text('dimension',report.width+' × '+report.height);
text('scene-name',report.world.display_name);
text('pipeline-gpu',scan?'Scan on CUDA':'Render on CUDA');
text('pipeline-description',scan?'NVIDIA Warp triangle raycasts.\nRaw depth + real hit statistics.':'gsplat Gaussian rasterization.\nRGB frames + explicit camera poses.');
text('explanation',scan?'The colored frames encode actual ray distances to the imported collision mesh: blue is near, red is far (clipped at the 95th percentile). Black means no return. The spatial cloud contains sampled hit positions from those same queries. Distances use '+report.world.units+'. The camera sweep is exploratory and is not a validated robot navigation path.':'The recorded frames were rasterized from the real Gaussian primitives using gsplat on CUDA. The interactive world uses Spark in your browser. CUDA timing includes rasterization, while saved images use SH degree 0 color; browser rendering may look different. No robot training or simulator validation is implied.');
text('run-id',report.run_id);
const maxTime=Math.max(...timing);
const path=report.metrics.cuda_frame_ms.map((v,i)=>(i?'L':'M')+(i/Math.max(1,report.frames-1)*300).toFixed(2)+','+(70-v/maxTime*60).toFixed(2)).join(' ');
el('chart').innerHTML='<path d="'+path+'" fill="none" stroke="var(--accent)" stroke-width="1.6"/><path d="M0,72 L300,72" stroke="#35424c"/>';
let index=0,playing=false,last=0,view='frames';
el('scrub').max=report.frames-1;
function showFrame(value){index=Number(value);el('frame').src='frames/'+String(index).padStart(4,'0')+'.jpg';el('scrub').value=index;text('time',String(index+1).padStart(3,'0')+' / '+String(report.frames).padStart(3,'0'));text('frame-label','FRAME '+String(index).padStart(4,'0')+' · '+report.metrics.cuda_frame_ms[index].toFixed(2)+' MS CUDA');}
const openingFrame=Math.floor(report.frames/2);
showFrame(openingFrame);
el('scrub').oninput=event=>showFrame(event.target.value);
el('play').onclick=()=>{playing=!playing;text('play',playing?'Ⅱ':'▶');};
for(let i=0;i<8;i++){const frame=Math.floor(i*report.frames/8),button=document.createElement('button'),img=document.createElement('img'),label=document.createElement('span');img.src='frames/'+String(frame).padStart(4,'0')+'.jpg';img.alt='Recorded GPU frame '+frame;label.textContent=String(frame).padStart(4,'0');button.append(img,label);button.onclick=()=>showFrame(frame);el('filmstrip').append(button);}
let THREE,renderer,scene,camera,controls,world,points,orbit;
async function initialize(){
 if(renderer)return;
 THREE=await import('three');
 renderer=new THREE.WebGLRenderer({canvas:el('viewport'),antialias:false});
 renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));scene=new THREE.Scene();scene.background=new THREE.Color(0x0b0e12);
 camera=new THREE.PerspectiveCamera(75,1,0.05,1000);
 const {SparkControls}=await import('@sparkjsdev/spark');controls=new SparkControls({canvas:el('viewport')});
 const {OrbitControls}=await import('three/addons/controls/OrbitControls.js');orbit=new OrbitControls(camera,el('viewport'));orbit.enabled=false;
 new ResizeObserver(()=>{const r=el('viewport').getBoundingClientRect();renderer.setSize(r.width,r.height,false);camera.aspect=r.width/r.height;camera.updateProjectionMatrix();}).observe(el('viewport'));
}
async function loadWorld(){
 if(world)return;
 const {SplatMesh}=await import('@sparkjsdev/spark');world=new SplatMesh({url:'world.spz'});
 world.scale.fromArray(report.world.splat_transform.scale);world.position.fromArray(report.world.splat_transform.translation);scene.add(world);await world.initialized;
}
async function loadPoints(){
 if(points)return;
 const response=await fetch('scan-points.json');if(!response.ok)throw new Error('Point-cloud artifact unavailable');
 const data=await response.json(),geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(data.flat(),3));
 const colors=data.flatMap(p=>[0.25+Math.min(Math.abs(p[1])/6,.7),.75,.95]);geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));
 points=new THREE.Points(geometry,new THREE.PointsMaterial({size:.025,vertexColors:true}));scene.add(points);
}
function resetView(){if(!camera)return;const pose=new THREE.Matrix4().fromArray(report.camera_to_world[openingFrame].flat()).transpose();pose.multiply(new THREE.Matrix4().makeScale(1,-1,-1));pose.decompose(camera.position,camera.quaternion,camera.scale);if(view==='points'){camera.position.set(7,6,9);orbit.target.set(0,0,0);orbit.update();}}
el('reset').onclick=resetView;
async function switchView(next){
 view=next;for(const button of document.querySelectorAll('[data-view]'))button.classList.toggle('selected',button.dataset.view===view);
 el('frame').style.display=view==='frames'?'block':'none';el('viewport').style.display=view==='frames'?'none':'block';
 text('view-status',view==='frames'?'RECORDED ON GPU':view==='world'?'DRAG TO LOOK · WASD TO MOVE':'DRAG TO ORBIT · SCROLL TO ZOOM');
 if(view==='frames'){el('loading').hidden=true;return;}
 el('loading').hidden=false;text('loading','Loading real world assets…');
 try{await initialize();if(view==='world')await loadWorld();else await loadPoints();if(world)world.visible=view==='world';if(points)points.visible=view==='points';orbit.enabled=view==='points';resetView();el('loading').hidden=true;}
 catch(error){text('loading','Viewer could not load: '+error.message+'. Serve this bundle over HTTP. The recorded GPU frames remain available.');}
}
for(const button of document.querySelectorAll('[data-view]'))button.onclick=()=>switchView(button.dataset.view);
function animate(now){requestAnimationFrame(animate);if(playing&&view==='frames'&&now-last>1000/24){showFrame((index+1)%report.frames);last=now;}if(renderer&&controls&&orbit&&view!=='frames'){if(view==='world')controls.update(camera);else orbit.update();renderer.render(scene,camera);}}
requestAnimationFrame(animate);
"""
