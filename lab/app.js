'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const views = [
    ['overview','Performance overview','◫','Overview'],
    ['timeline','GPU timeline','≋','Timeline'],
    ['bottlenecks','Bottleneck explorer','⌁','Bottlenecks'],
    ['hypotheses','Hypothesis laboratory','⌘','Hypotheses'],
    ['evolution','Optimization evolution','⑂','Evolution'],
    ['code','Code comparison','⌥','Code'],
    ['atlas','Performance atlas','▦','Atlas'],
    ['research','Research results','↗','Research']
  ];
  const titles = {
    overview:['Evidence before optimization.','A workbench for asking why a workload is slow, testing a better explanation, and keeping the proof.'],
    timeline:['Follow the execution.','Host and device activity from recorded traces. A timing sample is not a kernel execution trace.'],
    bottlenecks:['Find the limiting factor.','Separate measured characteristics from plausible explanations. Missing evidence stays visible.'],
    hypotheses:['A better question. A controlled test.','Competing explanations, disconfirming experiments, and the evidence that changes the next decision.'],
    evolution:['Keep the unsuccessful experiments.','Trace every attempted intervention. Rejected candidates are part of the result.'],
    code:['Inspect what actually changed.','Source, configuration, numerical checks and performance evidence, side by side.'],
    atlas:['A memory for experiments.','Search recorded investigations by workload, shape, device and evidence type.'],
    research:['Put the selection policy to the test.','Held-out comparisons, ablations and resource costs. Software validation and GPU research results are distinct.']
  };
  const state = {record:null, records:[], view:'overview', chosen:null, evaluation:null};
  const el = (tag, attrs = {}, ...children) => {
    const node = document.createElement(tag);
    for (const [key,value] of Object.entries(attrs)) {
      if (key === 'class') node.className = value;
      else if (key === 'onclick') node.addEventListener('click',value);
      else node.setAttribute(key,value);
    }
    for (const child of children.flat(Infinity)) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  };
  const num = (v,d=2) => typeof v === 'number' && Number.isFinite(v) ? v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d}) : '—';
  const finite = v => typeof v === 'number' && Number.isFinite(v);
  const arr = v => Array.isArray(v) ? v : [];
  const fmt = v => typeof v === 'object' && v !== null ? JSON.stringify(v) : String(v ?? 'Not recorded');
  const badge = (text,style='') => el('span',{class:`badge ${style}`},text==='accepted'?'accepted · reported':text);
  const median = a => {const sorted = a.filter(finite).sort((a,b)=>a-b); const n=sorted.length; return n ? n%2 ? sorted[(n-1)/2] : (sorted[n/2-1]+sorted[n/2])/2 : undefined;};
  const samples = x => arr(x).filter(v=>finite(v)&&v>0);
  const baselineSamples = r => samples(r?.baseline?.samples_us);
  const measured = e => e?.status === 'accepted' && e?.confirmation?.comparison ? e.confirmation : e?.timing;
  const caseCount = v => Array.isArray(v) ? v.length : v;
  const bestExperiment = r => arr(r?.experiments).find(x=>x.id===r?.winner_id) || arr(r?.experiments).filter(x=>x.correctness?.passed && samples(x.timing?.candidate_us).length).sort((a,b)=>median(samples(a.timing.candidate_us))-median(samples(b.timing.candidate_us)))[0];
  const chosenExperiment = () => arr(state.record?.experiments).find(x=>x.id===state.chosen) || bestExperiment(state.record) || arr(state.record?.experiments)[0];
  const evidenceLabel = r => r.evidence_kind === 'cuda_run' ? 'CUDA run' : r.evidence_kind === 'cpu_control' ? 'CPU control' : 'Reference record';
  const recordName = r => r.title || `${r.workload.family} · ${r.workload.shape.join(' × ')}`;
  const empty = (title,body,symbol='∅') => el('div',{class:'empty'},el('div',{class:'empty-symbol','aria-hidden':'true'},symbol),el('h3',{},title),el('p',{},body));
  const panel = (title,subtitle,body,tag) => el('section',{class:'panel'},el('div',{class:'panel-head'},el('div',{},el('h2',{},title),subtitle&&el('p',{class:'panel-subtitle'},subtitle)),tag),body);
  const body = (...children) => el('div',{class:'panel-body'},children);
  const dl = entries => el('dl',{class:'definition-list'},entries.map(([k,v])=>[el('dt',{},k),el('dd',{},fmt(v))]));
  const notice = (title,text) => el('div',{class:'notice'},el('strong',{},title),text);
  function message(text,error=false) {const n=$('feedback');n.textContent=text;n.hidden=false;n.className=error?'error':'';}
  const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
  function validatePhase(phase,label) {
    if(!object(phase)) throw Error(`${label} requires paired timing evidence.`);
    const b=phase.baseline_us,c=phase.candidate_us,comp=phase.comparison;
    if(!Array.isArray(b)||!Array.isArray(c)||b.length!==c.length||b.length<20||[...b,...c].some(v=>!finite(v)||v<=0)) throw Error(`${label} requires at least 20 positive matched timing pairs.`);
    if(!object(comp)||![comp.speedup,comp.ci_low,comp.ci_high,comp.min_speedup??1.03].every(finite)) throw Error(`${label} needs finite comparison statistics.`);
    const threshold=comp.min_speedup??1.03,observed=median(b)/median(c);
    if(comp.accepted!==true||threshold<1.03||comp.ci_low<=threshold||comp.ci_high<comp.ci_low||comp.n!==b.length||Math.abs(comp.speedup-observed)>Math.max(1e-12,Math.abs(observed)*1e-9)) throw Error(`${label} does not support the declared promotion gate.`);
  }
  function validate(r) {
    if (!object(r) || r.schema_version !== 1 || typeof r.id !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/.test(r.id) || !object(r.workload)) throw Error('Expected a version 1 KERNELDARWIN investigation record with a safe identifier.');
    if (!['accepted','no_improvement','blocked','failed'].includes(r.status)) throw Error('The record has an unsupported investigation status.');
    if (!['cuda_run','cpu_control','reference'].includes(r.evidence_kind)) throw Error('The record needs an explicit CUDA, CPU control, or reference evidence label.');
    const w=r.workload;
    if (!['sum','softmax','layernorm','fused'].includes(w.family) || !Array.isArray(w.shape) || w.shape.length !== 2 || w.shape.some(v=>!Number.isSafeInteger(v)||v<1)) throw Error('The workload must declare a supported family and two positive dimensions.');
    if(!['float32','float16','bfloat16'].includes(w.dtype)||!Number.isSafeInteger(w.stride)||w.stride<1||!finite(w.eps)||w.eps<=0||!Number.isSafeInteger(w.seed)||w.seed<0) throw Error('The workload has invalid precision, stride, epsilon or seed metadata.');
    if(!object(r.device)||!['cpu','cuda'].includes(r.device.kind)) throw Error('The record must declare a CPU or CUDA device.');
    if(w.actual_strides!==undefined&&(!Array.isArray(w.actual_strides)||w.actual_strides.length!==2||w.actual_strides.some(v=>!Number.isSafeInteger(v)||v<1)))throw Error('Actual strides must contain two positive element strides.');
    for(const key of ['environment','provenance','trace','diagnosis','budget','selection','baseline','compiled_baseline'])if(r[key]!==undefined&&!object(r[key]))throw Error(`${key} must be an object.`);
    if (r.evidence_kind === 'cuda_run' && (r.device.kind !== 'cuda'||r.provenance?.gpu_execution!==true)) throw Error('A CUDA evidence label requires a CUDA device and explicit GPU execution provenance.');
    if (r.evidence_kind === 'cpu_control' && r.device.kind !== 'cpu') throw Error('A CPU control must declare a CPU device.');
    if(r.evidence_kind==='reference'&&r.status==='accepted') throw Error('A reference record cannot claim an accepted optimization.');
    for(const key of ['experiments','hypotheses']) if(r[key]!==undefined&&(!Array.isArray(r[key])||r[key].some(x=>!object(x)))) throw Error(`${key} must be an array of objects.`);
    if(r.trace?.events!==undefined&&(!Array.isArray(r.trace.events)||r.trace.events.some(x=>!object(x)))) throw Error('Trace events must be an array of objects.');
    const checkSamples=(a,label)=>{if(a!==undefined&&(!Array.isArray(a)||a.some(v=>!finite(v)||v<=0)))throw Error(`${label} must be positive finite timing samples.`);};
    checkSamples(r.baseline?.samples_us,'Baseline samples');checkSamples(r.compiled_baseline?.samples_us,'Compiled baseline samples');
    const ids=new Set();let winner=null;
    for(const e of arr(r.experiments)) {
      if(typeof e.id!=='string'||!e.id||ids.has(e.id)) throw Error('Each experiment needs a unique string identifier.');ids.add(e.id);
      for(const key of ['config','correctness','timing','confirmation','source','safety'])if(e[key]!==undefined&&!object(e[key]))throw Error(`Experiment ${key} must be an object.`);
      if(e.config!==undefined&&(!Number.isSafeInteger(e.config.block_size)||e.config.block_size<1||!Number.isSafeInteger(e.config.num_warps)||e.config.num_warps<1))throw Error('Candidate configuration requires positive integer block size and warp count.');
      if(e.source&&['baseline','candidate'].some(key=>e.source[key]!==undefined&&typeof e.source[key]!=='string'))throw Error('Candidate sources must be text.');
      checkSamples(e.timing?.baseline_us,'Candidate baseline samples');checkSamples(e.timing?.candidate_us,'Candidate samples');
      if(e.status==='accepted'&&(e.correctness?.passed!==true||e.timing?.comparison?.accepted!==true)) throw Error('An accepted experiment requires passing correctness and performance gates.');
      if(e.id===r.winner_id)winner=e;
    }
    if(r.winner_id!=null&&(!winner||winner.status!=='accepted')) throw Error('The winner must identify an accepted experiment.');
    if(r.status==='accepted'&&!winner)throw Error('An accepted investigation requires a winner.');
    if(winner){
      validatePhase(winner.timing,'Discovery');validatePhase(winner.confirmation,'Independent confirmation');
      const confirmation=winner.confirmation;
      if(confirmation.correctness_passed!==true)throw Error('Independent confirmation needs a recorded correctness pass.');
      if(!Number.isSafeInteger(confirmation.seed)||confirmation.seed===winner.timing.comparison.seed)throw Error('Independent confirmation requires a different random seed.');
      if(!Array.isArray(confirmation.actual_strides)||confirmation.actual_strides.length!==2||confirmation.actual_strides.some(v=>!Number.isSafeInteger(v)||v<1)||JSON.stringify(confirmation.actual_strides)!==JSON.stringify(w.actual_strides))throw Error('Independent confirmation strides must match the declared workload.');
    }
    const check = (value,depth=0) => {
      if (depth>40) throw Error('The record is nested too deeply.');
      if (typeof value==='number' && !Number.isFinite(value)) throw Error('The record contains a non-finite number.');
      if (Array.isArray(value)) value.forEach(x=>check(x,depth+1));
      else if (value && typeof value==='object') Object.values(value).forEach(x=>check(x,depth+1));
    }; check(r);
    return r;
  }
  function activate(r,announce=false) {
    validate(r);
    const previous={record:state.record,chosen:state.chosen,records:state.records.slice()};
    state.record=r;state.chosen=null;
    const i=state.records.findIndex(x=>x.id===r.id);
    if(i<0)state.records.push(r);else state.records[i]=r;
    try {render();} catch(error) {Object.assign(state,previous);render();throw error;}
    if(announce)message(`Loaded ${recordName(r)}. Run kd validate record.json to independently recompute timing gates. This viewer checks consistency only; source code is never executed.`);
  }
  async function importFile(file) {
    if(!file)return;
    if(file.size>20*1024*1024){message('This record exceeds the 20 MB import limit. Export a smaller investigation record.',true);return;}
    try {const r=validate(JSON.parse(await file.text()));activate(r,true);}
    catch(e){message(`Could not import record: ${e.message} Your current investigation is preserved.`,true);}
    $('file').value='';
  }
  function svg(tag,attrs={},children=[]) {
    const node=document.createElementNS('http://www.w3.org/2000/svg',tag);
    Object.entries(attrs).forEach(([k,v])=>node.setAttribute(k,String(v)));
    children.flat().forEach(c=>node.append(c instanceof Node?c:document.createTextNode(String(c))));return node;
  }
  function distribution(base,cand) {
    if(!base.length)return empty('No latency samples recorded.','Import a completed run to compare measurement distributions.');
    const all=[...base,...cand];let low=all.reduce((a,b)=>Math.min(a,b),Infinity),high=all.reduce((a,b)=>Math.max(a,b),-Infinity);if(high===low){low*=.95;high*=1.05;}
    const bins=24,counts=a=>{const c=Array(bins).fill(0);a.forEach(v=>c[Math.min(bins-1,Math.floor((v-low)/(high-low)*bins))]++);return c;};
    const bc=counts(base),cc=counts(cand),maxCount=Math.max(...bc,...cc,1),w=window.innerWidth<=650?Math.max(280,window.innerWidth-72):600,h=230,left=window.innerWidth<=650?32:45,right=15,top=20,bottom=37,pw=w-left-right,ph=h-top-bottom;
    const s=svg('svg',{viewBox:`0 0 ${w} ${h}`,class:'chart',role:'img','aria-label':`Latency histogram in microseconds. Baseline median ${num(median(base))}; candidate median ${num(median(cand))}.`});
    for(let i=0;i<4;i++){const y=top+ph*i/3;s.append(svg('line',{x1:left,y1:y,x2:w-right,y2:y,class:'axis'}));s.append(svg('text',{x:left-8,y:y+3,'text-anchor':'end'},[String(Math.round(maxCount*(1-i/3)))]));}
    [bc,cc].forEach((data,series)=>data.forEach((n,i)=>{if(!n)return;const bh=n/maxCount*ph;const bw=pw/bins;s.append(svg('rect',{x:left+i*bw+series*bw*.39+1,y:top+ph-bh,width:bw*.42,height:bh,fill:series?'#c63f27':'#414a3d',opacity:series?'.9':'.75'}));}));
    for(let i=0;i<5;i++){const x=left+pw*i/4;s.append(svg('text',{x,y:h-13,'text-anchor':'middle'},[num(low+(high-low)*i/4,1)]));}
    s.append(svg('text',{x:w-right,y:12,'text-anchor':'end'},['SAMPLES / µs']));
    return el('div',{},s,el('div',{class:'legend'},el('span',{},el('i',{}),`Baseline · n=${base.length}`),cand.length&&el('span',{},el('i',{class:'candidate'}),`Candidate · n=${cand.length}`)));
  }
  function stat(label,value,unit,note,cls=''){return el('div',{class:'stat'},el('div',{class:'stat-label'},label,el('span',{'aria-hidden':'true'},'↗')),el('div',{class:`stat-value ${cls}`},value,unit&&el('span',{class:'stat-unit'},unit)),el('div',{class:'stat-note'},note));}
  function renderOverview() {
    const r=state.record,e=bestExperiment(r),m=measured(e),b=samples(m?.baseline_us).length?samples(m.baseline_us):baselineSamples(r),c=samples(m?.candidate_us),comp=m?.comparison;
    const accepted=!!r?.winner_id && e?.status==='accepted' && e?.correctness?.passed && comp?.accepted;
    const holder=el('div',{},el('div',{class:'stats'},stat('BASELINE LATENCY',num(median(b)),'µs',b.length?`${b.length} ${e?.confirmation?'confirmation':'recorded'} measurements`:'Awaiting measurements'),stat(accepted?'REPORTED ACCEPTED CANDIDATE':'BEST MEASURED CANDIDATE',num(median(c)),'µs',e?e.id:'No candidate recorded'),stat('MEASURED SPEEDUP',num(comp?.speedup),'×',comp?`95% CI ${num(comp.ci_low)}–${num(comp.ci_high)}`:'Requires paired measurements',accepted?'good-text':''),stat('CANDIDATES TESTED',r?String(arr(r.experiments).length):'—','',r?`${arr(r.experiments).filter(x=>x.correctness?.passed).length} passed numerical checks`:'Nothing inferred from empty data')));
    if(r && r.evidence_kind!=='cuda_run')holder.append(notice('GPU performance is not established.',r.evidence_kind==='cpu_control'?'This is a real CPU control experiment. Its measurements validate the investigation workflow and do not establish CUDA speedups or GPU policy quality.':'This is a reference record. The displayed source is not an independently reproduced GPU result.'));
    holder.append(el('div',{class:'grid-two'},panel('The latency distribution','Repeated measurements, not a single favorable sample.',body(distribution(b,c)),badge('RECORDED SAMPLES')),panel('Investigation brief','What the experiment supports.',body(el('p',{class:'summary'},r?.diagnosis?.summary||'Start with a recorded workload. Follow each hypothesis to its test, each candidate to its numerical checks, and each improvement to its measurement uncertainty.'),dl([['Decision',r?.status||'No record'],['Evidence',r?evidenceLabel(r):'Awaiting import'],['Winner',r?.winner_id||'None accepted']]),el('div',{class:'divider'}),el('div',{class:'micro muted'},'LIMITS OF THE RESULT'),el('ul',{class:'limit-list'},arr(r?.diagnosis?.limitations).length?arr(r.diagnosis.limitations).map(x=>el('li',{},fmt(x))):el('li',{},'No GPU measurements are bundled unless a genuine CUDA run is explicitly recorded.'))))));
    holder.append(el('div',{class:'process'},['Profile','Hypothesize','Experiment','Explain','Optimize','Verify','Learn'].map((name,i)=>el('div',{class:'process-step'},el('span',{},`0${i+1}`),el('strong',{},name),el('small',{},['Establish a baseline','Name a falsifier','Change one factor','Keep uncertainty','Test an intervention','Gate the result','Retain the outcome'][i])))));
    holder.append(el('div',{class:'grid-two equal'},panel('Workload fingerprint','Enough context to reproduce the question.',body(dl([['Operation',r?.workload?.family],['Shape',r?.workload?.shape?.join(' × ')],['Precision',r?.workload?.dtype],['Input stride',r?.workload?.stride],['Seed',r?.workload?.seed],['Device',r?.device?.name]]))),panel('Verification ledger','An improvement earns its status.',body(dl([['Numerical correctness',e?e.correctness?.passed?'Passed (record)':'Failed / unrecorded':'Not run'],['Validation cases',caseCount(e?.correctness?.tests)],['Memory safety',e?.safety?.status||'Not recorded'],['Independent timing gate',comp?comp.accepted?'Passed (record)':'Not accepted':'Not run'],['torch.compile baseline',r?.compiled_baseline?.status||'Not recorded'],['Execution trace',r?.trace?.status||'Unavailable']])))));
    if(!r)holder.append(el('div',{class:'intro-card'},el('div',{class:'micro'},'START WITH ONE QUESTION'),el('h2',{},'Why is this workload slow?'),el('p',{},'Run an investigation with the CLI, then import its JSON record. The laboratory reads the evidence without compiling or running uploaded code.'),el('code',{},'kerneldarwin investigate --help')));
    return holder;
  }
  function renderTimeline() {
    const trace=state.record?.trace;
    if(trace?.status!=='recorded'||!arr(trace.events).length)return panel('GPU execution trace','Only actual recorded host/device activity appears here.',empty('No execution trace is available.','Latency samples do not reveal kernel overlap, host gaps or synchronization. Collect a supported profiler trace and retain its events in the investigation record.','≋'),badge('UNAVAILABLE'));
    const events=arr(trace.events).filter(x=>finite(x.ts??x.start_us)&&finite(x.dur??x.duration_us)&&(x.dur??x.duration_us)>=0);
    if(!events.length)return panel('GPU execution trace','Trace event schema is not supported.',empty('No interpretable duration events.','Expected events with ts and dur, or start_us and duration_us, in microseconds.'));
    const start=Math.min(...events.map(x=>x.ts??x.start_us)),end=Math.max(...events.map(x=>(x.ts??x.start_us)+(x.dur??x.duration_us))),span=end-start||1;
    return panel('Recorded activity','Position and duration are derived from recorded timestamps.',body(events.slice(0,300).map(x=>{const a=x.ts??x.start_us,d=x.dur??x.duration_us;const bar=el('div',{class:'timeline-bar'});bar.style.left=`${Math.max(0,(a-start)/span*100)}%`;bar.style.width=`${Math.max(0,d/span*100)}%`;return el('div',{class:'timeline-event'},el('span',{class:'timeline-name'},x.name||x.kind||'Event'),el('div',{class:'timeline-track'},bar),el('span',{},`${num(d,1)} µs`));}),el('div',{class:'timeline-scale'},el('span',{},'0 µs'),el('span',{},`${num(span,1)} µs`)),el('p',{class:'caption'},`Showing ${Math.min(events.length,300)} of ${events.length} events. Relative time from the first event; trace source: ${trace.source||'see record provenance'}.`)),badge('RECORDED'));
  }
  function renderBottlenecks() {
    const r=state.record,counters=r?.counters||r?.profile?.counters||{};
    const metrics=Object.entries(counters).filter(([,v])=>finite(v));
    const holder=el('div',{},notice('A hypothesis is a testable explanation.', 'Timing, tensor shape and analytical estimates can guide an experiment. They do not independently prove occupancy, register pressure or bandwidth saturation.'));
    holder.append(el('div',{class:'grid-two equal'},panel('Hardware observations','Values retain their original metric names and units.',metrics.length?body(dl(metrics)):empty('Hardware counters were not recorded.','Collect Nsight Compute metrics on a supported NVIDIA GPU before making utilization claims.','⌁')),panel('Roofline analysis','A roofline needs measured or declared hardware ceilings.',empty('Insufficient evidence for a roofline.','Measured arithmetic throughput, memory traffic and defensible device compute/bandwidth ceilings are required. No illustrative utilization curve is substituted.','⌜'))));
    holder.append(panel('Diagnostic leads','Read the evidence and run the falsifying test.',arr(r?.hypotheses).length?el('div',{},arr(r.hypotheses).map((h,i)=>el('div',{class:'hypothesis'},el('div',{class:'hypothesis-num'},`LEAD ${String(i+1).padStart(2,'0')}`),el('h3',{},h.title||h.id),el('p',{class:'summary'},arr(h.evidence).map(fmt).join(' ')||'No supporting evidence recorded.'),el('div',{class:'micro muted'},'WHAT WOULD CHANGE THE EXPLANATION'),el('p',{class:'summary'},fmt(h.falsifier))))):empty('No diagnostic leads yet.','An investigation record will connect observed workload properties to controlled tests.')));
    return holder;
  }
  function renderHypotheses() {
    const hs=arr(state.record?.hypotheses);
    if (!hs.length) return panel('Competing explanations', 'Confidence values describe the recorded assessment, not proof of causality.', empty('No hypotheses recorded.', 'Run an investigation to preserve competing explanations, the evidence they use, and the experiments that could disprove them.', '?'), badge('0 HYPOTHESES'));
    const cards=hs.map((h,i)=> {
      const supporting=arr(h.evidence).length ? arr(h.evidence).map(v=>el('li',{},fmt(v))) : el('li',{},'No supporting evidence recorded.');
      const missing=arr(h.missing_evidence).length ? arr(h.missing_evidence).map(v=>el('li',{},fmt(v))) : el('li',{},'No missing evidence declared.');
      return el('article',{class:'hypothesis'},
        el('div',{class:'hypothesis-head'},
          el('div',{},el('div',{class:'hypothesis-num'},`H${String(i+1).padStart(2,'0')} / ${h.id||'UNNAMED'}`),el('h3',{},h.title||'Untitled hypothesis')),
          badge(h.status||'untested',h.status==='supported'?'good':'')),
        el('div',{class:'evidence-grid'},
          el('div',{class:'evidence-box'},el('h4',{},'SUPPORTING EVIDENCE'),el('ul',{},supporting)),
          el('div',{class:'evidence-box'},el('h4',{},'MISSING EVIDENCE'),el('ul',{},missing))),
        el('div',{class:'experiment-test'},
          el('p',{},el('strong',{},'Controlled test: '),fmt(h.experiment)),
          el('p',{},el('strong',{},'Would disprove it: '),fmt(h.falsifier))),
        el('div',{class:'caption'},`Recorded confidence: ${finite(h.confidence)?num(h.confidence,2):fmt(h.confidence)}. Interpret in the context of its evidence.`));
    });
    return panel('Competing explanations','Confidence values describe the recorded assessment, not proof of causality.',el('div',{},cards),badge(`${hs.length} HYPOTHESES`));
  }
  function experimentTable(experiments){return el('div',{class:'table-wrap'},el('table',{},el('thead',{},el('tr',{},['Candidate','Intervention','Correctness','Median / µs','Speedup','Decision'].map(t=>el('th',{scope:'col'},t)))),el('tbody',{},experiments.map(e=>el('tr',{},el('td',{class:'mono'},el('button',{class:'row-button',onclick:()=>{state.chosen=e.id;location.hash='code';if(state.view==='code')render();}},e.id)),el('td',{},e.intervention||fmt(e.config)),el('td',{},e.correctness?.passed?'Passed':'Failed / not run'),el('td',{class:'mono'},num(median(samples(e.timing?.candidate_us)))),el('td',{class:'mono'},`${num(e.timing?.comparison?.speedup)}×`),el('td',{},badge(e.status||'unrecorded',e.status==='accepted'?'good':'')))))));}
  function renderEvolution(){const r=state.record,es=arr(r?.experiments);return el('div',{},panel('Experimental lineage','Parent identifiers retain branching when the record provides it.',es.length?el('div',{class:'evolution-list'},el('div',{class:'evolution-node'},el('div',{class:'evolution-node-head'},el('h3',{},r.baseline?.name||'Reference baseline'),badge('BASELINE')),el('p',{},`${num(median(baselineSamples(r)))} µs median · ${r.device?.name||'device unrecorded'}`)),es.map(e=>el('div',{class:`evolution-node ${e.status||''}`},el('div',{class:'evolution-node-head'},el('h3',{},e.id),badge(e.status||'unrecorded',e.status==='accepted'?'good':'')),el('p',{},`${e.intervention||'Configuration experiment'} · parent ${e.parent_id||'baseline'}`),el('p',{},e.error||e.timing?.comparison?.reason||'See numerical checks and timing samples.'),el('p',{},el('button',{class:'row-button',onclick:()=>{state.chosen=e.id;location.hash='code';}},'Inspect candidate →'))))):empty('No experiments recorded.','Accepted, rejected, invalid and failed candidates all belong in the investigation history.','⑂')),es.length&&panel('Candidate ledger','A gate failure is preserved alongside the measurement.',experimentTable(es),badge(`${es.length} ATTEMPTS`)));}
  function renderCode(){const e=chosenExperiment(),es=arr(state.record?.experiments);if(!e)return panel('Code comparison','Inspect a measured candidate against its declared reference.',empty('No candidate source is available.','Candidate records carry source, compiler details, numerical checks and timing evidence.','{ }'));
    const select=el('select',{class:'candidate-select',id:'candidate-select'},es.map(x=>el('option',{value:x.id},x.id)));select.value=e.id;select.addEventListener('change',()=>{state.chosen=select.value;render();});
    return el('div',{},el('div',{class:'code-controls'},el('label',{for:'candidate-select'},'Candidate',select),badge(e.status||'unrecorded',e.status==='accepted'?'good':'')),el('div',{class:'grid-two equal'},panel('Declared reference','Source from this record.',el('div',{class:'code-panel'},el('div',{class:'code-label'},'BASELINE'),el('pre',{},el('code',{},e.source?.baseline||'Source not recorded.')))),panel('Candidate implementation','Source is displayed as text and never executed.',el('div',{class:'code-panel'},el('div',{class:'code-label'},'CANDIDATE'),el('pre',{},el('code',{},e.source?.candidate||'Source not recorded.'))))),el('div',{class:'grid-two equal'},panel('Numerical & safety checks','A fast result must still compute the declared operation.',body(dl([['Correctness',e.correctness?.passed?'Passed':'Failed / not run'],['Cases',caseCount(e.correctness?.tests)],['Max absolute error',e.correctness?.max_abs_error],['Tolerance',e.correctness?.tolerance||e.correctness?.atol],['Memory safety',e.safety?.status],['Safety note',e.safety?.reason]]))),panel('Compiler & measurement','Compilation cost is separate from steady-state timing.',body(dl([['Configuration',e.config],['Compilation / µs',e.compile_us],['Timing gate',e.timing?.comparison?.reason],['Speedup',e.timing?.comparison?.speedup],['95% interval',e.timing?.comparison?`${num(e.timing.comparison.ci_low)}–${num(e.timing.comparison.ci_high)}`:undefined]])))));
  }
  function renderAtlas(){
    const content=el('div',{}),search=el('input',{type:'search',placeholder:'Search operation, shape, GPU, intervention…','aria-label':'Search investigations'}),filter=el('select',{'aria-label':'Filter by evidence'},el('option',{value:''},'All evidence'),el('option',{value:'cuda_run'},'CUDA runs'),el('option',{value:'cpu_control'},'CPU controls'),el('option',{value:'reference'},'Reference records')),list=el('div',{});
    const update=()=>{const q=search.value.toLowerCase();const records=state.records.filter(r=>(!filter.value||r.evidence_kind===filter.value)&&JSON.stringify([r.id,r.title,r.workload,r.device,arr(r.experiments).map(e=>e.intervention)]).toLowerCase().includes(q));list.replaceChildren(records.length?panel('Recorded investigations',`${records.length} matching record${records.length===1?'':'s'}. Imported files remain in this browser session.`,el('div',{class:'table-wrap'},el('table',{},el('thead',{},el('tr',{},['Investigation','Workload / shape','Device','Evidence','Decision'].map(t=>el('th',{scope:'col'},t)))),el('tbody',{},records.map(r=>el('tr',{},el('td',{},el('button',{class:'row-button',onclick:()=>{activate(r);location.hash='overview';}},recordName(r))),el('td',{class:'mono'},`${r.workload.family} / ${r.workload.shape.join(' × ')}`),el('td',{},r.device?.name||'Not recorded'),el('td',{},badge(evidenceLabel(r),r.evidence_kind==='cuda_run'?'good':'accent')),el('td',{},r.status||'Not recorded'))))))):panel('Recorded investigations','No matching investigations.',empty(state.records.length?'No records match this search.':'The atlas is ready for evidence.',state.records.length?'Try a different operation, shape or evidence filter.':'Import a JSON record from the CLI. A searchable experimental history will appear here.','▦')));};search.addEventListener('input',update);filter.addEventListener('change',update);content.append(el('div',{class:'search-row'},search,filter),list);update();return content;
  }
  function evaluationPanel(evaluation) {
    if(!evaluation) return panel('Held-out policy evaluation','CPU replay and GPU research claims remain separate.',empty('No evaluation result is attached.','Grouped holdouts, random selection, rules and ablations should be recorded with their budgets and limitations. GPU generalization requires GPU evidence.','↗'));
    const aliases={learned_ucb:'Learned UCB',heuristic:'Rules',random:'Random',no_context:'No context',no_uncertainty:'No uncertainty',bayesian:'Bayesian',bayesian_ei:'Bayesian EI',bayesian_ucb:'Bayesian GP-UCB'};
    const splitNames=Object.keys(evaluation.splits||{}).filter(k=>arr(evaluation.splits[k]?.aggregates).length);
    const selector=el('select',{'aria-label':'Evaluation holdout split'},splitNames.map(k=>el('option',{value:k},`Held-out ${k}`)));
    const budget=el('select',{'aria-label':'Evaluation candidate budget'}),results=el('div',{});
    const section=panel('Held-out policy evaluation',`${evaluation.device_kind==='cpu'?'CPU control':'CUDA'} candidate replay · ${evaluation.observations??'unrecorded'} observations`,body(el('div',{class:'search-row'},selector,budget),results),badge(evaluation.device_kind==='cpu'?'CPU REPLAY':'RECORDED EVALUATION','accent'));
    const draw=()=>{
      const split=evaluation.splits?.[selector.value]||{},rows=arr(split.aggregates).filter(x=>object(x)&&x.budget===Number(budget.value)&&finite(x.mean_best_reward));
      if(!rows.length){results.replaceChildren(empty('No aggregate results for this selection.','Choose a recorded holdout split and candidate budget.'));return;}
      const w=window.innerWidth<=650?Math.max(280,window.innerWidth-72):650,h=rows.length*39+45,left=window.innerWidth<=650?105:139,right=22,top=16,pw=w-left-right;
      let lo=Math.min(0,...rows.map(x=>x.mean_best_reward)),hi=Math.max(0,...rows.map(x=>x.mean_best_reward));if(lo===hi)hi=lo+1;
      const x=value=>left+(value-lo)/(hi-lo)*pw;
      const chart=svg('svg',{viewBox:`0 0 ${w} ${h}`,class:'chart',role:'img','aria-label':`Mean best log-speedup reward by selection policy with ${budget.value} candidates; held-out ${selector.value}.`});
      chart.append(svg('line',{x1:x(0),y1:top,x2:x(0),y2:h-28,class:'axis'}));
      rows.forEach((row,i)=>{const y=top+i*39;chart.append(svg('text',{x:left-13,y:y+14,'text-anchor':'end'},[aliases[row.policy]||row.policy]));chart.append(svg('rect',{x:Math.min(x(0),x(row.mean_best_reward)),y,width:Math.max(1,Math.abs(x(row.mean_best_reward)-x(0))),height:21,fill:row.policy==='learned_ucb'?'#c63f27':'#747d6c'}));});
      [0,.5,1].forEach(f=>chart.append(svg('text',{x:left+pw*f,y:h-7,'text-anchor':'middle'},[num(lo+(hi-lo)*f,3)])));
      const table=el('div',{class:'table-wrap'},el('table',{},el('thead',{},el('tr',{},['Policy','Workloads','Mean best reward','Oracle gap','Mean cost / s'].map(t=>el('th',{scope:'col'},t)))),el('tbody',{},rows.map(row=>el('tr',{},el('td',{},aliases[row.policy]||row.policy),el('td',{class:'mono'},row.workloads),el('td',{class:'mono'},num(row.mean_best_reward,4)),el('td',{class:'mono'},num(row.mean_oracle_gap,4)),el('td',{class:'mono'},num(row.mean_cost,4)))))));
      results.replaceChildren(el('p',{class:'caption'},`${split.folds??arr(split.groups).length} holdout groups · ${budget.value} candidate${Number(budget.value)===1?'':'s'} per workload · reward = log(speedup), higher is better.`),chart,table,el('p',{class:'caption'},'Offline replay uses the recorded discovery outcomes. Oracle gap is the difference from the best observed candidate; it is not an estimate of unexplored improvements. Costs are measured experiment seconds. No CUDA benefit follows from CPU replay.'));
    };
    const setBudgets=()=>{const split=evaluation.splits?.[selector.value]||{},budgets=[...new Set(arr(split.aggregates).filter(object).map(x=>x.budget).filter(Number.isSafeInteger))].sort((a,b)=>a-b);budget.replaceChildren(...budgets.map(x=>el('option',{value:x},`${x} candidate${x===1?'':'s'}`)));draw();};
    selector.addEventListener('change',setBudgets);budget.addEventListener('change',draw);setBudgets();
    return section;
  }
  function renderResearch() {
    const r=state.record,s=r?.selection||{},evaluation=state.evaluation||s.evaluation||r?.research?.evaluation;
    const holder=el('div',{},notice('The research question remains open.', 'Can learned intervention selection find verified GPU improvements with fewer expensive experiments? CPU control experiments and a fitted model alone do not answer that question.'));
    holder.append(evaluationPanel(evaluation));
    const policy=panel('Selection policy','The strategy used by the current investigation.',body(
      dl([['Strategy',s.strategy||s.method||s.policy],['Training observations',s.training_observations??s.training_count],['Training device kind',s.device_kind],['Candidate budget',r?.budget?.max_candidates],['Candidates evaluated',r?.budget?.used_candidates],['Wall time / seconds',r?.budget?.wall_seconds]]),
      el('div',{class:'divider'}),el('p',{class:'summary'},s.note||'A policy must be evaluated on held-out workloads and compared with simpler strategies. No pretrained GPU policy is implied.')));
    const checks=[['CUDA execution',r?.evidence_kind==='cuda_run'?'Recorded':'Not established'],['Independent hardware replication','Not established in this view'],['Held-out policy evaluation',evaluation?`${evaluation.device_kind||'Recorded'} replay`:'Not recorded'],['Triton autotuning comparison',r?.research?.triton_autotune?'See record':'Not recorded'],['Bayesian optimization comparison',evaluation&&JSON.stringify(evaluation.splits||{}).includes('bayesian')?'Offline replay baseline only':'Not recorded'],['Agent baseline reproduction',r?.research?.agent_baseline?'See record':'Not recorded']];
    const readiness=panel('Evidence needed for a GPU claim','Research readiness is stated explicitly.',body(checks.map(([label,status])=>el('div',{class:'research-check'},el('span',{},label),el('span',{class:'muted'},status)))));
    holder.append(el('div',{class:'grid-two equal','data-spacing':'after-evaluation'},policy,readiness));
    if(evaluation?.limitations?.length)holder.append(panel('Evaluation limitations','Preserved from the recorded evaluation.',body(el('ul',{class:'limit-list'},arr(evaluation.limitations).map(x=>el('li',{},fmt(x)))))));
    if(r) holder.append(el('div',{class:'section-title'},el('h2',{},'Reproducibility details'),el('span',{class:'micro'},'FROM THIS RECORD')),el('div',{class:'grid-two equal'},
      panel('Environment','Versions and execution settings as recorded.',body(el('pre',{class:'json-pre'},JSON.stringify(r.environment||{},null,2)))),
      panel('Provenance','Record origin and collection details.',body(el('pre',{class:'json-pre'},JSON.stringify(r.provenance||{},null,2))))));
    return holder;
  }
  const renderers={overview:renderOverview,timeline:renderTimeline,bottlenecks:renderBottlenecks,hypotheses:renderHypotheses,evolution:renderEvolution,code:renderCode,atlas:renderAtlas,research:renderResearch};
  function render(){
    const view=views.find(x=>x[0]===state.view)||views[0],r=state.record;
    $('breadcrumb').textContent=view[3];$('page-title').textContent=titles[view[0]][0];$('page-description').textContent=titles[view[0]][1];document.title=`${view[3]} · KERNELDARWIN LAB`;
    document.querySelectorAll('.nav-link').forEach(n=>{if(n.dataset.view===state.view)n.setAttribute('aria-current','page');else n.removeAttribute('aria-current');});
    $('download').disabled=!r;
    $('record-strip').replaceChildren(r?badge(evidenceLabel(r),r.evidence_kind==='cuda_run'?'good':'accent'):badge('NO RECORD','dark'),el('strong',{},r?`${r.workload.family} / ${r.workload.shape.join(' × ')}`:'One investigation. All the evidence.'),r&&el('span',{class:'muted'},r.device?.name||'Device unrecorded'),el('span',{class:'spacer'}),el('span',{class:'record-id'},r?`${r.id} · ${r.created_at?String(r.created_at).slice(0,10):'date unrecorded'}`:'IMPORT A JSON RECORD TO BEGIN'));
    $('view').replaceChildren(renderers[view[0]]());
  }
  views.forEach(([id,label,icon],i)=>{const a=el('a',{href:`#${id}`,class:'nav-link'},el('span',{class:'nav-icon','aria-hidden':'true'},icon),el('span',{},label),el('span',{class:'nav-number','aria-hidden':'true'},String(i+1).padStart(2,'0')));a.dataset.view=id;$('navigation').append(a);});
  $('import').addEventListener('click',()=>$('file').click());$('file').addEventListener('change',()=>importFile($('file').files[0]));
  $('download').addEventListener('click',()=>{if(!state.record)return;const blob=new Blob([JSON.stringify(state.record,null,2)+'\n'],{type:'application/json'}),a=el('a',{href:URL.createObjectURL(blob),download:`${state.record.id.replace(/[^a-zA-Z0-9_.-]/g,'_')}.json`});a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);});
  window.addEventListener('hashchange',()=>{const next=location.hash.slice(1);state.view=renderers[next]?next:'overview';render();});
  let dragDepth=0;window.addEventListener('dragenter',e=>{if([...e.dataTransfer.types].includes('Files')){e.preventDefault();dragDepth++;$('drop-overlay').hidden=false;}});window.addEventListener('dragover',e=>{e.preventDefault();});window.addEventListener('dragleave',e=>{e.preventDefault();if(--dragDepth<=0){$('drop-overlay').hidden=true;dragDepth=0;}});window.addEventListener('drop',e=>{e.preventDefault();dragDepth=0;$('drop-overlay').hidden=true;importFile(e.dataTransfer.files[0]);});
  state.view=renderers[location.hash.slice(1)]?location.hash.slice(1):'overview';render();
  (async()=>{
    try{const response=await fetch('data/index.json');if(!response.ok)throw Error(`manifest HTTP ${response.status}`);const manifest=await response.json();
      let loaded=0;const failures=[];
      for(const path of arr(manifest.records).slice(0,100)){try{if(typeof path!=='string')throw Error('invalid record path');const url=new URL(path,new URL('data/',location.href));if(url.origin!==location.origin)throw Error('external record URLs are not allowed');const res=await fetch(url);if(!res.ok)throw Error(`HTTP ${res.status}`);const raw=await res.text();if(raw.length>20*1024*1024)throw Error('record too large');const r=validate(JSON.parse(raw));state.records.push(r);loaded++;}catch(e){failures.push(e.message);}}
      if(typeof manifest.evaluation==='string') {
        try {
          const url=new URL(manifest.evaluation,new URL('data/',location.href));
          if(url.origin!==location.origin)throw Error('external evaluation URLs are not allowed');
          const res=await fetch(url);if(!res.ok)throw Error(`HTTP ${res.status}`);
          const value=await res.json();
          if(!object(value)||value.schema_version!==1||!['cpu','cuda'].includes(value.device_kind)||!object(value.splits))throw Error('invalid evaluation schema');
          for(const split of Object.values(value.splits))if(!object(split)||split.aggregates!==undefined&&(!Array.isArray(split.aggregates)||split.aggregates.some(x=>!object(x))))throw Error('invalid evaluation aggregate');
          state.evaluation=value;
        }catch(e){failures.push(`evaluation: ${e.message}`);}
      }
      if(!state.record&&state.records.length)activate(state.records[0]);else if(['atlas','research'].includes(state.view))render();
      if(failures.length)message(`Loaded ${loaded} bundled records; ${failures.length} bundled files could not be read. Import a valid JSON record to continue.`,true);
    }catch(e){message(location.protocol==='file:'?'To load bundled records, serve this folder with the CLI or a local HTTP server. You can still import a JSON file directly.':`Bundled records could not load (${e.message}). You can import a JSON file directly.`,true);}
  })();
})();
