"""Deterministic editable SVGs; quantitative marks come from actual replay."""
from pathlib import Path
import json
from html import escape
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/assets';OUT.mkdir(parents=True,exist_ok=True)

def text(x,y,value,size=20,color='#252923',weight=400,anchor='start'):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>'

def svg(width,height,content,title,desc):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc><rect width="100%" height="100%" fill="#f5f3ec"/><g font-family="Arial,Helvetica,sans-serif">{content}</g></svg>\n'

phases=[('PROFILE','Shape · dtype · baseline'),('HYPOTHESIZE','Evidence + falsifier'),('EXPERIMENT','Bounded configurations'),('VERIFY','Reference + uncertainty'),('CONFIRM','Fresh input + timing'),('LEARN','Grouped recorded replay')]
content=text(44,52,'KERNELDARWIN / THE EVIDENCE LOOP',16,'#b83d26',700)+text(44,98,'An improvement earns its status.',38,weight=700)
for i,(title,subtitle) in enumerate(phases):
    x=44+i*222
    content+=f'<g id="phase-{i}"><rect x="{x}" y="140" width="202" height="134" fill="#fffefa" stroke="#c9cabe"/>'
    content+=text(x+16,169,f'0{i+1}',13,'#b83d26',700)+text(x+16,204,title,19,weight=700)+text(x+16,236,subtitle,13,'#50564c')+'</g>'
    if i<5:content+=f'<path d="M{x+204} 207h16m-5-5 5 5-5 5" fill="none" stroke="#70786d" stroke-width="2"/>'
content+='<path id="feedback-loop" d="M1250 282v35H358v-35m-5 5 5-5 5 5" fill="none" stroke="#b83d26" stroke-width="2"/>'
content+=text(805,341,'Preserve failures. Update the next experiment.',15,'#b83d26',anchor='middle')
content+='<line x1="44" y1="375" x2="1356" y2="375" stroke="#c9cabe"/>'
content+=text(44,407,'IMPLEMENTED',13,'#38634a',700)+text(190,407,'Authored templates, numerical checks, paired timing, confirmation, storage, offline selector.',16)
content+=text(44,442,'RESEARCH OPEN',13,'#b83d26',700)+text(190,442,'Causal diagnosis, learned GPU advantage, hardware transfer and NVIDIA validation.',16)
(OUT/'workflow.svg').write_text(svg(1400,475,content,'KERNELDARWIN experiment workflow','Six implementation stages and a feedback loop preserve failures and improve future experiment selection. Causal diagnosis and GPU learning remain unvalidated.'))
report=json.loads((ROOT/'evidence/cpu-2026-10-26/cpu-selector.evaluation.json').read_text())
rows=[r for r in report['splits']['family']['aggregates'] if r['budget']==1]
labels={'learned_ucb':'Learned UCB','heuristic':'Fixed heuristic','random':'Random (32 orders)','bayesian_ucb':'GP-UCB (32 starts)','no_context':'No context','no_uncertainty':'No uncertainty'}
order=['learned_ucb','heuristic','random','bayesian_ucb','no_context','no_uncertainty'];data={r['policy']:r for r in rows}
content=text(44,48,'MEASURED CPU CONTROL / OFFLINE REPLAY',15,'#b83d26',700)+text(44,92,'Learning matched the simple rule.',35,weight=700)+text(44,126,'Leave-one-family-out · 24 workloads · one candidate per task · higher log-speedup is better',16,'#50564c')
left,width=270,740;maximum=.4
for i,key in enumerate(order):
    row=data[key];y=174+i*47;value=row['mean_best_reward'];bar=width*value/maximum
    content+=text(250,y+22,labels[key],16,anchor='end')
    content+=f'<rect id="bar-{key}" x="{left}" y="{y}" width="{bar:.4f}" height="29" fill="{"#bd422a" if key=="learned_ucb" else "#596451"}"/>'
    content+=text(left+bar+12,y+21,f'{value:.4f}',16)
for tick in (0,.1,.2,.3,.4):
    x=left+width*tick/maximum
    content+=text(x,490,f'{tick:.1f}',14,'#50564c',anchor='middle')
content+=text(left+width/2,520,'Mean best observed log(speedup)',16,anchor='middle')+text(44,563,'Three CPU row-batch choices are a weak test. No GPU selection benefit or causal finding is established.',15,'#b83d26')
(OUT/'cpu-selection.svg').write_text(svg(1180,590,content,'CPU held-out selection comparison','The learned model and fixed heuristic have the same mean log-speedup. All marks derive from recorded CPU replay. This does not establish GPU performance.'))
