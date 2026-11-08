"""Second exact audit figure, generated only from run_context_demo()."""
import base64
import html
import json
from pathlib import Path
import sys
import zlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fisherlab import encode, run_context_demo


def render(report):
    r=encode(report)
    s=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="880" viewBox="0 0 1120 880">',
       '<title>FisherLab: sharing parameters creates context tradeoffs</title>',
       '<desc>Exact local derivatives and metric projection certificates under fixed exogenous context frequencies.</desc>',
       '<rect width="1120" height="880" fill="#0f172a"/>']
    packed=base64.b64encode(zlib.compress(json.dumps(r,sort_keys=True,separators=(',',':')).encode(),9)).decode()
    s.append(f'<metadata id="full-witness" encoding="zlib-base64-json">{packed}</metadata>')
    def text(x,y,value,size=15,color='#dbeafe',bold=False,ident=None,mono=False):
        tag=f' id="{ident}"' if ident else ''
        family='monospace' if mono else 'DejaVu Sans,sans-serif'
        s.append(f'<text{tag} x="{x}" y="{y}" fill="{color}" font-size="{size}" font-weight="{700 if bold else 400}" font-family="{family}">{html.escape(str(value))}</text>')
    def panel(x,y,w,h):s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#18253b" stroke="#334155"/>')
    cyan='#67e8f9';muted='#94a3b8';amber='#fbbf24'
    text(28,43,'FisherLab | sharing creates context tradeoffs',28,bold=True)
    text(28,73,'Known exogenous context weights, conditional softmax policies and one shared local parameter.',color=muted)
    panel(25,95,523,119);panel(565,95,530,119)
    text(44,125,'Context 0',18,cyan,True);text(44,153,'weight 3/4   p=(1/2,1/2)   rewards=(0,1)',mono=True)
    text(44,183,'A rows=(0),(1): action-1 probability rises with theta',14,muted)
    text(584,125,'Context 1',18,cyan,True);text(584,153,'weight 1/4   p=(1/2,1/2)   rewards=(0,-2)',mono=True)
    text(584,183,'Same A and theta: action-1 probability also rises here',14,muted)
    panel(25,231,1070,211)
    headers=['model / objective','rank / dim','direction v','dJ context0','dJ context1','weighted dJ','residual^2']
    xx=[44,321,445,580,715,848,985]
    for x,h in zip(xx,headers):text(x,262,h,13,cyan,True)
    keys=['weighted_shared','weighted_independent','uniform_shared_different_objective']
    labels=['weighted, shared','weighted, independent','uniform, shared (new objective)']
    for i,(key,label) in enumerate(zip(keys,labels)):
        a=r[key];n=a['natural']
        values=[label,f"{a['system']['rank']} / {len(n['parameter_direction'])}",','.join(n['parameter_direction']),n['contexts'][0]['reward_derivative'],n['contexts'][1]['reward_derivative'],n['objective_derivative'],n['residual_length_squared']]
        for j,(x,v) in enumerate(zip(xx,values)):text(x,302+43*i,v,13 if j==0 else 15,amber if i==2 else '#dbeafe',ident=f'chart-{i}-{j}',mono=j!=0)
    text(44,424,'Positive aggregate derivative does not mean every context improves. These are local derivatives, not training curves.',13,muted)
    panel(25,460,586,208);panel(628,460,467,208)
    text(44,490,'Exact projection audit under weights (3/4,1/4)',17,cyan,True)
    text(44,520,'Full target length squared = within-context variance sum',14,muted)
    a=r['weighted_shared'];n=a['natural']
    text(44,552,'full = reachable + residual',15)
    text(333,552,f"{a['full_conditional_target_length_squared']} = {n['fisher_length_squared']} + {n['residual_length_squared']}",17,ident='pyth-equation',mono=True)
    text(44,584,'projection inner product',14,muted);text(435,584,n['projection_inner_product'],17,ident='projection',mono=True)
    text(44,615,'Pythagorean equality residual',14,muted);text(435,615,n['pythagorean_residual'],17,ident='pyth-residual',mono=True)
    text(44,645,'Independent parameters reach both context targets exactly.',13,muted)
    text(646,490,'Reweighting changes the objective',17,cyan,True)
    text(646,522,'Uniform-objective direction: v = -1/2',15)
    text(646,552,'Derivative of original weighted objective:',14)
    text(646,586,r['uniform_direction_evaluated_on_original_weighted_objective'],25,amber,True,'wrong-objective')
    text(646,618,'Negative here, although its own objective rises.',14,amber)
    text(646,645,'Not an approximation to the original weighted objective.',13,muted)
    panel(25,686,1070,133)
    text(44,715,'A global softmax over joint outcomes is a different model',17,cyan,True)
    frozen=r['context_gauge_conditional_frozen'];naive=r['naive_flattened_joint_softmax_different_model']
    text(44,742,'Constant logit rows within each context: A0=(0),(0), A1=(1),(1). Conditional policies cannot move.',14,muted)
    vals=[frozen['pullback_fisher'][0][0],frozen['parameter_gradient'][0],naive['pullback_fisher'][0][0],naive['parameter_gradient'][0]]
    labels=['conditional G','conditional g','naive joint G','naive joint g']
    for i,(label,v) in enumerate(zip(labels,vals)):
        text(44+260*i,775,label,14,muted);text(194+260*i,775,v,16,amber if i>1 else '#dbeafe',ident=f'gauge-{i}',mono=True)
    text(44,802,'Naive joint logits move context mass, violating the fixed-context model. No such update is used by audit_contexts.',13,muted)
    text(28,849,'All displayed derivatives and certificates are exact rationals. Strict positive action and context probabilities.',13,muted)
    text(28,873,'Supplied contextual expected rewards, not MDP occupancies, training guarantees or a new natural-gradient algorithm.',13,muted)
    s.append('</svg>')
    return '\n'.join(s)+'\n'


if __name__=='__main__':
    p=ROOT/'assets/context-audit.svg';svg=render(run_context_demo())
    if sys.argv[1:]==['--check']:
        if p.read_text()!=svg:raise SystemExit('context SVG differs from fresh audit')
    elif not sys.argv[1:]:p.write_text(svg)
    else:raise SystemExit('usage: render_contexts.py [--check]')
