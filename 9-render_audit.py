"""Rebuild the SVG from fresh calculations. No plotting dependency."""
import base64
from fractions import Fraction as Q
import html
import json
from pathlib import Path
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fisherlab import encode, run_demo


def number(x):
    return f'{float(Q(x)):.6f}'


def render(report):
    r=encode(report)
    s=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="940" viewBox="0 0 1120 940">',
       '<title>FisherLab: local geometry is not a finite-step optimizer guarantee</title>',
       '<desc>Exact rational categorical Fisher systems, coordinate counterexamples, a nonlinear finite-step audit and a local KL comparison.</desc>',
       '<style>text{font-family:system-ui,sans-serif;fill:#dbeafe;font-size:15px}.muted{fill:#94a3b8}.mono{font-family:monospace;font-size:15px}.head{fill:#67e8f9;font-weight:600}.title{font-size:28px;font-weight:700}.small{font-size:13px}.warn{fill:#fbbf24}</style>',
       '<rect width="1120" height="940" fill="#0f172a"/>']
    packed=base64.b64encode(zlib.compress(json.dumps(r,sort_keys=True,separators=(',',':')).encode(),9)).decode()
    s.append(f'<metadata id="full-witness" encoding="zlib-base64-json">{packed}</metadata>')
    def text(x,y,t,cls='',ident=None):
        tag=f' id="{ident}"' if ident else ''
        color = '#fbbf24' if 'warn' in cls else '#67e8f9' if 'head' in cls else '#94a3b8' if 'muted' in cls else '#dbeafe'
        size = 28 if 'title' in cls else 13 if 'small' in cls else 15
        family = 'monospace' if 'mono' in cls else 'DejaVu Sans,sans-serif'
        weight = 700 if 'title' in cls or 'head' in cls else 400
        s.append(f'<text{tag} x="{x}" y="{y}" class="{cls}" fill="{color}" font-size="{size}" font-family="{family}" font-weight="{weight}">{html.escape(str(t))}</text>')
    def panel(x,y,w,h):
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#18253b" stroke="#334155"/>')
    text(30,43,'FisherLab | exact geometry, visible limits','title')
    text(30,73,'p=(1/2,1/3,1/6), rewards=(0,1,3), expected reward=5/6, variance=41/36','muted')
    text(30,99,'Four full-tangent charts agree exactly. A restricted chart changes the reachable model.','muted')
    panel(25,120,1070,303)
    xs=[42,205,290,374,478,588,706,838,975]
    heads=['chart','rank / dim','dp[0]','dp[1]','dp[2]','natural gap','diag gap','damped gap']
    hx=[42,205,348,455,565,680,815,950]
    for x,h in zip(hx,heads):text(x,150,h,'head small')
    text(42,174,'Gaps are L1 to full-simplex natural tangent; damping uses G+(1/10)I. All cells exact.','muted small')
    for index,row in enumerate(r['charts']):
        a=row['audit'];y=210+37*index
        values=[row['name'],f"{a['system']['rank']} / {len(a['coordinates'][0])}",*a['natural']['probability_tangent'],a['natural']['full_simplex_gap_l1'],a['diagonal']['full_simplex_gap_l1'],a['damped']['full_simplex_gap_l1']]
        for col,(x,v) in enumerate(zip(hx,values)):
            text(x,y,v,'mono'+(' warn' if index==4 else ''),f'chart-{index}-{col}')
    text(42,403,'Free variables are set to zero by exact RREF. This is not a minimum-norm pseudoinverse.','muted small')
    panel(25,440,587,219);panel(628,440,467,219)
    text(43,470,'A nonlinear chart: same tangent, different step','head')
    text(43,495,'Binary p(1)=sigmoid(z), z=theta^3, theta=1, eta=1/2','small')
    n=r['nonlinear']
    rows=[('natural dz / natural dtheta',n['natural_logit_direction']+' / '+n['natural_theta_direction']),
          ('direct / mapped Euler logit',n['direct_euler_logit']+' / '+n['curved_euler_logit']),
          ('logit difference (exact)',n['logit_difference']),
          ('direct / curved p(1)',number(n['direct_probability_action1'])+' / '+number(n['curved_probability_action1']))]
    for i,(label,value) in enumerate(rows):
        text(43,523+27*i,label,'small muted');text(335,523+27*i,value,'mono',f'nonlinear-{i}')
    text(646,470,'Singular metric, inspectable certificates','head')
    text(646,499,'G = sum over i&lt;j p[i] p[j] d[i,j] d[i,j]^T'.replace('&lt;','<'),'small')
    text(646,525,'d[i,j] = A[i,:] - A[j,:]','mono')
    text(646,553,'Every weight is positive. Null directions map to dp=0.','small')
    text(646,582,'Full charts: derivative = metric length = reward variance.','small')
    text(646,610,'Restricted chart: orthogonal projection, not a failed solve.','small warn')
    text(646,639,'Complete rational matrices, nullspaces and Gram witness in JSON.','small muted')
    panel(25,678,1070,200)
    text(43,708,'Local KL approximation is not an exact finite-step constraint','head')
    xx=[43,190,392,620,850]
    for x,t in zip(xx,['step','actual KL(p || p_new)','local eta^2 Var(r)/2','actual new reward','linear prediction']):text(x,735,t,'head small')
    for i,row in enumerate(r['finite_steps']):
        vals=[row['step'],f"{row['actual_kl_old_to_new']:.6f}",number(row['quadratic_kl_prediction']),f"{row['expected_reward']:.6f}",number(row['linear_reward_prediction'])]
        for j,(x,v) in enumerate(zip(xx,vals)):text(x,762+29*i,v,'mono',f'step-{i}-{j}')
    text(30,906,'Exact rational geometry; finite-step probabilities and KL use floating point. Positive support required.','muted small')
    text(30,929,'Standard natural-gradient ideas: Kakade (2001), Martens (2020). Not an MDP, neural trainer or superiority benchmark.','muted small')
    s.append('</svg>')
    return '\n'.join(s)+'\n'


if __name__=='__main__':
    target=ROOT/'assets/fisher-audit.svg';content=render(run_demo())
    if sys.argv[1:]==['--check']:
        if target.read_text()!=content:raise SystemExit('SVG differs from fresh audit')
    elif not sys.argv[1:]:target.write_text(content)
    else:raise SystemExit('usage: render_audit.py [--check]')
