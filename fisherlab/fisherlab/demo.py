"""Fixed rational geometry fixtures plus clearly separated floating finite steps."""
import math
from fractions import Fraction as Q
from .core import audit, dot


def nonlinear_demo():
    # z=theta^3 at theta=1 is a smooth local chart (theta>0).
    theta, eta = Q(1), Q(1, 2)
    dz, dtheta = Q(1), Q(1, 3)
    direct_z = theta**3+eta*dz
    curved_z = (theta+eta*dtheta)**3
    def sigmoid(z):
        return 1/(1+math.exp(-float(z)))
    p = sigmoid(theta**3)
    return {'map': 'z=theta^3, theta>0; binary reward(0,1), p(action1)=sigmoid(z)',
            'theta': theta, 'step': eta, 'natural_logit_direction': dz,
            'natural_theta_direction': dtheta, 'mapped_logit_tangent': 3*theta*theta*dtheta,
            'initial_probability_action1': p, 'probability_tangent_action1': p*(1-p),
            'direct_euler_logit': direct_z, 'curved_euler_logit': curved_z,
            'logit_difference': curved_z-direct_z,
            'direct_probability_action1': sigmoid(direct_z),
            'curved_probability_action1': sigmoid(curved_z),
            'scope': 'same initial tangent, different finite Euler steps; probabilities are floating point'}


def step_audit(p, rewards):
    mean = dot(p, rewards)
    var = dot(p, [(r-mean)**2 for r in rewards])
    rows = []
    for eta in (Q(1, 64), Q(1, 16), Q(1, 4), Q(1)):
        logits = [math.log(float(q))+float(eta*r) for q, r in zip(p, rewards)]
        peak = max(logits)
        exp = [math.exp(x-peak) for x in logits]
        updated = [x/math.fsum(exp) for x in exp]
        kl = math.fsum(float(q)*(math.log(float(q))-math.log(s)) for q, s in zip(p, updated))
        rows.append({'step': eta, 'updated_probabilities': updated,
                     'expected_reward': math.fsum(float(r)*q for r, q in zip(rewards, updated)),
                     'linear_reward_prediction': mean+eta*var,
                     'actual_kl_old_to_new': kl, 'quadratic_kl_prediction': eta*eta*var/2})
    return rows


def run_demo():
    p, r = [Q(1, 2), Q(1, 3), Q(1, 6)], [Q(0), Q(1), Q(3)]
    charts = [
        ('baseline', [[1,0], [0,1], [0,0]]),
        ('shear', [[1,2], [0,1], [0,0]]),
        ('full_logits', [[1,0,0], [0,1,0], [0,0,1]]),
        ('redundant', [[1,2,1,0], [0,1,0,1], [0,0,0,0]]),
        ('restricted', [[1], [0], [0]])]
    audits = [{'name': name, 'audit': audit(p, r, a)} for name, a in charts]
    assert all(x['audit']['gram_agrees'] for x in audits)
    assert all(not x['audit']['natural']['full_simplex_gap_l1'] for x in audits[:4])
    assert audits[-1]['audit']['natural']['full_simplex_gap_l1'] > 0
    assert audits[0]['audit']['diagonal']['probability_tangent'] != audits[1]['audit']['diagonal']['probability_tangent']
    assert audits[0]['audit']['damped']['probability_tangent'] != audits[1]['audit']['damped']['probability_tangent']
    return {'version': 1, 'charts': audits, 'nonlinear': nonlinear_demo(),
            'finite_steps': step_audit(p, r),
            'limits': 'Exact rational local systems; finite-step probabilities/KL use floats. No general optimizer superiority or finite-step invariance claim.'}
