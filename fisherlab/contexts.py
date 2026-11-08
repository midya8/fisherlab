"""Exact context-weighted categorical geometry with shared parameters."""
from fractions import Fraction as Q
from .core import audit, dot, mv, rational, solve


def audit_contexts(weights, contexts, damping='1/10'):
    """Fixed positive context frequencies, conditional policies and one shared chart.

    This is a contextual expected-reward objective, not state occupancy differentiation.
    Every context must specify probabilities/rewards/coordinates; all charts share d.
    """
    if (not isinstance(contexts, (list, tuple)) or not 1 <= len(contexts) <= 6
            or not isinstance(weights, (list, tuple)) or len(weights) != len(contexts)):
        raise ValueError('require1..6 contexts with aligned weights')
    w = list(map(rational, weights))
    if any(q <= 0 for q in w) or sum(w) != 1:
        raise ValueError('context weights must be strictly positive and sum exactly to1')
    local = []
    for c in contexts:
        if not isinstance(c, dict) or set(c) != {'probabilities', 'rewards', 'coordinates'}:
            raise ValueError('each context requires exactly probabilities/rewards/coordinates')
        local.append(audit(c['probabilities'], c['rewards'], c['coordinates'], damping))
    d = len(local[0]['parameter_gradient'])
    if any(len(c['parameter_gradient']) != d for c in local):
        raise ValueError('all contexts must share the same coordinate dimension')
    lam = rational(damping)
    metric = [[sum((q*c['pullback_fisher'][i][j] for q,c in zip(w,local)), Q(0))
               for j in range(d)] for i in range(d)]
    gradient = [sum((q*c['parameter_gradient'][i] for q,c in zip(w,local)), Q(0)) for i in range(d)]
    system = solve(metric, gradient)
    full_length = sum((q*c['reward_variance'] for q,c in zip(w,local)), Q(0))
    pairs = [dict(context=index, weight=q*t['weight'], actions=t['actions'], difference=t['difference'])
             for index,(q,c) in enumerate(zip(w,local)) for t in c['pair_certificate']]
    pair_gram = [[sum((t['weight']*t['difference'][i]*t['difference'][j] for t in pairs), Q(0))
                  for j in range(d)] for i in range(d)]
    score_gram = [[sum((q*c['score_gram'][i][j] for q,c in zip(w,local)), Q(0))
                   for j in range(d)] for i in range(d)]

    def direction(v):
        rows = []
        for q,c in zip(w,local):
            tangent = mv(c['probability_jacobian'], v)
            residual = [t-a for t,a in zip(c['full_simplex_natural_tangent'], tangent)]
            rows.append({'probability_tangent': tangent,
                         'joint_probability_tangent': [q*t for t in tangent],
                         'logit_direction': mv(c['coordinates'], v),
                         'reward_derivative': dot(c['parameter_gradient'], v),
                         'probability_length_squared': sum((t*t/p for t,p in zip(tangent,c['probabilities'])), Q(0)),
                         'full_target_residual': residual,
                         'residual_length_squared': sum((t*t/p for t,p in zip(residual,c['probabilities'])), Q(0))})
        projected = sum((q*c['probability_length_squared'] for q,c in zip(w,rows)), Q(0))
        residual = sum((q*c['residual_length_squared'] for q,c in zip(w,rows)), Q(0))
        return {'parameter_direction': v, 'contexts': rows,
                'objective_derivative': dot(gradient,v), 'fisher_length_squared': dot(v,mv(metric,v)),
                'probability_length_squared': projected, 'residual_length_squared': residual,
                'projection_inner_product': sum((q*sum((t*e/p for t,e,p in zip(
                    row['probability_tangent'],row['full_target_residual'],c['probabilities'])),Q(0))
                    for q,row,c in zip(w,rows,local)),Q(0)),
                'pythagorean_residual': full_length-projected-residual}
    natural = direction(system['solution'])
    damped_metric = [[metric[i][j]+(lam if i==j else 0) for j in range(d)] for i in range(d)]
    # Full context-local witnesses are retained, including their own RREF solutions.
    # Those local natural representatives are NOT combined to make the shared solve.
    return {'version': 1, 'context_weights': w, 'local_audits': local,
            'pullback_fisher': metric, 'parameter_gradient': gradient,
            'score_gram': score_gram, 'pair_certificate': pairs, 'pair_gram': pair_gram,
            'gram_agrees': metric==score_gram==pair_gram,
            'expected_reward': sum((q*c['expected_reward'] for q,c in zip(w,local)),Q(0)),
            'full_conditional_target_length_squared': full_length,
            'full_conditional_rank': sum(len(c['probabilities'])-1 for c in local),
            'full_conditional_reachable': system['rank']==sum(len(c['probabilities'])-1 for c in local),
            'system': system, 'solve_residual': [a-b for a,b in zip(mv(metric,system['solution']),gradient)],
            'null_probability_tangents': [[mv(c['probability_jacobian'],v) for c in local] for v in system['null_basis']],
            'natural': natural, 'ordinary': direction(gradient),
            'diagonal': direction([gradient[i]/metric[i][i] if metric[i][i] else Q(0) for i in range(d)]),
            'damped': direction(solve(damped_metric,gradient)['solution']), 'damping': lam,
            'scope': 'fixed exogenous positive context weights, supplied conditional probabilities/rewards/local logit Jacobians; no MDP state dynamics or learned policy'}


def run_context_demo():
    shared = [{'probabilities': ['1/2','1/2'], 'rewards': [0,1], 'coordinates': [[0],[1]]},
              {'probabilities': ['1/2','1/2'], 'rewards': [0,-2], 'coordinates': [[0],[1]]}]
    independent = [dict(shared[0], coordinates=[[0,0],[1,0]]),
                   dict(shared[1], coordinates=[[0,0],[0,1]])]
    weighted = audit_contexts(['3/4','1/4'], shared)
    separate = audit_contexts(['3/4','1/4'], independent)
    uniform = audit_contexts(['1/2','1/2'], shared)
    gauge = [dict(shared[0], coordinates=[[0],[0]]), dict(shared[1], coordinates=[[1],[1]])]
    frozen = audit_contexts(['3/4','1/4'], gauge)
    # A global softmax on flattened joint outcomes incorrectly allows context mass to change.
    joint_p = [Q(3,8),Q(3,8),Q(1,8),Q(1,8)]
    joint_r = [0,1,0,-2]
    naive_joint = audit(joint_p,joint_r,[[0],[0],[1],[1]])
    v = uniform['natural']['parameter_direction']
    wrong_objective_derivative = dot(weighted['parameter_gradient'],v)
    assert weighted['natural']['objective_derivative'] > 0
    assert weighted['natural']['contexts'][1]['reward_derivative'] < 0
    assert separate['full_conditional_reachable']
    assert separate['natural']['residual_length_squared']==0
    assert wrong_objective_derivative < 0
    assert weighted['natural']['projection_inner_product']==weighted['natural']['pythagorean_residual']==0
    return {'version': 1, 'weighted_shared': weighted, 'weighted_independent': separate,
            'uniform_shared_different_objective': uniform,
            'uniform_direction_evaluated_on_original_weighted_objective': wrong_objective_derivative,
            'context_gauge_conditional_frozen': frozen,
            'naive_flattened_joint_softmax_different_model': naive_joint,
            'limits': 'Exact local derivatives, not finite-step training results. Shared parameters can trade off contexts. Changing exogenous context weights changes the objective and metric.'}
