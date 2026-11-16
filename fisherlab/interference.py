"""Local context interactions measured with one fixed aggregate Fisher metric."""
from fractions import Fraction as Q
from .contexts import audit_contexts
from .core import dot, mv, solve


def audit_interference(weights, contexts, damping='1/10'):
    """Decompose local reward derivatives, not finite-step reward changes.

    Each context gradient is solved against the same aggregate Fisher. These
    are not separate context-Fisher natural directions averaged afterwards.
    """
    original = audit_contexts(weights, contexts, damping)
    metric = original['pullback_fisher']
    gradients = [row['parameter_gradient'] for row in original['local_audits']]
    systems = [solve(metric, gradient) for gradient in gradients]
    directions = [system['solution'] for system in systems]
    matrix = [[dot(g, v) for v in directions] for g in gradients]
    frequencies = original['context_weights']
    combined = [sum((w*v[j] for w,v in zip(frequencies,directions)), Q(0))
                for j in range(len(metric))]
    response = [sum((w*h for w,h in zip(frequencies,row)), Q(0)) for row in matrix]
    objective = sum((w*d for w,d in zip(frequencies,response)), Q(0))
    solve_residuals = [[a-b for a,b in zip(mv(metric,v),g)]
                       for v,g in zip(directions,gradients)]
    null_residuals = [[dot(g,n) for n in original['system']['null_basis']]
                     for g in gradients]
    tangent_residuals = [[a-b for a,b in zip(mv(c['probability_jacobian'],combined),
                         row['probability_tangent'])]
                         for c,row in zip(original['local_audits'],original['natural']['contexts'])]
    if (any(any(row) for row in solve_residuals+null_residuals+tangent_residuals)
            or matrix != [list(row) for row in zip(*matrix)]
            or combined != original['natural']['parameter_direction']
            or response != [c['reward_derivative'] for c in original['natural']['contexts']]
            or objective != original['natural']['objective_derivative']):
        raise ArithmeticError('context interference decomposition failed')
    return {'version': 1, 'context_audit': original,
            'context_gradient_systems': systems,
            'interaction_matrix': matrix, 'solve_residuals': solve_residuals,
            'null_gradient_residuals': null_residuals,
            'combined_parameter_direction': combined,
            'combined_tangent_residuals': tangent_residuals,
            'context_reward_derivatives': response,
            'objective_derivative': objective,
            'scope': 'exact infinitesimal reward interactions under the fixed weighted aggregate Fisher; no finite-step, training, causal, fairness or optimizer ranking claim'}
