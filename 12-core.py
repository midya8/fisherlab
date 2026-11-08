"""Exact finite categorical Fisher geometry. Not an MDP or pseudoinverse solver."""
from fractions import Fraction as Q


def rational(x):
    if isinstance(x, bool) or not isinstance(x, (int, str, Q)):
        raise ValueError('rationals must be integers or rational/decimal strings, never floats/bools')
    if isinstance(x, str) and (not x or len(x) > 100):
        raise ValueError('invalid rational text length')
    try:
        z = Q(x)
    except (ValueError, ZeroDivisionError, OverflowError) as e:
        raise ValueError('invalid rational') from e
    if max(z.numerator.bit_length(), z.denominator.bit_length()) > 128:
        raise ValueError('rational exceeds 128-bit input limit')
    return z


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), Q(0))


def transpose(a):
    return [list(r) for r in zip(*a)]


def mv(a, v):
    return [dot(r, v) for r in a]


def mm(a, b):
    return [[dot(r, c) for c in transpose(b)] for r in a]


def solve(a, b):
    """Exact square-system RREF, free variables zero; returns every null direction.

    This representative is NOT the Moore-Penrose/minimum-Euclidean-norm solution.
    """
    n = len(a)
    if n < 1 or n > 12 or len(b) != n or any(len(r) != n for r in a):
        raise ValueError('system must be square with 1..12 variables')
    m = [[Q(x) for x in r] + [Q(y)] for r, y in zip(a, b)]
    pivots = []
    for col in range(n):
        row = next((r for r in range(len(pivots), n) if m[r][col]), None)
        if row is None:
            continue
        k = len(pivots)
        m[k], m[row] = m[row], m[k]
        scale = m[k][col]
        m[k] = [x/scale for x in m[k]]
        for r in range(n):
            if r != k:
                scale = m[r][col]
                m[r] = [x-scale*y for x, y in zip(m[r], m[k])]
        pivots.append(col)
    if any(not any(r[:n]) and r[n] for r in m):
        raise ValueError('inconsistent system')
    solution = [Q(0)]*n
    for row, col in enumerate(pivots):
        solution[col] = m[row][n]
    free = [i for i in range(n) if i not in pivots]
    basis = []
    for f in free:
        v = [Q(0)]*n
        v[f] = Q(1)
        for row, col in enumerate(pivots):
            v[col] = -m[row][f]
        basis.append(v)
    return {'solution': solution, 'rank': len(pivots), 'pivots': pivots,
            'free_columns': free, 'null_basis': basis, 'rref_augmented': m}


def encode(obj):
    if isinstance(obj, Q):
        return str(obj)
    if isinstance(obj, dict):
        return {k: encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode(v) for v in obj]
    return obj


def audit(probabilities, rewards, coordinates, damping='1/10'):
    """A rows map local parameter tangents into logits; offsets need not be supplied.

    For rank-deficient charts, the induced natural tangent is the Fisher projection
    into that chart's reachable tangent space. Full simplex reachability is reported.
    """
    if not isinstance(probabilities, (list, tuple)) or not 2 <= len(probabilities) <= 8:
        raise ValueError('require 2..8 actions')
    k = len(probabilities)
    if not isinstance(rewards, (list, tuple)) or len(rewards) != k:
        raise ValueError('one reward per action required')
    if (not isinstance(coordinates, (list, tuple)) or len(coordinates) != k
            or any(not isinstance(r, (list, tuple)) for r in coordinates)):
        raise ValueError('coordinates require one row per action')
    d = len(coordinates[0])
    if not 1 <= d <= 12 or any(len(r) != d for r in coordinates):
        raise ValueError('coordinate matrix must have 1..12 aligned columns')
    p, r = list(map(rational, probabilities)), list(map(rational, rewards))
    a = [list(map(rational, row)) for row in coordinates]
    lam = rational(damping)
    if any(x <= 0 for x in p) or sum(p) != 1 or lam <= 0:
        raise ValueError('probabilities must be strictly positive and sum exactly to1; damping>0')
    f = [[(p[i] if i == j else Q(0))-p[i]*p[j] for j in range(k)] for i in range(k)]
    j = mm(f, a)
    g = mv(transpose(a), mv(f, r))
    metric = mm(transpose(a), j)
    system = solve(metric, g)
    expectation = dot(p, r)
    target = [p[i]*(r[i]-expectation) for i in range(k)]
    mean_a = mv(transpose(a), p)
    scores = [[a[i][h]-mean_a[h] for h in range(d)] for i in range(k)]
    gram = [[sum((p[i]*scores[i][u]*scores[i][v] for i in range(k)), Q(0))
             for v in range(d)] for u in range(d)]
    pairs = []
    for i in range(k):
        for h in range(i+1, k):
            pairs.append({'actions': [i, h], 'weight': p[i]*p[h],
                          'difference': [a[i][v]-a[h][v] for v in range(d)]})
    pair_gram = [[sum((t['weight']*t['difference'][u]*t['difference'][v] for t in pairs), Q(0))
                  for v in range(d)] for u in range(d)]

    def direction(v):
        tangent = mv(j, v)
        return {'parameter_direction': v, 'logit_direction': mv(a, v),
                'probability_tangent': tangent,
                'full_simplex_gap_l1': sum(abs(x-y) for x, y in zip(tangent, target)),
                'objective_derivative': dot(g, v),
                'fisher_length_squared': dot(v, mv(metric, v)),
                'probability_length_squared': sum((x*x/q for x, q in zip(tangent, p)), Q(0))}

    damped = [[metric[u][v]+(lam if u == v else 0) for v in range(d)] for u in range(d)]
    natural = direction(system['solution'])
    diagonal = direction([g[i]/metric[i][i] if metric[i][i] else Q(0) for i in range(d)])
    damp = direction(solve(damped, g)['solution'])
    residual = [x-y for x, y in zip(mv(metric, system['solution']), g)]
    null_tangents = [mv(j, v) for v in system['null_basis']]
    return {'version': 1, 'probabilities': p, 'rewards': r, 'coordinates': a,
            'damping': lam, 'expected_reward': expectation,
            'reward_variance': dot(p, [(x-expectation)**2 for x in r]),
            'logit_fisher': f, 'probability_jacobian': j, 'pullback_fisher': metric,
            'parameter_gradient': g, 'score_rows': scores,
            'score_gram': gram, 'pair_certificate': pairs, 'pair_gram': pair_gram,
            'gram_agrees': gram == metric == pair_gram,
            'system': system, 'solve_residual': residual, 'null_probability_tangents': null_tangents,
            'full_simplex_rank': k-1, 'full_simplex_reachable': system['rank'] == k-1,
            'full_simplex_natural_tangent': target,
            'natural': natural, 'ordinary': direction(g), 'diagonal': diagonal, 'damped': damp,
            'representative': 'exact RREF with free variables zero, not a pseudoinverse',
            'scope': 'local categorical expected reward; no state dynamics, sampled estimator, trust-region guarantee or training benchmark'}
