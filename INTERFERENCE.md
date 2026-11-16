# Local reward interactions across fixed contexts

`audit_interference` uses the same fixed-context model as `audit_contexts`.
It asks which local reward objectives help or oppose each other under one
aggregate Fisher metric. This is an infinitesimal calculation, not a prediction
of finite-step training outcomes.

For fixed positive context frequencies `w_c`, let `G_c` and `g_c` be each
context's pullback Fisher and expected-reward gradient. Define

```text
G = sum_c w_c G_c
g = sum_c w_c g_c
G v_d = g_d
H_cd = g_c^T v_d.
```

Every column uses the same `G`. Solving `G_d v_d = g_d` separately and averaging
those directions is a different calculation. The API retains every column's
complete RREF system, free columns and nullspace, along with residuals.

## Singular systems are allowed

Since all frequencies are positive and each `G_c` is positive semidefinite,
`null(G)` is the intersection of the context nullspaces. Each local gradient
annihilates that intersection: a direction invisible to a context's action
probabilities cannot change its expected reward. Thus every column system is
consistent, including frozen and redundant charts. Adding a null vector to a
column does not change any entry of `H`.

The representative sets RREF free variables to zero. It is not a pseudoinverse
or a minimum-Euclidean-norm direction. Because the same pivot convention is
used for every right-hand side, its weighted sum equals the original shared
natural representative; the induced probability tangents agree too.

The matrix is symmetric because
`g_c^T v_d = v_c^T G v_d = g_d^T v_c`. For any vector `a`,
`a^T H a = (sum_c a_c v_c)^T G (sum_c a_c v_c) >= 0`.
Positive semidefiniteness does not require each off-diagonal entry to be
positive. A negative `H_cd` means that the metric direction for context `d`
has a negative local derivative for context `c`. A zero diagonal means its
local gradient is zero on the reachable model, not that its reward is constant
under every imaginable model.

## Reconstructing the shared update

```text
v = sum_d w_d v_d
context_c derivative along v = sum_d H_cd w_d
weighted objective derivative = w^T H w.
```

The implementation checks these identities against the existing shared-context
report, including every conditional probability tangent. It checks symmetry,
column solve residuals and each gradient against the aggregate nullspace. A
failed check raises an error instead of returning a report.

For the existing binary fixture with frequencies `(3/4,1/4)`, equal action
probabilities, shared logit coordinate and reward differences `1` and `-2`:

```text
G = 1/4
H = [[1/4, -1/2], [-1/2, 1]]
H w = [1/16, -1/8]
w^T H w = 1/64.
```

The shared update improves the weighted objective locally while reducing the
second context's reward locally. Giving the contexts separate parameters
instead gives `H = diag(1/3,4)` and no cross-context terms. That is a larger
reachable model, not a fair optimizer-performance comparison.

## Use and checks

```sh
python -m fisherlab interference examples/contexts-input.json
```

```python
from fisherlab import audit_interference
report = audit_interference(weights, contexts)
```

The API accepts the same positive frequencies, strict rational inputs, context
shapes, dimension bounds and damping argument as `audit_contexts`. The new CLI
uses the unchanged context version-one input contract. Damping remains part of
the retained original report; `H` uses the undamped aggregate Fisher.

Eight tests include exact binary closed forms over 20 seeded inputs, a separate
permutation-determinant/Cramer reference for 16 six-context two/three-parameter
systems with unequal action counts, singular and frozen charts, independent
parameters, redundant/invertible coordinate changes, reward scaling/shifts,
context order/splitting, strict CLI/API validation and input/RNG isolation.
The original four complete CLI reports and both figures stay unchanged.

These are bounded finite-model checks. The matrix describes supplied local
reward gradients under fixed exogenous context frequencies, not causal effects,
estimated task affinity, fairness, sampled-gradient noise, finite-step gains,
learning speed or an optimizer ranking. The identities are standard Fisher
geometry and linear algebra, not new theory. See the original attributions in
README and the shared-context derivation in CONTEXTS.md.
