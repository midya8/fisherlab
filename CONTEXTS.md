# Shared parameters across fixed contexts

`audit_contexts(weights, contexts, damping='1/10')` extends the local categorical audit to 1-6 contexts sharing the same parameter vector. Context frequencies are positive rational constants summing exactly to 1. They are not MDP stationary distributions or functions of policy parameters. Each context has 2-8 positive action probabilities and its own reward vector/local logit Jacobian with the same 1-12 parameter columns.

## Model and exact system

For each context c, `p_c` is its action distribution, `r_c` its rewards, `A_c` its logit Jacobian, and `mu_c` its fixed context frequency. The objective is

```text
L = sum_c mu_c E_{p_c}[r_c]
F_c = diag(p_c) - p_c p_c^T
J_c = F_c A_c
G = sum_c mu_c A_c^T F_c A_c
g = sum_c mu_c A_c^T F_c r_c.
```

Solve `G v=g` in exact rational arithmetic. The same RREF representative, full nullspace and residual contracts as the single-context API apply. Do not average separately solved context directions: shared parameters require one aggregate normal equation. The report retains each full local audit for inspection, but constructs the shared solve from its weighted matrices/gradients, not from local solutions.

The PSD witness is a positive sum of context-tagged pairs `mu_c p_ci p_cj (A_ci-A_cj)(A_ci-A_cj)^T`. The weighted score Gram and pair Gram reconstruct the same aggregate Fisher. Positive context frequencies ensure an aggregate null direction is invisible within every context, rather than allowing cancellation between indefinite terms. Frozen/overcomplete shared charts remain valid.

## Projection and Pythagorean certificate

The independent full-conditional target is `t_ci=p_ci(r_ci-E_{p_c}[r_c])`. Define the metric on a collection of conditional probability tangents by

```text
<u,w> = sum_c mu_c sum_i u_ci w_ci / p_ci.
```

The shared Jacobian maps `v` to `dp_c=J_c v`; its joint outcome tangent is `mu_c dp_c`, because context mass is fixed. The shared system is exactly the normal equation for the metric projection of t into this Jacobian's reachable space. The reported residual is `e_c=t_c-dp_c`.

For the natural direction, all quantities below are exact:

```text
sum_c mu_c <dp_c,e_c>_{p_c} = 0
g^T v = v^T G v = sum_c mu_c ||dp_c||_{p_c}^2
sum_c mu_c Var_{p_c}(r_c)
    = sum_c mu_c ||dp_c||_{p_c}^2 + sum_c mu_c ||e_c||_{p_c}^2.
```

The left side is a weighted sum of within-context reward variances, not the variance of the flattened joint reward. The latter also includes between-context mean variation, which cannot be changed by a conditional policy with fixed context frequencies. Full conditional reachability requires rank `sum_c (k_c-1)`. A shared lower-rank model can still exactly realize a particular reward target; inspect residuals, not rank alone.

The output exposes the full target length, reachable rank, per-context conditional/joint tangents, local reward derivatives, weighted objective derivative, Fisher/probability lengths, residual lengths, projection inner product and Pythagorean residual. Natural projection identities do not generally hold for the ordinary/diagonal/damped directions, although those directions also have diagnostics. A nonzero approximation residual is not mislabeled a natural projection certificate.

## Chosen fixture: aggregate improvement can hide a local loss

Both contexts have binary probabilities `(1/2,1/2)` and logit Jacobian rows `(0),(1)`, sharing one scalar. Rewards are `(0,1)` in context 0 and `(0,-2)` in context 1. Frequencies are `(3/4,1/4)`.

```text
G = 1/4,  g = 1/16,  natural v = 1/4
context 0 reward derivative = 1/16
context 1 reward derivative = -1/8
weighted objective derivative = 1/64
full conditional target length = 7/16
reachable length = 1/64, residual length = 27/64
7/16 = 1/64 + 27/64.
```

The independent-parameter comparison gives each context its own coordinate, so its direction `(1,-2)` reaches both context targets: derivatives `1/4` and `1`, weighted derivative `7/16` and zero residual. This is a larger reachable model, not a fair speed/parameter-efficiency benchmark.

Changing frequencies to `(1/2,1/2)` changes both objective and metric. Its shared natural direction is `-1/2`, with positive derivative `1/16` for that uniform objective. Evaluated against the original weighted objective, the same direction has derivative `-1/32`. Reweighting is not an innocent implementation change, even though both solves are correct for their respective objectives.

## Why a naive global softmax is wrong here

A policy conditioned on a fixed context can add a different constant to every action logit within each context without changing any conditional probability. With context 0 rows `(0),(0)` and context 1 rows `(1),(1)`, the correct conditional Fisher/gradient are both zero. All shared directions are invisible.

Instead, naively treating the four joint outcomes as one global softmax, with probabilities `(3/8,3/8,1/8,1/8)`, produces a nonzero Fisher `3/16` and gradient `-9/32`. This alternate model moves context mass. It is not the fixed-context model. The demo labels this control as a different model, and never uses it to compute `audit_contexts` updates.

A valid alternate reconstruction first centers the Jacobian rows separately within each context, then forms the joint score Gram. Tests compare this independent-form construction against the conditional implementation. Merely flattening uncentered logits loses the model's conditional normalization.

## Use and validation

```sh
python -m fisherlab context-demo > context-report.json
python -m fisherlab contexts examples/contexts-input.json > custom.json
python examples/render_contexts.py --check
```

JSON requires exactly integer `version:1`, `weights`, `contexts` and positive `damping`. Each context requires exactly `probabilities`, `rewards`, `coordinates`. Existing rational/dimension/100KB/duplicate-field rules apply; zero context weights are rejected instead of silently dropping observations. Python contexts are dictionaries with these same three fields. Context order is preserved and all reports retain supplied context frequencies.

Fourteen new behavior tests include independent weighted reward and KL-Hessian finite differences, direct joint-score reconstruction with contextwise centering, exact shared/independent/reweighted/gauge controls, per-context gauge/reward-shift invariance, context reordering/weight splitting, 5 seeded systems including 6 contexts / 12 parameters, projection optimality against 8 rational competitors per system, nullspace/PSD/metric identities, strict CLI/API and no mutation/RNG side effects. Three new SVG tests verify all chart cells/positions, projection/gauge controls, full compressed metadata, hidden local-score tampering and byte-exact regeneration. Existing 28 tests remain unchanged and passing.

This is local supplied-model geometry, not policy training, an MDP occupancy derivative, estimated counterfactual reward, fairness guarantee, finite-step reward bound or optimizer-superiority claim. Context weights are explicitly exogenous. The original single-context demo and SVG remain byte-identical. Natural-gradient/Fisher foundations retain Kakade (2001) and Martens (2020) attribution in README; the shared-context normal equation and projection are small explicit instances of established geometry, not new theory.
