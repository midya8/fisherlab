# Geometry, singular systems and scope

## Categorical model

At a supplied positive distribution `p` on `k` actions, let logits vary locally as `z=z0+A theta`, where `z0_i=log(p_i)` at the origin. `A` is `k x d`. Reward is a fixed vector `r`. This is a one-state expected-reward problem, not an MDP policy-gradient implementation.

Define `F=diag(p)-p p^T`, `J=F A`, `G=A^T F A` and `g=A^T F r`. Softmax differentiation gives `dp=J dtheta` and reward derivative `g^T dtheta`. The score row is `s_i=A_i-sum_j p_j A_j`; consequently `G=sum_i p_i s_i s_i^T`. An independent-form PSD certificate is

```text
G = sum_{i<j} p_i p_j (A_i-A_j)(A_i-A_j)^T.
```

All weights are strictly positive. The report retains each weighted difference, the score Gram, pair Gram and exact equality against the direct pullback. A checker can reconstruct any quadratic form as a sum of nonnegative weighted squares. A diagonal sign check alone would not certify PSD.

## Singular solve and uniqueness of the induced tangent

Solve `G v=g` exactly using rational RREF. The returned representative has all free variables zero, together with pivot/free columns, full augmented RREF, a basis of every null direction and exact residual. It is not a pseudoinverse or minimum-norm representative.

If `G n=0`, the PSD certificate implies `A n` is constant over actions, since all `p_i p_j>0`. Therefore `F A n=0` and `g^T n=0`. The system is consistent because `g` is orthogonal to `ker(G)` for symmetric `G`, hence in its range. Every solution has the same induced probability tangent, objective derivative and Fisher length, despite possibly different parameter norms.

`F` is singular even with all positive probabilities: the common-logit gauge is invisible. Zero/constant coordinate matrices are valid frozen charts. Overcomplete charts have additional null coordinates. No epsilon is added to make the natural solve invertible.

## Full versus restricted tangent spaces

The full-simplex natural probability tangent is

```text
t_i = p_i (r_i - E_p[r]).
```

It sums to zero. If `rank(G)=k-1`, the chart spans all simplex tangents; `J v=t` for every solution. Then `g^T v=v^T G v=sum_i t_i^2/p_i=Var_p(r)`.

If the chart has smaller rank, `J v` is instead the projection of `t` into `range(J)` under the categorical Fisher metric `<u,w>_p=sum_i u_i w_i/p_i`. Indeed `J^T diag(1/p)J=G` and `J^T diag(1/p)t=g`, so the normal equation is exactly the natural system. The residual is metric-orthogonal to the chart. Its natural objective derivative equals its squared metric length, lies between 0 and reward variance, and need not reach the full-simplex tangent. Lower rank is a smaller model, not just alternative coordinates.

## Coordinate comparisons

The demo uses a minimal baseline chart, a shear of it, full logits and a redundant full-tangent chart. Their rational induced natural tangents agree. Under linear affine logit charts, their finite logit Euler directions differ only by an invisible common shift, so their softmax finite updates also agree. That special case does not imply general nonlinear finite-step invariance.

Ordinary ascent uses `v=g`. Diagonal Fisher ascent uses `v_j=g_j/G_jj`, with zero assigned when `G_jj=0` (then `g_j=0`). Coordinate damping solves `(G+lambda I)v=g` for positive rational `lambda`. Identity damping and a diagonal approximation depend on coordinates; a transformed regularizer need not equal identity in another chart. Exact gaps are counterexamples to invariance, not universal method rankings, speed results or arguments against practical damping.

For an arbitrary rectangular or rank-deficient local Jacobian, we claim equality only of the full reachable induced tangent, not a general theorem about arbitrary redundant parameter-space paths.

## Nonlinear finite-step control

Separately define a binary policy `p1=sigmoid(z)`, reward `(0,1)` and smooth local chart `z=theta^3` at `theta=1>0`. The map is a diffeomorphism on the positive domain, not at theta0. In direct logits, reward gradient and Fisher are both `p1(1-p1)`, so the natural direction is 1. In theta coordinates, gradient is `3 theta^2 p1(1-p1)` and Fisher `9 theta^4 p1(1-p1)`, so the direction is `1/(3 theta^2)`.

Their mapped initial tangents agree. At theta1, step eta gives direct logit `1+eta` but curved logit `(1+eta/3)^3`. Their difference is `eta^2/3+eta^3/27`, nonzero at finite positive eta and second order as eta tends to zero. Eta1/2 yields exact `19/216`. This does not require training or numerical inversion.

A separate finite affine-logit step has `p_new_i proportional to p_i exp(eta r_i)`. The report compares actual floating `KL(p_old||p_new)` to local `eta^2 Var(r)/2`, and actual reward to `E[r]+eta Var(r)`. These are expansions, not finite constraints or certified bounds. Local rational arithmetic is distinguished from floating exponentials/logs.

## Implementation verification

Independent tests compute softmax reward/probability finite differences and KL mixed Hessian entries without production matrix helpers. Exact references cover the categorical Fisher, full and projected tangents, score outer products and gauge shifts. Sixteen seeded dimension/chart combinations (2/3/5/8 actions, 1/k-1/k/12 columns) verify all residuals/null vectors and random-vector sum-of-squares certificates. Tests also cover zero/constant charts, every nullspace dimension in the fixtures, action/coordinate reorderings, reward shifts, strict rationals, parser limits and isolated wheel CLI parity.

The demo is a chosen educational fixture, not an untouched test set. The SVG embeds the complete canonical report compressed as base64 zlib JSON; table coordinates/values, nonlinear exact controls and even undisplayed nullspace changes are tested. The visualization does not replace the witness.

## Deliberate boundaries

Strict positive support; no support-boundary extension or silently injected probability. Exact arithmetic only for small bounded dimensions. No sampled/empirical Fisher, learning curves, state occupancy, approximate critics, TRPO implementation, natural neural trainer, reward improvement guarantee across arbitrary finite steps, or claimed optimizer superiority.

See README attribution: Kakade (2001) for the score-Fisher natural policy gradient, and Martens (2020) Sections 5/9/10/12 for Fisher pullbacks, practical damping and finite-step qualifications. These are established ideas. The implementation and small exact controls are original code, not novel theory or paper benchmark reproductions.
