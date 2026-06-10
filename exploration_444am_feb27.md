# 4:44 AM Feb 27 — Where the Geometry Actually Breaks

## The Situation

We've iterated through three architectural versions:

1. **Independent integration** — Flattened (B*T, d), processed all tokens independently through the spray. No path dependence at all. Worked as an ODE integrator but had nothing to do with Finsler geometry.

2. **Additive holonomy** — Sequential processing, but holonomy was a *vector* that got added to each token: `h_t = x_t + holonomy`. This is a translation, not a frame transformation. Direction-independent. Still Riemannian in character.

3. **Matrix holonomy** (current) — The connection `N = dG/dy` is a d-by-d matrix computed via autograd Jacobian. Transport is `H_{t+1} = exp(-N*dt) @ H_t`. Representation is `h_t = H_t @ x_t`. Properly Finslerian — the connection depends on direction.

Version 3 is structurally correct, but there are deeper problems.

---

## Problem 1: The Spray Is Not Derived from a Metric

The blueprint says:

> G^i = (1/4) g^{il} (2 dg_jl/dx^k - dg_jk/dx^l) y^j y^k

The spray is supposed to be *derived* from the fundamental tensor g_ij(x, y), which is itself derived from the Randers metric F(x, y) = alpha(x, y) + beta(x, y). The spray isn't a free function — it has specific algebraic structure:

- **2-homogeneity**: G(x, lambda*y) = lambda^2 * G(x, y). A real spray scales quadratically with velocity magnitude. Our MLP doesn't enforce this.
- **Metric compatibility**: The spray determines geodesics that extremize the arc length functional defined by F. An arbitrary MLP spray determines curves that don't extremize anything — they're just trajectories of a learned vector field.
- **Consistency**: The fundamental tensor, spray, and connection form a mutually consistent triple. Change one and the others follow. Our spray is disconnected from any metric.

We have a `RandersMetric` class (cell 7) for visualization, but the actual model uses a standalone `SprayNetwork` MLP that has no relationship to any metric. The "Finsler" label is aspirational, not descriptive.

**What this means**: The connection `N = dG/dy` is a well-defined matrix (it's a Jacobian), and it does depend on direction. But it's not the non-linear connection of any Finsler manifold. It's just the Jacobian of a neural network. The "holonomy" it produces doesn't correspond to parallel transport on any manifold.

---

## Problem 2: The Training Signal Is Inverted

The blueprint says:

> The loss function measures how much "force" is required to keep the tokens on the actual observed linguistic paths. If the metric is perfect, the sequences are natural geodesics, and the loss is zero.
>
> L = sum_t || accel_empirical + 2G(x_t, v_t) ||^2

This is the **geodesic deviation loss**. It says: "learn a spray (metric) under which the observed data IS a geodesic." The spray adapts to the data, not the other way around.

But in the full model, we flipped the objective. We're training a *predictor MLP* that uses the holonomy-enriched representation to predict next positions. The spray is only trained indirectly — it gets gradient signal from whatever holonomy happens to help the predictor. The spray doesn't learn to make data geodesic; it learns to be a useful feature extractor for a downstream MLP.

This is the root of the instability issues documented in the notes: the spray creates holonomy, holonomy shifts the predictor's input, the predictor compensates, and the gradient pushes the spray to collapse to zero. The system fights itself because the spray's geometric role (defining the manifold) and the predictor's role (making predictions) are entangled in contradictory ways.

**The standalone spray trainer (cells 19-25) gets this right.** It uses the geodesic deviation loss directly:

```
L_geodesic = ((accel + 2.0 * G_pw) ** 2).mean()
```

The full model abandoned this in favor of the predictor loss, losing the geometric grounding.

---

## Problem 3: What Does "Routes Baked In" Actually Mean?

The user's insight: trained trajectories should become geodesics of the learned manifold. At inference, starting along a familiar trajectory should naturally continue because the metric creates a channel — a region of low curvature where the geodesic equation carries the particle along the learned path.

For this to work:

1. **Training**: Minimize geodesic deviation on observed sequences. The metric (a_ij, b_i) adapts until each training trajectory is a geodesic. The spray at each observed (x, v) pair matches the empirical acceleration.

2. **Inference**: Given a seed (initial position and velocity), integrate the geodesic equation. The spray tells the particle how to accelerate. If the seed matches a training trajectory's initial conditions, the particle follows that trajectory because it's a geodesic of the learned metric.

3. **Novel sequences**: The metric interpolates between learned trajectories. A novel initial condition follows a geodesic that's a blend of nearby learned ones — the metric creates a smooth landscape where all trajectories are natural.

The current architecture doesn't achieve (1) because it doesn't use geodesic deviation loss for the full model. It doesn't achieve (2) because prediction comes from an MLP predictor, not geodesic integration. And it doesn't achieve (3) because the spray isn't constrained to correspond to any metric.

---

## Proposed Architecture: Metric-First Design

### Core Principle

**Learn the metric, derive everything else.**

The Randers metric F(x, y) = alpha + beta is the single source of truth. The spray, connection, and transport are all derived from it. Training minimizes geodesic deviation. Prediction integrates the geodesic equation.

### Components

#### 1. Learned Randers Manifold

Two neural networks parametrize the position-dependent metric:

- `a_net(x) -> L` (Cholesky factor, so `a = L @ L^T` is positive definite)
- `b_net(x) -> b` (1-form, projected to satisfy `||b||_a < 1`)

These define F(x, y) = sqrt(y^T a(x) y) + b(x)^T y at every point.

For d=2: a is 2x2 (3 free params via Cholesky), b is 2-vector. Small MLPs (2 -> 32 -> 32 -> 3 and 2 -> 32 -> 32 -> 2) are sufficient.

Initialize to near-Euclidean: a ~ I, b ~ 0. The manifold starts flat and learns curvature from data.

#### 2. Spray from the Metric

Compute the geodesic spray G^i(x, y) from F(x, y) using the energy formulation:

```
E = F^2 / 2
dE/dx, dE/dy via autograd
g_ij = d^2E / dy^i dy^j  (fundamental tensor, via autograd)
mixed = d^2E / dy^i dx^j  (via autograd)
G = (1/2) * g_inv @ (mixed @ y - dE/dx)
```

For d=2, this requires ~5 backward passes per spray evaluation. Cheap.

The spray is NOT a free neural network. It's a deterministic function of the metric. Training the metric parameters shapes the spray, which shapes the geodesics, which shapes the manifold.

#### 3. Connection from the Spray

`N^i_j = dG^i / dy^j` via autograd Jacobian.

For d=2: 2 backward passes per connection evaluation.

This is the same as the current implementation, but now the connection is derived from a metric rather than an arbitrary MLP. It's automatically:
- Direction-dependent (Finslerian)
- Metric-compatible (consistent with the Randers structure)
- 1-homogeneous in y (because the spray is 2-homogeneous)

#### 4. Training Objective: Geodesic Deviation

Primary loss — the data should be geodesics:

```
velocities = (x_seq[:, 1:] - x_seq[:, :-1]) / dt
accelerations = (velocities[:, 1:] - velocities[:, :-1]) / dt
G_at_data = spray(x[:, 1:-1], velocities[:, :-1])
L_geodesic = ((accelerations + 2 * G_at_data) ** 2).mean()
```

This is what the standalone spray trainer already does (cell 19). The difference: G is now derived from the Randers metric, not a freestanding MLP.

Secondary loss — rollout stability:

```
Integrate the geodesic equation from (x_0, v_0)
Compare trajectory to ground truth
```

Again, exactly what cell 19 does. The metric parameters get gradients through the spray, through the integration, through the loss.

#### 5. Sequential Transport for Context (Language Model Extension)

For the language model case, we need path-conditioned representations. The connection matrix `N(x, y)` enables this:

```
H_0 = I  (identity matrix)
For each token transition:
    N = connection(x_t, v_t)
    H_{t+1} = exp(-N * dt) @ H_t

h_t = H_t @ x_t  (transported representation)
logits = output_proj(h_t)
```

The transport is a DOWNSTREAM consumer of the learned geometry, not the training objective. The metric learns from geodesic deviation. The transport uses the learned metric to create path-conditioned representations. The output projection maps these to predictions.

This separation resolves the training signal problem: the metric/spray are trained by geodesic deviation (clean, direct signal), and the transport/projection are trained by next-token cross-entropy (standard LM objective). No conflicting gradients.

### Autograd Cost Analysis (d=2)

Per token transition:

| Operation | Backward passes | Notes |
|-----------|----------------|-------|
| E = F^2/2 | 0 | Forward only |
| dE/dx, dE/dy | 1 | Standard grad |
| g_ij (Hessian of E w.r.t. y) | 2 | One per column |
| mixed partials (d(dE/dy)/dx) | 2 | One per column |
| G^i (spray) | 0 | Tensor algebra |
| N^i_j (Jacobian of G w.r.t. y) | 2 | One per row |
| **Total** | **7** | Per transition |

For T=50 sequence: 49 * 7 = 343 backward passes. Each is a small MLP (2 -> 32 -> 32 -> 3). On CPU, ~0.5ms per pass. Total: ~0.17s per sequence. With batch_size=32 and 4 batches per epoch: ~2.2s per epoch. 500 epochs: ~18 minutes. Entirely feasible.

With n_sub_steps=4: multiply by 4 -> ~72 minutes. Still feasible for a research notebook.

---

## What Stays, What Changes

### Keep
- `RandersMetric` (cell 7) — but promote it from visualization tool to the actual model
- `SprayNetwork` (cell 9) — keep for the standalone spray experiments
- Standalone spray training (cells 19-25) — still valid as a baseline
- Data generation (cell 17) — unchanged
- All visualization code — reusable

### Replace
- `FinslerGeodesicTransformer` (cell 13) — new version derives spray from learned Randers metric
- `FullGeodesicPredictor` (cell 27) — prediction via geodesic integration, not MLP predictor
- Training loop (cell 27) — geodesic deviation as primary loss, rollout as secondary

### Add
- `LearnedRandersManifold` class — position-dependent Randers metric with autograd-derived spray and connection
- Indicatrix visualization at various points post-training — show how the metric varies across space
- Connection matrix visualization — show the anisotropy directly

---

## The Key Difference

Current: "Learn a spray MLP. Hope it behaves like a Finsler spray. Use its Jacobian as a connection. Train via prediction loss on an MLP decoder."

Proposed: "Learn a Randers metric. Derive the spray analytically. Derive the connection analytically. Train via geodesic deviation — make the data be geodesics. Use the connection for transport. Predict via geodesic integration."

The first approach uses Finsler language to describe what is essentially a learned RNN with fancy vocabulary. The second approach actually builds a Finsler manifold and uses it as the theory intends.

---

## Open Questions

1. **Training stability**: Third-order derivatives (loss -> spray -> energy -> metric parameters). The spray involves second derivatives of E, and loss.backward() adds another. For d=2 this should be fine. For d=64 it might not be. Can we use the explicit Randers spray formula instead of autograd to avoid higher-order derivatives?

2. **Expressivity**: Is the Randers metric rich enough? It's the simplest non-Riemannian Finsler metric. More complex Finsler metrics (e.g., (alpha, beta)-metrics) offer more flexibility but are harder to implement. For the 2D toy case, Randers should be sufficient. For language, we might need something richer.

3. **Scalability to d=64**: The fundamental tensor g_ij is d-by-d. Computing it via autograd Hessian requires d backward passes. The spray requires a matrix inverse. The connection requires another d backward passes. Total: O(d^2) backward passes per token transition. For d=64, that's ~8000 backward passes per transition. This needs approximations (low-rank metric, diagonal connection, etc.) for practical use.

4. **Initialization sensitivity**: If the metric starts too far from Euclidean, the spray could produce wild geodesics from epoch 1. The near-identity initialization (a ~ I, b ~ 0) should handle this, but it needs testing.

5. **The prediction/transport split**: For language modeling, we need both geodesic deviation (to learn the metric) AND next-token prediction (to learn the output projection). Can these be trained jointly, or do they need alternating phases?

---

## Why Spirals Work and Wind Walks Don't (and What That Actually Means)

### The spirals are deterministic

A spiral trajectory is a smooth, deterministic curve. Given initial conditions (position, velocity), the entire trajectory is uniquely determined. Every point on the spiral has a well-defined velocity and acceleration. The geodesic deviation loss asks: "learn a spray G such that accel + 2G = 0 at every observed point." For spirals, there EXISTS a smooth spray field that satisfies this exactly — the acceleration field of the spiral. The loss can go to zero. The spray learns the curvature, the metric shapes itself to make spirals into geodesics, and prediction via integration reproduces the spiral.

This is why the standalone spray on spirals converges cleanly. The problem is well-posed: there's a deterministic signal to learn, and enough capacity in the spray to learn it.

### Wind walks are stochastic — and that's not a bug

Let's look at what wind walks actually are:

```
wind = [0.05, 0.015]
step_t = 0.1 * randn(2) + wind
trajectory = cumsum(steps)
```

Each step is a Gaussian random variable with mean [0.05, 0.015] and std 0.1 in each component. The signal-to-noise ratio is 0.05 / 0.1 = 0.5. The noise is 2x the signal.

Now compute the empirical acceleration (what the geodesic deviation loss sees):

```
velocity_t = (x_{t+1} - x_t) / dt = step_t / dt
acceleration_t = (v_{t+1} - v_t) / dt = (step_{t+1} - step_t) / dt^2
```

Since the wind is constant, it cancels in the acceleration difference. The acceleration is:

```
accel_t = (noise_{t+1} - noise_t) / dt^2
```

This is pure noise. Mean zero, variance proportional to noise_std^2 / dt^4. There is no systematic acceleration to learn.

The best the spray can do is G = 0 (predict zero acceleration), which gives a geodesic deviation equal to the noise variance. This isn't a failure — it's the correct answer. The spray correctly identifies that the acceleration is unpredictable noise.

### But the Randers metric CAN capture the drift

The wind doesn't appear in the acceleration — it appears in the *velocity bias*. And this is exactly what the Randers 1-form b_i is for. The Randers metric was literally invented by physicist Gunnar Randers in 1941 to model "navigation in the presence of wind."

In a Randers metric F(x, y) = sqrt(y^T a y) + b^T y:
- a_ij defines the base geometry (how hard it is to move in each direction)
- b_i defines the wind (a directional bias that makes F(x, y) != F(x, -y))

Geodesics on a Randers manifold with constant wind b naturally drift in the b direction. A particle released with zero velocity will accelerate along b. This is exactly the wind walk's drift component.

So the decomposition is:
- **The 1-form b_i captures the wind** (the deterministic drift [0.05, 0.015])
- **The spray G^i captures the curvature** (which is ~zero for constant wind walks)
- **The noise is irreducible** (no deterministic model can predict it)

### What "works" means for stochastic data

For spirals: "works" = geodesic deviation goes to zero, rollout reproduces exact trajectories. Achievable.

For wind walks: "works" = the Randers metric learns b ~ [proportional to wind], the spray learns G ~ 0 (flat space with wind), rollout reproduces the MEAN trajectory (straight drift in wind direction), and geodesic deviation plateaus at a floor proportional to the noise variance. Each individual rollout prediction won't match any specific stochastic walk, but it will follow the correct statistical trend.

**This is not a failure. This is correct behavior.** The model correctly separates signal (wind direction) from noise (random perturbations). A model that overfits to individual noise realizations would be worse, not better.

### The real diagnostic: what does the learned metric look like?

After training on wind walks, we should inspect:

1. **The learned b_i(x)**: Should point approximately in the [1, 0.3] direction (proportional to wind) everywhere. If it does, the metric correctly learned the drift.

2. **The indicatrix at various points**: Should be an offset ellipse, shifted in the wind direction. The shift magnitude tells us the learned wind strength.

3. **The spray field G(x, y)**: Should be approximately zero for constant-wind walks (no position-dependent curvature). If the wind varies spatially, the spray should be non-zero in regions of wind gradient.

4. **The geodesic deviation distribution**: Should be centered at zero with variance matching the noise level. If the spray is trying to fit noise, the distribution will be biased.

### So is training the manifold to find matching geodesics a "mathematical certainty"?

For the deterministic component: **yes, absolutely.** The Randers metric can represent any smooth drift field. If the wind is [0.05, 0.015], the metric can learn b proportional to that, and the mean trajectory becomes a geodesic. This is guaranteed by the universality of the Randers parametrization for this class of problems.

For the stochastic component: **no, and it shouldn't be.** Individual noise realizations are not geodesics of ANY smooth manifold. A trajectory that zigs left then right due to noise is not extremizing any arc length functional. The geodesic deviation loss correctly identifies this residual as unlearnable.

For the combined system: the metric learns the deterministic structure and correctly ignores the noise. The geodesic deviation loss has a non-zero floor, and this floor equals the noise variance of the data. If you reduce noise_std, the floor drops. If you increase wind_strength, the learned b becomes more pronounced. If you make the wind spatially varying (e.g., a vortex), the spray will learn non-zero curvature to capture the spatial variation.

### The language modeling implication

Language is NOT like wind walks. Language has:
- High predictability given context (low effective noise)
- Complex, path-dependent structure (high curvature)
- Strong directional bias (syntax constrains what follows what)

The SNR for language is much more favorable than 0.5. Given "the cat sat on the", the next word is highly predictable. The Finsler manifold should be able to create tight geodesic channels for common constructions, with the curvature encoding syntactic/semantic constraints.

Wind walks are actually the HARDEST test case for the framework because they have the lowest SNR. Spirals are the easiest because they're deterministic. Language is somewhere in between — highly structured but with genuine ambiguity at branch points.

### Concrete recommendation

To separate the signal-learning question from the noise-floor question, test with:

1. **Zero noise wind walks** (noise_std=0): Pure drift. The metric should learn perfect geodesics matching every trajectory. Geodesic deviation -> 0. This verifies the Randers wind mechanism.

2. **Low noise** (noise_std=0.02, wind_strength=0.1): SNR=5. The metric should learn accurate drift with small residual. Rollout should track close to ground truth.

3. **Current noise** (noise_std=0.1, wind_strength=0.05): SNR=0.5. The metric learns the drift but rollout diverges quickly because noise dominates. The floor is high. This is correct.

4. **Spatially varying wind** (wind that changes direction across the plane): Now the spray needs to learn non-zero curvature. The Finsler structure earns its keep — the direction-dependent metric creates different geodesics from different starting angles in different regions.

Test (4) is the most informative for the language modeling case, because language has spatially varying "curvature" — different words have different grammatical affordances, and the metric should learn this variation.
