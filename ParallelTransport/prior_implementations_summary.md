# Prior Implementations Summary & Analysis

## Overview of All Implementations

Five distinct architectures have been built and tested across the notebook iterations. This document catalogs what each actually did, what the numbers say, and how the directionality problem from `directionality_problem.md` manifests in each.

---

## Implementation 1: Standalone Spray MLP (cells 19–25 in main notebook)

**Architecture**: A `SprayNetwork` MLP trained directly on trajectory data using the original geodesic deviation loss: `||accel + 2G(x, v)||^2`, plus a rollout integration loss.

**What it does right**: This is the closest to the blueprint's intent. The spray learns pointwise — at each observed (x, v), it learns the acceleration that makes the data a geodesic. The rollout loss teaches stability under iteration.

**Results on spirals** (main notebook):
- Geodesic deviation: 15.49 → 6.52 over 500 epochs (significant reduction)
- Rollout loss: 0.058 → 0.012 (converges well)
- Spray field visualizations show clear direction-dependence across the 4 momentum panels — the geometry IS Finslerian, not just Riemannian

**Results on wind walks** (main notebook):
- Geodesic deviation: 102.21 → 99.75 over 500 epochs (barely moves)
- Rollout loss: 0.26 → 0.25 (stuck)
- Floor is ~100, matching the theoretical noise variance floor (see `exploration_444am_feb27.md` analysis)

**Directionality**: Not applicable — this is a pointwise spray trainer, not a sequence model. It has no sequential processing, no holonomy, no path dependence. It's a diagnostic tool for the spray, not a language model architecture.

**Key insight**: The geodesic deviation on spirals plateaus at ~6.5, not zero. The notes attribute this to the global MLP being too smooth to capture the sharp, radially varying curvature of spirals. The `README_042726.md` explicitly states: "A 2-layer MLP metric field with SiLU activations can't create the sharp, localized channels needed to guide trajectories along specific paths."

---

## Implementation 2: Independent Integration (original FinslerGeodesicTransformer)

**Architecture**: Flattened (B*T, d) → processed all tokens independently through K spray layers → reshaped back. The "full model" wrapper.

**What it does right**: Nothing geometrically. This treats every token as an independent particle on the same global spray field. There is zero path dependence — token 5 gets the same spray evaluation regardless of tokens 1–4.

**Directionality**: N/A — there is no sequential structure at all. Every token is processed identically. This is neither directional nor bidirectional — it's context-free.

**Outcome**: Abandoned. The rollout loss exploded to 10^17 due to a K-step mismatch bug (model predicted K steps ahead but rollout compared 1 step ahead). After fixing, it trained but the architecture was recognized as fundamentally wrong — no path dependence means no holonomy, no attention replacement.

---

## Implementation 3: Additive Holonomy (sequential transport, vector holonomy)

**Architecture**: Sequential processing through the token sequence. At each step, the spray evaluates at the holonomy-shifted state `h_t = x_t + holonomy_vector`, integrates to produce a deflection, and the deflection accumulates additively into the holonomy vector. A predictor MLP maps the transported representation to displacements for next-position prediction.

**Results on spirals** (main notebook, Pred/Roll format):
- Prediction: 0.0025 → 0.0010 over 500 epochs (good convergence)
- Rollout: 0.059 → 0.011 (good convergence)

**Results on wind walks** (main notebook):
- Prediction: 0.010 → 0.010 (stuck from epoch 1)
- Rollout: 0.170 → 0.169 (essentially flat)

**Directionality problem**: Severe. Information flows strictly left-to-right. Token 0 has `holonomy = 0` (no context). Token 49 has the richest context but can't influence earlier tokens. This IS the RNN bottleneck.

**Additional problems**:
- The holonomy is a vector (d numbers), not a matrix (d^2 numbers). It can only SHIFT the representation, not rotate/scale/shear it. This is geometrically wrong — parallel transport should be a frame transformation.
- The spray is an MLP disconnected from any metric. The holonomy doesn't correspond to transport on any manifold.
- The training signal problem: the predictor MLP is what gets trained, not the spray/holonomy directly. The spray only gets indirect gradient signal, creating the competing-objectives issue documented in the notes.

**Key finding**: Spirals work because the predictor MLP can learn position-dependent displacements even without meaningful holonomy. Wind walks fail because there's nothing position-dependent to learn — all positions have the same dynamics.

---

## Implementation 4: Matrix Holonomy (current, sequential transport with connection Jacobian)

**Architecture**: Sequential processing. The connection `N = dG/dy` is computed as the Jacobian of the spray MLP w.r.t. velocity via `torch.autograd`. Transport is `H_{t+1} = exp(-N*dt) @ H_t`. Representation is `h_t = H_t @ x_t`. Predictor MLP maps to displacements.

**Results on spirals** (from most recent run in main notebook):
- Not yet fully trained with the matrix holonomy — the `torch.enable_grad()` fix was just applied

**Directionality problem**: Same as Implementation 3 — still strictly left-to-right. The matrix holonomy is richer per step (d^2 vs d context), but the information flow direction is identical.

**Improvements over Implementation 3**:
- Holonomy is a d×d matrix, not a d-vector. Can rotate, scale, shear.
- Connection depends on direction (velocity), making it properly Finslerian.
- `torch.linalg.matrix_exp` guarantees invertibility of each transport step.

**Remaining problems**:
- Spray is still an MLP, not derived from a metric.
- Training still goes through the predictor MLP, not geodesic deviation.
- Unidirectional — the directionality problem persists.

---

## Implementation 5: Metric-First Design (Experiments notebook)

**Architecture**: `RandersMetricField` — neural networks output position-dependent `a_ij(x)` and `b_i(x)`. The spray is computed analytically from the energy `E = F^2/2` via autograd. Training uses geodesic deviation loss directly. No predictor MLP — prediction is via geodesic integration.

**Results on spirals** (Experiments notebook):
- Geodesic deviation: 32.3 → 25.2 over 300 epochs
- Rollout: 0.087 → 0.066

**Results on wind walks** (Experiments notebook):
- Geodesic deviation: 164.7 → 149.8 over 300 epochs
- Rollout: 0.080 → 0.053

**Comparison**: The geodesic deviation is HIGHER than Implementation 1's standalone spray (25.2 vs 6.5 for spirals). This is surprising — the metric-derived spray should be more geometrically principled. Possible reasons:
1. Shorter training (300 vs 500 epochs)
2. The metric field may need more capacity (deeper networks, more parameters)
3. The autograd chain (loss → spray → energy → metric) creates a longer gradient path, making optimization harder
4. The analytic spray from a smooth metric field may be less expressive than a direct MLP spray for the same parameter count

**Directionality**: The experiments notebook uses the same sequential transport as Implementation 3/4. The metric is better but the directionality problem is identical.

**Key visualizations produced**:
- Spatially varying indicatrices — showing how the metric shape/orientation changes across space
- Wind field plot — showing the learned Randers 1-form `b_i(x)` as arrows
- Direction-dependent spray — showing how G varies with velocity direction (Finslerian signature)
- Geodesic predictions — rollout from seed points

---

## Cross-Implementation Comparison

| | Impl 1 | Impl 2 | Impl 3 | Impl 4 | Impl 5 |
|---|---|---|---|---|---|
| **Spray source** | MLP | MLP | MLP | MLP | Randers metric |
| **Sequential** | No | No | Yes | Yes | Yes |
| **Holonomy type** | None | None | Additive (d) | Matrix (d×d) | Depends on variant |
| **Training signal** | Geodesic dev | Prediction | Prediction | Prediction | Geodesic dev |
| **Metric derived** | No | No | No | No | Yes |
| **Directionality** | N/A | N/A | Left→Right | Left→Right | Left→Right |
| **Spiral geo dev** | 6.5 | N/A | N/A | N/A | 25.2 |
| **Spiral predict** | N/A | N/A | 0.001 | TBD | N/A |
| **Wind geo dev** | ~100 | N/A | N/A | N/A | ~150 |
| **Wind predict** | N/A | N/A | 0.010 | TBD | N/A |

---

## What the Numbers Actually Tell Us

### 1. The standalone spray MLP (Impl 1) is the best spray learner

Geodesic deviation of 6.5 on spirals vs 25.2 for the metric-derived spray. The direct MLP has more capacity to fit arbitrary acceleration fields than a spray constrained to come from a Randers metric. This is the tension identified in `README_042726.md`: geometric consistency vs expressivity.

### 2. The predictor MLP (Impl 3) achieves low prediction error but it's misleading

Prediction loss of 0.001 on spirals sounds great, but the predictor is doing the work, not the holonomy. The spray/holonomy could be zero and the predictor would still learn position-dependent displacements. The low loss doesn't validate the geometric framework.

To test whether holonomy actually contributes, we'd need an ablation: train the same predictor WITHOUT holonomy (just `h_t = x_t`) and compare. If the prediction loss is the same, holonomy is dead weight. This ablation has not been run.

### 3. Wind walks plateau everywhere, confirming the noise floor analysis

- Impl 1: geodesic deviation ~100 (noise floor)
- Impl 3: prediction 0.010 (stuck from epoch 1 — the predictor quickly learns the mean displacement and then can't improve because the noise is unpredictable)
- Impl 5: geodesic deviation ~150 (even higher — the metric's smoothness constraint prevents fitting even the mean as well as the raw MLP)

These numbers are all consistent with the analysis in `exploration_444am_feb27.md`: the noise variance creates an irreducible floor, and no architecture can beat it.

### 4. No implementation addresses the directionality problem

Every sequential implementation (3, 4, 5) processes left-to-right only. None implements bidirectional transport or geodesic loops. The holonomy at token 0 is always I (identity) — the first token has zero geometric context in every version.

For causal (GPT-style) language modeling, this is actually correct. But it means:
- The model can't use future context to refine past representations
- Information from early tokens must survive through the entire holonomy chain to reach late tokens
- The effective context window is limited by the stability of the matrix product chain

---

## Gap Analysis: Are These Experiments Sufficient to Validate the Language Modeling Claim?

**No. The experiments test something fundamentally different from language modeling.**

### What the experiments actually test

"Given a 2D position and velocity, predict the next position." Every trajectory is a sequence of continuous coordinates in R^2. There are no discrete tokens, no vocabulary, no embedding lookup, no softmax over a distribution of possible next tokens. The prediction is a deterministic point in continuous space.

### What language modeling requires

"Given a sequence of discrete tokens from a vocabulary of size V, predict a probability distribution over V for the next token." The output isn't a single point — it's a distribution. "The cat sat on the ___" could be "mat" (high probability), "floor" (medium), "roof" (low), etc. The model must represent this ambiguity.

### The five gaps

1. **Discrete vs continuous**: Token embeddings live in R^d but the tokens themselves are discrete. The manifold's geodesics connect continuous points, but language jumps between discrete embedding vectors. A geodesic passing through the embedding for "cat" doesn't smoothly interpolate to "dog" — there's nothing meaningful at the midpoint.

2. **One-to-many mapping**: From any point in the spiral experiments, there's exactly one correct continuation (deterministic) or one statistical mean (wind walks). In language, "the cat" can continue as "sat", "ran", "slept", "is", etc. The geometry needs to support *branching* geodesics, not just single channels. This is more like a vector field with multiple attractors than a single geodesic.

3. **No vocabulary projection**: The experiments go from R^2 to R^2. Language modeling goes from R^d to R^V (vocabulary logits). The entire output projection layer — where the transported representation becomes token probabilities — is untested.

4. **Sequence length and context**: The trajectories have T=50 points with smooth, local dependencies. Language has complex long-range dependencies ("The cat that the dog chased ___ fast" — the verb must agree with "cat", not "dog", across an intervening clause). The holonomy chain has never been tested on this kind of structure.

5. **The actual hard problem is untested**: Can the holonomy matrix H_t encode enough context to disambiguate the next token? A 2D holonomy is a 2x2 matrix — 4 numbers. Language models need to track subject-verb agreement, coreference, topic, tense, negation, etc. simultaneously. Even at d=64, the question is whether a 64x64 holonomy matrix (4096 numbers) encodes enough distinct "path signatures" to replace what attention does with N*d numbers (sequence_length * d).

### What would be a sufficient test

The cell 35 `FinslerLanguageModel` exists but was only used for a gradient-flow check on random tokens. It was never trained on actual text. The minimum viable test would be: train on a small corpus (character-level Shakespeare, byte-level TinyStories, or even just memorizing 100 sentences), measure perplexity, and compare against a baseline (LSTM or small Transformer on the same data). Without this, the framework is a mathematical proposition with 2D visualizations, not a validated language modeling architecture.

---

## Recommendations Going Forward

### 1. Run the holonomy ablation

Compare predictor performance WITH and WITHOUT holonomy on spirals. This is the single most informative experiment. If holonomy doesn't help, the geometric framework isn't contributing — the predictor is doing all the work.

### 2. Test spatially varying dynamics

The constant-wind walks are a bad test case (see `exploration_444am_feb27.md`). Generate walks with position-dependent drift, attractors, or spatially varying curvature. These require genuine Finsler structure to model.

### 3. Implement geodesic loops (from `directionality_problem.md`)

The geodesic loop holonomy — computing pairwise holonomies from precomputed forward/backward transport — gives O(N^2) all-to-all communication with geometric meaning. This addresses the directionality problem while staying within the Finsler framework. No other proposed remedy is as clean.

### 4. Reconcile the spray capacity gap

Impl 1 (MLP spray) achieves 6.5 geodesic deviation. Impl 5 (metric-derived spray) achieves 25.2. The metric constraint reduces capacity. The hybrid approach from `README_042726.md` — `G = G_analytic + G_residual` — deserves implementation and testing. It gives geometric grounding from the metric plus flexibility from the residual.

### 5. Longer training + deeper metric for Impl 5

300 epochs and a shallow metric field may be insufficient. The Experiments notebook should be re-run with 1000+ epochs and a deeper metric network (3–4 layers) to see if the geodesic deviation can approach the MLP spray's 6.5 level. If it can't, the Randers metric may be insufficiently expressive for spiral curvature (which varies with radius and angle).
