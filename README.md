# The Finsler Transformer

**What if attention is not a mechanism you compute, but a curvature you move through?**

The Finsler Transformer is an applied-math research project that replaces the transformer's $O(T^2)$ attention with **geodesic flow on a learned Finsler manifold**. Instead of scoring every pair of tokens, the model learns a *geometry* in which a sentence is a single trajectory — a geodesic — and "attention" becomes the cumulative deformation of space the trajectory experiences along its route. The aim is an $O(T)$ autoregressive generator whose context-handling is baked into the metric rather than recomputed at every step.

---

## 1. Why Finsler, and not Euclidean or Riemannian

A standard transformer implicitly treats representation space as **Euclidean**: distances are symmetric and direction-independent, $g_{ij}(x) = \delta_{ij}$ everywhere. Attention scores inherit that symmetry — $\text{relevance}(A\!\to\!B)$ and $\text{relevance}(B\!\to\!A)$ are computed by the same inner product. But language is **directional and asymmetric**: *A implies B* is not *B implies A*; syntax sharply constrains what may follow what. A geometry that is symmetric by construction is the wrong prior.

A **Riemannian** metric $g_{ij}(x)$ lets distance vary with *position* but keeps it symmetric in *direction*, and isotropic diffusion on a Riemannian manifold tends to **over-smooth** — tokens drift toward generic attractors and lose their specific identity (the same rank-collapse pathology seen in deep GNNs).

**Finsler geometry** is the natural generalization: a Finsler norm

$$F(x, v) : TM \to \mathbb{R}_{>0}, \qquad F(x,v) \neq F(x,-v)\ \text{in general}$$

assigns a *direction-dependent* cost to motion at every point of the manifold. The project uses the simplest non-Riemannian Finsler metric, the **Randers metric**:

$$F(x, y) = \sqrt{a_{ij}(x)\, y^i y^j} \;+\; b_i(x)\, y^i, \qquad \|b\|_a < 1,$$

where $a_{ij}(x)$ is a learned Riemannian background (baseline semantic similarity) and $b_i(x)$ is a learned 1-form — a *"wind"* that biases flow toward the next logical token. The geometry can **stiffen** to block information flow in irrelevant directions and **loosen** to channel it toward grammatically or semantically licensed continuations. Asymmetry, the thing language most needs and Euclidean attention least provides, is built into the metric for free.

---

## 2. The core idea: a sentence is a geodesic

The governing slogan: **context is not an additive matrix operation; it is the geometric deformation of the latent space.**

- **Sequence as trajectory.** A sentence is a path $x(t)$ through the manifold. The exact ordering of tokens defines a unique differential equation whose solution that path is.
- **Attention is curvature.** As the state moves $w_1 \to w_2$, the non-linear connection distorts the tangent space seen by $w_3$. The "attention" a token pays to its predecessors is *organically encoded* in the accumulated distortion along the route — no $T\times T$ score matrix is ever formed.
- **Path-dependent meaning is holonomy.** Two different routes arriving at the same point produce different **holonomy** — different accumulated frame rotation — and therefore different representations. Where standard attention is position-dependent only, Finsler transport is genuinely *path*-dependent, which is the correct inductive bias for language.

Concretely, generation integrates the **geodesic equation**

$$\frac{d^2 x^i}{dt^2} + 2\,G^i(x, \dot{x}) = 0,$$

where the spray coefficients $G^i(x,y)$ — the "laws of physics" of the sequence — are derived analytically from the Randers metric via the energy functional $E = \tfrac12 F^2$ and the Euler–Lagrange equation, using autograd. The non-linear connection $N^i_j = \partial G^i/\partial y^j$ drives frame parallel transport, $H_{t+1} = \exp(-N\,dt)\,H_t$, which is where holonomy accumulates. Training minimizes the **geodesic deviation** — the residual of the geodesic equation on observed sequences — plus a rollout term that forces the learned metric to produce *stable* trajectories under iteration, not merely pointwise fits.

---

## 3. The efficiency argument

For autoregressive generation, cost is **$O(T)$ total** rather than attention's $O(T^2)$: each step evaluates the spray (one autograd pass over $F^2/2$), forms the connection Jacobian, applies a small matrix exponential ($O(d^3)$ with $d$ small), and integrates a handful of ODE sub-steps. There is no growing pairwise score matrix and no quadratic blow-up — the context lives in the holonomy and the metric, both of fixed size. In the long-sequence regime $T \gg d$, where attention hurts most, geometric transport is asymptotically favorable.

---

## 4. What was built, what worked, and what didn't

This is an exploration repo, and the failure analysis is the most valuable part of it.

- **Synthetic 2D validation.** On deterministic spirals (high signal-to-noise, where acceleration is uniquely determined by $(x, v)$) the metric-first model learns a sensible geometry. On constant-"wind" stochastic walks, geodesic deviation correctly plateaus at the **noise floor** — the Randers $b$-form learns the drift direction and the spray learns $\approx$ flat space; reproducing the *mean* trajectory is the right answer, not a failure.
- **Failure modes that shaped the design:**
  - A free-form **MLP spray** is expressive but geometrically ungrounded — no guarantee of 2-homogeneity, metric compatibility, or consistent geodesics.
  - **Additive holonomy** (accumulating a vector onto position) is a translation, not a frame transformation — it silently discards the direction-dependence that is the whole point.
  - A **space mismatch** between where accelerations were measured (raw data) and where the spray predicted them (encoded space) produced meaningless gradients.
  - **Compounding** all geodesic layers per step exploded rollout error far outside the training regime.
- **The lesson → metric-first architecture.** Learn the Randers metric explicitly; derive the spray analytically from it (geometry stays grounded); train on geodesic deviation directly. The current frontier is metric **expressiveness**: a smooth 2–3 layer metric net cannot carve the sharp, localized channels language needs. The proposed fix is a **Hybrid Randers manifold** — analytic spray plus a residual network *conditioned on the local metric parameters*, so the network corrects the geometry rather than replacing it — targeting the gap from analytic-only deviation toward MLP-level fit while keeping geometric guarantees.

---

## 5. Where more resources go

1. **Spatially varying test geometry** — rotating-wind fields and multi-attractor dynamics, the synthetic analog of *different sentence contexts inducing different dynamics in different regions*. This is the direct test of the core claim that position-and-direction-dependent metrics learn trajectory structure.
2. **Language-model proof of concept** — train on tokenized text, compare perplexity against a parameter- and FLOP-matched transformer, and verify that path-conditioned representations measurably help next-token prediction, especially at long context where the $O(T)$ advantage is largest.
3. **Multi-chart manifolds** — partition embedding space into overlapping local coordinate charts $M_q = \sum_k w_k(q) M_q^{(k)}$, each with its own metric. Manifolds are *locally* Euclidean and patched together; a single global metric is too smooth for high-dimensional language, and charts restore local specialization.
4. **Richer Finsler structure** — move beyond Randers toward more general $(\alpha,\beta)$ metrics, and connect the construction to information geometry (Fisher metrics on the output simplex), where the asymmetry of $F$ aligns naturally with the asymmetry of KL divergence.

The thesis the project is testing: **bake linguistic structure into geometry** so that similar meanings flow along similar geodesics and genuine branch points (where multiple continuations are valid) appear as regions of geodesic divergence — yielding a generator whose context cost is a property of the space, not a quadratic tax paid per token.

---

## 6. Relation to Sparse Geometric Signal Transport

This is the sister project to [**Sparse Geometric Signal Transport (SGST)**](https://github.com/dledbetter123/SparseGeometricSignalTransport). Both pursue efficient generative AI by treating sequence processing as transport on a learned geometry, from two complementary angles:

- **SGST** works in the **spectral fiber bundle**: tokens are sparse constellations of Fourier modes, transport is a gauge-covariant Wilson line, and the open frontier is closing attention's state-capacity gap with complex, sparse, delta-rule transport.
- **The Finsler Transformer** works on the **base manifold itself**: the metric is asymmetric and direction-dependent, the sequence *is* the geodesic, and holonomy replaces the attention matrix.

SGST contributes efficient spectral computation and the fiber-bundle decomposition; Finsler contributes the geometric grounding for *why* a given transport is the right one and *how* directionality arises. Together they sketch a Finsler-geometric, spectrally-transported, path-dependent generative model — the long-horizon target of this research line.

---

## Repository map

| Path | What it holds |
|---|---|
| `LedbetterFinslerTransformer.md` | The original architectural blueprint — geometry, holonomy, the "attention is curvature" thesis. |
| `README_042726.md` | Extended design notes and the metric-first rationale. |
| `exploration_444am_feb27.md` | The failure analysis and the wind-walk vs. spiral SNR argument — why test-case choice matters. |
| `ParallelTransport/` | Implementation plan, prior-implementation summary, Christoffel/spray bugfix notes, the directionality problem writeup. |
| `*.ipynb` | Experiment notebooks (large; clean versions tracked selectively). |

*Heavy notebook outputs and any thesis LaTeX are intentionally git-ignored.*
