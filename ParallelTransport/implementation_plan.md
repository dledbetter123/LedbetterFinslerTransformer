# Finsler Parallel Transport: PhD-Level Implementation Plan

## Preamble

This plan synthesizes the findings from `directionality_problem.md`, `prior_implementations_summary.md`, `exploration_444am_feb27.md`, and `README_042726.md` into a staged research program. Each phase produces independently publishable results while building toward the full system: **a Finsler-geometric attention mechanism derived from geodesic loop holonomy on a learned Randers manifold**.

The five prior implementations established that:

- The math works (spray, connection, matrix holonomy are geometrically sound)
- The metric-first design (Implementation 5) is architecturally correct but undertrained
- The spray MLP (Implementation 1) fits better but lacks geometric grounding
- No implementation addresses the directionality problem
- No implementation has been tested on actual language data

This plan closes every gap.

---

## Phase 0: Foundation — Validated Metric-First Engine

**Goal:** A single, clean `LearnedRandersManifold` module that produces spray, connection, and transport from a learned Randers metric, validated to match or exceed the standalone MLP spray on spirals.

**Why first:** Every subsequent phase depends on a working, expressive metric field. Implementation 5 achieved geodesic deviation of 25.2 on spirals vs the MLP's 6.5. That gap must close before anything else matters.

### 0.1 — `LearnedRandersManifold` class

A single module that owns the entire geometric pipeline:

```python
class LearnedRandersManifold(nn.Module):
    """
    Position-dependent Randers metric field with autograd-derived
    spray, connection, and parallel transport.
    
    The metric F(x, y) = sqrt(y^T a(x) y) + b(x)^T y defines
    everything. No free MLP spray.
    """
    def __init__(self, d, hidden=64, n_layers=3, n_sub_steps=4):
        super().__init__()
        self.d = d
        self.n_sub_steps = n_sub_steps
        
        # Deeper metric networks (3+ layers, not 2)
        cholesky_dim = d * (d + 1) // 2
        layers_a = []
        layers_b = []
        for i in range(n_layers):
            in_dim = d if i == 0 else hidden
            layers_a += [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU()]
            layers_b += [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        layers_a.append(nn.Linear(hidden, cholesky_dim))
        layers_b.append(nn.Linear(hidden, d))
        
        self.a_net = nn.Sequential(*layers_a)
        self.b_net = nn.Sequential(*layers_b)
        
        # Near-identity initialization
        nn.init.zeros_(self.a_net[-1].weight)
        nn.init.zeros_(self.a_net[-1].bias)
        nn.init.zeros_(self.b_net[-1].weight)
        nn.init.zeros_(self.b_net[-1].bias)
    
    def metric_components(self, x):
        """Returns (a_ij, b_i) at position x."""
        raw_L = self.a_net(x)
        L = torch.zeros(*x.shape[:-1], self.d, self.d, device=x.device)
        idx = torch.tril_indices(self.d, self.d)
        L[..., idx[0], idx[1]] = raw_L
        L[..., range(self.d), range(self.d)] = F.softplus(
            L[..., range(self.d), range(self.d)]
        ) + 0.5  # ensure positive diagonal, init near 1
        a = L @ L.transpose(-1, -2)
        
        b_raw = self.b_net(x)
        # Project to ||b||_a < 1 (Randers admissibility)
        a_inv = torch.linalg.inv(a)
        b_norm_sq = torch.einsum('...i,...ij,...j->...', b_raw, a_inv, b_raw)
        scale = torch.where(
            b_norm_sq > 0.81,  # 0.9^2, leave margin
            0.9 / (b_norm_sq.sqrt() + 1e-8),
            torch.ones_like(b_norm_sq)
        )
        b = b_raw * scale.unsqueeze(-1)
        
        return a, b
    
    def finsler_energy(self, x, y):
        """E = F^2 / 2 where F = alpha + beta."""
        a, b = self.metric_components(x)
        alpha_sq = torch.einsum('...i,...ij,...j->...', y, a, y)
        alpha = torch.sqrt(alpha_sq.clamp(min=1e-12))
        beta = torch.einsum('...i,...i->...', b, y)
        F_val = alpha + beta
        return 0.5 * F_val ** 2
    
    def spray(self, x, y):
        """
        Geodesic spray G^i(x, y) from the energy E = F^2/2.
        
        Uses the identity: 2G^i = g^{ij}(dE/dx^j * y^k - dE/dx^k)
        derived from the Euler-Lagrange equation of E.
        Computed via autograd — no manual derivatives.
        """
        x_g = x.detach().requires_grad_(True)
        y_g = y.detach().requires_grad_(True)
        
        E = self.finsler_energy(x_g, y_g)
        E_sum = E.sum()
        
        dE_dx = torch.autograd.grad(E_sum, x_g, create_graph=True)[0]
        dE_dy = torch.autograd.grad(E_sum, y_g, create_graph=True)[0]
        
        # Fundamental tensor g_ij = d^2E/dy^i dy^j
        g = []
        for i in range(self.d):
            g_row = torch.autograd.grad(
                dE_dy[..., i].sum(), y_g, create_graph=True
            )[0]
            g.append(g_row)
        g_matrix = torch.stack(g, dim=-2)
        
        # Mixed partials d^2E/dy^i dx^j
        mixed = []
        for i in range(self.d):
            m_row = torch.autograd.grad(
                dE_dy[..., i].sum(), x_g, create_graph=True
            )[0]
            mixed.append(m_row)
        mixed_matrix = torch.stack(mixed, dim=-2)
        
        g_inv = torch.linalg.inv(g_matrix)
        
        # G^i = 0.5 * g^{ij} (mixed_{jk} y^k - dE/dx_j)
        rhs = torch.einsum('...jk,...k->...j', mixed_matrix, y_g) - dE_dx
        G = 0.5 * torch.einsum('...ij,...j->...i', g_inv, rhs)
        
        return G
    
    def connection(self, x, y):
        """Non-linear connection N^i_j = dG^i/dy^j."""
        y_c = y.detach().requires_grad_(True)
        G = self.spray(x, y_c)
        
        N = []
        for i in range(self.d):
            row = torch.autograd.grad(
                G[..., i].sum(), y_c, create_graph=True
            )[0]
            N.append(row)
        return torch.stack(N, dim=-2)
    
    def transport_frame(self, x_seq, dt=1.0):
        """
        Parallel transport a frame along a token sequence.
        
        Returns:
            H_fwd: list of d x d holonomy matrices, one per token
                   H_fwd[0] = I, H_fwd[t] = prod_{s<t} exp(-N_s * dt)
        """
        B, T, d = x_seq.shape
        H = torch.eye(d, device=x_seq.device).unsqueeze(0).expand(B, -1, -1)
        frames = [H]
        
        for t in range(T - 1):
            x_t = x_seq[:, t]
            x_next = x_seq[:, t + 1]
            
            sub_dt = dt / self.n_sub_steps
            for s in range(self.n_sub_steps):
                alpha = (s + 0.5) / self.n_sub_steps
                x_interp = x_t + alpha * (x_next - x_t)
                v_interp = (x_next - x_t) / dt
                
                N = self.connection(x_interp, v_interp)
                H = torch.linalg.matrix_exp(-N * sub_dt) @ H
            
            frames.append(H.clone())
        
        return frames  # length T
```

### 0.2 — Hybrid spray (metric-conditioned residual)

To close the 6.5 vs 25.2 gap without abandoning geometric grounding:

```python
class HybridRandersManifold(LearnedRandersManifold):
    """
    G_total = G_analytic(metric) + G_residual(x, y, a(x), b(x))
    
    The residual MLP receives local metric parameters as conditioning.
    Regularize ||G_residual|| toward zero so the metric does most of
    the work, with the residual capturing sharp local corrections.
    """
    def __init__(self, d, hidden=64, n_layers=3, n_sub_steps=4):
        super().__init__(d, hidden, n_layers, n_sub_steps)
        
        # Residual spray conditioned on local metric
        cholesky_dim = d * (d + 1) // 2
        residual_in = d + d + cholesky_dim + d  # x, y, flat(L), b
        self.residual_spray = nn.Sequential(
            nn.Linear(residual_in, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d)
        )
        # Initialize to near-zero output
        nn.init.zeros_(self.residual_spray[-1].weight)
        nn.init.zeros_(self.residual_spray[-1].bias)
    
    def spray(self, x, y):
        G_analytic = super().spray(x, y)
        
        raw_L = self.a_net(x.detach())
        b = self.b_net(x.detach())
        conditioning = torch.cat([x, y, raw_L, b], dim=-1)
        G_residual = self.residual_spray(conditioning)
        
        return G_analytic + G_residual
```

### 0.3 — Training protocol

```python
def train_phase0(manifold, data, epochs=1000, lr=1e-3):
    """
    Geodesic deviation + rollout stability.
    Two-phase schedule: geodesic-only for 200 epochs, then add rollout.
    """
    optimizer = torch.optim.AdamW(manifold.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    for epoch in range(epochs):
        # Extract triples
        x, v, a = extract_geodesic_triples(data, dt=1.0)
        
        G = manifold.spray(x, v)
        L_geo = ((a + 2.0 * G) ** 2).mean()
        
        L_rollout = torch.tensor(0.0)
        if epoch >= 200:
            rollout_weight = min(0.1, 0.01 * (epoch - 200) / 100)
            # ... rollout from seed, compare to ground truth
            L_rollout = rollout_loss(manifold, data) * rollout_weight
        
        # Residual regularizer (hybrid only)
        L_residual = torch.tensor(0.0)
        if hasattr(manifold, 'residual_spray'):
            # Penalize residual magnitude to keep analytic spray dominant
            L_residual = 0.01 * (G_residual ** 2).mean()
        
        loss = L_geo + L_rollout + L_residual
        loss.backward()
        torch.nn.utils.clip_grad_norm_(manifold.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
```

### 0.4 — Validation targets

| Metric | Target | Current best |
|--------|--------|--------------|
| Spiral geodesic deviation | < 8.0 | 6.5 (MLP), 25.2 (metric) |
| Spiral rollout MSE (20-step) | < 0.02 | 0.012 (MLP) |
| Variable-wind geo deviation | < noise floor + 5% | untested |
| Metric residual ratio | ||G_residual|| / ||G_total|| < 0.3 | N/A |

**Phase 0 is complete when the hybrid manifold matches or beats the MLP spray on spirals while maintaining metric grounding.**

---

## Phase 1: Spatially Varying Test Cases

**Goal:** Validate that the Finsler structure earns its keep by testing on data where position-dependent, direction-dependent geometry is required.

**Why now:** Prior experiments used spirals (deterministic, good SNR) and constant-wind walks (no spatial structure, bad SNR). Neither tests the core claim: that a Finsler manifold can learn spatially varying anisotropic geometry from trajectory data. Phase 1 provides the right test cases.

### 1.1 — Data generators

```python
def generate_variable_wind_walks(n=128, T=50, noise=0.03):
    """
    Wind direction rotates with position.
    Tests whether b_i(x) learns spatially varying drift.
    """
    traj = torch.zeros(n, T, 2)
    traj[:, 0] = torch.randn(n, 2) * 0.5
    for t in range(T - 1):
        x = traj[:, t]
        angle = torch.atan2(x[:, 1], x[:, 0])
        wind = 0.08 * torch.stack([
            torch.cos(angle + torch.pi / 4),
            torch.sin(angle + torch.pi / 4)
        ], dim=-1)
        traj[:, t + 1] = x + wind + noise * torch.randn(n, 2)
    return traj


def generate_attractor_walks(n=128, T=50, noise=0.03):
    """
    Drift toward one of K attractors. Different walks → different
    geodesic channels. Tests whether the metric creates distinct
    channels analogous to different sentence structures.
    """
    attractors = torch.tensor([[1., 1.], [-1., 1.], [0., -1.5]])
    traj = torch.zeros(n, T, 2)
    traj[:, 0] = torch.randn(n, 2) * 0.3
    targets = attractors[torch.randint(0, 3, (n,))]
    for t in range(T - 1):
        drift = 0.05 * (targets - traj[:, t])
        traj[:, t + 1] = traj[:, t] + drift + noise * torch.randn(n, 2)
    return traj


def generate_noisy_spirals(n=128, T=50, noise=0.06):
    """
    Spirals with higher noise. Tests separation of deterministic
    geometry from stochastic noise at intermediate SNR.
    """
    # Same as generate_spirals but noise_std=0.06 instead of 0.02
    pass


def generate_bifurcating_walks(n=128, T=50, noise=0.02):
    """
    Walks that start together then split based on initial velocity.
    Tests whether the Finslerian direction-dependence captures
    velocity-dependent branching (the 'one-to-many' problem).
    """
    traj = torch.zeros(n, T, 2)
    traj[:, 0] = torch.randn(n, 2) * 0.1
    v0 = F.normalize(torch.randn(n, 2), dim=-1) * 0.1
    for t in range(T - 1):
        x = traj[:, t]
        # Drift direction depends on VELOCITY direction, not just position
        v = v0 if t == 0 else (traj[:, t] - traj[:, t - 1])
        v_angle = torch.atan2(v[:, 1], v[:, 0])
        # Two attractor basins split by velocity angle
        drift_angle = torch.where(v_angle > 0, torch.pi / 4, -torch.pi / 4)
        drift = 0.06 * torch.stack([
            torch.cos(drift_angle), torch.sin(drift_angle)
        ], dim=-1)
        traj[:, t + 1] = x + drift + noise * torch.randn(n, 2)
    return traj
```

### 1.2 — Diagnostic suite

For each dataset, after training the manifold, produce:

1. **Learned b_i(x) wind field**: Quiver plot of the Randers 1-form across space. For variable-wind, should show rotating arrows. For attractors, should show convergent flow.

2. **Spatially varying indicatrices**: Plot the unit ball of F at a grid of points. Shape and orientation reveal local anisotropy. For spirals, indicatrices should elongate along spiral tangent direction.

3. **Geodesic deviation heatmap**: At each (x, y) in the training data, color by local geodesic deviation magnitude. Reveals where the metric fits well vs poorly.

4. **Direction-dependent spray difference**: At a single point x, plot ||G(x, y) - G(x, -y)|| as a function of direction. Measures Finslerian (not just Riemannian) content. Should be non-zero for all datasets.

5. **Holonomy anisotropy**: For forward transport along a trajectory, measure ||H_T - I|| (deviation from identity). For different trajectories through the same endpoints, compare holonomies. If H depends on path, the manifold has curvature. If all holonomies are similar, the manifold is effectively flat.

### 1.3 — Validation targets

| Dataset | Key metric | Target |
|---------|-----------|--------|
| Variable wind | b_i(x) angular correlation with true wind | > 0.85 |
| Attractors | Rollout reaches correct attractor basin | > 80% of trajectories |
| Noisy spirals | Geo deviation within 2x of clean spiral floor | < 15.0 |
| Bifurcating | G(x,y) != G(x,-y) by > 20% relative magnitude | verified |

---

## Phase 2: The Holonomy Ablation

**Goal:** Definitively answer: does matrix holonomy contribute to prediction, or is the predictor MLP doing all the work?

**Why now:** The `prior_implementations_summary.md` identifies this as "the single most informative experiment." Without it, the entire geometric framework might be dead weight.

### 2.1 — Experimental design

Three conditions, identical architecture except for the holonomy:

```
Condition A (full):      h_t = H_t @ x_t     (matrix holonomy, d^2 context)
Condition B (additive):  h_t = x_t + h_vec_t  (vector holonomy, d context)
Condition C (ablation):  h_t = x_t            (no holonomy, zero context)
```

All three use the same predictor MLP: `displacement = predictor(h_t)`, same training protocol, same data, same seeds (5 random seeds per condition for statistical significance).

### 2.2 — Metrics

- Prediction MSE (one-step)
- Rollout MSE (5, 10, 20 steps)
- Gradient magnitude through the holonomy chain (does the spray receive gradient signal?)
- Holonomy magnitude: ||H_T - I||_F averaged over sequences

### 2.3 — Expected outcomes and decision criteria

| Outcome | Interpretation | Action |
|---------|---------------|--------|
| A >> B >> C | Holonomy helps, matrix form matters | Proceed to Phase 3 |
| A ~ B >> C | Holonomy helps but matrix form doesn't matter | Investigate why d^2 doesn't beat d |
| A ~ B ~ C | Holonomy doesn't help at all | The predictor MLP is doing the work; re-examine training signal |
| A ~ C, B > both | Additive holonomy is better (unlikely) | Architecture error in matrix transport |

### 2.4 — Implementation

```python
class AblationPredictor(nn.Module):
    """Wraps any holonomy scheme with an identical predictor."""
    def __init__(self, d, holonomy_mode='matrix', manifold=None):
        super().__init__()
        self.mode = holonomy_mode
        self.manifold = manifold
        self.predictor = nn.Sequential(
            nn.Linear(d, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, d)
        )
    
    def forward(self, x_seq):
        B, T, d = x_seq.shape
        
        if self.mode == 'matrix':
            frames = self.manifold.transport_frame(x_seq)
            h = torch.stack([
                frames[t] @ x_seq[:, t].unsqueeze(-1)
                for t in range(T)
            ], dim=1).squeeze(-1)
        
        elif self.mode == 'additive':
            # Accumulate spray as additive offset
            h_vec = torch.zeros(B, d, device=x_seq.device)
            h_list = [x_seq[:, 0]]
            for t in range(T - 1):
                v = x_seq[:, t + 1] - x_seq[:, t]
                G = self.manifold.spray(x_seq[:, t], v)
                h_vec = h_vec + G * 0.1
                h_list.append(x_seq[:, t + 1] + h_vec)
            h = torch.stack(h_list, dim=1)
        
        elif self.mode == 'none':
            h = x_seq  # No holonomy at all
        
        displacements = self.predictor(h[:, :-1])
        predictions = x_seq[:, :-1] + displacements
        return predictions
```

---

## Phase 3: Geodesic Loop Holonomy (The Novel Contribution)

**Goal:** Implement all-to-all geometric communication via pairwise holonomy of closed loops. This is the mechanism that replaces attention with curvature.

**Why this is the key contribution:** `directionality_problem.md` identifies geodesic loops as "the most geometrically principled approach and the most novel." The holonomy of a loop connecting tokens i and j measures the curvature enclosed between them — a geometrically meaningful analog of the attention weight. Cost is O(N^2), same as standard attention, but each "weight" is a d x d matrix with geometric content.

### 3.1 — Mathematical specification

Given a sequence [x_0, ..., x_{T-1}]:

**Step 1: Precompute forward and backward transport**

```
H_fwd[0] = I
H_fwd[t] = prod_{s=0}^{t-1} exp(-N(x_s, v_s^fwd) * dt)    for t = 1..T-1

H_bwd[T-1] = I  
H_bwd[t] = prod_{s=T-1}^{t+1} exp(-N(x_s, v_s^bwd) * dt)  for t = T-2..0
```

where `v_s^fwd = (x_{s+1} - x_s) / dt` and `v_s^bwd = (x_{s-1} - x_s) / dt`.

Cost: O(T) matrix operations for each direction.

**Step 2: Pairwise loop holonomy**

```
Hol(i, j) = H_fwd[j] @ inv(H_fwd[i]) @ H_bwd[i] @ inv(H_bwd[j])
```

This transports from i to j forward, then from j back to i backward, forming a closed loop. For flat space, Hol(i, j) = I. Deviation from I encodes enclosed curvature.

Cost: O(1) per pair (matrix multiplies of precomputed matrices), O(N^2) total.

**Step 3: Holonomy-weighted aggregation**

```
h_t = sum_j alpha(t, j) * (Hol(t, j) @ x_j)

where alpha(t, j) = softmax_j(f(Hol(t, j)))
```

The attention weight `alpha(t, j)` is derived from the holonomy matrix. Possible extraction functions `f`:

- Frobenius deviation: `f(H) = ||H - I||_F` (scalar, measures total curvature)
- Trace: `f(H) = tr(H) - d` (scalar, measures rotation)  
- Learned projection: `f(H) = w^T vec(H - I)` (scalar, trainable)
- Full matrix: skip scalar extraction, use `H @ x_j` directly

### 3.2 — Implementation

```python
class GeodesicLoopAttention(nn.Module):
    """
    All-to-all communication via pairwise geodesic loop holonomy.
    
    This replaces standard dot-product attention with geometric
    attention: the 'weight' between tokens i and j is the holonomy
    of the closed loop from i to j and back, which measures the
    curvature enclosed between them.
    """
    def __init__(self, manifold, n_heads=1, extraction='learned'):
        super().__init__()
        self.manifold = manifold
        self.d = manifold.d
        self.n_heads = n_heads
        self.extraction = extraction
        
        if extraction == 'learned':
            # Project vectorized (H - I) to scalar attention weight
            self.weight_proj = nn.Linear(self.d * self.d, n_heads, bias=False)
        
        # Output projection (combine heads if multiple)
        self.out_proj = nn.Linear(self.d * n_heads, self.d)
    
    def forward(self, x_seq, causal=False):
        """
        Args:
            x_seq: (B, T, d) token embeddings
            causal: if True, mask future tokens (for autoregressive)
        
        Returns:
            h: (B, T, d) holonomy-attended representations
        """
        B, T, d = x_seq.shape
        dt = 1.0
        
        # --- Step 1: Precompute forward transport ---
        H_fwd = self._precompute_forward(x_seq, dt)   # list of T matrices (B, d, d)
        H_fwd_inv = [torch.linalg.inv(H) for H in H_fwd]
        
        # --- Step 2: Precompute backward transport ---
        H_bwd = self._precompute_backward(x_seq, dt)  # list of T matrices (B, d, d)
        H_bwd_inv = [torch.linalg.inv(H) for H in H_bwd]
        
        # --- Step 3: Pairwise holonomy ---
        # Hol(i, j) = H_fwd[j] @ H_fwd_inv[i] @ H_bwd[i] @ H_bwd_inv[j]
        
        # Stack for batch operations: (B, T, d, d)
        Hf = torch.stack(H_fwd, dim=1)        # (B, T, d, d)
        Hfi = torch.stack(H_fwd_inv, dim=1)
        Hb = torch.stack(H_bwd, dim=1)
        Hbi = torch.stack(H_bwd_inv, dim=1)
        
        # For each (i, j) pair:
        # Hol(i,j) = Hf[:,j] @ Hfi[:,i] @ Hb[:,i] @ Hbi[:,j]
        # Expand for broadcasting: (B, T_i, T_j, d, d)
        Hf_j  = Hf.unsqueeze(1)    # (B, 1, T, d, d)  — index j
        Hfi_i = Hfi.unsqueeze(2)   # (B, T, 1, d, d)  — index i
        Hb_i  = Hb.unsqueeze(2)    # (B, T, 1, d, d)
        Hbi_j = Hbi.unsqueeze(1)   # (B, 1, T, d, d)
        
        Hol = Hf_j @ Hfi_i @ Hb_i @ Hbi_j  # (B, T, T, d, d)
        
        # --- Step 4: Extract attention weights ---
        Hol_dev = Hol - torch.eye(d, device=x_seq.device)  # deviation from identity
        
        if self.extraction == 'frobenius':
            # Scalar: ||Hol - I||_F
            weights = torch.norm(Hol_dev.flatten(-2, -1), dim=-1)  # (B, T, T)
        elif self.extraction == 'trace':
            weights = torch.diagonal(Hol, dim1=-2, dim2=-1).sum(-1) - d  # (B, T, T)
        elif self.extraction == 'learned':
            flat = Hol_dev.flatten(-2, -1)    # (B, T, T, d*d)
            weights = self.weight_proj(flat)   # (B, T, T, n_heads)
            weights = weights.squeeze(-1) if self.n_heads == 1 else weights
        
        # Causal mask
        if causal:
            mask = torch.triu(
                torch.ones(T, T, device=x_seq.device), diagonal=1
            ).bool()
            weights = weights.masked_fill(mask.unsqueeze(0), float('-inf'))
        
        alpha = torch.softmax(weights, dim=-1)  # (B, T, T)
        
        # --- Step 5: Aggregate ---
        # Transform x_j by Hol(i,j) then weight by alpha
        x_expanded = x_seq.unsqueeze(1).unsqueeze(-1)  # (B, 1, T, d, 1)
        transported = (Hol @ x_expanded).squeeze(-1)    # (B, T, T, d)
        
        h = torch.einsum('btj,btjd->btd', alpha, transported)  # (B, T, d)
        h = self.out_proj(h)
        
        return h
    
    def _precompute_forward(self, x_seq, dt):
        B, T, d = x_seq.shape
        H = torch.eye(d, device=x_seq.device).unsqueeze(0).expand(B, -1, -1)
        frames = [H]
        for t in range(T - 1):
            v = (x_seq[:, t + 1] - x_seq[:, t]) / dt
            N = self.manifold.connection(x_seq[:, t], v)
            H = torch.linalg.matrix_exp(-N * dt) @ H
            frames.append(H.clone())
        return frames
    
    def _precompute_backward(self, x_seq, dt):
        B, T, d = x_seq.shape
        H = torch.eye(d, device=x_seq.device).unsqueeze(0).expand(B, -1, -1)
        frames = [None] * T
        frames[T - 1] = H
        for t in range(T - 2, -1, -1):
            v = (x_seq[:, t] - x_seq[:, t + 1]) / dt  # reversed velocity
            N = self.manifold.connection(x_seq[:, t + 1], v)
            H = torch.linalg.matrix_exp(-N * dt) @ H
            frames[t] = H.clone()
        return frames
```

### 3.3 — Comparison against standard attention

Build an identical model with standard dot-product attention replacing `GeodesicLoopAttention`:

```python
class StandardAttentionBaseline(nn.Module):
    """Same architecture but with dot-product attention for A/B comparison."""
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)
    
    def forward(self, x_seq, causal=False):
        Q = self.q_proj(x_seq)
        K = self.k_proj(x_seq)
        V = self.v_proj(x_seq)
        d_k = Q.shape[-1]
        
        weights = (Q @ K.transpose(-1, -2)) / (d_k ** 0.5)
        if causal:
            mask = torch.triu(torch.ones(T, T, device=x_seq.device), diagonal=1).bool()
            weights = weights.masked_fill(mask.unsqueeze(0), float('-inf'))
        alpha = torch.softmax(weights, dim=-1)
        h = alpha @ V
        return self.out_proj(h)
```

Test both on the Phase 1 datasets. The geodesic loop attention should outperform on tasks where geometric structure matters (spatially varying dynamics, direction-dependent branching) and may underperform on tasks where the geometry is flat (constant drift).

### 3.4 — Validation targets

| Metric | Target |
|--------|--------|
| Pairwise holonomy Hol(i,j) != I for i != j | ||Hol - I||_F > 0.01 on average |
| Learned attention weights correlate with curvature | Pearson r > 0.5 between alpha(i,j) and ||Hol(i,j) - I||_F |
| Prediction MSE (attractor walks) | Geodesic loop <= standard attention |
| Causal masking preserves autoregressive property | Verified by construction |

---

## Phase 4: Bidirectional Transport

**Goal:** Implement bidirectional transport as a simpler, more immediately useful alternative to geodesic loops, and benchmark both.

**Why include both:** Geodesic loops (Phase 3) are the theoretically richer mechanism but computationally heavier. Bidirectional transport is the "BiLSTM analog" — simpler, cheaper (2x unidirectional), and sufficient for many tasks. Having both allows cost-quality tradeoffs.

### 4.1 — Architecture

```python
class BidirectionalFinslerTransport(nn.Module):
    """
    Forward + backward transport along the sequence.
    Combines holonomies via learned weighting.
    
    Key insight: forward and backward transports use the SAME metric
    but traverse it in opposite directions. In Finsler geometry,
    F(x, y) != F(x, -y), so the forward and backward holonomies
    are genuinely different (unlike Riemannian geometry where they
    would be exact inverses).
    """
    def __init__(self, manifold, combine='learned'):
        super().__init__()
        self.manifold = manifold
        self.combine = combine
        d = manifold.d
        
        if combine == 'learned':
            self.W_fwd = nn.Linear(d, d, bias=False)
            self.W_bwd = nn.Linear(d, d, bias=False)
        elif combine == 'concat':
            self.reduce = nn.Linear(2 * d, d)
    
    def forward(self, x_seq):
        B, T, d = x_seq.shape
        
        fwd_frames = self.manifold.transport_frame(x_seq)
        
        # Backward transport: reverse the sequence
        x_rev = x_seq.flip(1)
        bwd_frames_rev = self.manifold.transport_frame(x_rev)
        bwd_frames = bwd_frames_rev[::-1]  # un-reverse
        
        h_fwd = torch.stack([
            fwd_frames[t] @ x_seq[:, t].unsqueeze(-1)
            for t in range(T)
        ], dim=1).squeeze(-1)
        
        h_bwd = torch.stack([
            bwd_frames[t] @ x_seq[:, t].unsqueeze(-1)
            for t in range(T)
        ], dim=1).squeeze(-1)
        
        if self.combine == 'sum':
            return h_fwd + h_bwd
        elif self.combine == 'learned':
            return self.W_fwd(h_fwd) + self.W_bwd(h_bwd)
        elif self.combine == 'concat':
            return self.reduce(torch.cat([h_fwd, h_bwd], dim=-1))
```

### 4.2 — Asymmetry analysis

A critical diagnostic specific to Finsler (vs Riemannian) geometry:

```python
def measure_finsler_asymmetry(manifold, x_seq):
    """
    Quantify how much the forward and backward holonomies differ.
    In Riemannian geometry: H_fwd @ H_bwd = I exactly.
    In Finsler geometry: H_fwd @ H_bwd != I due to b_i asymmetry.
    The deviation measures the 'Finslerian content' of the learned metric.
    """
    fwd = manifold.transport_frame(x_seq)
    bwd_rev = manifold.transport_frame(x_seq.flip(1))
    bwd = bwd_rev[::-1]
    
    products = [fwd[t] @ bwd[t] for t in range(len(fwd))]
    deviations = [
        torch.norm(p - torch.eye(manifold.d, device=p.device), dim=(-2, -1))
        for p in products
    ]
    return torch.stack(deviations, dim=1)  # (B, T), should be > 0
```

---

## Phase 5: Multi-Scale Transport

**Goal:** Address long-range dependencies via hierarchical transport at O(T log T) cost.

**Why:** For sequences longer than ~100 tokens, sequential transport suffers from the same vanishing gradient / information bottleneck as RNNs. Multi-scale transport builds a dyadic hierarchy where each level captures context at a different scale. This is the Finsler analog of Longformer / hierarchical attention.

### 5.1 — Architecture

```python
class MultiScaleTransport(nn.Module):
    """
    Hierarchical transport: log(T) levels, each covering 2x the range.
    
    Level 0: transport between adjacent tokens (local)
    Level 1: transport between every 2nd token (phrase-level)
    Level 2: transport between every 4th token (clause-level)
    ...
    Level L: transport across the full sequence (global)
    
    At each level, the holonomy is computed at the COARSENED scale
    using the metric field. Higher-level transport sees "summarized"
    trajectories, not individual tokens.
    """
    def __init__(self, manifold, max_levels=None):
        super().__init__()
        self.manifold = manifold
        self.max_levels = max_levels
        self.level_weights = nn.Parameter(torch.ones(16))  # up to 2^16 seq len
    
    def forward(self, x_seq):
        B, T, d = x_seq.shape
        n_levels = min(
            int(torch.log2(torch.tensor(T, dtype=torch.float)).item()) + 1,
            self.max_levels or 100
        )
        
        level_outputs = []
        for level in range(n_levels):
            stride = 2 ** level
            if stride >= T:
                break
            
            # Subsample sequence at this scale
            indices = torch.arange(0, T, stride, device=x_seq.device)
            x_sub = x_seq[:, indices]
            
            # Transport at this scale
            frames = self.manifold.transport_frame(x_sub, dt=stride)
            
            # Interpolate holonomies back to full resolution
            H_full = self._interpolate_frames(frames, indices, T, B, d, x_seq.device)
            
            h_level = torch.stack([
                H_full[t] @ x_seq[:, t].unsqueeze(-1)
                for t in range(T)
            ], dim=1).squeeze(-1)
            
            level_outputs.append(h_level)
        
        # Weighted combination across scales
        weights = torch.softmax(self.level_weights[:len(level_outputs)], dim=0)
        h = sum(w * out for w, out in zip(weights, level_outputs))
        return h
    
    def _interpolate_frames(self, frames, indices, T, B, d, device):
        """Linearly interpolate holonomy matrices to full resolution."""
        H_full = []
        for t in range(T):
            # Find surrounding indices in subsampled sequence
            pos = t / (indices[1] - indices[0]).float() if len(indices) > 1 else 0
            lo = int(pos.floor().item()) if isinstance(pos, torch.Tensor) else int(pos)
            lo = min(lo, len(frames) - 2)
            hi = lo + 1
            alpha = pos - lo if isinstance(pos, (int, float)) else (pos - lo).item()
            
            H_interp = (1 - alpha) * frames[lo] + alpha * frames[hi]
            H_full.append(H_interp)
        return H_full
```

### 5.2 — Complexity comparison

| Mechanism | Time | Space | Context range |
|-----------|------|-------|---------------|
| Sequential transport | O(T) | O(d^2) | Unidirectional, decays |
| Bidirectional transport | O(T) | O(d^2) | Both directions, decays |
| Geodesic loop holonomy | O(T^2) | O(T^2 * d^2) | All-to-all, geometric |
| Multi-scale transport | O(T log T) | O(log(T) * d^2) | All scales, hierarchical |
| Standard attention | O(T^2) | O(T^2) | All-to-all, learned |

---

## Phase 6: Language Model Integration

**Goal:** Train on actual text and measure perplexity. This is the experiment that validates or falsifies the entire framework for language modeling.

**Why last:** Every prior phase builds a validated component. Phase 6 assembles them into a language model. Attempting language modeling before the geometry is validated wastes effort diagnosing whether failures are geometric or architectural.

### 6.1 — Architecture

```python
class FinslerLanguageModel(nn.Module):
    """
    Full language model with Finsler geometric attention.
    
    Architecture:
        Token IDs → Embedding → Geodesic Loop Attention → Output Projection
    
    The manifold operates in the d-dimensional embedding space.
    The geodesic loop holonomy provides context-dependent representations.
    The output projection maps to vocabulary logits.
    
    Training has TWO objectives:
        1. Geodesic deviation: the metric should make embedding trajectories
           into geodesics (trains the manifold)
        2. Cross-entropy: the output projection should predict the next token
           (trains the projection + fine-tunes the manifold)
    
    These can be trained jointly or in alternating phases.
    """
    def __init__(self, vocab_size, d_model, manifold_type='loop',
                 n_layers=1, n_heads=1):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        
        self.manifold = LearnedRandersManifold(d_model, hidden=256, n_layers=3)
        
        self.attention_layers = nn.ModuleList()
        for _ in range(n_layers):
            if manifold_type == 'loop':
                self.attention_layers.append(
                    GeodesicLoopAttention(self.manifold, n_heads=n_heads)
                )
            elif manifold_type == 'bidirectional':
                self.attention_layers.append(
                    BidirectionalFinslerTransport(self.manifold)
                )
            elif manifold_type == 'multiscale':
                self.attention_layers.append(
                    MultiScaleTransport(self.manifold)
                )
        
        # Feedforward (standard Transformer FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, token_ids):
        B, T = token_ids.shape
        
        x = self.embedding(token_ids) + self.pos_encoding[:, :T]
        
        for attn in self.attention_layers:
            h = attn(self.ln1(x), causal=True)
            x = x + h  # residual
            x = x + self.ffn(self.ln2(x))  # FFN + residual
        
        logits = self.output_proj(x)
        return logits
    
    def geodesic_deviation_loss(self, token_ids):
        """Auxiliary loss: embedding trajectories should be geodesics."""
        x = self.embedding(token_ids) + self.pos_encoding[:, :token_ids.shape[1]]
        
        dt = 1.0
        v = (x[:, 1:] - x[:, :-1]) / dt
        a = (v[:, 1:] - v[:, :-1]) / dt
        
        G = self.manifold.spray(x[:, 1:-1], v[:, :-1])
        return ((a + 2.0 * G) ** 2).mean()
```

### 6.2 — Training protocol

**Phase A: Metric pre-training (optional but recommended)**

Pre-train the manifold on embedding trajectories using geodesic deviation only. Feed sequences of token embeddings (from a pre-trained or random embedding) and train the metric to make them geodesics. This gives the manifold a head start before the full LM loss is applied.

```python
def pretrain_metric(model, dataloader, epochs=50, lr=1e-3):
    optimizer = torch.optim.AdamW(model.manifold.parameters(), lr=lr)
    for epoch in range(epochs):
        for batch in dataloader:
            loss = model.geodesic_deviation_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.manifold.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
```

**Phase B: Joint LM training**

```python
def train_lm(model, dataloader, epochs=100, lr=5e-4,
             geo_weight=0.1, geo_warmdown=0.995):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for batch in dataloader:
            logits = model(batch[:, :-1])
            targets = batch[:, 1:]
            
            L_ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1)
            )
            
            L_geo = model.geodesic_deviation_loss(batch)
            
            loss = L_ce + geo_weight * L_geo
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        geo_weight *= geo_warmdown  # gradually reduce geometric regularization
```

### 6.3 — Datasets (ordered by complexity)

1. **Memorization test** (10 sentences, repeated): Can the model memorize and reproduce? If not, the architecture has fundamental capacity issues. Trivial but necessary.

2. **Character-level tiny Shakespeare** (~1M chars): Standard small-scale LM benchmark. Compare perplexity against LSTM and 1-layer Transformer baselines.

3. **TinyStories** (byte-level or BPE): Structured narratives with clear grammatical patterns. Tests whether the holonomy captures syntactic dependencies.

4. **WikiText-2** (standard benchmark): Direct comparison to published Transformer results at the same model size.

### 6.4 — Baselines

All baselines use the same embedding dimension, number of parameters (approximately), and training budget:

| Model | Parameters | Description |
|-------|-----------|-------------|
| LSTM (1-layer) | ~d^2 + 4d^2 | Sequential baseline |
| BiLSTM | ~2x LSTM | Bidirectional sequential baseline |
| Transformer (1-layer, 1-head) | ~4d^2 | Attention baseline |
| FinslerLM (loop) | ~manifold + d^2 | Geodesic loop attention |
| FinslerLM (bidir) | ~manifold + d^2 | Bidirectional transport |
| FinslerLM (multiscale) | ~manifold + d^2 | Multi-scale transport |

### 6.5 — Validation targets

| Metric | Minimum viable | Competitive |
|--------|---------------|-------------|
| Char-Shakespeare perplexity | < 2x Transformer | Within 1.2x Transformer |
| Memorization (10 sentences) | 100% after 500 epochs | 100% after 50 epochs |
| Geodesic deviation (embedding trajectories) | Decreasing over training | Converges below noise floor |
| Finsler asymmetry (forward != backward) | Measurably non-zero | Correlates with syntactic structure |
| Holonomy path-dependence | Different paths → different H | H encodes enough context to predict |

---

## Phase 7: Analysis and Interpretability

**Goal:** Understand what the learned geometry means. This is what separates a PhD thesis from a workshop paper.

### 7.1 — Geometric interpretability

1. **Indicatrix atlas of the embedding space**: After training on text, visualize the metric's unit ball at the embeddings of specific words. Do semantically similar words have similar indicatrices? Do function words (the, a, is) have different geometric profiles than content words (cat, run, blue)?

2. **Geodesic channels**: Compute geodesics from a given word embedding in multiple directions. Do they pass near embeddings of grammatically/semantically compatible words? E.g., starting from "the" with forward momentum, does the geodesic pass near nouns?

3. **Holonomy as syntax**: For specific sentences, visualize the pairwise holonomy matrix Hol(i, j). Does it reveal syntactic structure? E.g., for "the cat that the dog chased ran fast", is the holonomy between "cat" and "ran" different from the holonomy between "dog" and "ran"? It should be — "cat" is the subject of "ran", "dog" is not.

4. **Curvature map**: Compute the Ricci curvature scalar at various points in embedding space. Do high-curvature regions correspond to semantically rich or syntactically complex regions?

### 7.2 — Ablation matrix

| Ablation | What it tests |
|----------|--------------|
| Remove b_i (set to zero) | Is Finslerian asymmetry needed, or is Riemannian enough? |
| Remove residual spray (hybrid only) | How much does the residual contribute? |
| Freeze metric, train only projection | Can a fixed (random) geometry work? |
| Replace geodesic loop with dot-product | Is the geometric attention better than learned attention? |
| Remove geodesic deviation loss | Does the geometric training signal help LM performance? |
| Single-direction transport only | How much does bidirectional/loop help? |
| Reduce d from 64 to 16 to 4 | How does holonomy capacity (d^2) affect LM quality? |

### 7.3 — Scaling analysis

For the paper, characterize:

- **Parameter efficiency**: Parameters vs perplexity curve, compared to Transformer at the same parameter counts.
- **Sequence length scaling**: How does perplexity change with context length? Multi-scale transport should degrade more gracefully than sequential.
- **Dimensionality scaling**: The holonomy is d x d. At what d does it carry enough context? Compare to attention, which scales as T * d.
- **Computational cost**: Wall-clock time per training step. The autograd chain for the metric-derived spray is expensive. Profile it. Identify the bottleneck.

---

## Dependency Graph

```
Phase 0: Validated Metric Engine
    |
    +---> Phase 1: Spatially Varying Test Cases
    |         |
    |         +---> Phase 2: Holonomy Ablation
    |                   |
    |                   +---> Phase 3: Geodesic Loop Holonomy
    |                   |         |
    |                   |         +---> Phase 6: Language Model Integration
    |                   |                    |
    |                   |                    +---> Phase 7: Analysis
    |                   |
    |                   +---> Phase 4: Bidirectional Transport
    |                   |         |
    |                   |         +---> Phase 6 (alternate attention)
    |                   |
    |                   +---> Phase 5: Multi-Scale Transport
    |                             |
    |                             +---> Phase 6 (alternate attention)
```

Phases 3, 4, 5 are parallel branches (different attention mechanisms). All feed into Phase 6. Phase 7 follows Phase 6.

**Critical path:** 0 → 1 → 2 → 3 → 6 → 7

---

## Estimated Timeline

| Phase | Duration | Cumulative | Deliverable |
|-------|----------|-----------|-------------|
| 0 | 2-3 weeks | 2-3 weeks | Validated manifold engine, hybrid spray matching MLP |
| 1 | 1-2 weeks | 4-5 weeks | Four new test datasets, diagnostic suite |
| 2 | 1 week | 5-6 weeks | Ablation results (holonomy contribution quantified) |
| 3 | 2-3 weeks | 7-9 weeks | Geodesic loop attention, comparison to standard attention |
| 4 | 1 week | 8-10 weeks | Bidirectional transport (simpler parallel effort) |
| 5 | 1-2 weeks | 9-12 weeks | Multi-scale transport |
| 6 | 3-4 weeks | 12-16 weeks | Language model training + baselines |
| 7 | 2-3 weeks | 14-19 weeks | Analysis, figures, interpretability |

Total: ~4-5 months for the full research program.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Autograd chain too deep for d > 16 (training instability) | Medium | High | Use explicit Randers spray formula, or low-rank approximations |
| Holonomy ablation shows no benefit | Medium | Critical | If holonomy is dead weight, the project pivots to curvature-based local context (Phase 7 contingency) |
| Geodesic loop is O(N^2) in d^2 (memory) | High for d=64, T=512 | High | Sparse loops (only compute for nearby or salient pairs), low-rank holonomy approximation |
| Language model perplexity much worse than Transformer | Medium | High | Start with very small scale (d=16, T=32, 10 sentences) to debug before scaling |
| Metric-derived spray remains less expressive than MLP | Medium | Medium | The hybrid (Phase 0.2) covers this; if residual dominates, the metric adds regularization but not the core model |
| Finsler asymmetry is negligible (Riemannian is sufficient) | Low | Medium | The ablation in Phase 7 tests this directly; if true, simplify to Riemannian geometry |

---

## File Organization

```
ParallelTransport/
    implementation_plan.md          <- this document
    directionality_problem.md       <- existing: problem statement
    prior_implementations_summary.md <- existing: prior work review
    
    src/
        manifold.py                 <- LearnedRandersManifold, HybridRandersManifold
        attention.py                <- GeodesicLoopAttention, BidirectionalTransport
        multiscale.py               <- MultiScaleTransport
        language_model.py           <- FinslerLanguageModel
        data.py                     <- All data generators
        diagnostics.py              <- Visualization + ablation utilities
        baselines.py                <- LSTM, Transformer baselines
    
    notebooks/
        phase0_metric_validation.ipynb
        phase1_spatial_datasets.ipynb
        phase2_holonomy_ablation.ipynb
        phase3_geodesic_loops.ipynb
        phase4_bidirectional.ipynb
        phase5_multiscale.ipynb
        phase6_language_model.ipynb
        phase7_analysis.ipynb
    
    results/
        figures/
        checkpoints/
        metrics/
```

---

## Summary of Novel Contributions

1. **Geodesic loop holonomy as attention**: The pairwise holonomy of closed loops connecting tokens derives an all-to-all communication mechanism from Finsler curvature. The attention weight between tokens i and j is the curvature enclosed by the geodesic loop between them. This is a genuinely new attention mechanism with geometric interpretation and O(N^2) cost.

2. **Metric-conditioned hybrid spray**: Analytic spray from a learned Randers metric plus a metric-conditioned residual MLP. Geometric grounding plus expressivity.

3. **Finsler asymmetry as directional bias**: The Randers 1-form b_i creates different forward and backward holonomies. This asymmetry naturally encodes the directional bias of language (syntax flows left-to-right) without an explicit causal mask — the geometry itself is directional.

4. **Multi-scale hierarchical transport**: O(T log T) geometric context at all scales, connecting to the broader literature on efficient attention.

5. **Complete geometric pipeline from metric to language model**: Training the metric via geodesic deviation, deriving attention from curvature, predicting tokens from transported representations. The narrative is: "The metric defines the geometry. The curvature enclosed by token-pair loops defines attention. Training makes data into geodesics. Attention emerges from curvature."
