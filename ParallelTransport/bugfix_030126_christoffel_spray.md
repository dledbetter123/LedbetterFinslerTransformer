# Bugfix: Christoffel-Based Spray — 2026-03-01

## Issues Fixed

1. **Pure Metric** — Geodesic loss stuck at 28.9015 (never decreased)
2. **Hybrid Manifold** — All losses `nan` from epoch 1

## Root Cause

`spray()` used **triple-nested autograd** (E → ∂E/∂y → ∂²E/∂y² → invert → contract) through deep networks with LayerNorm. This is numerically catastrophic — gradients either vanish (stuck loss) or explode (NaN).

## Fix: Analytic Christoffel Spray

Replaced the energy-based spray with one using **analytic Christoffel symbols** — only **one** level of autograd needed (for `∂a/∂x`, `∂b/∂x`):

### `LearnedRandersManifold.spray()` — Full Rewrite

**Before:** Triple autograd through `finsler_energy`:
```python
E = self.finsler_energy(x_g, y_g)
dE_dx, dE_dy = autograd.grad(E.sum(), [x_g, y_g], create_graph=True)
# Then second derivatives of dE_dy w.r.t y and x...
# Then invert g_matrix and contract...
```

**After:** Christoffel symbols + Randers correction:
```python
# 1. Spatial derivatives of a_ij via single autograd
da_dx[l, m, k] = autograd.grad(a[l,m].sum(), x_g, create_graph=True)

# 2. Riemannian spray from Christoffel contraction
term1 = einsum('...ljk,...j,...k->...l', da_dx, y, y)
term2 = einsum('...jkl,...j,...k->...l', da_dx, y, y)
G_riem = 0.5 * einsum('...il,...l->...i', a_inv, term1 - 0.5*term2)

# 3. Randers correction from b_i derivatives
e_ij = 0.5 * (db_dx + db_dx^T)  # symmetric
s_ij = 0.5 * (db_dx - db_dx^T)  # antisymmetric
G = G_riem + (e_00/2F)*y - alpha*s^i_0
```

### `HybridRandersManifold.spray()` — Minor Fix

Kept `torch.no_grad()` on conditioning (correct — residual shouldn't drive metric). The hybrid now inherits the stable Christoffel spray from the base class.

### `train_manifold()` — NaN-safe Gradients

Added `torch.isnan(total_norm)` check after `clip_grad_norm_` to skip optimizer steps with NaN gradients.

### Hybrid Training LR

Lowered from `1e-3` to `5e-4` for stability.

## Verification

50-epoch integration test on spiral data:

| Model | Before | After |
|-------|--------|-------|
| Pure Metric | Stuck at 28.9 | 28.4 → 19.7 ↓ |
| Hybrid | `nan` | 28.4 → 22.2 ↓ |

## Files Modified

- `ParallelTransport/phase0_metric_validation.ipynb` — cells 5, 7, 12, 15
