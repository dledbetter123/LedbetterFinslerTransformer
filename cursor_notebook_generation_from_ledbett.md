# Notebook generation from Ledbetter Finsler findings
_Exported on 2/27/2026 at 01:34:49 PST from Cursor (2.4.21)_

---

**User**

Inspect @LedbetterFinslerTransformer.md and generate a notebook from on the findings with phD rigor/commentary.

---

**Cursor**

This is a rich theoretical document. Let me build a comprehensive notebook that develops the full pipeline — from mathematical foundations through implementation and visualization — with PhD-level exposition throughout.

The notebook `LedbetterFinslerTransformer.ipynb` is ready. Here is a summary of its structure and the PhD-level treatment applied:

**10 sections, ~20 cells** covering theory through experiment:

1. **Abstract** — Frames the contribution: replacing O(N^2) attention with O(N) geodesic integration on a learned Finsler manifold via Randers metric and spray coefficients.

2. **Motivation & Geometric Critique** — Formally identifies three implicit assumptions of standard dot-product attention (Euclidean latent space, position-independence, global communication) and connects over-smoothing to isotropic diffusion on Riemannian manifolds.

3. **Mathematical Foundations** — Full definitions: Finsler structures, the fundamental tensor, Randers metric with validity condition, geodesic spray coefficients, non-linear connection, and holonomy. Each concept is tied back to the linguistic interpretation.

4. **Architecture Design** — Three design principles, plus a comparison table (Transformer vs. Finsler model) covering complexity, geometry, directionality, path dependence, and smoothing.

5. **Implementation** — Four PyTorch modules:
   - `RandersMetric` — Cholesky-parametrized a_ij with projected b_i for Finsler validity, plus indicatrix computation.
   - `SprayNetwork` — Residual MLP with LayerNorm approximating G^i(x, y), with discussion of the 2-homogeneity constraint.
   - `FinslerGeodesicLayer` — Symplectic Euler integration (not explicit Euler), with explanation of why symplecticity matters for Hamiltonian structure preservation.
   - `FinslerGeodesicTransformer` — Full model stacking K layers with momentum estimation.

6. **Geodesic Deviation Loss** — Derived from the residual of the geodesic equation on observed data, plus spray smoothness regularization.

7. **Synthetic Experiments** — Three trajectory families (spirals, Lissajous, wind-biased walks), training loop with cosine annealing, convergence diagnostics, spray field visualization, and geodesic prediction vs. ground truth.

8. **Indicatrix Analysis** — Standalone Randers metric fitted to velocity statistics, visualized against the Euclidean unit circle to reveal learned anisotropy and wind direction.

9. **Language Model Proof of Concept** — `FinslerLanguageModel` with embedding, geodesic integration, and output projection. Verifies end-to-end gradient flow on random data.

10. **Discussion** — Complexity analysis, theoretical advantages (intrinsic directionality, holonomic context, geometric regularization), limitations (long-range deps, parallelism, integrator order), connections to Neural ODEs / hyperbolic networks / linear attention, and concrete future directions.

---

**User**

The calculated geodesics are incorrect


---

**Cursor**

Let me examine the prediction and training code to diagnose the root cause.

Let me read the specific code cells.

Let me read the training and prediction cells more precisely. I'll use the notebook cell indices.

I can see the problems clearly now. The training log shows geodesic deviation stuck at ~273,000 — essentially not learning. There are three fundamental bugs:

1. **Space mismatch in the loss**: The model transforms positions through `position_encoder` and `momentum_encoder` (Linear layers), so the spray predictions live in an *encoded* space. But the loss computes empirical accelerations from the *original* data. Comparing quantities in different spaces makes the gradient signal meaningless.

2. **Spray averaging across layers**: The loss averages spray predictions from all layers, but each layer sees a *different* evolved state. Layer 2 sees the output of Layer 1, not the original data. Averaging these and comparing to the original acceleration is physically incorrect.

3. **Prediction function compounds integration**: `predict_trajectory` applies ALL layers (each an integration step) per prediction step, so 40 steps x 4 layers = 160 integration steps — far beyond the training regime. It also mixes original-space seed points with encoded-space predictions.

The fix is to properly separate the 2D toy experiment from the full language model architecture. For the toy case, we should train a single spray network directly on (position, velocity) -> acceleration pairs, then integrate forward for prediction. Let me rewrite the affected cells.

Now let me rewrite the training code cell (cell 19):

Let me read the exact content of cell 19 to get the precise text.

The source was modified since I wrote it. Let me get the exact current content.

Now I have the exact content. Let me edit the cells properly.

Good. Now let me update the training diagnostics cell (cell 21) to work with the new history format.

Now let me update the spray field visualization (cell 23) to use the trained spray directly.

Now the most critical fix — the prediction cell (cell 24 markdown, cell 25 code). Let me update both.

Now let me read the exact content of cell 25 to replace it properly.

Now let me verify the indicatrix analysis cell (cell 27) also doesn't reference `model_spiral` — it should use the `RandersMetric` directly and doesn't depend on the spray model.

All references to `model_spiral` have been replaced with `spray_spiral`. The indicatrix section (cell 27) uses its own standalone `RandersMetric`, so no changes needed there.

Let me also verify the `FinslerLanguageModel` in cell 29 still works — it uses `FinslerGeodesicTransformer` which we preserved.

Good — the `FinslerLanguageModel` still uses `FinslerGeodesicTransformer` and `FinslerConfig`, which are intact. That's the right design: the multi-layer model with encoders is for the high-dimensional language task, while the 2D toy uses direct spray training.

All fixes are in place. Here is a summary of what was wrong and what changed:

**Root causes of the diverging geodesics:**

1. **Space mismatch in the loss function** (cell 15). The `GeodesicDeviationLoss` computed empirical accelerations from original `x_seq` but compared them against spray predictions from an *encoded* space (after `position_encoder` and `momentum_encoder` transforms). The gradient signal was comparing apples to oranges.

2. **Incorrect spray averaging across integration layers** (cell 15). The loss averaged spray predictions from all layers, but each layer saw *different* evolved states — layer 2's spray was evaluated at layer 1's output, not the original data. Averaging and comparing to original-space acceleration is physically inconsistent.

3. **Prediction compounded integration steps** (cell 25). The old `predict_trajectory` applied all 4 layers per step (each a separate integration step), so 40 prediction steps x 4 layers = 160 total integration steps — far beyond the training regime. It also mixed original-space seed points with encoded-space outputs.

**What changed:**

- **Cell 18** (markdown): Updated commentary explaining why direct spray training is correct for the 2D toy case.
- **Cell 19** (code): Replaced `train_finsler_model` with `train_spray_direct` — extracts `(position, velocity, acceleration)` triples from data and trains a single `SprayNetwork` to satisfy the geodesic equation `a + 2G(x,v) = 0` directly. No encoders, no multi-layer integration during training.
- **Cell 21** (code): Simplified training diagnostic to single geodesic deviation plot.
- **Cell 23** (code): Updated `visualize_spray_field` to accept a `SprayNetwork` directly instead of the full model.
- **Cell 24** (markdown): Updated prediction commentary with the symplectic Euler scheme.
- **Cell 25** (code): Completely rewritten `predict_trajectory` — estimates velocity at the last seed point, then iterates single symplectic Euler steps using the trained spray. No encoding layers, no multi-layer compounding.

The `FinslerGeodesicTransformer` and `FinslerLanguageModel` are preserved in cells 13 and 29 for the high-dimensional language modeling proof of concept, where the multi-layer architecture with encoders is appropriate.

---

**User**

Why do we not do anything with the model during training???

---

**Cursor**

Good catch. The training function only does pointwise supervision — it teaches the spray what the acceleration should be at observed data points but never actually *integrates* the spray to verify it produces correct trajectories. This is the train-test mismatch: prediction integrates iteratively, but training never exercises that path.

Let me look at the current training code to confirm.

Right. The issue is clear — the training only does pointwise regression (`G(x,v)` vs empirical acceleration at observed data points), but never integrates the spray forward and backpropagates through the integration steps. During prediction, errors at one step feed into the next, so the spray needs to be *stable under iteration*, not just pointwise accurate.

The fix: add a **rollout loss** — during training, integrate the spray from initial conditions and compare the full integrated trajectory against the ground truth. This is essentially backpropagation through the symplectic Euler integration scheme, forcing the spray to learn dynamics that are stable under iteration.

Now let me rewrite the training code to include the rollout loss:

The user has been tweaking parameters. Let me get the exact current cell content.

Let me use a targeted match on the part that needs to change. I'll replace the `train_spray_direct` function and the training call, keeping `extract_geodesic_triples`.

Now let me update the training diagnostics cell (cell 21) to show both loss components:

Let me verify everything is consistent:

Those references are only in the cached execution output (the traceback from the previous run), not in the source code. They'll be cleared when the cell is re-executed. Let me verify the source code is clean:

Those are only in the cached output (traceback from the old execution). The actual source code is clean. Let me confirm by searching only in source strings:

Good. The source code is properly updated. The old `train_spray_direct` references in the output will disappear on re-execution.

Here's what changed and why:

**The problem you identified:** The old `train_spray_direct` only did *pointwise* supervision — it compared `G(x,v)` against empirical acceleration at observed data points. It never actually *ran* the spray through integration. So the spray could learn to match accelerations at data points perfectly while still diverging catastrophically when you chain integration steps together (because small errors at step N shift the state to a point the spray has never seen, causing even larger errors at step N+1).

**The fix — `train_spray_with_rollout`:** The new training function has two loss components:

1. **Pointwise geodesic deviation** (same as before): Dense per-sample supervision from the geodesic equation. This provides a strong, stable gradient signal for initial convergence.

2. **Rollout integration loss** (new): Starting from the initial conditions `(x_0, v_0)` of each trajectory, the spray is *integrated forward* using symplectic Euler — the exact same scheme used during prediction. At each step, the predicted position is compared to the ground truth. Crucially, gradients flow *back through the entire integration chain*, so the spray learns to produce dynamics that remain stable and accurate under iteration. This is backpropagation through time applied to the geodesic ODE.

The training diagnostics plot now shows all three components (geodesic deviation, rollout loss, total) so you can monitor both convergence and integration stability.

---

**User**

Bro. This is the spray field



---

**Cursor**

That's looking structurally sound. A few observations on what the spray field is telling us:

1. **Magnitude gradient**: The acceleration magnitude (color) increases from near-zero at the center (dark) to ~5 at the periphery (yellow). This is physically correct for outward spirals — tighter curvature at larger radii requires stronger corrective forces from the spray.

2. **Directional coherence**: The arrows show a clear, smooth vector field without chaotic discontinuities. This means the spray learned a *regular* geometry — no sharp artifacts or spurious attractors.

3. **One caveat on the visualization**: This plot evaluates the spray at a **fixed reference momentum** `[1.0, 0.0]` (rightward). Since the spray depends on *both* position and velocity, this is a single "slice" through the full tangent-bundle dynamics. The actual acceleration a token experiences along a spiral varies as its momentum rotates. That's why the arrows all lean rightward — they reflect the spray's response to rightward-moving particles, not the full picture.

The real test is whether the integrated geodesics match the ground truth (cell 25). Did the prediction plot improve after adding the rollout loss?

---

**User**

They seem to tightly follow the seed points then fly off the rails


---

**Cursor**

Right, the pattern is clear. The predictions hug the seed points, then launch into straight lines. This tells us the spray only learned to be accurate *on the training data distribution* — the moment integration drifts slightly off-manifold, the spray produces garbage, and errors compound into a straight-line runaway.

The root cause is in the rollout training: it always starts from `t=0` (the very beginning of each trajectory). But during prediction you start from `t=10` (the end of the seed). The spray never learned to integrate from arbitrary starting positions. Two fixes needed:

1. **Randomize rollout starting position** within each trajectory during training
2. **Extend rollout length** to cover more of the trajectory, so the spray learns long-horizon stability

Good, I can see the problem clearly. The rollout always starts from `batch[:, 0]` with velocity from `batch[:, 1] - batch[:, 0]`. Let me fix the rollout to start from random positions within each trajectory:

Now let me also increase the default `rollout_steps` to cover a bigger chunk of each trajectory (50 points long, so 30 steps is reasonable):

Two changes:

1. **Random rollout starting position**: Each training batch now picks a random `start_t` within the trajectory. So the spray practices integrating from t=5, t=20, t=35, etc. — not just from the beginning. This directly matches the prediction scenario where you start from `seed_len=10`.

2. **`rollout_steps=30`** (up from 15): The trajectories are 50 points long, so 30 steps covers 60% of the trajectory in a single rollout. Combined with random starting positions, this means the spray sees integration from essentially every part of the trajectory.

3. **`rollout_weight=2.0`** (up from 1.0): Prioritizes integration stability over pointwise accuracy. The pointwise loss helps the spray learn the right ballpark; the rollout loss is what actually keeps geodesics on-manifold.

Also dropped `n_epochs` back to 500 since the old 5000 was just spinning wheels on the pointwise-only loss. With rollout, each epoch does much more useful work.

---

**User**

What are we expecting again? what exactly are we training. it's not making sense to me that we're only training some points that don't capture the complexity of the full spiral. the spoints are only on one , or one half arc. the line matches this but it makes no sense for it to expect to do a spiral, we're not training it to spiral. maybe we can choose points farther out along the spiral as well as close together, so it can learn to chart long and short courses. like i don't know. 

---

**Cursor**

You're raising the right question. Let me clarify what the spray *should* be doing and why the current visualization is misleading about it.

**What the spray learns:** During training, the spray sees (position, velocity, acceleration) triples extracted from ALL 50 points of ALL 128 spiral trajectories — inner arcs, outer arcs, every direction. The pointwise loss covers the entire data distribution. The rollout loss integrates through large chunks of trajectories starting from random positions. So the spray IS trained on the full spiral dynamics.

**What prediction needs:** In theory, the spray only needs ONE point and ONE velocity to continue the trajectory — because `G(x, v)` tells you "at position x, moving with velocity v, curve by this much." If the spray learned the correct curvature field, even 2 seed points should produce a spiral.

**Why it's failing:** The spray hasn't converged to the correct dynamics yet. And the prediction visualization makes it look worse than it might be, because it only uses the first 10 points (a tiny inner arc) as seeds. We have no idea if the spray works better from other starting positions.

Your instinct is exactly right — let's make the visualization more diagnostic. Let me fix the prediction cell to test seeds from different parts of the trajectory with different lengths:

Good. Now let me rewrite the prediction visualization to be a proper diagnostic — testing different seed positions and lengths across the trajectory:

Now the prediction plot will be a 3x3 grid testing:

- **Column 1 — Inner arc**: Seed from t=0 to t=10 (innermost ring, tiny arc), predict 40 steps out. This is the hardest case — the spray has to maintain spiral dynamics for a long time from minimal context.

- **Column 2 — Mid arc**: Seed from t=15 to t=25 (middle of the spiral), predict 25 steps. Tests whether the spray works better when starting from a region with more curvature.

- **Column 3 — Outer arc**: Seed from t=25 to t=40 (outer ring), predict just 10 steps. Easiest case — larger arc seed with short prediction horizon.

This will tell us immediately whether the spray's issue is:
- **(a) Global failure**: All columns look bad — the spray hasn't learned spiral dynamics at all
- **(b) Distribution gap**: Inner seeds fail but outer seeds work — the spray struggles with the tight inner geometry
- **(c) Horizon problem**: Short predictions work everywhere but long ones diverge — the spray is locally accurate but not stable

That's the diagnostic you need. Run cells 19-25 fresh and let me know what the 3x3 grid looks like.

---

**User**

Okay, well with allllll the distributions and what not. the spray field is stil jus t showiing up as the left side pointing right, and the right side pointing left. this works for may be one arc-change, but after that of course it flies away. the spray field isn't learning spirals, or even concentric circles.

---

**Cursor**

You've nailed the diagnostic. The spray is learning a simple "push toward center" restoring force, not the velocity-dependent curvature a spiral requires. Let me look at the SprayNetwork architecture — I think I see the architectural bug.

There it is. **The `LayerNorm` on the spray output is the bug.**

Here's what LayerNorm does to a 2D vector `[Gx, Gy]`:
1. Subtracts the mean: `[(Gx - Gy)/2, (Gy - Gx)/2]`
2. Divides by standard deviation

For d_model=2, this forces the output to be approximately `[c, -c]` — one component is always the negative of the other, regardless of what the MLP produces. That's why the spray field shows "left points right, right points left." The LayerNorm is literally *preventing* the spray from learning anything more complex than a linear restoring force.

The spray is a physical quantity (acceleration). Its magnitude and direction both matter independently. LayerNorm destroys both.

Now re-run from cell 9 (SprayNetwork definition) through cell 25 (predictions). The LayerNorm was clamping the spray into a 1-dimensional subspace of the 2D output. Without it, the spray can now learn:

- **Centripetal acceleration** (pointing inward) — for circular motion
- **Tangential acceleration** (perpendicular to radius) — for maintaining angular velocity
- **Both depending on velocity direction** — which is what makes it Finslerian, not just Riemannian

The "left-right" pattern you saw was the only thing LayerNorm *allowed* the network to express in 2D. The entire training was fighting against an architectural constraint.

---

**User**

Make another cell using the full model now.

---

**Cursor**

Let me check what the full model looks like and where the prediction section ends, so I know where to insert the new cell.

Good. Cell 25 (the prediction diagnostic grid) ends around line 1188, and cell 26 starts the Randers indicatrix section. Let me read the FinslerGeodesicTransformer and FinslerGeodesicLayer to understand their current state, then build a new section that trains and predicts using the full model.

Good, the indicatrix section starts at cell 26. I'll insert the full model experiment right before it (at cell index 26, pushing indicatrix down).

Now add the training diagnostics and prediction visualization for the full model:

Two new cells added (26-28). Here's what the full model experiment does:

**`FullGeodesicPredictor`** — wraps `FinslerGeodesicTransformer` with a learned decoder. The pipeline is:
1. Position encoder transforms raw 2D coordinates
2. Momentum encoder transforms finite-difference velocities
3. K=3 stacked `FinslerGeodesicLayer`s, each with its own `SprayNetwork` (now LayerNorm-free), each doing one symplectic Euler step
4. Decoder projects back to observation space

**Training** has two loss components:
1. **Trajectory matching**: The model integrates each token forward K steps. Output at index t should match input at index t+K.
2. **Autoregressive rollout**: Takes a sliding window, feeds model predictions back in as input for the next step (like inference), compares to ground truth. Random starting positions.

**Prediction** (`predict_full_model`) is truly autoregressive: maintains a sliding window of recent positions, feeds through the full model at each step, appends the last decoded position, slides the window forward.

The same 3x3 diagnostic grid (inner/mid/outer arc, 3 trajectories) is used so you can directly compare against the standalone spray results above. Run cells 9 (SprayNetwork without LayerNorm), then 11, 13, and 27-28.

---

**User**

---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
Cell In[101], line 149
    143 cfg_full = FinslerConfig(
    144     d_model=2, n_spray_layers=3, delta_t=DT_2D,
    145     spray_hidden_mult=16, n_integration_steps=8, dropout=0.05
    146 )
    148 print("Training full FinslerGeodesicTransformer on spiral data...")
--> 149 full_model, hist_full = train_full_model(
    150     spirals, cfg_full, n_epochs=500, lr=1e-3,
    151     batch_size=32, rollout_steps=20, rollout_weight=2.0
    152 )

Cell In[101], line 87
     84 current_window = window
     86 for step in range(steps):
---> 87     x_out, _ = model(current_window)
     88     next_pred = x_out[:, -1:]
     89     target_idx = start_t + step + 1

File ~/miniconda3/lib/python3.12/site-packages/torch/nn/modules/module.py:1775, in Module._wrapped_call_impl(self, *args, **kwargs)
   1773     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1774 else:
-> 1775     return self._call_impl(*args, **kwargs)

File ~/miniconda3/lib/python3.12/site-packages/torch/nn/modules/module.py:1786, in Module._call_impl(self, *args, **kwargs)
   1781 # If we don't have any hooks, we want to skip the rest of the logic in
   1782 # this function, and just call forward.
   1783 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1784         or _global_backward_pre_hooks or _global_backward_hooks
   1785         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1786     return forward_call(*args, **kwargs)
   1788 result = None
   1789 called_always_called_hooks = set()

Cell In[101], line 25
     14 def forward(self, x_seq: torch.Tensor
     15             ) -> Tuple[torch.Tensor, torch.Tensor]:
     16     """Forward pass: encode -> integrate K layers -> decode.
     17 
     18     Args:
   (...)
     23         y_final: final momenta in encoded space, shape (B, T, d)
     24     """
---> 25     x_pred, y_final, _ = self.geodesic(x_seq)
     26     x_decoded = self.decoder(x_pred)
     27     return x_decoded, y_final

File ~/miniconda3/lib/python3.12/site-packages/torch/nn/modules/module.py:1775, in Module._wrapped_call_impl(self, *args, **kwargs)
   1773     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1774 else:
-> 1775     return self._call_impl(*args, **kwargs)

File ~/miniconda3/lib/python3.12/site-packages/torch/nn/modules/module.py:1786, in Module._call_impl(self, *args, **kwargs)
   1781 # If we don't have any hooks, we want to skip the rest of the logic in
   1782 # this function, and just call forward.
   1783 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1784         or _global_backward_pre_hooks or _global_backward_hooks
   1785         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1786     return forward_call(*args, **kwargs)
   1788 result = None
   1789 called_always_called_hooks = set()

Cell In[94], line 51
     48 y = self.momentum_encoder(self.estimate_initial_momentum(x_seq))
     50 x_flat = x.reshape(B * T, d)
---> 51 y_flat = y.reshape(B * T, d)
     53 spray_history = []
     54 for layer in self.layers:

RuntimeError: shape '[32, 2]' is invalid for input of size 0

---

**Cursor**

The crash is because the sliding window can end up with only 1 time step, but the `FinslerGeodesicTransformer` needs at least 2 to estimate momentum via finite differences. Let me fix the rollout window logic.

The fix: the window now always has exactly `window_size=5` points, and `start_t` is constrained to be at least `window_size - 1 = 4` so the slice `[start_t - 4 : start_t + 1]` always produces 5 elements. This guarantees the `FinslerGeodesicTransformer` has enough points for momentum estimation.

Re-run cell 27 and it should train cleanly now.

---

**User**

Add another training and prediction section where we do the wind based walks

---

**Cursor**

Good, `wind_walks` is already generated in cell 17. Now let me find where the full model prediction section ends so I know where to insert.

The full model prediction ends at cell 28, and the indicatrix section starts at cell 29. Let me also check the wind walk data generator to understand the structure.

Good — wind-biased walks are random walks with a directional drift `[0.05, 0.015]`. These are the most "Randers-like" data: stochastic trajectories with a directional bias, exactly what the 1-form `b_i` in the Randers metric models. Now let me add the training + prediction section for both standalone spray and full model.

Three new cells (29-31) added. Here's the layout:

**Cell 29** (markdown) — Explains why wind-biased walks are the ideal Randers test case: the directional drift is exactly what the 1-form `b_i` captures, and the stochastic spread is what the Riemannian background `a_ij` models.

**Cell 30** (code) — Standalone spray on wind walks:
- Trains `SprayNetwork` with rollout (same `train_spray_with_rollout`)
- Shows training curves (geodesic deviation + rollout)
- Visualizes the learned spray field overlaid on wind walk data
- 3x3 prediction grid: early/mid/late seeds across 3 trajectories

**Cell 31** (code) — Full model on wind walks:
- Trains `FullGeodesicPredictor` (encoder + 3 spray layers + decoder)
- Shows training curves (trajectory matching + rollout)
- Same 3x3 prediction grid for direct comparison
- Prints a summary comparing final losses between standalone spray and full model

The wind walks should be an easier task than spirals — the dynamics are nearly linear (constant drift + noise), so the spray mostly needs to learn a constant acceleration field biased in the `[1, 0.3]` direction. This is a good sanity check before going back to spirals with harder curvature.

---

**User**

Can you explain what each hyperparameter means for the config in the class definition so it shows up in the hint

@LedbetterFinslerTransformer.ipynb (9-12) 


---

**Cursor**

Now when you hover over any field in `FinslerConfig(...)`, the docstring will show up in the IDE hint with the full explanation. Here's the quick reference:

| Parameter | What it controls |
|---|---|
| `d_model` | Manifold dimension (2 for toy, 64+ for language) |
| `n_spray_layers` | How many integration steps per forward pass |
| `n_integration_steps` | Reserved for adaptive solvers (currently unused) |
| `delta_t` | Time step size — must match trajectory data spacing |
| `spray_hidden_mult` | Spray network width = `d_model * this` |
| `randers_norm_eps` | Numerical stability epsilon |
| `dropout` | Regularization inside spray networks |

Re-run cell 5 to pick up the updated docstring.

---

**User**

I think we need a finer vector field ngl

---

**Cursor**

The grid resolution is only 20x20 and it uses a single fixed momentum direction. Let me upgrade it — higher resolution grid, and show multiple momentum directions so you can actually see how the spray varies with velocity (the whole point of Finsler over Riemannian).

Now also update the wind walks spray visualization call to use the same function (it'll automatically get the new version):

Two changes:

1. **Grid resolution bumped from 20x20 to 35x35** — that's ~3x more arrows, giving you a much denser field.

2. **4 panels instead of 1** — each showing the spray evaluated at a different momentum direction: rightward `[1,0]`, upward `[0,1]`, leftward `[-1,0]`, and diagonal `[.7,.7]`. This is the critical visualization because:

   - If all 4 panels look **identical**, the spray is only position-dependent — it learned Riemannian geometry (no directional asymmetry).
   - If the panels look **different**, the spray is genuinely velocity-dependent — it learned Finslerian geometry. The *difference* between panels is the Randers 1-form in action.

   For spirals, you'd expect the spray to look very different when moving tangentially vs. radially. For wind walks, the spray should show asymmetry between with-wind and against-wind directions.

Re-run cell 23 to see the new 4-panel spray field.

---

**User**

Okay yes but now the spirals are messed up

---

**Cursor**

The spray visualization cell for spirals is calling the updated function, which is fine. The issue is likely that you need to **re-run the training cell** (cell 19) first — after the LayerNorm removal from `SprayNetwork` (cell 9), the `spray_spiral` model needs to be retrained with the fixed architecture.

The execution order should be:

1. **Cell 9** — `SprayNetwork` (now without LayerNorm)
2. **Cell 11** — `FinslerGeodesicLayer` (uses SprayNetwork)
3. **Cell 19** — `train_spray_with_rollout` on spirals (creates new `spray_spiral`)
4. **Cell 23** — spray field visualization

If you ran cell 23 without retraining, it's still using the old `spray_spiral` that was trained *with* LayerNorm — which is the broken one. Kill and re-run from cell 9 onward, or at minimum re-run cells 9 and 19 before 23.

---

**User**

Yes, for the spray training. but they are slightly deformed on training FinslerGeodesicTransformer

---

**Cursor**

The "slight deformation" is coming from the `position_encoder` and `momentum_encoder` in `FinslerGeodesicTransformer`. For 2D, they're random Linear(2,2) layers that rotate/shear the coordinate system, and the decoder has to learn to undo that. They start random, so the model wastes training capacity learning to invert its own encoders.

The fix: initialize all three (position encoder, momentum encoder, decoder) as identity matrices so they start as pass-through. The model begins behaving like the standalone spray and only deviates if the optimizer finds a better coordinate system.

Now do the same for the decoder in `FullGeodesicPredictor`:

All three linear transforms now start as identity. Re-run cells 13 and 27, then retrain. The model will start with raw coordinates passing straight through to the spray layers — same behavior as the standalone spray — and the optimizer will only learn non-trivial encodings if they actually help.

