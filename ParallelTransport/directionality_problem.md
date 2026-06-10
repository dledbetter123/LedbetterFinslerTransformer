# The Directionality Problem in Sequential Finsler Transport

## The Concern

Parallel transport along a curve is inherently directional. In our implementation:

```
H_t = prod_{s=0}^{t-1} exp(-N_s * dt)
h_t = H_t @ x_t
```

Token t's representation depends on tokens 0 through t-1. Token 0 has H_0 = I — no context at all. Token T-1 has the richest context. Information flows strictly left-to-right.

This is the same limitation that killed RNNs for long sequences: by the time you reach token 50, the information from token 1 has been compressed through 49 matrix multiplications, suffering from vanishing/exploding gradients and information bottleneck. Attention solved this with O(N^2) all-to-all communication — every token directly sees every other token.

If we simply replace attention with sequential transport, we've built a geometrically flavored RNN and inherited all its problems.

## But Wait — Is It Really the Same Problem?

No. There are important structural differences between our transport and an RNN. Whether they're sufficient is the question.

### Difference 1: The spray is a global geometric object, not a generic recurrence

In an RNN, the transition matrix W_hh is the same at every step. It's a fixed set of weights that compress all context into a fixed-size vector. The representational bottleneck is fundamental — all information about the past must fit through a single matrix.

Our spray G(x, y) is a function on the entire tangent bundle TM. It sees BOTH position AND direction, and produces different outputs at different (x, y) pairs. During training, the spray learns from ALL trajectories simultaneously. The spray at point x with velocity y was trained on every sequence that ever passed through a nearby (x, y) state.

This means: when the model processes token t, the spray at (x_t, v_t) already "knows" statistical properties of what typically precedes and follows tokens at this embedding position moving in this direction. The curvature was shaped by that data distribution. The spray is a learned geometric field that encodes statistical context implicitly.

An RNN has no equivalent of this. Its transition weights don't depend on the current state's position in embedding space. Our spray is like having a different transition matrix at every point — one that was shaped by the data distribution at that point.

**But this only captures statistical regularities.** The spray can't capture specific, token-level interactions within a single novel sequence (like resolving a pronoun to a particular antecedent mentioned 20 tokens ago). For that, you need something that actually examines the specific tokens in the current input.

### Difference 2: The connection is a MATRIX, not a bottleneck

An RNN hidden state is a d-dimensional vector. All context is compressed into d numbers. Our holonomy is a d-by-d MATRIX — d^2 numbers. For d=64, that's 4096 numbers of context vs 64 in an RNN.

The matrix holonomy can encode:
- Which directions in embedding space have been rotated (and by how much)
- Which dimensions have been scaled (amplified or suppressed)
- Which pairs of dimensions have been coupled (shearing)

This is a much richer representation of path history than a vector. Whether d^2 is enough context for real language is an open question, but it's a substantial upgrade over d.

### Difference 3: For causal language modeling, unidirectionality is correct

GPT-style autoregressive models use a causal mask: token t only attends to tokens 0 through t-1. This is exactly what our transport does. The left-to-right directionality isn't a bug for causal LM — it's a feature. You SHOULD only condition on past context when predicting the next token.

The problem only arises for:
- Bidirectional understanding tasks (BERT, T5 encoder)
- Sequence classification (where the label depends on the full sequence)
- Tasks requiring long-range backward dependencies

For pure next-token prediction, sequential forward transport is architecturally correct.

## Remedies Within the Finsler Framework

If we DO need bidirectional context (for understanding tasks, or because forward-only transport is too limiting even for causal LM), here are options ranked by how naturally they fit the geometry:

### 1. Bidirectional Transport (simplest, most direct)

Run transport in both directions along the sequence:

```
Forward:  H_fwd_t = prod_{s=0}^{t-1}   exp(-N(x_s, v_s_fwd) * dt)
Backward: H_bwd_t = prod_{s=T-1}^{t+1} exp(-N(x_s, v_s_bwd) * dt)

h_t = combine(H_fwd_t @ x_t, H_bwd_t @ x_t)
```

where v_s_bwd = (x_{s-1} - x_s) / dt (reversed velocity).

This is geometrically valid: transport along a curve in either direction is well-defined. Forward and backward transports produce DIFFERENT holonomies (that's the point of Finsler geometry — F(x, y) != F(x, -y) in general). The Randers 1-form b_i creates asymmetry: forward transport feels the wind as tailwind, backward transport feels it as headwind. This asymmetry naturally captures the directional bias of language while still allowing bidirectional information flow.

The `combine` function could be:
- Concatenation: h_t = [H_fwd @ x_t ; H_bwd @ x_t], doubling the representation dimension
- Sum: h_t = H_fwd @ x_t + H_bwd @ x_t
- Learned combination: h_t = W_f @ (H_fwd @ x_t) + W_b @ (H_bwd @ x_t)

Cost: 2x the computation of unidirectional transport. Still O(T) per sequence.

**This is the BiLSTM analogue, but with geometry.** The key advantage over BiLSTM: the forward and backward sprays share the same metric. The geometry is one consistent manifold, just traversed in two directions. The resulting holonomies are geometrically related (they're inverses for flat space, and the DIFFERENCE between forward and backward transport encodes curvature). A BiLSTM has two independent parameter sets with no geometric relationship.

### 2. Geodesic Loops (holonomy as all-to-all communication)

This is the most geometrically principled approach and the most novel.

In differential geometry, holonomy is defined for CLOSED LOOPS, not open paths. The holonomy of a loop measures the total curvature enclosed by the loop. Different loops produce different holonomies.

For a sequence [x_0, ..., x_T], define loops that connect each token to every other:

```
Loop(i, j): transport from x_i to x_j (forward along the sequence)
            then transport from x_j back to x_i (backward along the sequence)

Holonomy(i, j) = H_forward(i->j) @ H_backward(j->i)
```

If the manifold is flat, Holonomy(i, j) = I for all pairs. If the manifold has curvature, Holonomy(i, j) != I, and the deviation from identity encodes the curvature enclosed between tokens i and j.

This gives an N x N "holonomy matrix" — one d-by-d holonomy for each pair of tokens. This is directly analogous to the N x N attention matrix, but instead of scalar attention weights, we have d-by-d geometric transformations.

The representation of token t incorporates all pairwise holonomies:

```
h_t = sum_j f(Holonomy(t, j)) @ x_j
```

where f extracts information from the holonomy matrix (e.g., its trace, its eigenvalues, or a learned projection).

**This IS attention, derived from geometry.** The "attention weight" between tokens i and j is the holonomy of the loop connecting them. Nearby tokens with low curvature between them have holonomy ~ I (low attention). Distant tokens with high curvature between them have holonomy far from I (high attention — the geometry says these tokens are geometrically "interesting" relative to each other).

Cost: O(N^2) holonomy computations (same as attention). Each holonomy computation requires O(N) transport steps. Total: O(N^3) — worse than attention. BUT: the forward and backward transport can be precomputed in O(N), and the pairwise holonomies can be derived from the precomputed transports in O(N^2). So total cost is O(N^2), same as attention.

Specifically:
```
Precompute H_fwd[t] for all t  (one forward pass, O(N))
Precompute H_bwd[t] for all t  (one backward pass, O(N))

Holonomy(i, j) = H_fwd[j] @ inv(H_fwd[i]) @ H_bwd[i] @ inv(H_bwd[j])
```

This is O(1) per pair (just matrix multiplies of precomputed matrices), so O(N^2) total. Same asymptotic cost as attention, but with d-by-d matrices instead of scalar weights.

### 3. Multi-Scale Transport (logarithmic depth)

Instead of token-by-token transport, build a hierarchy:

```
Level 0: transport between adjacent tokens (t, t+1)
Level 1: transport between every 2nd token (t, t+2) using level-0 holonomies
Level 2: transport between every 4th token (t, t+4)
...
Level log(T): transport between first and last token
```

At each level, the transport uses the connection evaluated at the coarser scale. The holonomy at each level captures context at that scale — local context at level 0, paragraph-level context at higher levels.

This is analogous to hierarchical attention (Transformer-XL, Longformer) but built from geometric transport. The total depth is O(log T) instead of O(T), which addresses the vanishing gradient problem.

Cost: O(T log T) — better than attention's O(T^2) for long sequences.

### 4. Curvature-Based Local Context (O(1) per token)

The curvature tensor R^i_jkl at each point encodes how transport around infinitesimal loops deforms vectors. It's a LOCAL property — it doesn't require integrating along the sequence.

For each token, evaluate the curvature tensor at (x_t, y_t). This tells you the local geometric structure: how much the embedding space is curved at this point, in which directions, and how the curvature depends on the velocity direction.

The curvature provides a kind of "local attention" — it captures the geometric relationships in the neighborhood of each token without needing to integrate along the full sequence.

For the Randers metric, the curvature involves derivatives of the spray:
```
R^i_k = 2 * dG^i/dx^k - y^j * d^2G^i/(dx^j dy^k) + 2G^j * d^2G^i/(dy^j dy^k) - dG^i/dy^j * dG^j/dy^k
```

This is computable via autograd but expensive (second derivatives of the spray, which involves fourth derivatives of the metric). For d=2, feasible. For d=64, needs approximation.

The curvature "flag" R^i_k captures: "how much does the transported vector rotate if you take an infinitesimal detour in the k-direction?" This is exactly the information that attention captures — the relevance of nearby tokens to the current one.

## Recommendation

For the notebook's 2D toy case: **bidirectional transport** is sufficient and easy to implement. Run forward and backward transport, combine the holonomies.

For a research paper: **geodesic loops** (Option 2) are the most interesting contribution because they derive attention-like all-to-all communication from the geometry itself. The holonomy of loops connecting token pairs is a genuinely novel attention mechanism with geometric meaning. And the cost is the same O(N^2) as standard attention.

For scaling to long sequences: **multi-scale transport** (Option 3) gives O(T log T) complexity and addresses both the directionality problem and the long-range dependency problem simultaneously.

For the cleanest theoretical story: combine **geodesic deviation training** (to learn the metric) with **geodesic loop holonomy** (to derive attention from the metric). This gives a complete narrative: "The metric defines the geometry. The curvature enclosed by token-pair loops defines attention. Training makes data into geodesics. Attention emerges from curvature."

## The Key Insight

The RNN problem isn't really about directionality — it's about BOTTLENECK. An RNN compresses all past context into a fixed-size vector. Attention solves this by giving every token direct access to every other token.

Our transport has the same bottleneck IF we only use the final holonomy matrix. But the geodesic loop construction sidesteps this: each token pair has its own holonomy matrix. There's no bottleneck because the context isn't compressed into a single state — it's distributed across N^2 pairwise holonomies, just like attention distributes context across N^2 attention weights.

The geometry doesn't force us into sequential-only processing. It provides a richer structure (the holonomy GROUP, not just the holonomy along one path) that we can exploit for all-to-all communication. We just haven't been using it.
