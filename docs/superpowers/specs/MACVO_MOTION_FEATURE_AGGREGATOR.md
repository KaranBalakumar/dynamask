# MAC-VO Motion Feature Aggregator (GMA `Aggregate`) - In-Depth Guide

This note explains what the "Motion Feature Aggregator" is in MAC-VO, where it sits in the pipeline, and exactly what data flows through it.

---

## 1. What it is (short answer)

In this codebase, the motion feature aggregator is the class:

- `Aggregate` in `dynamask_vio/references/MAC-VO/Module/Network/FlowFormer/core/gma.py`

It is used by:

- `GMAUpdateBlock` in `.../FlowFormer/core/gru.py`
- the FlowFormerCov decoder path (`MemoryCovDecoder`) in `.../FlowFormerCov/covhead.py`

Its job is to turn **local motion features** into **globally informed motion features** using an attention map.

---

## 2. Where it sits in the full MAC-VO path

### High-level runtime path

1. `FlowFormerCovFrontend` builds and calls `FlowFormerCov`.
2. `FlowFormerCov` runs:
   - context encoder
   - memory encoder
   - covariance-aware memory decoder (`MemoryCovDecoder`)
3. Inside each decoder iteration, motion features are encoded, globally aggregated, then fed to GRU update blocks.

### Key files

- Frontend:
  - `.../Module/Frontend/Frontend.py`
- Model wrapper:
  - `.../Module/Network/FlowFormerCov/flownet.py`
  - `.../Module/Network/FlowFormerCov/__init__.py`
- Decoder core:
  - `.../Module/Network/FlowFormer/core/decoder.py`
  - `.../Module/Network/FlowFormerCov/covhead.py`
  - `.../Module/Network/FlowFormer/core/gru.py`
  - `.../Module/Network/FlowFormer/core/gma.py`

---

## 3. End-to-end flow diagram

```text
Stereo inputs (t1, t2)
        |
        v
FlowFormerCovFrontend.estimate_pair(...)
        |
        v
FlowFormerCov.inference(imageA, imageB)
        |
        +--> Context Encoder (TwinsSVT truncation) ------------------+
        |                                                             |
        +--> Memory Encoder (cost memory + cost maps)                 |
        |                                                             |
        v                                                             |
MemoryCovDecoder (iterative, depth = decoder_depth)                   |
        |                                                             |
        |  flow_inp = relu(split(context_proj)[1])                    |
        |  attention = Attention(flow_inp)   [computed once]          |
        |                                                             |
        |  for each iteration:                                        |
        |    1) cost_forward = encode_flow_token(cost_maps, coords1)  |
        |    2) cost_global  = cross-attn(query, cost_memory)         |
        |    3) corr = concat(cost_global, cost_forward)              |
        |    4) motion_feat = BasicMotionEncoder(flow, corr)          |
        |    5) motion_feat_global = Aggregate(attention, motion_feat)|
        |    6) inp_cat = concat(flow_inp, motion_feat, motion_feat_global)
        |    7) Flow branch: GRU -> delta_flow -> flow upsample       |
        |    8) Cov  branch: GRU -> delta_cov  -> cov upsample        |
        |
        v
Final flow/cov -> frontend converts to depth + match outputs
```

---

## 4. Zoom-in: what the aggregator computes

The aggregator consumes:

- `attn`: global attention matrix over spatial positions
- `fmap`: motion features at each spatial position

and computes:

```text
V = Conv1x1(fmap)
G = attn @ V               # global weighted mixing over all positions
out = fmap + gamma * G     # residual + learnable gate
```

### Internal flow (`gma.py`)

```text
fmap [B,C,H,W]
  |
  +--> to_v (1x1 conv) -> v [B, heads*dim_head, H, W]
  +--> reshape v -> [B, heads, HW, d]

attn [B, heads, HW, HW]
  |
  +--> matmul(attn, v) -> [B, heads, HW, d]
  +--> reshape/project -> [B, C, H, W]
  +--> residual fusion: fmap + gamma * out
```

Important detail: `gamma` is initialized to `0`.  
So early in training, behavior is close to identity (`out ~= fmap`), and the model gradually learns how much global aggregation to apply.

---

## 5. Where `attn` comes from

`Attention` in `gma.py` generates `attn` from `flow_inp`:

1. `to_qk` (1x1 conv) creates `Q` and `K`
2. similarity `QK^T` over all `(H*W)` positions
3. softmax over keys to get attention weights

This gives each pixel a weighted view of all other pixels.

---

## 6. Decoder iteration dataflow with shapes

Typical shapes in this config:

- `query_latent_dim = 64`
- `cost_heads_num = 1`
- local correlation patch channels = `9x9x1 = 81`

So:

1. `cost_forward`: `[B, 81, H1, W1]`
2. `cost_global`: `[B, 64, H1, W1]`
3. `corr = concat(...)`: `[B, 145, H1, W1]`
4. `motion_feat = BasicMotionEncoder(flow, corr)`: `[B, 128, H1, W1]`
5. `motion_feat_global = Aggregate(attn, motion_feat)`: `[B, 128, H1, W1]`
6. `flow_inp`: `[B, 128, H1, W1]`
7. `inp_cat`: `[B, 384, H1, W1]`

`inp_cat` drives both:

- flow update GRU/head
- covariance update GRU/head (in FlowFormerCov)

So the same globally aggregated motion context influences both prediction and uncertainty updates.

---

## 7. Why this module exists

Without aggregation, updates rely mostly on local evidence at each pixel.  
With aggregation, each pixel's motion update can borrow information from globally consistent regions (large structures, repeated texture handling, broad motion patterns).

In short:

- **Local encoder** gives fine detail.
- **Aggregator** injects scene-level consistency.
- **GRU** fuses both over iterations.

---

## 8. Difference from plain RAFT-style local update

RAFT-style update: local corr + recurrent refinement.  
This MAC-VO FlowFormer path adds:

1. latent memory/cross-attention cost retrieval
2. explicit global motion aggregation (`Aggregate`)
3. covariance branch sharing the same aggregated context

So it is a local+global recurrent decoder, not only local recurrent refinement.

---

## 9. Minimal pseudocode (exact logic pattern)

```python
# once before loop
flow_net, flow_inp = split(context_proj, [128, 128])
attention = Attention(flow_inp)

for _ in range(depth):
    flow = flow_coords1 - flow_coords0
    cost_forward = encode_flow_token(cost_maps, flow_coords1)
    cost_global, key, value = decoder_layer(query(cost_forward), key, value, cost_memory, flow_coords1)
    corr = cat(cost_global, cost_forward)

    motion_feat = BasicMotionEncoder(flow, corr)
    motion_feat_global = Aggregate(attention, motion_feat)
    inp_cat = cat(flow_inp, motion_feat, motion_feat_global)

    flow_net = flow_gru(flow_net, inp_cat)
    delta_flow = flow_head(flow_net)

    cov_net = cov_gru(cov_net, inp_cat)
    delta_cov = cov_head(cov_net)
```

---

## 10. Graphify mapping (helpful node names)

If you inspect `graphify-out/graph.json`, relevant nodes are:

- `gma_aggregate`
- `gru_gmaupdateblock`
- `covhead_memorycovdecoder`
- `flownet_flowformercov`
- `frontend_flowformercovfrontend`

These appear in the FlowFormer/FlowFormerCov cluster (Community 9 in the current graph report extraction for these modules).

