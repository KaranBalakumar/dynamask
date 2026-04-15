---
type: "query"
date: "2026-04-14T19:30:40.615125+00:00"
question: "Why does ScoreHead connect Flow Decoder Pipeline to RAFT Feature Encoding, Motion Score Head?"
contributor: "graphify"
source_nodes: ["ScoreHead", "FlowDecoder", "DynaMaskVIO", "BasicEncoder"]
---

# Q: Why does ScoreHead connect Flow Decoder Pipeline to RAFT Feature Encoding, Motion Score Head?

## Answer

ScoreHead acts as a bridge because flow_decoder.py imports score_head.py with EXTRACTED evidence, and ScoreHead has many INFERRED uses edges to FlowDecoder pipeline components like CorrBlock, SepConvGRU, and FlowHead. The shortest RAFT chain in the graph is ScoreHead to FlowDecoder to DynaMaskVIO to BasicEncoder, which links decoder logic to RAFT feature encoding. ScoreHead also connects directly to score_head.py and its methods with EXTRACTED edges, anchoring the motion score head side. The cross-community bridge exists, but several key hops are INFERRED and should be verified in source code.

## Source Nodes

- ScoreHead
- FlowDecoder
- DynaMaskVIO
- BasicEncoder