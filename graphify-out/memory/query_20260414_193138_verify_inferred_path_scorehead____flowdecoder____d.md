---
type: "query"
date: "2026-04-14T19:31:38.198070+00:00"
question: "Verify inferred path ScoreHead -> FlowDecoder -> DynaMaskVIO -> BasicEncoder"
contributor: "graphify"
source_nodes: ["ScoreHead", "FlowDecoder", "DynaMaskVIO", "BasicEncoder"]
---

# Q: Verify inferred path ScoreHead -> FlowDecoder -> DynaMaskVIO -> BasicEncoder

## Answer

Direct source verification shows the path exists but the first two inferred edge directions are reversed. flow_decoder.py imports ScoreHead at line 23, instantiates it at lines 259-263, and calls it at line 310, so FlowDecoder points to ScoreHead. dynamask.py instantiates FlowDecoder at lines 59-65 and calls it at lines 100-102, so DynaMaskVIO points to FlowDecoder. dynamask.py also instantiates BasicEncoder at lines 43-54 and uses it at lines 93-95, so DynaMaskVIO points to BasicEncoder. The bridge is real structurally as ScoreHead <- FlowDecoder <- DynaMaskVIO -> BasicEncoder, but not as a strictly directed ScoreHead -> FlowDecoder -> DynaMaskVIO chain.

## Source Nodes

- ScoreHead
- FlowDecoder
- DynaMaskVIO
- BasicEncoder