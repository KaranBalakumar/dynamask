---
type: "query"
date: "2026-04-14T19:35:59.027506+00:00"
question: "Audit inferred ScoreHead relationships"
contributor: "graphify"
source_nodes: ["ScoreHead", "FlowDecoder", "GradientClip", "CorrBlock", "MotionEncoder", "SepConvGRU", "FlowHead"]
---

# Q: Audit inferred ScoreHead relationships

## Answer

Audit result: none of the 18 inferred ScoreHead uses edges are explicit in that directed form. All 18 originate from flow_decoder.py line 23, indicating co-occurrence around the ScoreHead import. One edge is structurally related but directionally reversed: ScoreHead uses FlowDecoder should be FlowDecoder imports instantiates and calls ScoreHead at flow_decoder.py lines 23, 259-263, and 310. One edge is a likely name-collision mismatch: inferred target GradientClip points to flow_decoder GradientClip (line 26), while ScoreHead actually uses score_head GradientClip at score_head.py lines 21 and 38. The remaining inferred links to GradClipModule, CorrBlock, MotionEncoder, SepConvGRU, FlowHead, and multiple flow_decoder rationale nodes are speculative association edges rather than direct code dependencies.

## Source Nodes

- ScoreHead
- FlowDecoder
- GradientClip
- CorrBlock
- MotionEncoder
- SepConvGRU
- FlowHead