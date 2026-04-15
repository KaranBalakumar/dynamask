Graphify output guide for dynamask_vio/models
============================================

Recommended default
-------------------
Use graph.corrected.json for query and traversal work.
It keeps the filtered low-noise edge set and adds direction-corrected bridge edges
validated from source code.

Graph variants
--------------
1) graph.json
   - Original graph output from graphify run.
   - Includes raw inferred edges (higher recall, lower precision).

2) graph.filtered.json
   - Removes low-signal inferred edges sourced only from:
     dynamask_vio/models/flow_decoder.py:L23
   - Better precision around ScoreHead bridge analysis.

3) graph.corrected.json   (RECOMMENDED)
   - Starts from graph.filtered.json.
   - Applies direction-corrected, extracted bridge edges:
     DynaMaskVIO -> IMUEncoder
     DynaMaskVIO -> FlowDecoder
     DynaMaskVIO -> BasicEncoder
     FlowDecoder -> ScoreHead
   - Best artifact for architecture navigation in this session.

Reports and visualizations
--------------------------
- GRAPH_REPORT.md            : report for graph.json
- GRAPH_REPORT.filtered.md   : report for graph.filtered.json
- GRAPH_REPORT.corrected.md  : report for graph.corrected.json
- graph.html                 : visualization for graph.json
- graph.filtered.html        : visualization for graph.filtered.json
- graph.corrected.html       : visualization for graph.corrected.json

Reference map
-------------
- bridge_map.corrected.json
  Direction-corrected bridge map with source-line evidence and correction notes.
