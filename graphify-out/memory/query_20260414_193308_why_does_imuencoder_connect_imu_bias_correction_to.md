---
type: "query"
date: "2026-04-14T19:33:08.037663+00:00"
question: "Why does IMUEncoder connect IMU Bias Correction to RAFT Feature Encoding?"
contributor: "graphify"
source_nodes: ["IMUEncoder", "AirIMUCorrector", "BasicEncoder", "FiLM", "DynaMaskVIO"]
---

# Q: Why does IMUEncoder connect IMU Bias Correction to RAFT Feature Encoding?

## Answer

IMUEncoder is the bridge because it turns corrected IMU motion into the f_imu conditioning vector, which is then injected into RAFT BasicEncoder via FiLM. In imu_encoder.py, IMUEncoder runs AirIMUCorrector and DifferentiablePreintegrator and computes f_imu at lines 79-113, returning it at line 116. In dynamask.py, DynaMaskVIO calls IMUEncoder at line 87, reads f_imu at line 88, and feeds it to BasicEncoder at lines 93-94. BasicEncoder consumes f_imu through film_layers at backbone.py lines 165-176, and FiLM applies gamma and beta modulation in film.py lines 46-48. So the cross-community connection is not accidental: bias-corrected IMU signals directly modulate RAFT visual features.

## Source Nodes

- IMUEncoder
- AirIMUCorrector
- BasicEncoder
- FiLM
- DynaMaskVIO