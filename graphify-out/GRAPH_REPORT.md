# Graph Report - dynamask_vio+references+docs  (2026-04-10)

## Corpus Check
- 337 files · ~1,849,871 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2045 nodes · 3255 edges · 65 communities detected
- Extraction: 83% EXTRACTED · 17% INFERRED · 0% AMBIGUOUS · INFERRED: 551 edges (avg confidence: 0.6)
- Token cost: 47,349 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `LieGroup` - 42 edges
2. `DPVO` - 29 edges
3. `pypose` - 26 edges
4. `OfflineDebugCallback` - 23 edges
5. `LieGroupParameter` - 23 edges
6. `ToMatrix` - 21 edges
7. `IMUAugmentor` - 20 edges
8. `Exp` - 20 edges
9. `Log` - 20 edges
10. `Inv` - 20 edges

## Surprising Connections (you probably didn't know these)
- `Reference: https://github.com/uzh-rpg/learned_inertial_model_odometry/blob/maste` --uses--> `Sequence`  [INFERRED]
  references/Air-IO/datasets/BlackBirddataset.py → references/Air-IO/datasets/dataset.py
- `Module size grows with function count` --shares_data_with--> `Functions`  [INFERRED]
  references/DPVO/DPViewer/pybind11/docs/pybind11_vs_boost_python2.png → /home/karan/dynamask/references/DPVO/DPRetrieval/pybind11/docs/pybind11_vs_boost_python1.png
- `Compilation time of module` --conceptually_related_to--> `Seconds`  [EXTRACTED]
  references/DPVO/DPViewer/pybind11/docs/pybind11_vs_boost_python1.png → /home/karan/dynamask/references/DPVO/DPRetrieval/pybind11/docs/pybind11_vs_boost_python1.png
- `Load poses from pose_lcam_front.txt.         Format: each line is tx ty tz qx qy` --uses--> `VisualAugmentor`  [INFERRED]
  dynamask_vio/data/tartanair_dataset.py → dynamask_vio/data/augmentations.py
- `Load poses from pose_lcam_front.txt.         Format: each line is tx ty tz qx qy` --uses--> `IMUAugmentor`  [INFERRED]
  dynamask_vio/data/tartanair_dataset.py → dynamask_vio/data/augmentations.py

## Hyperedges (group relationships)
- **Blackbird train/test split** — blackbird_dataset, seen_sequences, unseen_sequences [EXTRACTED 1.00]
- **Blackbird method comparison** — ronin, imo_with_thrust, airio_ours [EXTRACTED 1.00]
- **IMU Navigation Benchmark** — raw_imu, airimu, ground_truth, raw_integration_trajectory, long_range_helicopter_imu_integration [INFERRED 0.93]
- **Trajectory Comparison Suite** — traj_a_position_error, traj_b_position_error, avg_velocity_error [EXTRACTED 1.00]
- **Mission Map Context** — takeoff, landing_event, lake_erie, cleveland, pittsburgh [INFERRED 0.78]
- **Traditional Pipeline** — imu_sensor, noisy_signal, integrator, pose_graph_optimization [EXTRACTED 0.95]
- **AirIMU Pipeline** — imu_sensor, network, differentiable_covariance_propagation, differentiable_integrator, pose_graph_optimization [EXTRACTED 0.96]
- **Learning-based Pipeline** — imu_sensor, network, covariance_velocity_w, directly_regressed_velocity [EXTRACTED 0.93]
- **pybind11 logo wordmark** — pybind11_logo_image, python_motif, bind11_text [INFERRED 0.90]
- **Compile time series comparison** — compilation_time_of_module_chart, pybind11_series, boost_python_series [INFERRED 0.91]
- **Compilation time chart components** — pybind11_vs_boost_python1_chart, pybind11_vs_boost_python1_title, pybind11_vs_boost_python1_xaxis, pybind11_vs_boost_python1_yaxis [INFERRED 0.93]
- **Module size comparison** — pybind11_vs_boost_python2_image, boost_python, pybind11, module_file_size, function_count [EXTRACTED 0.96]
- **Module size comparison** — pybind11_vs_boost_python2_chart, boost_python, pybind11, functions_axis, bytes_axis [INFERRED 0.92]
- **Module compilation comparison** — compilation_time_of_module, blue_data_series, orange_data_series, pybind11, boost_python [INFERRED 0.78]
- **Module compilation comparison** — pybind11, boost_python, compilation_time_of_module, functions_axis [EXTRACTED 0.93]
- **Module size scaling comparison** — pybind11_module, boost_python2_module, functions_axis, bytes_axis [INFERRED 0.90]
- **Module Size Benchmark Comparison** — pybind11, boost_python, module_file_size [INFERRED 0.90]
- **pybind11 Documentation Suite** — pybind11_index, pybind11_installing, pybind11_basics, pybind11_classes, pybind11_compiling, pybind11_reference, pybind11_benchmark, pybind11_faq, pybind11_limitations, pybind11_changelog, pybind11_upgrade [EXTRACTED 1.00]
- **DPRetrieval Build Stack** — dpretrieval_cmake, pybind11_subdirectory, opencv_dependency, dbow2_dependency, dpretrieval_module [EXTRACTED 1.00]
- **EuRoC Ground Truth Sequence Set** — euroc_trajectory_format, mh_01_easy, mh_02_easy, mh_03_medium, mh_04_difficult, mh_05_difficult, v1_01_easy, v1_02_medium, v1_03_difficult, v2_01_easy, v2_02_medium, v2_03_difficult [INFERRED 0.88]
- **pybind11 docs suite** — docs_index, docs_installing, docs_basics, docs_classes, docs_compiling, docs_faq, docs_limitations, docs_reference, docs_release, docs_changelog, docs_upgrade, docs_benchmark [EXTRACTED 1.00]
- **camera calibration parameters** — calib_barn, calib_eth, calib_euroc, calib_iphone, calib_kitti, calib_monovo, calib_tartan, calib_tum3, calib_viper [INFERRED 0.90]
- **DPV-SLAM benchmark logs** — log_dpv_slam_euroc, log_dpv_slam_tartan, log_dpv_slam_tum_rgbd, concept_5_runs_per_sequence [INFERRED 0.88]
- **Pybind11 type-conversion family** — file:references/DPVO/DPRetrieval/pybind11/docs/advanced/cast/chrono.rst, file:references/DPVO/DPRetrieval/pybind11/docs/advanced/cast/custom.rst, file:references/DPVO/DPRetrieval/pybind11/docs/advanced/cast/eigen.rst, file:references/DPVO/DPRetrieval/pybind11/docs/advanced/cast/functional.rst, file:references/DPVO/DPRetrieval/pybind11/docs/advanced/cast/index.rst, file:references/DPVO/DPRetrieval/pybind11/docs/advanced/cast/overview.rst, file:references/DPVO/DPRetrieval/pybind11/docs/advanced/cast/stl.rst, file:references/DPVO/DPRetrieval/pybind11/docs/advanced/cast/strings.rst, file:references/DPVO/DPViewer/pybind11/docs/advanced/cast/chrono.rst, file:references/DPVO/DPViewer/pybind11/docs/advanced/cast/custom.rst, file:references/DPVO/DPViewer/pybind11/docs/advanced/cast/eigen.rst, file:references/DPVO/DPViewer/pybind11/docs/advanced/cast/functional.rst, file:references/DPVO/DPViewer/pybind11/docs/advanced/cast/index.rst, file:references/DPVO/DPViewer/pybind11/docs/advanced/cast/overview.rst, file:references/DPVO/DPViewer/pybind11/docs/advanced/cast/stl.rst, file:references/DPVO/DPViewer/pybind11/docs/advanced/cast/strings.rst, concept:pybind11_type_conversions [INFERRED 0.95]
- **DynaMask-VIO design stack** — file:docs/DynaMask_V2_Method.md, file:docs/DynaMask_VIO_FINAL_Implementation_Spec.md, file:docs/DynaMask_VIO_Project_Plan.md, concept:dynamask_vio_system, concept:dynamask_v2 [INFERRED 0.90]
- **pybind11 documentation bundle** — readme_dpviewer, readme_dpretrieval, classes_doc_dpviewer, embedding_doc_dpviewer, exceptions_doc_dpviewer, functions_doc_dpviewer, misc_doc_dpviewer, smart_ptrs_doc_dpviewer, pycpp_index_dpviewer, pycpp_numpy_dpviewer, pycpp_object_dpviewer, pycpp_utilities_dpviewer, pycpp_index_dpretrieval, pycpp_numpy_dpretrieval, pycpp_object_dpretrieval, pycpp_utilities_dpretrieval, cmakelists_dpviewer, cmakelists_dpretrieval [INFERRED 0.88]
- **DPVO evaluation log suite** — euroc_log, euroc_fast_log, icl_nuim_log, tartan_log, tum_rgbd_log [EXTRACTED 1.00]
- **AirIMU dataset references** — airimu_readme, subt_mrs_readme, euroc_dataset, tum_vi_dataset, kitti_dataset, subt_mrs_dataset [INFERRED 0.85]
- **AirIO Dataset Options** — euroc_dataset, blackbird_dataset, pegasus_dataset [INFERRED 0.90]
- **AirIO Runtime Workflow** — airio_pretrained_models, airio_runtime_workflows, airio_custom_config [INFERRED 0.86]
- **DPVO Viewer Toolchain** — dpvo_setup_install, dpviewer_root_cmake, dpviewer_module_cmake, pytorch_dependency, pangolin_dependency, cuda_architecture_detection [INFERRED 0.83]

## Communities

### Community 0 - "DPVO Corpus"
Cohesion: 0.02
Nodes (64): Reference: https://github.com/uzh-rpg/learned_inertial_model_odometry/blob/maste, broadcast_inputs(), check_broadcastable(), Automatic broadcasting of missing dimensions, gradientvelo(), interp_xyz(), plot_bias_subplots(), Plots the bias of three axes in subplots arranged vertically.      Parameters: (+56 more)

### Community 1 - "DPVO Corpus"
Cohesion: 0.02
Nodes (38): CorrLayer, PatchLayer, _Extension, flow_mag(), iproj(), point_cloud(), proj(), generate point cloud from patches (+30 more)

### Community 2 - "Pybind11 Advanced Docs"
Cohesion: 0.02
Nodes (99): Automatic Downcasters, Blue data series, Boost.Python, Boost.Python2, Boost.Python, BSD-style License, Build performance comparison, Build System (+91 more)

### Community 3 - "DynaMask Models"
Cohesion: 0.03
Nodes (55): BasicEncoder, load_raft_encoder_weights(), RAFT-style BasicEncoder for DynaMask V2.  Two separate encoder instances are use, Args:             x: [B, 3, H, W] input image             f_imu: [B, imu_dim] IM, Load pretrained RAFT weights into a BasicEncoder.      Args:         encoder: th, Two 3x3 convs with InstanceNorm and residual connection.      When in_ch != out_, RAFT BasicEncoder — produces 128-ch features at 1/8 resolution.      Optionally, ResidualBlock (+47 more)

### Community 4 - "DPVO Corpus"
Cohesion: 0.03
Nodes (10): call(), multiple_values_error(), nameless_argument_error(), process(), instance_simple_holder_in_ptrs(), size_in_ptrs(), global_state(), options() (+2 more)

### Community 5 - "DPVO Corpus"
Cohesion: 0.09
Nodes (44): Act3, Act4, Adj, AdjT, Exp, FromVec, GroupOp, Inv (+36 more)

### Community 6 - "DPVO Corpus"
Cohesion: 0.05
Nodes (34): ate(), evaluate(), run(), show_image(), video_iterator(), _append_jsonl(), OfflineDebugCallback, OfflineModelWatchCallback (+26 more)

### Community 7 - "DPVO Benchmarks"
Cohesion: 0.04
Nodes (49): IMUAugmentor, All data augmentations — visual and IMU.  Visual augmentations are applied ident, Augments IMU window data., Args:             imu_window: [N, 7] — (dt, ax, ay, az, gx, gy, gz), Augments a pair of images + mask consistently., Args:             img_prev, img_curr: [H, W, 3] float32 in [0, 255], Apply identical color jitter to both images (expects [0, 255] range)., VisualAugmentor (+41 more)

### Community 8 - "DPVO Corpus"
Cohesion: 0.04
Nodes (23): advance(), bytearray(), bytes(), capsule(), dec_ref(), discard_as_unraisable(), ellipsis(), error_string() (+15 more)

### Community 9 - "DPVO Corpus"
Cohesion: 0.05
Nodes (26): DPVO, local correlation volume, reproject patch k from i -> j, kinda hacky way to ensure enough motion for initialization, Global bundle adjustment          Includes both active and inactive edges, run(), show_image(), run() (+18 more)

### Community 10 - "AirIMU Corpus"
Cohesion: 0.06
Nodes (16): CNNcorrection, CNNEncoder, CNNPOS, The input feature shape [B, F, Duration, 6], Only correct the accelerometer, CNNEncoder, CodeNet, CodeNetKITTI (+8 more)

### Community 11 - "AirIMU Corpus"
Cohesion: 0.06
Nodes (17): ABC, BlackBird, Updates the data (imu / velocity) based on the required mode.         :param coo, Sets the ground truth orientation based on the provided rotation.         :param, Sequence, Euroc, Updates the data (imu / velocity) based on the required mode.         :param coo, Sets the ground truth orientation based on the provided rotation.         :param (+9 more)

### Community 12 - "DPVO Corpus"
Cohesion: 0.07
Nodes (20): GatedResidual, GradClip, GradientClip, GradientZero, GradMag, GradZero, LayerNorm1D, SoftAgg (+12 more)

### Community 13 - "Pybind11 Docs"
Cohesion: 0.05
Nodes (54): Benchmark Suite, breathe, Build Systems, CMake, Compile Test Cases, conda-forge, const_name, Convenience Classes (+46 more)

### Community 14 - "DPVO Corpus"
Cohesion: 0.05
Nodes (24): ba(), block_matmul(), block_solve(), CholeskySolver, disp_retr(), pose_retr(), block matrix multiply, safe_scatter_add_mat() (+16 more)

### Community 15 - "DPVO Corpus"
Cohesion: 0.05
Nodes (37): object, auto_cpp_level(), build_ext, cxx_std(), has_flag(), intree_extensions(), naive_recompile(), no_recompile() (+29 more)

### Community 16 - "DPVO Corpus"
Cohesion: 0.06
Nodes (27): cropping and resizing, perform augmentation on RGB-D video, RGBDAugmentor, depth_read(), image_read(), Base class for RGBD dataset, compute optical flow distance between all pairs of frames, return training video (+19 more)

### Community 17 - "Air-IO Corpus"
Cohesion: 0.08
Nodes (12): IMUEKF, r'''     Performs Batched Extended Kalman Filter (EKF).      Args:         model, r'''         Set the covariance matrices of transition noise and observation noi, r'''         Performs one step estimation.          Args:             state (:ob, r'''         Propogate the system model without observation., EKF_runner, Take the     state: R, V, P, bg, ba     propogate(Ri, hat(Vi), Pi) - Pj;  hat(Vj, Don't add noise in this function, as it will be used for automatically         l (+4 more)

### Community 18 - "Pybind11 Advanced Docs"
Cohesion: 0.07
Nodes (37): Pybind11 Advanced Bindings, Chrono Conversions, Classes and Virtuals, Custom Type Casters, Eigen Conversions, Embedding the Interpreter, Exception Translation, Functional Callbacks (+29 more)

### Community 19 - "DPVO Corpus"
Cohesion: 0.08
Nodes (22): _as_tuple(), _differentiable_outputs(), get_analytical_jacobian(), get_numerical_jacobian(), gradcheck(), gradgradcheck(), iter_tensors(), make_jacobian() (+14 more)

### Community 20 - "Pybind11 Advanced Docs"
Cohesion: 0.09
Nodes (33): AirIMU, AirIMU README, All-weather Environments, Ambiguous Estimation, Arrays, Buffer Protocol, Citation, Covariance / Velocity (+25 more)

### Community 21 - "Pybind11 Docs"
Cohesion: 0.13
Nodes (30): AirIMU Result File Format, AirIO Citation, AirIO Custom Configuration Guide, AirIO Dataset Setup, AirIO Installation & Dataset Setup, AirIO (Ours), AirIO Pre-trained Models & Results, AirIO Repository Guide (+22 more)

### Community 22 - "AirIMU Corpus"
Cohesion: 0.08
Nodes (7): SeqeuncesMotionDataset, For the purpose of training and inferering     1. Abandon the features of the la, For the purpose of training and inferering     1. Abandon the features of the la, SeqDataset, SeqeuncesDataset, SeqInfDataset, SeqeuncesDataset

### Community 23 - "DynaMask Core"
Cohesion: 0.09
Nodes (17): _error_map(), _flow_magnitude_image(), _get_submodules(), _grad_norm(), _mask_overlay(), _param_norm(), W&B in-depth debugging and logging system for DynaMask V2.  Provides a Lightning, Convert [2, H, W] flow to a magnitude heatmap [H, W, 3] uint8. (+9 more)

### Community 24 - "AirIMU Corpus"
Cohesion: 0.12
Nodes (27): AirIMU, ALTO: Navigation-Grade IMU, Avg. velocity Error, Cleveland, Differentiable, Drifting Reduced, Empirical Tuning, Generalizable (+19 more)

### Community 25 - "DPVO SLAM Benchmarks"
Cohesion: 0.1
Nodes (27): Barn Calibration Parameters, ETH Calibration Parameters, EuRoC Calibration Parameters, iPhone Calibration Parameters, KITTI Calibration Parameters, MonoVO Calibration Parameters, Tartan Calibration Parameters, TUM3 Calibration Parameters (+19 more)

### Community 26 - "DynaMask Models"
Cohesion: 0.1
Nodes (23): compute_reprojection_jacobian(), differentiable_ba(), gradient_clip(), _GradientClipFn, Differentiable Two-Frame Bundle Adjustment — training-only module.  Zero learnab, Exponential map so(3) -> SO(3). omega: [B, 3] -> [B, 3, 3].      Rodrigues formu, Logarithmic map SO(3) -> so(3). R: [B, 3, 3] -> [B, 3].      Inverse Rodrigues w, Update SE(3) pose via retraction (left-multiply by exp(δξ)).      Convention: δξ (+15 more)

### Community 27 - "DPVO Corpus"
Cohesion: 0.11
Nodes (13): clear_instance(), enable_buffer_protocol(), enable_dynamic_attributes(), get_fully_qualified_tp_name(), make_default_metaclass(), make_new_python_type(), make_object_base_type(), make_static_property_type() (+5 more)

### Community 28 - "Project Docs"
Cohesion: 0.18
Nodes (26): Backend Integration, Data Augmentations, Datasets and Supervision, Differentiable Bundle Adjustment, DynaMask-V2 Self-Supervised Masking, DynaMask-VIO Project Plan, DynaMask-VIO Implementation, DynaMask-VIO System (+18 more)

### Community 29 - "Pybind11 Advanced Docs"
Cohesion: 0.11
Nodes (13): Docstrings, dec_ref(), gil_scoped_acquire(), get_internals(), get_internals_pp(), get_shared_data(), raise_err(), set_shared_data() (+5 more)

### Community 30 - "DPVO Corpus"
Cohesion: 0.11
Nodes (18): build(), docs(), lint(), make_changelog(), Lint the codebase (except for clang-format/tidy)., Lint the codebase (except for clang-format/tidy)., Run the tests (requires a compiler)., Run the tests (requires a compiler). (+10 more)

### Community 31 - "AirIMU Corpus"
Cohesion: 0.19
Nodes (17): _forward_pair(), inference(), _load_config(), load_model(), main(), _make_overlay(), _prob_to_heatmap(), Single-sequence inference script for DynaMask V2.  Supports two input modes: 1) (+9 more)

### Community 32 - "AirIMU Corpus"
Cohesion: 0.13
Nodes (8): get_loss(), get_motion_loss(), get_motion_RMSE(), get_RMSE(), loss_(), motion_loss_(), get the RMSE of the last state in one segment, get the RMSE of the last state in one segment

### Community 33 - "DPVO Corpus"
Cohesion: 0.13
Nodes (8): Store the image into the frame buffer, Once we keyframe an image, we can safely cache all images          before & incl, Add frames to the image-retrieval database, Record the loop closure so we don't have redundant edges, Check that we've retrieved <num_repeat> consecutive frames, Keep popping off the queue until the it is empty          or we find a positive, Pop retrived pairs off the queue. Return if they have non-trivial score, RetrievalDBOW

### Community 34 - "Air-IO Corpus"
Cohesion: 0.2
Nodes (3): CasADIEKF, CasADIMU, EKF_runner

### Community 35 - "DPVO Corpus"
Cohesion: 0.15
Nodes (8): build_ext, CMakeBuild, CMakeExtension, get_and_replace(), Prepare a temporary directory, cleanup when done, # TODO: use literals & overload (typing extensions or Python 3.8), SDist, TemporaryDirectory()

### Community 36 - "Evaluation Pipeline"
Cohesion: 0.23
Nodes (12): compute_mask_metrics(), evaluate_imu(), evaluate_mask(), evaluate_trajectory(), _load_config(), main(), _merge_configs(), Evaluation: mask quality metrics + VIO trajectory metrics.  Mask metrics (on VIO (+4 more)

### Community 37 - "DPVO Corpus"
Cohesion: 0.22
Nodes (4): ImageCache, Wait until the previous image is finished writing, Save the image to disk (asynchronously), Pop images from the buffer and write them to disk

### Community 38 - "DynaMask Losses"
Cohesion: 0.17
Nodes (11): flow_smoothness_loss(), mask_regularisation_loss(), photometric_loss(), pose_consistency_loss(), Self-supervised loss functions for DynaMask V2.  No GT masks. No GT flow. Only:, BA reprojection error on static correspondences should be low.      Args:, Prevent trivial mask solutions + spatial smoothness.      Args:         mask: [B, Edge-aware flow smoothness (UnFlow/DDFlow style).      Flow should be smooth exc (+3 more)

### Community 39 - "DPVO Corpus"
Cohesion: 0.23
Nodes (10): cam_read(), Read depth data from file, return as numpy array., Read camera data, return (M,N) tuple.     M is the intrinsic matrix, N is the ex, Read .flo file in Middlebury format, Write optical flow to file.          If v is None, uv is assumed to contain both, read_gen(), readDPT(), readFlow() (+2 more)

### Community 40 - "DPVO Corpus"
Cohesion: 0.17
Nodes (12): EuRoC Ground Truth Trajectory Format, MH 01 Easy Trajectory, MH 02 Easy Trajectory, MH 03 Medium Trajectory, MH 04 Difficult Trajectory, MH 05 Difficult Trajectory, V1 01 Easy Trajectory, V1 02 Medium Trajectory (+4 more)

### Community 41 - "Pybind11 Docs"
Cohesion: 0.35
Nodes (11): First Steps, Benchmark, Changelog, Object-oriented Code, Build Systems, Frequently Asked Questions, pybind11 Documentation Index, Installing the Library (+3 more)

### Community 42 - "DynaMask Core"
Cohesion: 0.31
Nodes (6): DynaMaskONNXWrapper, export(), _load_config(), main(), Export DynaMask-VIO to ONNX for deployment.  Usage:     python -m dynamask_vio.e, Thin wrapper that flattens the dict output for ONNX export.

### Community 43 - "DynaMask Losses"
Cohesion: 0.29
Nodes (7): covariance_nll_loss(), imu_integration_loss(), IMU integration error loss and covariance calibration (NLL) loss.  Uses PyPose f, Geodesic angle ‖Log(R)‖ for batch of rotation matrices [B, 3, 3] → [B]., Supervise preintegrated motion against ground truth.      Args:         pred_R,, Negative log-likelihood — teaches calibrated uncertainty.      NLL = 0.5 * (e^T, _rotation_angle()

### Community 44 - "DynaMask Data"
Cohesion: 0.43
Nodes (6): convert_bag(), find_bags(), _get_ros1_decoder(), main(), Recursively find all .bag files under root, return list of (rel_path, abs_path)., Convert a single .bag to .h5.  Streams data — peak RAM is one frame.

### Community 45 - "DPVO Corpus"
Cohesion: 0.33
Nodes (2): cast(), cast_impl()

### Community 46 - "DPVO Corpus"
Cohesion: 0.33
Nodes (1): Logger

### Community 47 - "AirIMU Corpus"
Cohesion: 0.38
Nodes (5): custom_collate(), motion_collate(), motion_collate_data(), padding_collate(), # TODO: Implement data augmentation if needed

### Community 48 - "DynaMask Core"
Cohesion: 0.4
Nodes (4): download_raft_weights(), main(), Download pretrained weights required for DynaMask V2.  RAFT pretrained weights (, Download or locate RAFT-Things pretrained weights.      Returns path to the chec

### Community 49 - "Pybind11 Docs"
Cohesion: 0.4
Nodes (0): 

### Community 50 - "Pybind11 Docs"
Cohesion: 0.4
Nodes (5): bind11 text, pybind11 logo image, pybind11 wordmark, Python binding library, Python motif in logo

### Community 51 - "DPVO Corpus"
Cohesion: 0.6
Nodes (5): DBoW2 Dependency, DPRetrieval CMake Configuration, dpretrieval Python Module, OpenCV Dependency, pybind11 Subdirectory

### Community 52 - "Pybind11 Docs"
Cohesion: 0.67
Nodes (0): 

### Community 53 - "DynaMask Models"
Cohesion: 1.0
Nodes (1): Solve H x = b via Cholesky decomposition.          Args:             H: [B, 6, 6

### Community 54 - "DPVO Corpus"
Cohesion: 1.0
Nodes (1): The CXX standard level. If set, will add the required flags. If left         at

### Community 55 - "DPVO Corpus"
Cohesion: 1.0
Nodes (0): 

### Community 56 - "DPVO Corpus"
Cohesion: 1.0
Nodes (0): 

### Community 57 - "DPVO Corpus"
Cohesion: 1.0
Nodes (1): The CXX standard level. If set, will add the required flags. If left at

### Community 58 - "DPVO Corpus"
Cohesion: 1.0
Nodes (0): 

### Community 59 - "Air-IO Corpus"
Cohesion: 1.0
Nodes (1): r'''         Linear/linearized system output matrix.          .. math::

### Community 60 - "Air-IO Corpus"
Cohesion: 1.0
Nodes (1): r'''         Linear/Linearized system observation matrix.          .. math::

### Community 61 - "Air-IO Corpus"
Cohesion: 1.0
Nodes (1): r'''         The covariance of system transition noise.

### Community 62 - "Air-IO Corpus"
Cohesion: 1.0
Nodes (1): r'''         The covariance of system transition noise.

### Community 63 - "Air-IO Corpus"
Cohesion: 1.0
Nodes (1): r'''         The covariance of system observation noise.

### Community 64 - "DPVO SLAM Benchmarks"
Cohesion: 1.0
Nodes (1): config/fast.yaml

## Ambiguous Edges - Review These
- `Blue Series` → `Orange Series`  [AMBIGUOUS]
  references/DPVO/DPRetrieval/pybind11/docs/pybind11_vs_boost_python1.svg · relation: semantically_similar_to

## Knowledge Gaps
- **345 isolated node(s):** `Evaluation: mask quality metrics + VIO trajectory metrics.  Mask metrics (on VIO`, `Compute IoU, Precision, Recall, F1 for a batch.      Args:         pred_mask: [B`, `Evaluate mask quality over a full dataloader.`, `Evaluate IMU preintegration quality (RTE, ROE).`, `Compute ATE RMSE and RPE using the evo package.      Args:         est_traj_file` (+340 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `DynaMask Models`** (1 nodes): `Solve H x = b via Cholesky decomposition.          Args:             H: [B, 6, 6`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `DPVO Corpus`** (1 nodes): `The CXX standard level. If set, will add the required flags. If left         at`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `DPVO Corpus`** (1 nodes): `make_changelog.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `DPVO Corpus`** (1 nodes): `libsize.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `DPVO Corpus`** (1 nodes): `The CXX standard level. If set, will add the required flags. If left at`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `DPVO Corpus`** (1 nodes): `config.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Air-IO Corpus`** (1 nodes): `r'''         Linear/linearized system output matrix.          .. math::`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Air-IO Corpus`** (1 nodes): `r'''         Linear/Linearized system observation matrix.          .. math::`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Air-IO Corpus`** (1 nodes): `r'''         The covariance of system transition noise.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Air-IO Corpus`** (1 nodes): `r'''         The covariance of system transition noise.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Air-IO Corpus`** (1 nodes): `r'''         The covariance of system observation noise.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `DPVO SLAM Benchmarks`** (1 nodes): `config/fast.yaml`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Blue Series` and `Orange Series`?**
  _Edge tagged AMBIGUOUS (relation: semantically_similar_to) - confidence is low._
- **Why does `Pybind11Extension` connect `DPVO Corpus` to `DPVO Corpus`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Why does `pypose` connect `DPVO Corpus` to `DynaMask Models`, `AirIMU Corpus`, `DPVO Corpus`, `Air-IO Corpus`, `Pybind11 Advanced Docs`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Why does `AirIO Dataset Setup` connect `Pybind11 Docs` to `DPVO Benchmarks`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `LieGroup` (e.g. with `Exp` and `Log`) actually correct?**
  _`LieGroup` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `DPVO` (e.g. with `Read in a calibration file and parse into a dictionary.` and `VONet`) actually correct?**
  _`DPVO` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `OfflineDebugCallback` (e.g. with `DynaMaskLitModule` and `PhaseDataModule`) actually correct?**
  _`OfflineDebugCallback` has 11 INFERRED edges - model-reasoned connections that need verification._