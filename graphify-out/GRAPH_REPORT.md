# Graph Report - dynamask_vio  (2026-04-10)

## Corpus Check
- Large corpus: 2904 files · ~6,206,179 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder, or use --no-semantic to run AST-only.

## Summary
- 6412 nodes · 10628 edges · 181 communities detected
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 1082 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `Timer` - 74 edges
2. `IStereoDepth` - 63 edges
3. `IMatcher` - 50 edges
4. `StereoFrame` - 49 edges
5. `StereoData` - 48 edges
6. `SequenceBase` - 48 edges
7. `LieGroup` - 42 edges
8. `DPVO` - 33 edges
9. `AutoScalingTensor` - 33 edges
10. `StereoInertialFrame` - 32 edges

## Surprising Connections (you probably didn't know these)
- `Load poses from pose_lcam_front.txt.         Format: each line is tx ty tz qx qy` --uses--> `VisualAugmentor`  [INFERRED]
  dynamask_vio/data/tartanair_dataset.py → dynamask_vio/data/augmentations.py
- `Load poses from pose_lcam_front.txt.         Format: each line is tx ty tz qx qy` --uses--> `IMUAugmentor`  [INFERRED]
  dynamask_vio/data/tartanair_dataset.py → dynamask_vio/data/augmentations.py
- `Load IMU data from imu/ directory.` --uses--> `VisualAugmentor`  [INFERRED]
  dynamask_vio/data/tartanair_dataset.py → dynamask_vio/data/augmentations.py
- `Load IMU data from imu/ directory.` --uses--> `IMUAugmentor`  [INFERRED]
  dynamask_vio/data/tartanair_dataset.py → dynamask_vio/data/augmentations.py
- `EuRoC MAV dataset loader (ASL format).  Used for:   1. IMU Head pretraining (Pha` --uses--> `IMUAugmentor`  [INFERRED]
  dynamask_vio/data/euroc_dataset.py → dynamask_vio/data/augmentations.py

## Communities

### Community 0 - "Test Submodule When"
Cohesion: 0.0
Nodes (168): deprecated_call(), pytest.deprecated_call() seems broken in pytest<3.9.x; concretely, it     doesn', # TODO: Remove this when testing requires pytest>=3.9., bind_ConstructorStats(), PYBIND11_MODULE(), get_await_result(), test_await(), test_await_missing() (+160 more)

### Community 1 - "Init   Run"
Cohesion: 0.01
Nodes (183): ba(), block_matmul(), block_solve(), CholeskySolver, disp_retr(), pose_retr(), block matrix multiply, safe_scatter_add_mat() (+175 more)

### Community 2 - "Cpp Pangolin"
Cohesion: 0.01
Nodes (105): BayerOutputFormat(), DebayerVideo(), DownsampleDebayer(), GrabNewest(), GrabNext(), PitchedImageCopy(), ProcessImage(), ProcessStreams() (+97 more)

### Community 3 - "Cpp Pangolin"
Cohesion: 0.01
Nodes (91): GetKeyModifierBitmask(), key_callback(), mod_key(), mouse_callback(), spec_key(), GetMouseModifierKey(), GetPangoKey(), HandleWinMessages() (+83 more)

### Community 4 - "Cpp Pangolin"
Cohesion: 0.01
Nodes (134): main(), sample(), main(), sample(), main(), sample(), main(), sample() (+126 more)

### Community 5 - "Init   Forward"
Cohesion: 0.01
Nodes (114): PSMNet, Chain, To have SML-like function chaining operator      (f1 >> f2)(x) = f2(f1(x)), get_cmake_dir(), get_include(), Return the path to the pybind11 CMake module directory., Return the path to the pybind11 include directory. The historical "user"     arg, ConfigTestable (+106 more)

### Community 6 - "Init   Getitem  "
Cohesion: 0.02
Nodes (111): build_intrinsic(), EuRoC_Sequence, EuRoC_StereoSequence, EurocIMULoader, EurocMonocularDataset, interpolate_rotate(), interpolate_vecN(), load_EurocGTPose() (+103 more)

### Community 7 - "Init   Estimate"
Cohesion: 0.02
Nodes (131): ABC, TartanVO, ConfigTestableSubclass, Plots Gaussian distribution for some confidence interval and compare the  predic, CUDAGraph_FlowFormerCovFrontend, CUDAGraphHandler, estimate_depth(), estimate_pair() (+123 more)

### Community 8 - "Pybind11 Namespace Begin Name"
Cohesion: 0.01
Nodes (72): call(), cast(), multiple_values_error(), nameless_argument_error(), none(), process(), clear_instance(), deregister_instance() (+64 more)

### Community 9 - "Cpp Pangolin"
Cohesion: 0.01
Nodes (72): AddSamples(), Clear(), DataLog(), FirstBlock(), Log(), Sample(), Samples(), Save() (+64 more)

### Community 10 - "Pangolin Cpp"
Cohesion: 0.02
Nodes (90): Expand(), FileExists(), FilesMatchingWildcard(), FindPath(), MakeUniqueFilename(), MatchesWildcard(), PathExpand(), PathOsNormaliseInplace() (+82 more)

### Community 11 - "Forward Init  "
Cohesion: 0.02
Nodes (81): AttentionLayer, BroadMultiHeadAttention, LinearPositionEmbeddingSine(), MultiHeadAttention, PositionalEncoding2D, :param channels: The last dimension of the tensor you want to apply pos emb to., :param tensor: A 4d tensor of size (batch_size, x, y, ch)         :return: Posit, # NOTE: For some reason, replace this with SDPA (FlashAttention 2) backend makes (+73 more)

### Community 12 - "Init   Forward"
Cohesion: 0.02
Nodes (72): GlobalConfig, HourglassDecoder, BasicBlock, conv(), linear(), VOFlowRes, ApplyGTMatchCov, ApplyGTMatchMask (+64 more)

### Community 13 - "Cpp Act4"
Cohesion: 0.02
Nodes (51): build_ext, _Extension, BindGlElement(), GlDraw(), ToGlGeometry(), ToGlGeometryElement(), UnbindGlElements(), flow_mag() (+43 more)

### Community 14 - "Pangolin Cpp"
Cohesion: 0.02
Nodes (51): changeStructure(), loadFeatures(), main(), testDatabase(), testVocCreation(), wait(), HighestPriScheme(), PrintFactoryDetails() (+43 more)

### Community 15 - "Init   Forward"
Cohesion: 0.02
Nodes (52): DeepPatchVO, # NOTE: DPVO will perform /255 operation internally., GatedResidual, GradClip, GradientClip, GradientZero, GradMag, GradZero (+44 more)

### Community 16 - "Stb Truetype Equal"
Cohesion: 0.04
Nodes (124): equal(), main(), my_stbtt_initfont(), my_stbtt_print(), stbrp_init_target(), stbrp_pack_rects(), stbtt__add_point(), stbtt_BakeFontBitmap() (+116 more)

### Community 17 - "Pangolin Cpp"
Cohesion: 0.02
Nodes (51): AdjustScale(), AdjustTranslation(), FixSelection(), glRenderOverlay(), glRenderTexture(), ImageToScreen(), ImageViewHandler(), Keyboard() (+43 more)

### Community 18 - "Pangolin Cpp"
Cohesion: 0.03
Nodes (66): PThreadConditionVariable, ConfigureColorNode(), ConfigureDepthNode(), ConfigureNodes(), DepthSenseContext(), DepthSenseVideo(), DeviceClosing(), EventLoop() (+58 more)

### Community 19 - "Init   Training"
Cohesion: 0.03
Nodes (46): _append_jsonl(), OfflineDebugCallback, OfflineModelWatchCallback, OfflineWriter, Offline debug logging callbacks for DynaMask V2.  These callbacks mirror the W&B, Offline equivalent of WandbDebugCallback with parity metric names., Append-only writer for offline training debug artifacts., Offline equivalent of wandb.watch periodic parameter/gradient snapshots. (+38 more)

### Community 20 - "Forward Init  "
Cohesion: 0.03
Nodes (55): BasicEncoder, load_raft_encoder_weights(), RAFT-style BasicEncoder for DynaMask V2.  Two separate encoder instances are use, Args:             x: [B, 3, H, W] input image             f_imu: [B, imu_dim] IM, Load pretrained RAFT weights into a BasicEncoder.      Args:         encoder: th, Two 3x3 convs with InstanceNorm and residual connection.      When in_ch != out_, RAFT BasicEncoder — produces 128-ch features at 1/8 resolution.      Optionally, ResidualBlock (+47 more)

### Community 21 - "Init   Convert"
Cohesion: 0.09
Nodes (44): Act3, Act4, Adj, AdjT, Exp, FromVec, GroupOp, Inv (+36 more)

### Community 22 - "Cpp Pangolin"
Cohesion: 0.03
Nodes (36): cast(), localtime_thread_safe(), GetNodeValStr(), GetParameter(), Initialise(), InitPangoDeviceProperties(), SetDeviceParams(), SetNodeValStr() (+28 more)

### Community 23 - "Std Stl"
Cohesion: 0.03
Nodes (26): auxiliaries(), data(), data_t(), index_at(), index_at_t(), offset_at(), offset_at_t(), Tests fix for #685 - ndarray shouldn't go to std::string overload (+18 more)

### Community 24 - "Init   Repr  "
Cohesion: 0.04
Nodes (29): Core method for IVisualOdometry. This method handles the incoming frames and per, Provides the VisualMap built across multiple calls of .run(...)., You can define additional operations on terminate. For instance, smoothing traje, ColoredTqdm, GlobalLog, save_as_csv(), create(), data() (+21 more)

### Community 25 - "Init   Frame"
Cohesion: 0.08
Nodes (30): AnalyticModule, FactorGraph, Analytic_ICP_TwoframePGO, Analytic_Reproj_TwoFramePGO, Analytic_ReprojDisp_TwoFramePGO, GraphInput, GraphOutput, ICP_TwoframePGO (+22 more)

### Community 26 - "Pangolin Hpp"
Cohesion: 0.04
Nodes (33): Apply(), ApplyNView(), EnableProjectiveTexturing(), Follow(), GetModelViewMatrix(), GetProjectionMatrix(), GetProjectionModelViewMatrix(), GetProjectiveTextureMatrix() (+25 more)

### Community 27 - "Flow Forward"
Cohesion: 0.04
Nodes (35): backward(), CorrLayer, cupy_kernel(), cupy_launch(), forward(), _FunctionCorrelation, ModuleCorrelation, PatchLayer (+27 more)

### Community 28 - "Imu Dataset"
Cohesion: 0.05
Nodes (40): IMUAugmentor, All data augmentations — visual and IMU.  Visual augmentations are applied ident, Augments IMU window data., Args:             imu_window: [N, 7] — (dt, ax, ay, az, gx, gy, gz), Augments a pair of images + mask consistently., Args:             img_prev, img_curr: [H, W, 3] float32 in [0, 255], Apply identical color jitter to both images (expects [0, 255] range)., VisualAugmentor (+32 more)

### Community 29 - "Init   Serialize"
Cohesion: 0.04
Nodes (17): AutoScalingBundle, DenseEdge_Multi, deserialize(), EdgeLike, init(), Provide graph infrastructure (node with arbitrary features and edges), An arbitrary one-to-multi mapping relationship., In case of one-to-multi mapping with continuous index. Can significantly reduce (+9 more)

### Community 30 - "Init   Forward"
Cohesion: 0.06
Nodes (16): CNNcorrection, CNNEncoder, CNNPOS, The input feature shape [B, F, Duration, 6], Only correct the accelerometer, CNNEncoder, CodeNet, CodeNetKITTI (+8 more)

### Community 31 - "Hpp Attribute Iterator"
Cohesion: 0.05
Nodes (25): compare(), attribute_iterator, node_iterator, memory_pool, parse_error, parse_node_contents(), copy_and_expand_chars(), copy_chars() (+17 more)

### Community 32 - "Compute Between"
Cohesion: 0.05
Nodes (33): cropping and resizing, perform augmentation on RGB-D video, RGBDAugmentor, depth_read(), image_read(), Base class for RGBD dataset, compute optical flow distance between all pairs of frames, return training video (+25 more)

### Community 33 - "Pangolin Cpp"
Cohesion: 0.06
Nodes (30): GetImageWrapper(), LoadGeometryObj(), AddVertexNormals(), AttachAssociatedTexturesPly(), LoadGeometryPly(), ParsePlyAscii(), ParsePlyBE(), ParsePlyHeader() (+22 more)

### Community 34 - "Ccompiler Compile"
Cohesion: 0.05
Nodes (44): auto_cpp_level(), build_ext, cxx_std(), has_flag(), intree_extensions(), naive_recompile(), no_recompile(), ParallelCompile (+36 more)

### Community 35 - "Display Android Cpp"
Cohesion: 0.07
Nodes (41): android_app_destroy(), android_app_entry(), android_app_free(), android_app_post_exec_cmd(), android_app_pre_exec_cmd(), android_app_read_cmd(), android_app_set_activity_state(), android_app_set_input() (+33 more)

### Community 36 - "Push Tensor"
Cohesion: 0.05
Nodes (7): AutoScalingTensor, Provides `AutoScalingTensor` class, an efficient tensor wrapper for data accumul, A circular buffer tensor., push a batch of values into the CircularTensor. Will trigger a cache eviction, Actual underlying circular buffer push algorithm., tensor(), TensorQueue

### Community 37 - "Eigen Numpy"
Cohesion: 0.05
Nodes (31): array_copy_but_one(), assert_equal_ref(), assert_keeps_alive(), assert_sparse_equal_ref(), assign_both(), get_elem(), Eigen doesn't support (as of yet) negative strides. When a function takes an Eig, Tests various ways of returning references and non-referencing copies (+23 more)

### Community 38 - "Pleora Cpp"
Cohesion: 0.08
Nodes (28): DeinitBuffers(), DeinitDevice(), DeinitStream(), DropNFrames(), GetAnalogBlackLevel(), GetExposure(), GetGain(), GetGamma() (+20 more)

### Community 39 - "Custom Eq  "
Cohesion: 0.06
Nodes (27): Capture, doc(), gc_collect(), _make_explanation(), msg(), Output, pytest_assertrepr_compare(), Extended `capsys` with context manager and custom equality operators (+19 more)

### Community 40 - "Pangolin Cpp"
Cohesion: 0.07
Nodes (23): GlFont(), InitialiseFont(), InitialiseGlTexture(), Text(), glColorMask(), glCullFace(), glDepthMask(), glDisable() (+15 more)

### Community 41 - "Init Factory Wrapper"
Cohesion: 0.07
Nodes (31): create_and_destroy(), PYBIND11_OVERRIDE(), PyTF6(), PyTF7(), Tests py::init_factory() wrapper with various upcasting and downcasting returns, Tests py::init_factory() wrapper around various ways of returning the object, Tests py::init_factory() wrapper with value conversions and alias types, Tests init factory functions with dual main/alias factory functions (+23 more)

### Community 42 - "Std Tuple"
Cohesion: 0.05
Nodes (29): Tests the ability to pass bytes to C++ string-accepting functions.  Note that th, Tests support for C++17 string_view arguments and return values, Tests unicode conversion and error reporting., Issue #929 - out-of-range integer values shouldn't be accepted, # TODO: Avoid DeprecationWarning in `PyLong_AsLong` (and similar), # TODO: Avoid DeprecationWarning in `PyLong_AsLong` (and similar), std::pair <-> tuple & std::tuple <-> tuple, Casters produced with PYBIND11_TYPE_CASTER() should convert nullptr to None (+21 more)

### Community 43 - "Display Wayland Cpp"
Cohesion: 0.06
Nodes (6): handle_configure(), MakeCurrent(), Resize(), SwapBuffers(), WaylandDisplay(), WaylandWindow()

### Community 44 - "Check Gradients"
Cohesion: 0.08
Nodes (22): _as_tuple(), _differentiable_outputs(), get_analytical_jacobian(), get_numerical_jacobian(), gradcheck(), gradgradcheck(), iter_tensors(), make_jacobian() (+14 more)

### Community 45 - "Test Numpy Dtypes Assert Equal"
Cohesion: 0.09
Nodes (9): assert_equal(), dt_fmt(), packed_dtype_fmt(), partial_dtype_fmt(), partial_ld_offset(), partial_nested_fmt(), simple_dtype_fmt(), test_dtype() (+1 more)

### Community 46 - "Pangolin Assert"
Cohesion: 0.12
Nodes (20): close_device(), GetExposure(), GetGain(), GrabNewest(), GrabNext(), init_device(), init_mmap(), init_read() (+12 more)

### Community 47 - "Path Input"
Cohesion: 0.11
Nodes (27): computeHash(), deployBinary(), fix_binary(), fix_dependency(), fix_main_binaries(), get_dependencies(), getMutex(), is_loader_path_lib() (+19 more)

### Community 48 - "Pose Differentiable Ba"
Cohesion: 0.1
Nodes (23): compute_reprojection_jacobian(), differentiable_ba(), gradient_clip(), _GradientClipFn, Differentiable Two-Frame Bundle Adjustment — training-only module.  Zero learnab, Exponential map so(3) -> SO(3). omega: [B, 3] -> [B, 3, 3].      Rodrigues formu, Logarithmic map SO(3) -> so(3). R: [B, 3, 3] -> [B, 3].      Inverse Rodrigues w, Update SE(3) pose via retraction (left-multiply by exp(δξ)).      Convention: δξ (+15 more)

### Community 49 - "Build Run"
Cohesion: 0.08
Nodes (24): build(), docs(), lint(), make_changelog(), Lint the codebase (except for clang-format/tidy)., Lint the codebase (except for clang-format/tidy)., Run the tests (requires a compiler)., Run the tests (requires a compiler). (+16 more)

### Community 50 - "Gil Makes"
Cohesion: 0.1
Nodes (21): _python_to_cpp_to_python(), _python_to_cpp_to_python_from_threads(), Calls different C++ functions that come back to Python., Calls different C++ functions that come back to Python, from Python threads., # TODO: FIXME, sometimes returns -11 (segfault) instead of 0 on macOS Python 3.9, Makes sure there is no GIL deadlock when running in a thread.      It runs in a, # TODO: FIXME on macOS Python 3.9, Makes sure there is no GIL deadlock when running in a thread multiple times in p (+13 more)

### Community 51 - "Operator Argagg"
Cohesion: 0.13
Nodes (12): cmd_line_arg_is_option_flag(), flag_is_short(), fmt_ostream(), fmt_string(), get_definition_for_long_flag(), get_definition_for_short_flag(), is_valid_flag_definition(), known_long_flag() (+4 more)

### Community 52 - "Flowformer Library"
Cohesion: 0.12
Nodes (18): asNamespace(), __build_dynamic_config(), IncludeLoader, load_config(), LoadFrom, namespace_to_cfgnode(), Design for Flowformer. Flowformer uses yacs.config.CfgNode as config container., close() (+10 more)

### Community 53 - "Image Keyframe"
Cohesion: 0.12
Nodes (8): Store the image into the frame buffer, Once we keyframe an image, we can safely cache all images          before & incl, Add frames to the image-retrieval database, Record the loop closure so we don't have redundant edges, Check that we've retrieved <num_repeat> consecutive frames, Keep popping off the queue until the it is empty          or we find a positive, Pop retrived pairs off the queue. Return if they have non-trivial score, RetrievalDBOW

### Community 54 - "Ffmpeg Output Cpp"
Cohesion: 0.2
Nodes (11): Close(), CreateStream(), CreateVideoCodecContext(), FfmpegVideoOutput(), FfmpegVideoOutputStream(), Flush(), Initialise(), StartStream() (+3 more)

### Community 55 - "Deploy Tlssettings"
Cohesion: 0.16
Nodes (3): DisableRC4(), Test-RegistryValueForFipsSettings(), Write-Log()

### Community 56 - "Display Headless Cpp"
Cohesion: 0.2
Nodes (4): EGLDisplayHL, makeCurrent(), swap(), SwapBuffers()

### Community 57 - "Select Version"
Cohesion: 0.21
Nodes (9): DocumentationUpdate, Get-AuthHeader(), Get-MergedPullRequests(), Get-PullRequestFileMap(), PortUpdate, PRFileMap, Select-UpdatedPorts(), Select-Version() (+1 more)

### Community 58 - "Read Return"
Cohesion: 0.23
Nodes (10): cam_read(), Read depth data from file, return as numpy array., Read camera data, return (M,N) tuple.     M is the intrinsic matrix, N is the ex, Read .flo file in Middlebury format, Write optical flow to file.          If v is None, uv is assumed to contain both, read_gen(), readDPT(), readFlow() (+2 more)

### Community 59 - "Export Onnx"
Cohesion: 0.31
Nodes (6): DynaMaskONNXWrapper, export(), _load_config(), main(), Export DynaMask-VIO to ONNX for deployment.  Usage:     python -m dynamask_vio.e, Thin wrapper that flattens the dict output for ONNX export.

### Community 60 - "Regenerate Ps1"
Cohesion: 0.33
Nodes (3): CMakeDocumentation, FinalDocFile(), RelativeUnixPathTo()

### Community 61 - "Logger Close"
Cohesion: 0.33
Nodes (1): Logger

### Community 62 - "Evaluates Error"
Cohesion: 0.29
Nodes (4): evaluateROE(), evaluateRPE(), Evaluates error of rotation, Evaluates error of se(3) pose

### Community 63 - "Cov Loss Depth Loss"
Cohesion: 0.57
Nodes (5): cov_loss(), final_cov_loss(), flow_loss(), sequence_loss(), sequence_metric()

### Community 64 - "Pretrained Weights"
Cohesion: 0.4
Nodes (4): download_raft_weights(), main(), Download pretrained weights required for DynaMask V2.  RAFT pretrained weights (, Download or locate RAFT-Things pretrained weights.      Returns path to the chec

### Community 65 - "Conf Clean Up"
Cohesion: 0.4
Nodes (0): 

### Community 66 - "Utility Prefix"
Cohesion: 0.7
Nodes (4): Get-TempFilePath(), InstallMSI(), InstallZip(), PrintMsiExitCodeMessage()

### Community 67 - "Generate Ports"
Cohesion: 0.7
Nodes (4): GeneratePort(), GeneratePortDependency(), GeneratePortManifest(), GeneratePortName()

### Community 68 - "Torch Float"
Cohesion: 0.4
Nodes (4): CalculateOnePatch(), depth_mean: (N,) torch.float     depth_var:  (N,) torch.float     selector_dist:, depth_mean: (N,) torch.float     depth_var:  (N,) torch.float     selector_dist:, SimulateOnePatch()

### Community 69 - "Test Setuphelper Test Intree Extensions"
Cohesion: 0.5
Nodes (0): 

### Community 70 - "File Script Gen All File Strings"
Cohesion: 0.83
Nodes (3): gen_all_file_strings(), getFiles(), main()

### Community 71 - "Point Filterpointsinrange"
Cohesion: 0.5
Nodes (0): 

### Community 72 - "Test Config Modules Test Frontend Config"
Cohesion: 0.5
Nodes (0): 

### Community 73 - "Evalseq Evaluatesequences"
Cohesion: 0.83
Nodes (3): EvaluateSequences(), EvaluateSequencesAvg(), mean()

### Community 74 - "Benchmark Generate Dummy Code Boost"
Cohesion: 0.67
Nodes (0): 

### Community 75 - "Flops Analyzer Getflops"
Cohesion: 1.0
Nodes (2): GetFlops(), main()

### Community 76 - "Evalflow Evaluate Flow"
Cohesion: 0.67
Nodes (0): 

### Community 77 - "Evaldepth Evaluate Depth"
Cohesion: 0.67
Nodes (0): 

### Community 78 - "Plotseq Plot Jointly"
Cohesion: 0.67
Nodes (0): 

### Community 79 - "Addposhvcpkgtopowershellprofile Ps1"
Cohesion: 1.0
Nodes (0): 

### Community 80 - "Deploy Windows"
Cohesion: 1.0
Nodes (0): 

### Community 81 - "Create Image"
Cohesion: 1.0
Nodes (0): 

### Community 82 - "Deploy Mpi"
Cohesion: 1.0
Nodes (0): 

### Community 83 - "Deploy Install"
Cohesion: 1.0
Nodes (0): 

### Community 84 - "Deploy Inteloneapi"
Cohesion: 1.0
Nodes (0): 

### Community 85 - "Disk Space"
Cohesion: 1.0
Nodes (0): 

### Community 86 - "Deploy Visual"
Cohesion: 1.0
Nodes (0): 

### Community 87 - "Qtdeploy Ps1"
Cohesion: 1.0
Nodes (0): 

### Community 88 - "Openni2Deploy Ps1"
Cohesion: 1.0
Nodes (0): 

### Community 89 - "Check Getcontext"
Cohesion: 1.0
Nodes (0): 

### Community 90 - "Gettimeofday"
Cohesion: 1.0
Nodes (0): 

### Community 91 - "Magnumdeploy Ps1"
Cohesion: 1.0
Nodes (0): 

### Community 92 - "Decomp Set Endian"
Cohesion: 2.0
Nodes (0): 

### Community 93 - "Generatefeatures Ps1"
Cohesion: 1.0
Nodes (0): 

### Community 94 - "K4Adeploy Ps1"
Cohesion: 1.0
Nodes (0): 

### Community 95 - "Optimization Ablation Run Frame"
Cohesion: 1.0
Nodes (0): 

### Community 96 - "Test Config Tartanvo Test Tartanvo Config"
Cohesion: 1.0
Nodes (0): 

### Community 97 - "Test Config Macvo Test Macvo Config"
Cohesion: 1.0
Nodes (0): 

### Community 98 - "Test Config Sequence Test Sequence Cfg"
Cohesion: 1.0
Nodes (0): 

### Community 99 - "Test Performance Macvo Test Macvo Performance"
Cohesion: 1.0
Nodes (0): 

### Community 100 - "Test Stereo Depth Test Matching"
Cohesion: 1.0
Nodes (0): 

### Community 101 - "Test Config Loadable"
Cohesion: 1.0
Nodes (0): 

### Community 102 - "Test Matching"
Cohesion: 1.0
Nodes (0): 

### Community 103 - "Test Frontend"
Cohesion: 1.0
Nodes (0): 

### Community 104 - "Experiment Dpvo Execute Experiment"
Cohesion: 1.0
Nodes (0): 

### Community 105 - "Experiment Macvo Execute Experiment"
Cohesion: 1.0
Nodes (0): 

### Community 106 - "Experiment Tartanvo Execute Experiment"
Cohesion: 1.0
Nodes (0): 

### Community 107 - "Matchestimator"
Cohesion: 1.0
Nodes (0): 

### Community 108 - "Tartanvodisparity Avgerror"
Cohesion: 1.0
Nodes (0): 

### Community 109 - "Submission Get Cfg"
Cohesion: 1.0
Nodes (0): 

### Community 110 - "Chartdir"
Cohesion: 1.0
Nodes (0): 

### Community 111 - "Solve Via"
Cohesion: 1.0
Nodes (1): Solve H x = b via Cholesky decomposition.          Args:             H: [B, 6, 6

### Community 112 - "Cxx Standard"
Cohesion: 1.0
Nodes (1): The CXX standard level. If set, will add the required flags. If left         at

### Community 113 - "Make Changelog"
Cohesion: 1.0
Nodes (0): 

### Community 114 - "Libsize"
Cohesion: 1.0
Nodes (0): 

### Community 115 - "Test Eval Call"
Cohesion: 1.0
Nodes (0): 

### Community 116 - "Cxx Standard"
Cohesion: 1.0
Nodes (1): The CXX standard level. If set, will add the required flags. If left         at

### Community 117 - "X11Glcontext"
Cohesion: 1.0
Nodes (0): 

### Community 118 - "Symbol Helper Hpp"
Cohesion: 1.0
Nodes (0): 

### Community 119 - "Dummy Cpp"
Cohesion: 1.0
Nodes (0): 

### Community 120 - "Bootstrap Ps1"
Cohesion: 1.0
Nodes (0): 

### Community 121 - "Modified Ports"
Cohesion: 1.0
Nodes (0): 

### Community 122 - "Create Prdiff"
Cohesion: 1.0
Nodes (0): 

### Community 123 - "Drop Admin"
Cohesion: 1.0
Nodes (0): 

### Community 124 - "Provision Entire"
Cohesion: 1.0
Nodes (0): 

### Community 125 - "Sysprep Ps1"
Cohesion: 1.0
Nodes (0): 

### Community 126 - "Deploy Pwsh"
Cohesion: 1.0
Nodes (0): 

### Community 127 - "Deploy Cuda"
Cohesion: 1.0
Nodes (0): 

### Community 128 - "Deploy Psexec"
Cohesion: 1.0
Nodes (0): 

### Community 129 - "Create Vmss"
Cohesion: 1.0
Nodes (0): 

### Community 130 - "Install Prerequisites"
Cohesion: 1.0
Nodes (0): 

### Community 131 - "Setup Vagrantmachines"
Cohesion: 1.0
Nodes (0): 

### Community 132 - "Vagrantfile"
Cohesion: 1.0
Nodes (0): 

### Community 133 - "Vagrantfile Box"
Cohesion: 1.0
Nodes (0): 

### Community 134 - "Rearrange Msvc"
Cohesion: 1.0
Nodes (0): 

### Community 135 - "Arith Osx"
Cohesion: 1.0
Nodes (0): 

### Community 136 - "Arith Win64"
Cohesion: 1.0
Nodes (0): 

### Community 137 - "Arith Win32"
Cohesion: 1.0
Nodes (0): 

### Community 138 - "Angle Commit"
Cohesion: 1.0
Nodes (0): 

### Community 139 - "Convert Lib Params Linux"
Cohesion: 1.0
Nodes (0): 

### Community 140 - "Generate Static Link Cmd Linux"
Cohesion: 1.0
Nodes (0): 

### Community 141 - "Generate Static Link Cmd Windows"
Cohesion: 1.0
Nodes (0): 

### Community 142 - "Generate Static Link Cmd Macos"
Cohesion: 1.0
Nodes (0): 

### Community 143 - "Convert Lib Params Macos"
Cohesion: 1.0
Nodes (0): 

### Community 144 - "Convert Lib Params Windows"
Cohesion: 1.0
Nodes (0): 

### Community 145 - "Cgnsconfig"
Cohesion: 1.0
Nodes (0): 

### Community 146 - "Fficonfig"
Cohesion: 1.0
Nodes (0): 

### Community 147 - "Modp B64 Data"
Cohesion: 1.0
Nodes (0): 

### Community 148 - "Libsecp256K1"
Cohesion: 1.0
Nodes (0): 

### Community 149 - "Openblas Common"
Cohesion: 1.0
Nodes (0): 

### Community 150 - "Linux"
Cohesion: 1.0
Nodes (0): 

### Community 151 - "Arith Linux64"
Cohesion: 1.0
Nodes (0): 

### Community 152 - "Magick Types"
Cohesion: 1.0
Nodes (0): 

### Community 153 - "B64 Static Config"
Cohesion: 1.0
Nodes (0): 

### Community 154 - "B64 Dynamic Config"
Cohesion: 1.0
Nodes (0): 

### Community 155 - "U2F Server"
Cohesion: 1.0
Nodes (0): 

### Community 156 - "Freeimageconfig Static"
Cohesion: 1.0
Nodes (0): 

### Community 157 - "Freeimageconfig Dynamic"
Cohesion: 1.0
Nodes (0): 

### Community 158 - "Cxx Standard"
Cohesion: 1.0
Nodes (1): The CXX standard level. If set, will add the required flags. If left at

### Community 159 - "Name Assign"
Cohesion: 1.0
Nodes (1): Assign a short name for the dataset class. By default will be the class name.

### Community 160 - "Is Valid Config Minimum"
Cohesion: 1.0
Nodes (1): `is_valid_config`                  This method is for minimum sanity check on co

### Community 161 - "Default Collate"
Cohesion: 1.0
Nodes (1): A default collate function that will handle torch.Tensor, pp.LieTensor and

### Community 162 - "Matchquality"
Cohesion: 1.0
Nodes (0): 

### Community 163 - "Depthquality"
Cohesion: 1.0
Nodes (0): 

### Community 164 - "Tartanvo"
Cohesion: 1.0
Nodes (0): 

### Community 165 - "Given Batch"
Cohesion: 1.0
Nodes (1): Given a batch of N observation (`TensorBundle`), the filter returns a boolean te

### Community 166 - "Given Sequence"
Cohesion: 1.0
Nodes (1): Given a sequence of frames, elaborate the trajectory (frame poses) and handle th

### Community 167 - "Given Initialize"
Cohesion: 1.0
Nodes (1): Given config, initialize a *mutable* context object that is preserved between op

### Community 168 - "Given Context"
Cohesion: 1.0
Nodes (1): Given context and argument, construct the optimization problem, solve it and ret

### Community 169 - "Returns Immediately"
Cohesion: 1.0
Nodes (1): Returns immediately, indicate the status of optimizer:         - true if there

### Community 170 - "Returns Concrete"
Cohesion: 1.0
Nodes (1): Returns the concrete type used for T_GraphInput. Raises TypeError if not explici

### Community 171 - "Returns Concrete"
Cohesion: 1.0
Nodes (1): Returns the concrete type used for T_GraphOutput. Raises TypeError if not explic

### Community 172 - "Should Implemented"
Cohesion: 1.0
Nodes (1): This function should be implemented by the user.         It should return the ja

### Community 173 - "Returns Jacobian"
Cohesion: 1.0
Nodes (1): Returns the jacobian of the model's previous forward call with respect to model'

### Community 174 - "Verifies Whether"
Cohesion: 1.0
Nodes (1): Verifies whether the input J_analytic coincides with autograd jacobian of the pr

### Community 175 - "Given Pixel Uv"
Cohesion: 1.0
Nodes (1): Given a pixel_uv (Nx2) tensor, retrieve the pixel values (1, N) from scalar_map

### Community 176 - "Linear Linearized"
Cohesion: 1.0
Nodes (1): r'''         Linear/linearized system output matrix.          .. math::

### Community 177 - "Linear Linearized"
Cohesion: 1.0
Nodes (1): r'''         Linear/Linearized system observation matrix.          .. math::

### Community 178 - "Covariance System"
Cohesion: 1.0
Nodes (1): r'''         The covariance of system transition noise.

### Community 179 - "Covariance System"
Cohesion: 1.0
Nodes (1): r'''         The covariance of system transition noise.

### Community 180 - "Covariance System"
Cohesion: 1.0
Nodes (1): r'''         The covariance of system observation noise.

## Knowledge Gaps
- **546 isolated node(s):** `Evaluation: mask quality metrics + VIO trajectory metrics.  Mask metrics (on VIO`, `Compute IoU, Precision, Recall, F1 for a batch.      Args:         pred_mask: [B`, `Evaluate mask quality over a full dataloader.`, `Evaluate IMU preintegration quality (RTE, ROE).`, `Compute ATE RMSE and RPE using the evo package.      Args:         est_traj_file` (+541 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Addposhvcpkgtopowershellprofile Ps1`** (2 nodes): `addPoshVcpkgToPowershellProfile.ps1`, `findExistingImportModuleDirectives()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Deploy Windows`** (2 nodes): `deploy-windows-sdks.ps1`, `InstallWindowsDK()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Create Image`** (2 nodes): `create-image.ps1`, `Invoke-ScriptWithPrefix()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Deploy Mpi`** (2 nodes): `deploy-mpi.ps1`, `InstallMpi()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Deploy Install`** (2 nodes): `deploy-install-disk.ps1`, `New-PhysicalDisk()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Deploy Inteloneapi`** (2 nodes): `deploy-inteloneapi.ps1`, `InstallInteloneAPI()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Disk Space`** (2 nodes): `disk-space.ps1`, `Format-Size()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Deploy Visual`** (2 nodes): `deploy-visual-studio.ps1`, `InstallVisualStudio()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Qtdeploy Ps1`** (2 nodes): `qtdeploy.ps1`, `deployPluginsIfQt()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Openni2Deploy Ps1`** (2 nodes): `openni2deploy.ps1`, `deployOpenNI2()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Check Getcontext`** (2 nodes): `check_getcontext.cc`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Gettimeofday`** (2 nodes): `gettimeofday.h`, `gettimeofday()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Magnumdeploy Ps1`** (2 nodes): `magnumdeploy.ps1`, `deployPluginsIfMagnum()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Decomp Set Endian`** (2 nodes): `decomp.c`, `set_endian()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Generatefeatures Ps1`** (2 nodes): `generateFeatures.ps1`, `GetDescription()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `K4Adeploy Ps1`** (2 nodes): `k4adeploy.ps1`, `deployAzureKinectSensorSDK()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Optimization Ablation Run Frame`** (2 nodes): `Optimization_Ablation.py`, `run_frame()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Config Tartanvo Test Tartanvo Config`** (2 nodes): `test_config_tartanvo.py`, `test_tartanvo_config()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Config Macvo Test Macvo Config`** (2 nodes): `test_config_macvo.py`, `test_macvo_config()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Config Sequence Test Sequence Cfg`** (2 nodes): `test_config_sequence.py`, `test_sequence_cfg()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Performance Macvo Test Macvo Performance`** (2 nodes): `test_performance_macvo.py`, `test_macvo_performance()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Stereo Depth Test Matching`** (2 nodes): `test_stereo_depth.py`, `test_matching()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Config Loadable`** (2 nodes): `test_config_loadable.py`, `test_config_loadable()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Matching`** (2 nodes): `test_matching.py`, `test_matching()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Frontend`** (2 nodes): `test_frontend.py`, `test_frontend()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Experiment Dpvo Execute Experiment`** (2 nodes): `Experiment_DPVO.py`, `execute_experiment()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Experiment Macvo Execute Experiment`** (2 nodes): `Experiment_MACVO.py`, `execute_experiment()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Experiment Tartanvo Execute Experiment`** (2 nodes): `Experiment_TartanVO.py`, `execute_experiment()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Matchestimator`** (2 nodes): `MatchEstimator.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Tartanvodisparity Avgerror`** (2 nodes): `TartanVODisparity_AvgError.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Submission Get Cfg`** (2 nodes): `submission.py`, `get_cfg()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Chartdir`** (1 nodes): `chartdir.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Solve Via`** (1 nodes): `Solve H x = b via Cholesky decomposition.          Args:             H: [B, 6, 6`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Cxx Standard`** (1 nodes): `The CXX standard level. If set, will add the required flags. If left         at`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Make Changelog`** (1 nodes): `make_changelog.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Libsize`** (1 nodes): `libsize.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Eval Call`** (1 nodes): `test_eval_call.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Cxx Standard`** (1 nodes): `The CXX standard level. If set, will add the required flags. If left         at`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `X11Glcontext`** (1 nodes): `X11GlContext.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Symbol Helper Hpp`** (1 nodes): `symbol_helper.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Dummy Cpp`** (1 nodes): `dummy.cpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Bootstrap Ps1`** (1 nodes): `bootstrap.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Modified Ports`** (1 nodes): `test-modified-ports.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Create Prdiff`** (1 nodes): `Create-PRDiff.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Drop Admin`** (1 nodes): `drop-to-admin-user-prefix.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Provision Entire`** (1 nodes): `provision-entire-image.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Sysprep Ps1`** (1 nodes): `sysprep.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Deploy Pwsh`** (1 nodes): `deploy-pwsh.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Deploy Cuda`** (1 nodes): `deploy-cuda.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Deploy Psexec`** (1 nodes): `deploy-psexec.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Create Vmss`** (1 nodes): `create-vmss.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Install Prerequisites`** (1 nodes): `Install-Prerequisites.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Setup Vagrantmachines`** (1 nodes): `Setup-VagrantMachines.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Vagrantfile`** (1 nodes): `Vagrantfile-vm.rb`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Vagrantfile Box`** (1 nodes): `Vagrantfile-box.rb`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Rearrange Msvc`** (1 nodes): `rearrange-msvc-drop-layout.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Arith Osx`** (1 nodes): `arith_osx.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Arith Win64`** (1 nodes): `arith_win64.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Arith Win32`** (1 nodes): `arith_win32.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Angle Commit`** (1 nodes): `angle_commit.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Convert Lib Params Linux`** (1 nodes): `convert_lib_params_linux.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Generate Static Link Cmd Linux`** (1 nodes): `generate_static_link_cmd_linux.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Generate Static Link Cmd Windows`** (1 nodes): `generate_static_link_cmd_windows.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Generate Static Link Cmd Macos`** (1 nodes): `generate_static_link_cmd_macos.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Convert Lib Params Macos`** (1 nodes): `convert_lib_params_macos.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Convert Lib Params Windows`** (1 nodes): `convert_lib_params_windows.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Cgnsconfig`** (1 nodes): `cgnsconfig.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Fficonfig`** (1 nodes): `fficonfig.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Modp B64 Data`** (1 nodes): `modp_b64_data.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Libsecp256K1`** (1 nodes): `libsecp256k1-config.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Openblas Common`** (1 nodes): `openblas_common.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Linux`** (1 nodes): `config.linux.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Arith Linux64`** (1 nodes): `arith_linux64.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Magick Types`** (1 nodes): `magick_types.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `B64 Static Config`** (1 nodes): `b64_static_config.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `B64 Dynamic Config`** (1 nodes): `b64_dynamic_config.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `U2F Server`** (1 nodes): `u2f-server-version.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Freeimageconfig Static`** (1 nodes): `FreeImageConfig-static.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Freeimageconfig Dynamic`** (1 nodes): `FreeImageConfig-dynamic.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Cxx Standard`** (1 nodes): `The CXX standard level. If set, will add the required flags. If left at`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Name Assign`** (1 nodes): `Assign a short name for the dataset class. By default will be the class name.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Is Valid Config Minimum`** (1 nodes): ``is_valid_config`                  This method is for minimum sanity check on co`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Default Collate`** (1 nodes): `A default collate function that will handle torch.Tensor, pp.LieTensor and`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Matchquality`** (1 nodes): `MatchQuality.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Depthquality`** (1 nodes): `DepthQuality.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Tartanvo`** (1 nodes): `TartanVO.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Given Batch`** (1 nodes): `Given a batch of N observation (`TensorBundle`), the filter returns a boolean te`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Given Sequence`** (1 nodes): `Given a sequence of frames, elaborate the trajectory (frame poses) and handle th`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Given Initialize`** (1 nodes): `Given config, initialize a *mutable* context object that is preserved between op`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Given Context`** (1 nodes): `Given context and argument, construct the optimization problem, solve it and ret`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Returns Immediately`** (1 nodes): `Returns immediately, indicate the status of optimizer:         - true if there`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Returns Concrete`** (1 nodes): `Returns the concrete type used for T_GraphInput. Raises TypeError if not explici`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Returns Concrete`** (1 nodes): `Returns the concrete type used for T_GraphOutput. Raises TypeError if not explic`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Should Implemented`** (1 nodes): `This function should be implemented by the user.         It should return the ja`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Returns Jacobian`** (1 nodes): `Returns the jacobian of the model's previous forward call with respect to model'`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Verifies Whether`** (1 nodes): `Verifies whether the input J_analytic coincides with autograd jacobian of the pr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Given Pixel Uv`** (1 nodes): `Given a pixel_uv (Nx2) tensor, retrieve the pixel values (1, N) from scalar_map`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Linear Linearized`** (1 nodes): `r'''         Linear/linearized system output matrix.          .. math::`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Linear Linearized`** (1 nodes): `r'''         Linear/Linearized system observation matrix.          .. math::`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Covariance System`** (1 nodes): `r'''         The covariance of system transition noise.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Covariance System`** (1 nodes): `r'''         The covariance of system transition noise.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Covariance System`** (1 nodes): `r'''         The covariance of system observation noise.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Timer` connect `Init   Estimate` to `Init   Frame`, `Cpp Pangolin`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._
- **Why does `TartanStereoVOMatch` connect `Init   Forward` to `Init   Estimate`?**
  _High betweenness centrality (0.010) - this node is a cross-community bridge._
- **Are the 73 inferred relationships involving `Timer` (e.g. with `MACVO` and `The main process that continuously running to manage different modules in MAC-VO`) actually correct?**
  _`Timer` has 73 INFERRED edges - model-reasoned connections that need verification._
- **Are the 53 inferred relationships involving `IStereoDepth` (e.g. with `Matplotlib_Visualizer` and `Register a classmethod of Matplotlib Visualizer`) actually correct?**
  _`IStereoDepth` has 53 INFERRED edges - model-reasoned connections that need verification._
- **Are the 38 inferred relationships involving `IMatcher` (e.g. with `Matplotlib_Visualizer` and `Register a classmethod of Matplotlib Visualizer`) actually correct?**
  _`IMatcher` has 38 INFERRED edges - model-reasoned connections that need verification._
- **Are the 46 inferred relationships involving `StereoFrame` (e.g. with `IDataTransform` and `NoTransform`) actually correct?**
  _`StereoFrame` has 46 INFERRED edges - model-reasoned connections that need verification._
- **Are the 46 inferred relationships involving `StereoData` (e.g. with `IDataTransform` and `NoTransform`) actually correct?**
  _`StereoData` has 46 INFERRED edges - model-reasoned connections that need verification._