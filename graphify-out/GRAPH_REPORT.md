# Graph Report - .  (2026-04-10)

## Corpus Check
- 1665 files · ~404,392,773 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 6438 nodes · 10663 edges · 182 communities detected
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 1079 edges (avg confidence: 0.5)
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
- `DynaMask V2.5 training script (single-phase, 3-loss stack).` --uses--> `OfflineDebugCallback`  [INFERRED]
  dynamask_vio/train.py → dynamask_vio/offline_debug.py
- `Lightning module for V2.5 single-phase training.` --uses--> `OfflineDebugCallback`  [INFERRED]
  dynamask_vio/train.py → dynamask_vio/offline_debug.py
- `Backward-compatible hook used by debug callbacks.` --uses--> `OfflineDebugCallback`  [INFERRED]
  dynamask_vio/train.py → dynamask_vio/offline_debug.py
- `Single-phase datamodule with optional multi-dataset concatenation.` --uses--> `OfflineDebugCallback`  [INFERRED]
  dynamask_vio/train.py → dynamask_vio/offline_debug.py
- `Load poses from pose_lcam_front.txt.         Format: each line is tx ty tz qx qy` --uses--> `VisualAugmentor`  [INFERRED]
  dynamask_vio/data/tartanair_dataset.py → dynamask_vio/data/augmentations.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.01
Nodes (161): deprecated_call(), pytest.deprecated_call() seems broken in pytest<3.9.x; concretely, it     doesn', # TODO: Remove this when testing requires pytest>=3.9., bind_ConstructorStats(), PYBIND11_MODULE(), load(), PYBIND11_NAMESPACE_BEGIN(), reserve_maybe() (+153 more)

### Community 1 - "Community 1"
Cohesion: 0.01
Nodes (171): BlackBird, Reference: https://github.com/uzh-rpg/learned_inertial_model_odometry/blob/maste, Updates the data (imu / velocity) based on the required mode.         :param coo, Sets the ground truth orientation based on the provided rotation.         :param, broadcast_inputs(), check_broadcastable(), Automatic broadcasting of missing dimensions, CasADIEKF (+163 more)

### Community 2 - "Community 2"
Cohesion: 0.01
Nodes (141): main(), sample(), main(), sample(), main(), sample(), main(), sample() (+133 more)

### Community 3 - "Community 3"
Cohesion: 0.01
Nodes (125): BayerOutputFormat(), DebayerVideo(), DownsampleDebayer(), GrabNewest(), GrabNext(), PitchedImageCopy(), ProcessImage(), ProcessStreams() (+117 more)

### Community 4 - "Community 4"
Cohesion: 0.01
Nodes (149): ABC, TartanVO, ConfigTestableSubclass, Plots Gaussian distribution for some confidence interval and compare the  predic, CUDAGraph_FlowFormerCovFrontend, CUDAGraphHandler, estimate_depth(), estimate_pair() (+141 more)

### Community 5 - "Community 5"
Cohesion: 0.02
Nodes (116): For the purpose of training and inferering     1. Abandon the features of the la, For the purpose of training and inferering     1. Abandon the features of the la, SeqDataset, SeqeuncesDataset, SeqInfDataset, build_intrinsic(), EuRoC_Sequence, EuRoC_StereoSequence (+108 more)

### Community 6 - "Community 6"
Cohesion: 0.01
Nodes (108): PSMNet, Chain, To have SML-like function chaining operator      (f1 >> f2)(x) = f2(f1(x)), ConfigTestable, custom_collate(), motion_collate(), motion_collate_data(), padding_collate() (+100 more)

### Community 7 - "Community 7"
Cohesion: 0.01
Nodes (54): call(), cast(), multiple_values_error(), nameless_argument_error(), none(), process(), constexpr_first(), first() (+46 more)

### Community 8 - "Community 8"
Cohesion: 0.01
Nodes (81): GlobalConfig, Capture, doc(), gc_collect(), _make_explanation(), msg(), Output, pytest_assertrepr_compare() (+73 more)

### Community 9 - "Community 9"
Cohesion: 0.02
Nodes (81): AttentionLayer, BroadMultiHeadAttention, LinearPositionEmbeddingSine(), MultiHeadAttention, PositionalEncoding2D, :param channels: The last dimension of the tensor you want to apply pos emb to., :param tensor: A 4d tensor of size (batch_size, x, y, ch)         :return: Posit, # NOTE: For some reason, replace this with SDPA (FlashAttention 2) backend makes (+73 more)

### Community 10 - "Community 10"
Cohesion: 0.02
Nodes (75): CheckPrintClearError(), EvalExec(), PyInterpreter(), connect(), connect_scoped(), connection, connection_blocker, copy_on_write (+67 more)

### Community 11 - "Community 11"
Cohesion: 0.02
Nodes (54): build_ext, changeStructure(), loadFeatures(), main(), testDatabase(), testVocCreation(), wait(), HighestPriScheme() (+46 more)

### Community 12 - "Community 12"
Cohesion: 0.02
Nodes (50): ba(), block_matmul(), block_solve(), CholeskySolver, disp_retr(), pose_retr(), block matrix multiply, safe_scatter_add_mat() (+42 more)

### Community 13 - "Community 13"
Cohesion: 0.02
Nodes (62): DeepPatchVO, # NOTE: DPVO will perform /255 operation internally., GatedResidual, GradClip, GradientClip, GradientZero, GradMag, GradZero (+54 more)

### Community 14 - "Community 14"
Cohesion: 0.02
Nodes (89): glColorMask(), glCullFace(), glDepthMask(), glDisable(), glEnable(), glLineWidth(), glPointSize(), glShadeModel() (+81 more)

### Community 15 - "Community 15"
Cohesion: 0.02
Nodes (77): main(), sample(), main(), sample(), cast(), cast_impl(), glDraw_x0(), glDraw_y0() (+69 more)

### Community 16 - "Community 16"
Cohesion: 0.02
Nodes (59): get_cmake_dir(), get_include(), Return the path to the pybind11 CMake module directory., Return the path to the pybind11 include directory. The historical "user"     arg, AppendColumns(), ReadRow(), SkipLines(), Expand() (+51 more)

### Community 17 - "Community 17"
Cohesion: 0.04
Nodes (124): equal(), main(), my_stbtt_initfont(), my_stbtt_print(), stbrp_init_target(), stbrp_pack_rects(), stbtt__add_point(), stbtt_BakeFontBitmap() (+116 more)

### Community 18 - "Community 18"
Cohesion: 0.02
Nodes (51): AdjustScale(), AdjustTranslation(), FixSelection(), glRenderOverlay(), glRenderTexture(), ImageToScreen(), ImageViewHandler(), Keyboard() (+43 more)

### Community 19 - "Community 19"
Cohesion: 0.03
Nodes (66): AirIMUCorrector, _assert_airimu_load_clean(), CNNEncoder, load_airimu_weights(), AirIMU CodeNet-compatible IMU correction module.  This module mirrors the core A, Broadcast interval-wise features back to per-sample sequence slots., 1D CNN encoder used by AirIMU CodeNet., Fail loudly on suspicious state-dict mismatches. (+58 more)

### Community 20 - "Community 20"
Cohesion: 0.02
Nodes (45): cast(), localtime_thread_safe(), clear_instance(), deregister_instance(), enable_buffer_protocol(), enable_dynamic_attributes(), get_fully_qualified_tp_name(), make_default_metaclass() (+37 more)

### Community 21 - "Community 21"
Cohesion: 0.03
Nodes (46): ate(), evaluate(), run(), show_image(), video_iterator(), _append_jsonl(), OfflineDebugCallback, OfflineModelWatchCallback (+38 more)

### Community 22 - "Community 22"
Cohesion: 0.09
Nodes (44): Act3, Act4, Adj, AdjT, Exp, FromVec, GroupOp, Inv (+36 more)

### Community 23 - "Community 23"
Cohesion: 0.08
Nodes (30): AnalyticModule, FactorGraph, Analytic_ICP_TwoframePGO, Analytic_Reproj_TwoFramePGO, Analytic_ReprojDisp_TwoFramePGO, GraphInput, GraphOutput, ICP_TwoframePGO (+22 more)

### Community 24 - "Community 24"
Cohesion: 0.04
Nodes (35): backward(), CorrLayer, cupy_kernel(), cupy_launch(), forward(), _FunctionCorrelation, ModuleCorrelation, PatchLayer (+27 more)

### Community 25 - "Community 25"
Cohesion: 0.04
Nodes (20): GetPixelFormat(), LoadExr(), OpenEXRPixelType(), SaveExr(), SetOpenEXRChannels(), StdIStream, GetMJpegOffsets(), LoadJpg() (+12 more)

### Community 26 - "Community 26"
Cohesion: 0.06
Nodes (35): Add(), AttachColour(), AttachDepth(), Bind(), bind_gl(), CheckResize(), CopyFrom(), Delete() (+27 more)

### Community 27 - "Community 27"
Cohesion: 0.05
Nodes (35): PThreadConditionVariable, ConfigureColorNode(), ConfigureDepthNode(), ConfigureNodes(), DepthSenseContext(), DepthSenseVideo(), DeviceClosing(), EventLoop() (+27 more)

### Community 28 - "Community 28"
Cohesion: 0.05
Nodes (27): Core method for IVisualOdometry. This method handles the incoming frames and per, Provides the VisualMap built across multiple calls of .run(...)., You can define additional operations on terminate. For instance, smoothing traje, ColoredTqdm, create(), data(), __get_curr_time(), __get_git_version() (+19 more)

### Community 29 - "Community 29"
Cohesion: 0.05
Nodes (40): IMUAugmentor, All data augmentations — visual and IMU.  Visual augmentations are applied ident, Augments IMU window data., Args:             imu_window: [N, 7] — (dt, ax, ay, az, gx, gy, gz), Augments a pair of images + mask consistently., Args:             img_prev, img_curr: [H, W, 3] float32 in [0, 255], Apply identical color jitter to both images (expects [0, 255] range)., VisualAugmentor (+32 more)

### Community 30 - "Community 30"
Cohesion: 0.04
Nodes (17): AutoScalingBundle, DenseEdge_Multi, deserialize(), EdgeLike, init(), Provide graph infrastructure (node with arbitrary features and edges), An arbitrary one-to-multi mapping relationship., In case of one-to-multi mapping with continuous index. Can significantly reduce (+9 more)

### Community 31 - "Community 31"
Cohesion: 0.06
Nodes (16): CNNcorrection, CNNEncoder, CNNPOS, The input feature shape [B, F, Duration, 6], Only correct the accelerometer, CNNEncoder, CodeNet, CodeNetKITTI (+8 more)

### Community 32 - "Community 32"
Cohesion: 0.03
Nodes (24): BreaksBase, BreaksTramp, Chimera, Dog, Hamster, Pet, ProtectedA, ProtectedB (+16 more)

### Community 33 - "Community 33"
Cohesion: 0.05
Nodes (25): compare(), attribute_iterator, node_iterator, memory_pool, parse_error, parse_node_contents(), copy_and_expand_chars(), copy_chars() (+17 more)

### Community 34 - "Community 34"
Cohesion: 0.05
Nodes (33): cropping and resizing, perform augmentation on RGB-D video, RGBDAugmentor, depth_read(), image_read(), Base class for RGBD dataset, compute optical flow distance between all pairs of frames, return training video (+25 more)

### Community 35 - "Community 35"
Cohesion: 0.05
Nodes (44): auto_cpp_level(), build_ext, cxx_std(), has_flag(), intree_extensions(), naive_recompile(), no_recompile(), ParallelCompile (+36 more)

### Community 36 - "Community 36"
Cohesion: 0.07
Nodes (41): android_app_destroy(), android_app_entry(), android_app_free(), android_app_post_exec_cmd(), android_app_pre_exec_cmd(), android_app_read_cmd(), android_app_set_activity_state(), android_app_set_input() (+33 more)

### Community 37 - "Community 37"
Cohesion: 0.05
Nodes (7): AutoScalingTensor, Provides `AutoScalingTensor` class, an efficient tensor wrapper for data accumul, A circular buffer tensor., push a batch of values into the CircularTensor. Will trigger a cache eviction, Actual underlying circular buffer push algorithm., tensor(), TensorQueue

### Community 38 - "Community 38"
Cohesion: 0.05
Nodes (10): auxiliaries(), data(), data_t(), index_at(), index_at_t(), offset_at(), offset_at_t(), Tests fix for #685 - ndarray shouldn't go to std::string overload (+2 more)

### Community 39 - "Community 39"
Cohesion: 0.08
Nodes (28): DeinitBuffers(), DeinitDevice(), DeinitStream(), DropNFrames(), GetAnalogBlackLevel(), GetExposure(), GetGain(), GetGamma() (+20 more)

### Community 40 - "Community 40"
Cohesion: 0.07
Nodes (31): create_and_destroy(), PYBIND11_OVERRIDE(), PyTF6(), PyTF7(), Tests py::init_factory() wrapper with various upcasting and downcasting returns, Tests py::init_factory() wrapper around various ways of returning the object, Tests py::init_factory() wrapper with value conversions and alias types, Tests init factory functions with dual main/alias factory functions (+23 more)

### Community 41 - "Community 41"
Cohesion: 0.07
Nodes (14): Dc1394ColorCodingToString(), Dc1394ModeDetails(), FirewireVideo(), get_firewire_mode(), GetNewest(), GetNext(), GrabNewest(), GrabNext() (+6 more)

### Community 42 - "Community 42"
Cohesion: 0.05
Nodes (29): Tests the ability to pass bytes to C++ string-accepting functions.  Note that th, Tests support for C++17 string_view arguments and return values, Tests unicode conversion and error reporting., Issue #929 - out-of-range integer values shouldn't be accepted, # TODO: Avoid DeprecationWarning in `PyLong_AsLong` (and similar), # TODO: Avoid DeprecationWarning in `PyLong_AsLong` (and similar), std::pair <-> tuple & std::tuple <-> tuple, Casters produced with PYBIND11_TYPE_CASTER() should convert nullptr to None (+21 more)

### Community 43 - "Community 43"
Cohesion: 0.06
Nodes (6): handle_configure(), MakeCurrent(), Resize(), SwapBuffers(), WaylandDisplay(), WaylandWindow()

### Community 44 - "Community 44"
Cohesion: 0.06
Nodes (4): C++ default and converting constructors are equivalent to type calls in Python, Tests implicit casting when assigning or appending to dicts and lists., test_constructors(), test_implicit_casting()

### Community 45 - "Community 45"
Cohesion: 0.08
Nodes (22): _as_tuple(), _differentiable_outputs(), get_analytical_jacobian(), get_numerical_jacobian(), gradcheck(), gradgradcheck(), iter_tensors(), make_jacobian() (+14 more)

### Community 46 - "Community 46"
Cohesion: 0.1
Nodes (15): AddDevice(), AddStream(), FindOpenNI2Mode(), GrabNewest(), GrabNext(), InitialiseOpenNI(), OpenNi2Video(), PrintOpenNI2Modes() (+7 more)

### Community 47 - "Community 47"
Cohesion: 0.09
Nodes (9): assert_equal(), dt_fmt(), packed_dtype_fmt(), partial_dtype_fmt(), partial_ld_offset(), partial_nested_fmt(), simple_dtype_fmt(), test_dtype() (+1 more)

### Community 48 - "Community 48"
Cohesion: 0.1
Nodes (18): Close(), Grab(), GrabNewest(), GrabNext(), InitialiseRecorder(), IsRecording(), Open(), Record() (+10 more)

### Community 49 - "Community 49"
Cohesion: 0.11
Nodes (27): computeHash(), deployBinary(), fix_binary(), fix_dependency(), fix_main_binaries(), get_dependencies(), getMutex(), is_loader_path_lib() (+19 more)

### Community 50 - "Community 50"
Cohesion: 0.1
Nodes (23): compute_reprojection_jacobian(), differentiable_ba(), gradient_clip(), _GradientClipFn, Differentiable Two-Frame Bundle Adjustment — training-only module.  Zero learnab, Exponential map so(3) -> SO(3). omega: [B, 3] -> [B, 3, 3].      Rodrigues formu, Logarithmic map SO(3) -> so(3). R: [B, 3, 3] -> [B, 3].      Inverse Rodrigues w, Update SE(3) pose via retraction (left-multiply by exp(δξ)).      Convention: δξ (+15 more)

### Community 51 - "Community 51"
Cohesion: 0.08
Nodes (24): build(), docs(), lint(), make_changelog(), Lint the codebase (except for clang-format/tidy)., Lint the codebase (except for clang-format/tidy)., Run the tests (requires a compiler)., Run the tests (requires a compiler). (+16 more)

### Community 52 - "Community 52"
Cohesion: 0.13
Nodes (12): cmd_line_arg_is_option_flag(), flag_is_short(), fmt_ostream(), fmt_string(), get_definition_for_long_flag(), get_definition_for_short_flag(), is_valid_flag_definition(), known_long_flag() (+4 more)

### Community 53 - "Community 53"
Cohesion: 0.1
Nodes (21): _python_to_cpp_to_python(), _python_to_cpp_to_python_from_threads(), Calls different C++ functions that come back to Python., Calls different C++ functions that come back to Python, from Python threads., # TODO: FIXME, sometimes returns -11 (segfault) instead of 0 on macOS Python 3.9, Makes sure there is no GIL deadlock when running in a thread.      It runs in a, # TODO: FIXME on macOS Python 3.9, Makes sure there is no GIL deadlock when running in a thread multiple times in p (+13 more)

### Community 54 - "Community 54"
Cohesion: 0.12
Nodes (18): asNamespace(), __build_dynamic_config(), IncludeLoader, load_config(), LoadFrom, namespace_to_cfgnode(), Design for Flowformer. Flowformer uses yacs.config.CfgNode as config container., close() (+10 more)

### Community 55 - "Community 55"
Cohesion: 0.21
Nodes (15): exportGroupsToShape(), InitMaterial(), LoadMtl(), LoadObj(), LoadObjWithCallback(), operator(), parseInt(), parseRawTriple() (+7 more)

### Community 56 - "Community 56"
Cohesion: 0.12
Nodes (8): Store the image into the frame buffer, Once we keyframe an image, we can safely cache all images          before & incl, Add frames to the image-retrieval database, Record the loop closure so we don't have redundant edges, Check that we've retrieved <num_repeat> consecutive frames, Keep popping off the queue until the it is empty          or we find a positive, Pop retrived pairs off the queue. Return if they have non-trivial score, RetrievalDBOW

### Community 57 - "Community 57"
Cohesion: 0.2
Nodes (11): Close(), CreateStream(), CreateVideoCodecContext(), FfmpegVideoOutput(), FfmpegVideoOutputStream(), Flush(), Initialise(), StartStream() (+3 more)

### Community 58 - "Community 58"
Cohesion: 0.16
Nodes (3): DisableRC4(), Test-RegistryValueForFipsSettings(), Write-Log()

### Community 59 - "Community 59"
Cohesion: 0.21
Nodes (9): DocumentationUpdate, Get-AuthHeader(), Get-MergedPullRequests(), Get-PullRequestFileMap(), PortUpdate, PRFileMap, Select-UpdatedPorts(), Select-Version() (+1 more)

### Community 60 - "Community 60"
Cohesion: 0.31
Nodes (6): DynaMaskONNXWrapper, export(), _load_config(), main(), Export DynaMask V2.5 model to ONNX., Flatten dict outputs into a deterministic ONNX output tuple.

### Community 61 - "Community 61"
Cohesion: 0.5
Nodes (7): download_airimu_weights(), download_raft_weights(), _download_url(), _extract_first_matching(), main(), Download pretrained weights required for DynaMask V2.5., _verify_torch_load()

### Community 62 - "Community 62"
Cohesion: 0.33
Nodes (3): CMakeDocumentation, FinalDocFile(), RelativeUnixPathTo()

### Community 63 - "Community 63"
Cohesion: 0.33
Nodes (1): Logger

### Community 64 - "Community 64"
Cohesion: 0.29
Nodes (4): evaluateROE(), evaluateRPE(), Evaluates error of rotation, Evaluates error of se(3) pose

### Community 65 - "Community 65"
Cohesion: 0.57
Nodes (5): cov_loss(), final_cov_loss(), flow_loss(), sequence_loss(), sequence_metric()

### Community 66 - "Community 66"
Cohesion: 0.4
Nodes (0): 

### Community 67 - "Community 67"
Cohesion: 0.7
Nodes (4): Get-TempFilePath(), InstallMSI(), InstallZip(), PrintMsiExitCodeMessage()

### Community 68 - "Community 68"
Cohesion: 0.7
Nodes (4): GeneratePort(), GeneratePortDependency(), GeneratePortManifest(), GeneratePortName()

### Community 69 - "Community 69"
Cohesion: 0.4
Nodes (4): CalculateOnePatch(), depth_mean: (N,) torch.float     depth_var:  (N,) torch.float     selector_dist:, depth_mean: (N,) torch.float     depth_var:  (N,) torch.float     selector_dist:, SimulateOnePatch()

### Community 70 - "Community 70"
Cohesion: 0.5
Nodes (0): 

### Community 71 - "Community 71"
Cohesion: 0.83
Nodes (3): gen_all_file_strings(), getFiles(), main()

### Community 72 - "Community 72"
Cohesion: 0.5
Nodes (0): 

### Community 73 - "Community 73"
Cohesion: 0.5
Nodes (0): 

### Community 74 - "Community 74"
Cohesion: 0.83
Nodes (3): EvaluateSequences(), EvaluateSequencesAvg(), mean()

### Community 75 - "Community 75"
Cohesion: 0.67
Nodes (0): 

### Community 76 - "Community 76"
Cohesion: 1.0
Nodes (2): GetFlops(), main()

### Community 77 - "Community 77"
Cohesion: 0.67
Nodes (0): 

### Community 78 - "Community 78"
Cohesion: 0.67
Nodes (0): 

### Community 79 - "Community 79"
Cohesion: 0.67
Nodes (0): 

### Community 80 - "Community 80"
Cohesion: 1.0
Nodes (0): 

### Community 81 - "Community 81"
Cohesion: 1.0
Nodes (0): 

### Community 82 - "Community 82"
Cohesion: 1.0
Nodes (0): 

### Community 83 - "Community 83"
Cohesion: 1.0
Nodes (0): 

### Community 84 - "Community 84"
Cohesion: 1.0
Nodes (0): 

### Community 85 - "Community 85"
Cohesion: 1.0
Nodes (0): 

### Community 86 - "Community 86"
Cohesion: 1.0
Nodes (0): 

### Community 87 - "Community 87"
Cohesion: 1.0
Nodes (0): 

### Community 88 - "Community 88"
Cohesion: 1.0
Nodes (0): 

### Community 89 - "Community 89"
Cohesion: 1.0
Nodes (0): 

### Community 90 - "Community 90"
Cohesion: 1.0
Nodes (0): 

### Community 91 - "Community 91"
Cohesion: 1.0
Nodes (0): 

### Community 92 - "Community 92"
Cohesion: 1.0
Nodes (0): 

### Community 93 - "Community 93"
Cohesion: 2.0
Nodes (0): 

### Community 94 - "Community 94"
Cohesion: 1.0
Nodes (0): 

### Community 95 - "Community 95"
Cohesion: 1.0
Nodes (0): 

### Community 96 - "Community 96"
Cohesion: 1.0
Nodes (0): 

### Community 97 - "Community 97"
Cohesion: 1.0
Nodes (0): 

### Community 98 - "Community 98"
Cohesion: 1.0
Nodes (0): 

### Community 99 - "Community 99"
Cohesion: 1.0
Nodes (0): 

### Community 100 - "Community 100"
Cohesion: 1.0
Nodes (0): 

### Community 101 - "Community 101"
Cohesion: 1.0
Nodes (0): 

### Community 102 - "Community 102"
Cohesion: 1.0
Nodes (0): 

### Community 103 - "Community 103"
Cohesion: 1.0
Nodes (0): 

### Community 104 - "Community 104"
Cohesion: 1.0
Nodes (0): 

### Community 105 - "Community 105"
Cohesion: 1.0
Nodes (0): 

### Community 106 - "Community 106"
Cohesion: 1.0
Nodes (0): 

### Community 107 - "Community 107"
Cohesion: 1.0
Nodes (0): 

### Community 108 - "Community 108"
Cohesion: 1.0
Nodes (0): 

### Community 109 - "Community 109"
Cohesion: 1.0
Nodes (0): 

### Community 110 - "Community 110"
Cohesion: 1.0
Nodes (0): 

### Community 111 - "Community 111"
Cohesion: 1.0
Nodes (0): 

### Community 112 - "Community 112"
Cohesion: 1.0
Nodes (1): Solve H x = b via Cholesky decomposition.          Args:             H: [B, 6, 6

### Community 113 - "Community 113"
Cohesion: 1.0
Nodes (1): The CXX standard level. If set, will add the required flags. If left         at

### Community 114 - "Community 114"
Cohesion: 1.0
Nodes (0): 

### Community 115 - "Community 115"
Cohesion: 1.0
Nodes (0): 

### Community 116 - "Community 116"
Cohesion: 1.0
Nodes (0): 

### Community 117 - "Community 117"
Cohesion: 1.0
Nodes (1): The CXX standard level. If set, will add the required flags. If left         at

### Community 118 - "Community 118"
Cohesion: 1.0
Nodes (0): 

### Community 119 - "Community 119"
Cohesion: 1.0
Nodes (0): 

### Community 120 - "Community 120"
Cohesion: 1.0
Nodes (0): 

### Community 121 - "Community 121"
Cohesion: 1.0
Nodes (0): 

### Community 122 - "Community 122"
Cohesion: 1.0
Nodes (0): 

### Community 123 - "Community 123"
Cohesion: 1.0
Nodes (0): 

### Community 124 - "Community 124"
Cohesion: 1.0
Nodes (0): 

### Community 125 - "Community 125"
Cohesion: 1.0
Nodes (0): 

### Community 126 - "Community 126"
Cohesion: 1.0
Nodes (0): 

### Community 127 - "Community 127"
Cohesion: 1.0
Nodes (0): 

### Community 128 - "Community 128"
Cohesion: 1.0
Nodes (0): 

### Community 129 - "Community 129"
Cohesion: 1.0
Nodes (0): 

### Community 130 - "Community 130"
Cohesion: 1.0
Nodes (0): 

### Community 131 - "Community 131"
Cohesion: 1.0
Nodes (0): 

### Community 132 - "Community 132"
Cohesion: 1.0
Nodes (0): 

### Community 133 - "Community 133"
Cohesion: 1.0
Nodes (0): 

### Community 134 - "Community 134"
Cohesion: 1.0
Nodes (0): 

### Community 135 - "Community 135"
Cohesion: 1.0
Nodes (0): 

### Community 136 - "Community 136"
Cohesion: 1.0
Nodes (0): 

### Community 137 - "Community 137"
Cohesion: 1.0
Nodes (0): 

### Community 138 - "Community 138"
Cohesion: 1.0
Nodes (0): 

### Community 139 - "Community 139"
Cohesion: 1.0
Nodes (0): 

### Community 140 - "Community 140"
Cohesion: 1.0
Nodes (0): 

### Community 141 - "Community 141"
Cohesion: 1.0
Nodes (0): 

### Community 142 - "Community 142"
Cohesion: 1.0
Nodes (0): 

### Community 143 - "Community 143"
Cohesion: 1.0
Nodes (0): 

### Community 144 - "Community 144"
Cohesion: 1.0
Nodes (0): 

### Community 145 - "Community 145"
Cohesion: 1.0
Nodes (0): 

### Community 146 - "Community 146"
Cohesion: 1.0
Nodes (0): 

### Community 147 - "Community 147"
Cohesion: 1.0
Nodes (0): 

### Community 148 - "Community 148"
Cohesion: 1.0
Nodes (0): 

### Community 149 - "Community 149"
Cohesion: 1.0
Nodes (0): 

### Community 150 - "Community 150"
Cohesion: 1.0
Nodes (0): 

### Community 151 - "Community 151"
Cohesion: 1.0
Nodes (0): 

### Community 152 - "Community 152"
Cohesion: 1.0
Nodes (0): 

### Community 153 - "Community 153"
Cohesion: 1.0
Nodes (0): 

### Community 154 - "Community 154"
Cohesion: 1.0
Nodes (0): 

### Community 155 - "Community 155"
Cohesion: 1.0
Nodes (0): 

### Community 156 - "Community 156"
Cohesion: 1.0
Nodes (0): 

### Community 157 - "Community 157"
Cohesion: 1.0
Nodes (0): 

### Community 158 - "Community 158"
Cohesion: 1.0
Nodes (0): 

### Community 159 - "Community 159"
Cohesion: 1.0
Nodes (1): The CXX standard level. If set, will add the required flags. If left at

### Community 160 - "Community 160"
Cohesion: 1.0
Nodes (1): Assign a short name for the dataset class. By default will be the class name.

### Community 161 - "Community 161"
Cohesion: 1.0
Nodes (1): `is_valid_config`                  This method is for minimum sanity check on co

### Community 162 - "Community 162"
Cohesion: 1.0
Nodes (1): A default collate function that will handle torch.Tensor, pp.LieTensor and

### Community 163 - "Community 163"
Cohesion: 1.0
Nodes (0): 

### Community 164 - "Community 164"
Cohesion: 1.0
Nodes (0): 

### Community 165 - "Community 165"
Cohesion: 1.0
Nodes (0): 

### Community 166 - "Community 166"
Cohesion: 1.0
Nodes (1): Given a batch of N observation (`TensorBundle`), the filter returns a boolean te

### Community 167 - "Community 167"
Cohesion: 1.0
Nodes (1): Given a sequence of frames, elaborate the trajectory (frame poses) and handle th

### Community 168 - "Community 168"
Cohesion: 1.0
Nodes (1): Given config, initialize a *mutable* context object that is preserved between op

### Community 169 - "Community 169"
Cohesion: 1.0
Nodes (1): Given context and argument, construct the optimization problem, solve it and ret

### Community 170 - "Community 170"
Cohesion: 1.0
Nodes (1): Returns immediately, indicate the status of optimizer:         - true if there

### Community 171 - "Community 171"
Cohesion: 1.0
Nodes (1): Returns the concrete type used for T_GraphInput. Raises TypeError if not explici

### Community 172 - "Community 172"
Cohesion: 1.0
Nodes (1): Returns the concrete type used for T_GraphOutput. Raises TypeError if not explic

### Community 173 - "Community 173"
Cohesion: 1.0
Nodes (1): This function should be implemented by the user.         It should return the ja

### Community 174 - "Community 174"
Cohesion: 1.0
Nodes (1): Returns the jacobian of the model's previous forward call with respect to model'

### Community 175 - "Community 175"
Cohesion: 1.0
Nodes (1): Verifies whether the input J_analytic coincides with autograd jacobian of the pr

### Community 176 - "Community 176"
Cohesion: 1.0
Nodes (1): Given a pixel_uv (Nx2) tensor, retrieve the pixel values (1, N) from scalar_map

### Community 177 - "Community 177"
Cohesion: 1.0
Nodes (1): r'''         Linear/linearized system output matrix.          .. math::

### Community 178 - "Community 178"
Cohesion: 1.0
Nodes (1): r'''         Linear/Linearized system observation matrix.          .. math::

### Community 179 - "Community 179"
Cohesion: 1.0
Nodes (1): r'''         The covariance of system transition noise.

### Community 180 - "Community 180"
Cohesion: 1.0
Nodes (1): r'''         The covariance of system transition noise.

### Community 181 - "Community 181"
Cohesion: 1.0
Nodes (1): r'''         The covariance of system observation noise.

## Knowledge Gaps
- **544 isolated node(s):** `Evaluation: mask quality metrics + VIO trajectory metrics.  Mask metrics (on VIO`, `Compute IoU, Precision, Recall, F1 for a batch.      Args:         pred_mask: [B`, `Evaluate mask quality over a full dataloader.`, `Evaluate IMU preintegration quality (RTE, ROE).`, `Compute ATE RMSE and RPE using the evo package.      Args:         est_traj_file` (+539 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 80`** (2 nodes): `addPoshVcpkgToPowershellProfile.ps1`, `findExistingImportModuleDirectives()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 81`** (2 nodes): `deploy-windows-sdks.ps1`, `InstallWindowsDK()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 82`** (2 nodes): `create-image.ps1`, `Invoke-ScriptWithPrefix()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 83`** (2 nodes): `deploy-mpi.ps1`, `InstallMpi()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 84`** (2 nodes): `deploy-install-disk.ps1`, `New-PhysicalDisk()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 85`** (2 nodes): `deploy-inteloneapi.ps1`, `InstallInteloneAPI()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 86`** (2 nodes): `disk-space.ps1`, `Format-Size()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 87`** (2 nodes): `deploy-visual-studio.ps1`, `InstallVisualStudio()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 88`** (2 nodes): `qtdeploy.ps1`, `deployPluginsIfQt()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 89`** (2 nodes): `openni2deploy.ps1`, `deployOpenNI2()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 90`** (2 nodes): `check_getcontext.cc`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 91`** (2 nodes): `gettimeofday.h`, `gettimeofday()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 92`** (2 nodes): `magnumdeploy.ps1`, `deployPluginsIfMagnum()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 93`** (2 nodes): `decomp.c`, `set_endian()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 94`** (2 nodes): `generateFeatures.ps1`, `GetDescription()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 95`** (2 nodes): `k4adeploy.ps1`, `deployAzureKinectSensorSDK()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 96`** (2 nodes): `Optimization_Ablation.py`, `run_frame()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 97`** (2 nodes): `test_config_tartanvo.py`, `test_tartanvo_config()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 98`** (2 nodes): `test_config_macvo.py`, `test_macvo_config()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 99`** (2 nodes): `test_config_sequence.py`, `test_sequence_cfg()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 100`** (2 nodes): `test_performance_macvo.py`, `test_macvo_performance()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 101`** (2 nodes): `test_stereo_depth.py`, `test_matching()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 102`** (2 nodes): `test_config_loadable.py`, `test_config_loadable()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 103`** (2 nodes): `test_matching.py`, `test_matching()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 104`** (2 nodes): `test_frontend.py`, `test_frontend()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 105`** (2 nodes): `Experiment_DPVO.py`, `execute_experiment()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 106`** (2 nodes): `Experiment_MACVO.py`, `execute_experiment()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 107`** (2 nodes): `Experiment_TartanVO.py`, `execute_experiment()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 108`** (2 nodes): `MatchEstimator.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 109`** (2 nodes): `TartanVODisparity_AvgError.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 110`** (2 nodes): `submission.py`, `get_cfg()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 111`** (1 nodes): `chartdir.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 112`** (1 nodes): `Solve H x = b via Cholesky decomposition.          Args:             H: [B, 6, 6`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 113`** (1 nodes): `The CXX standard level. If set, will add the required flags. If left         at`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 114`** (1 nodes): `make_changelog.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 115`** (1 nodes): `libsize.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 116`** (1 nodes): `test_eval_call.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 117`** (1 nodes): `The CXX standard level. If set, will add the required flags. If left         at`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 118`** (1 nodes): `X11GlContext.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 119`** (1 nodes): `symbol_helper.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 120`** (1 nodes): `dummy.cpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 121`** (1 nodes): `bootstrap.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 122`** (1 nodes): `test-modified-ports.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 123`** (1 nodes): `Create-PRDiff.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 124`** (1 nodes): `drop-to-admin-user-prefix.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 125`** (1 nodes): `provision-entire-image.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 126`** (1 nodes): `sysprep.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 127`** (1 nodes): `deploy-pwsh.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 128`** (1 nodes): `deploy-cuda.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 129`** (1 nodes): `deploy-psexec.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 130`** (1 nodes): `create-vmss.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 131`** (1 nodes): `Install-Prerequisites.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 132`** (1 nodes): `Setup-VagrantMachines.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 133`** (1 nodes): `Vagrantfile-vm.rb`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 134`** (1 nodes): `Vagrantfile-box.rb`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 135`** (1 nodes): `rearrange-msvc-drop-layout.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 136`** (1 nodes): `arith_osx.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 137`** (1 nodes): `arith_win64.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 138`** (1 nodes): `arith_win32.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 139`** (1 nodes): `angle_commit.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 140`** (1 nodes): `convert_lib_params_linux.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 141`** (1 nodes): `generate_static_link_cmd_linux.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 142`** (1 nodes): `generate_static_link_cmd_windows.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 143`** (1 nodes): `generate_static_link_cmd_macos.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 144`** (1 nodes): `convert_lib_params_macos.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 145`** (1 nodes): `convert_lib_params_windows.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 146`** (1 nodes): `cgnsconfig.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 147`** (1 nodes): `fficonfig.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 148`** (1 nodes): `modp_b64_data.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 149`** (1 nodes): `libsecp256k1-config.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 150`** (1 nodes): `openblas_common.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 151`** (1 nodes): `config.linux.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 152`** (1 nodes): `arith_linux64.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 153`** (1 nodes): `magick_types.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 154`** (1 nodes): `b64_static_config.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 155`** (1 nodes): `b64_dynamic_config.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 156`** (1 nodes): `u2f-server-version.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 157`** (1 nodes): `FreeImageConfig-static.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 158`** (1 nodes): `FreeImageConfig-dynamic.h`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 159`** (1 nodes): `The CXX standard level. If set, will add the required flags. If left at`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 160`** (1 nodes): `Assign a short name for the dataset class. By default will be the class name.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 161`** (1 nodes): ``is_valid_config`                  This method is for minimum sanity check on co`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 162`** (1 nodes): `A default collate function that will handle torch.Tensor, pp.LieTensor and`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 163`** (1 nodes): `MatchQuality.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 164`** (1 nodes): `DepthQuality.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 165`** (1 nodes): `TartanVO.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 166`** (1 nodes): `Given a batch of N observation (`TensorBundle`), the filter returns a boolean te`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 167`** (1 nodes): `Given a sequence of frames, elaborate the trajectory (frame poses) and handle th`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 168`** (1 nodes): `Given config, initialize a *mutable* context object that is preserved between op`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 169`** (1 nodes): `Given context and argument, construct the optimization problem, solve it and ret`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 170`** (1 nodes): `Returns immediately, indicate the status of optimizer:         - true if there`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 171`** (1 nodes): `Returns the concrete type used for T_GraphInput. Raises TypeError if not explici`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 172`** (1 nodes): `Returns the concrete type used for T_GraphOutput. Raises TypeError if not explic`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 173`** (1 nodes): `This function should be implemented by the user.         It should return the ja`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 174`** (1 nodes): `Returns the jacobian of the model's previous forward call with respect to model'`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 175`** (1 nodes): `Verifies whether the input J_analytic coincides with autograd jacobian of the pr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 176`** (1 nodes): `Given a pixel_uv (Nx2) tensor, retrieve the pixel values (1, N) from scalar_map`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 177`** (1 nodes): `r'''         Linear/linearized system output matrix.          .. math::`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 178`** (1 nodes): `r'''         Linear/Linearized system observation matrix.          .. math::`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 179`** (1 nodes): `r'''         The covariance of system transition noise.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 180`** (1 nodes): `r'''         The covariance of system transition noise.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 181`** (1 nodes): `r'''         The covariance of system observation noise.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Timer` connect `Community 4` to `Community 3`, `Community 23`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._
- **Why does `TartanStereoVOMatch` connect `Community 4` to `Community 8`?**
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