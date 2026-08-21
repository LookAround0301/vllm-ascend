#!/usr/bin/env bash
# env_a3_0260.sh — environment for vllm-ascend on Atlas A3 (Ascend 910), aarch64.
#
# SCOPE: process environment only. Sets variables that must exist BEFORE python
# starts. Launches nothing, defines no functions, knows nothing about any single
# run. Model path, parallelism, graph mode, speculative config, the
# expert_offload_config JSON and the `vllm serve` line belong to the test script.
# Only what is COMMON to all eight test configurations lives here.
#
# STACK (verified against the repos, not assumed):
#   vLLM          upstream vllm-project/vllm @ v0.26.0   (no vLLM fork anymore)
#   vllm-ascend   LookAround0301/vllm-ascend @ moe_offload_v0260rc_k3_0801
#   CANN / NNAL   9.0.1          torch 2.10.0 (aarch64: no "+cpu" suffix)
#   torch-npu     2.10.0.post2   torchvision 0.25.0 / torchaudio 2.10.0
#   triton-ascend 3.2.1          transformers 5.14.1 (hard pin, via requirements.txt)
#   Python        3.12           (memfabric/memcache_hybrid are cp312-only)
#
# SOURCES OF TRUTH (vllm-ascend @ moe_offload_v0260rc_k3_0801):
#   requirements.txt · Dockerfile.a3.openEuler · docs/source/installation.md
#
# USAGE — source, never execute; every shell, before building or running:
#   source env_a3_0260.sh
#   ./run_case.sh 6            # the test script owns everything per-run

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: source this file, do not run it ->  source ${BASH_SOURCE[0]}" >&2
  exit 1
fi

# ---- 1. Conda env ----------------------------------------------------------
# New env on purpose: 0.26.0 moves protobuf (3.20 -> >=5.29.6), pins
# transformers==5.14.1 and downgrades fastapi. Upgrading moeoffv5 in place
# leaves a half-resolved graph.
export VA_CONDA_ENV="${VA_CONDA_ENV:-moeoff0260-keyi}"
conda activate "${VA_CONDA_ENV}" || echo "WARN: could not activate ${VA_CONDA_ENV}"

# Set BEFORE sourcing CANN, deliberately: set_env.sh PREPENDS its paths, so this
# ordering leaves CANN's libs ahead of conda's. Reversing it lets conda's
# libstdc++ shadow CANN's. Do not move this line below section 2.
# export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

# ---- 2. CANN / NNAL --------------------------------------------------------
# VA_ASCEND_ROOT is the INSTALL ROOT (the dir containing ascend-toolkit/ and
# nnal/). Edit this one line to switch CANN installs.
#
# It is deliberately NOT called ASCEND_HOME_PATH. CANN's set_env.sh and
# vllm-ascend's envs.py both define ASCEND_HOME_PATH as the TOOLKIT dir
# (<root>/ascend-toolkit/latest), not the root. Deriving the root from it —
# `${ASCEND_HOME_PATH:-<root>}` — is self-corrupting: after one source,
# ASCEND_HOME_PATH is already <root>/ascend-toolkit/latest, so :- never fires,
# the old install sticks, and the script then hunts for
# <root>/ascend-toolkit/latest/ascend-toolkit/set_env.sh and reports it missing.
export VA_ASCEND_ROOT="/home/yangliu/Ascend901"

# set_env.sh CANNOT be un-sourced — it prepends to PATH / LD_LIBRARY_PATH /
# PYTHONPATH. Mixing two CANN roots in one shell yields a half-and-half
# environment whose symptom is missing or mismatched .so files. Refuse instead.
_stale="$(printf '%s:%s' "${LD_LIBRARY_PATH}" "${PATH}" | tr ':' '\n' \
  | sed -n 's|\(.*\)/ascend-toolkit/.*|\1|p' | sort -u \
  | grep -vx "${VA_ASCEND_ROOT}" || true)"
if [[ -n "${_stale}" ]]; then
  echo "ERROR: a DIFFERENT CANN root is already active in this shell:"
  echo "${_stale}" | sed 's/^/         /'
  echo "       wanted: ${VA_ASCEND_ROOT}"
  echo "       set_env.sh cannot be undone — open a FRESH shell and source this file there."
fi

# Idempotent: re-sourcing in the same shell would append duplicate paths.
if [[ "${VA_CANN_SOURCED}" == "${VA_ASCEND_ROOT}" ]]; then
  echo "[env] CANN already sourced from ${VA_ASCEND_ROOT} — skipping"
else
  for _f in "${VA_ASCEND_ROOT}/ascend-toolkit/set_env.sh" \
            "${VA_ASCEND_ROOT}/nnal/atb/set_env.sh"; do
    if [[ ! -f "${_f}" ]]; then
      echo "ERROR: not found: ${_f}"
    elif ! source "${_f}"; then
      # Explicit if/else, not `&& source || echo`: with that idiom a non-zero
      # return from set_env.sh prints a bogus "missing file" message.
      echo "WARN: ${_f} exists but returned non-zero"
    fi
  done
  export VA_CANN_SOURCED="${VA_ASCEND_ROOT}"
fi

# Now pin the two toolkit vars to CANN's OWN meaning. vllm-ascend's setup.py and
# envs.py read ASCEND_HOME_PATH during the CMake configure and at runtime, so it
# must point at the toolkit dir even if set_env.sh failed to load.
export ASCEND_TOOLKIT_HOME="${VA_ASCEND_ROOT}/ascend-toolkit/latest"
export ASCEND_HOME_PATH="${ASCEND_TOOLKIT_HOME}"

# Read the REAL toolkit version rather than asserting one. The kernels compile
# against CANN headers; a 9.0.0 toolkit under a 9.0.1-era branch fails obscurely.
_CANN_V="$(sed -n 's/^version=//p' \
  "${ASCEND_TOOLKIT_HOME}/ascend_toolkit_install.info" 2>/dev/null)"

# Prove the toolkit actually resolved, rather than trusting that it did.
for _lib in "${ASCEND_TOOLKIT_HOME}/lib64/libascendcl.so" \
            "${VA_ASCEND_ROOT}/nnal/atb/latest/atb/cxx_abi_1/lib/libatb.so"; do
  [[ -e "${_lib}" ]] || echo "WARN: not found: ${_lib}"
done

# Leftover TORCH_NPU_PATH=/torch_npu from a container image breaks the C++ build
# (NPUStream.h not found). Always clear it.
unset TORCH_NPU_PATH

# ---- 3. Build target -------------------------------------------------------
# CONFIRM with: npu-smi info -t board -i 0 -c 0
# Dockerfile.a3 defaults to ascend910_9391; this box reports ascend910_9362.
# Both are valid A3 SOCs — use what npu-smi says. On A3 the FULL AscendC kernel
# set compiles against this (CMakeLists only skips ascend310p*/ascend950*), so a
# wrong value builds cleanly and is wrong for the silicon.
export SOC_VERSION="ascend910_9362"

# ---- 4. Version pins -------------------------------------------------------
# INSTALL-TIME ONLY. Nothing at runtime reads these except VLLM_VERSION; they
# exist so INSTALL_A3_0260.md's commands and this file cannot drift apart.
#
# VLLM_VERSION is the single source of truth and IS functional: vllm-ascend's
# envs.py reads it to decide which vLLM it is patching, because an editable
# install from a git tag reports a setuptools-scm string like
# "0.26.1.dev0+g<sha>" that makes the version gates take the wrong branch.
# VLLM_TAG is derived from it — never set the two independently.
export VLLM_VERSION="0.26.0"
export VLLM_TAG="v${VLLM_VERSION}"
export VLLM_ASCEND_BRANCH="moe_offload_v0260rc_k3_0801"

export TORCH_VERSION="2.10.0"
export TORCH_NPU_VERSION="2.10.0.post2"   # .post2, NOT plain 2.10.0
export TORCHVISION_VERSION="0.25.0"
export TORCHAUDIO_VERSION="2.10.0"
export TRITON_ASCEND_VERSION="3.2.1"
# transformers / arctic-inference are pinned by requirements.txt, never by hand.

# ---- 5. Pip indexes --------------------------------------------------------
# Three layers, three behaviours:
#   index-url            PRIMARY, exactly one wins. Set it per-env with
#                        `pip config set --site global.index-url <url>`, which
#                        writes $CONDA_PREFIX/pip.conf and so cannot leak into
#                        moeoffv5 or any other env on this box.
#   PIP_EXTRA_INDEX_URL  additive, from the environment — exported below.
#   --extra-index-url    additive, per command.
# The last two MERGE (verified: extra_index_urls == [env, cli]); pip searches
# primary -> env -> cli. That merge is how Dockerfile.a3.openEuler pulls
# torch-npu/memfabric from Ascend and torch from the PyTorch CPU index at once.
# The old "never set a global extra-index-url" warning was about putting several
# URLs in the pip.conf KEY, which fans every resolution across all of them.
export TUNA_PYPI="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"  # Dockerfile's PIP_INDEX_URL
export HUAWEI_PYPI="https://mirrors.huaweicloud.com/repository/pypi/simple/"  # alternative primary
export ASCEND_IDX="https://mirrors.huaweicloud.com/ascend/repos/pypi"    # torch-npu, triton-ascend, memfabric/memcache

export PIP_EXTRA_INDEX_URL="${ASCEND_IDX}"   # the one extra you always want live

# TORCH_IDX_ARG is x86-only. The aarch64 PyPI torch wheel is ALREADY CPU-only
# (its nvidia-* deps carry platform_machine == "x86_64" markers), so there is no
# "+cpu" variant and download.pytorch.org is dead weight here. Empty on ARM.
export HOST_ARCH="$(uname -m)"
if [[ "${HOST_ARCH}" == "x86_64" ]]; then
  export TORCH_IDX_ARG="--extra-index-url https://download.pytorch.org/whl/cpu/"
else
  export TORCH_IDX_ARG=""
fi

# ---- 6. Toolchain ----------------------------------------------------------
# cmake  HARD. requirements.txt and pyproject.toml both declare cmake>=3.26 and
#        pip enforces it. (CMakeLists.txt asks only for 3.16; the pip floor bites.)
# clang  NOT hard. triton-ascend's _get_cxx() prefers clang++ but falls back to
#        g++, and only uses it for the host-side CPython launcher (C++17). NPU
#        codegen is bishengir-compile from CANN, which carries its own clang.
#        The clang-15/lld-15 in triton-ascend's docs is for building it FROM
#        SOURCE. The catch is only that clang++ is preferred WHEN PRESENT, so
#        openEuler 22.03's clang 12 would be chosen over a fine gcc 10.3.
#        Escape hatch without LLVM: export CC=g++  (yes, CC, used as C++).
export LLVM_HOME="${LLVM_HOME:-/home/yangliu/llvm-17}"
[[ -d "${LLVM_HOME}/bin" ]] && export PATH="${LLVM_HOME}/bin:${PATH}"

_CMAKE_V="$(cmake --version 2>/dev/null | head -n1 | awk '{print $3}')"
_CXX="$(command -v clang++ || command -v g++ || echo none)"   # what triton-ascend will pick
_CXX_V="$("${_CXX}" -dumpversion 2>/dev/null)"

[[ "$(printf '%s\n3.26.0\n' "${_CMAKE_V:-0}" | sort -V | head -n1)" == "3.26.0" ]] \
  || echo "ERROR: cmake ${_CMAKE_V:-not found} < 3.26 — pip will reject the build. pip install 'cmake==3.31.10'"
[[ "${_CXX}" == *clang++ && "${_CXX_V%%.*}" -lt 15 ]] \
  && echo "NOTE: ${_CXX} ${_CXX_V} is preferred over g++ by triton-ascend. Fix LLVM_HOME or 'export CC=g++'."

# ---- 7. Build-time knobs (read only while compiling vllm-ascend) -----------
# Changing any of these needs `rm -rf build/ csrc/build/` + reinstall.
export COMPILE_CUSTOM_KERNELS=1           # MUST stay 1 on A3 (0 = no NPU kernels)
export MAX_JOBS="${MAX_JOBS:-32}"         # kernel compile parallelism
# export CMAKE_BUILD_TYPE="Release"       # Release | Debug | RelWithDebugInfo
# export VERBOSE=1

# VLLM_BATCH_INVARIANT is read at BOTH build and run time under the same name.
# Build: csrc/build_aclnn.sh only compiles the batch-invariant AscendC op package
# when it is 1. Run: vLLM decides whether to route through those ops.
# Dockerfile.a3.openEuler BUILDS with 1, so the official image ships the ops and
# defers the choice to runtime. Build with 1 (INSTALL Phase 4), serve with 0.
# Building with 0 means you cannot enable it later without a full rebuild.
export VLLM_BATCH_INVARIANT=0

# ---- 8. Runtime knobs ------------------------------------------------------
export TASK_QUEUE_ENABLE=1                            # from Dockerfile.a3
export VLLM_ASCEND_ENABLE_NZ=1                        # int: 0=off, 1=quant-only, 2=always
export VLLM_ASCEND_ENABLE_MLAPO=1                     # fused MLA preprocess for DeepSeek W8A8;
                                                      # faster but costs NPU memory — 0 if OOM
export HCCL_OP_EXPANSION_MODE="AIV"                   # expand HCCL ops onto AI Vector cores
export HCCL_BUFFSIZE=1024                             # MB
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0               # deterministic scheduling
export VLLM_SERVER_DEV_MODE=1                         # extra endpoints (`vllm serve` only)

# CPU threading. The reference image uses 1; 10 is the tuned single-card value.
# Every EP rank is its own process, so 10 x N oversubscribes fast — the test
# script re-exports VA_OMP_MULTI for the multi-card cases. Both values live here
# so the two scripts agree on the numbers.
export OMP_PROC_BIND=false
export VA_OMP_SINGLE="${VA_OMP_SINGLE:-10}"
export VA_OMP_MULTI="${VA_OMP_MULTI:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${VA_OMP_SINGLE}}"

# Generous timeouts — avoid spurious failures on long large-model loads.
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export VLLM_ENGINE_READY_TIMEOUT_S=10000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000

# ---- 9. Expert-offload invariants ------------------------------------------
# Preconditions that must hold for EVERY offload run, in all eight configs.
# Not tuning knobs: each maps to a defect in 00b_TARGET_INTEGRATION_SCOPE_v2.md
# §9 that yields wrong results SILENTLY. The test script must not override them.

# D12 — dynamic EPLB deletes the exact weight/scale attributes the offload paging
# primitive (_load_expert_weights_into_slot) writes into, and skips pack_to_int32
# so even the byte layout diverges. No config guard exists in the target.
export DYNAMIC_EPLB=false

# D9 + D10 — fused MC2, two independent problems with offload on:
#   D9  W8A8's fused_w1_scale / fused_w2_scale / cann_mega_moe_fused_*_list are
#       derived ONCE in process_weights_after_loading; paging refreshes only
#       w13_weight_scale_fp32, so every expert paged in later carries a stale
#       fused scale. Numerically wrong, silently.
#   D10 FusedMC2CommImpl.fused_experts applies log2phy WITHOUT .clamp(min=0), so
#       an unmapped -1 reaches the fused op.
# Unreachable anyway once enable_multi_card is set (the multi-card branch of
# select_moe_comm_method pre-empts all hardware branches) — pinned so single-card
# runs are safe too.
export VLLM_ASCEND_ENABLE_FUSED_MC2=0

# O8 — _ExtraForwardContextProxy branches on this, and the whole lifecycle
# analysis (load_model -> _finalize_offload_setup, _skip_prefill, capture path)
# was traced through model_runner_v1.py. Pin V1 so analysis and runtime agree.
# Flip to 1 only as a deliberate experiment; re-verify the offload wiring first.
export VLLM_USE_V2_MODEL_RUNNER=0

# Must stay 0 for the four GRAPH configs. Blocking dispatch serialises every
# kernel launch on the host, defeating the prefetch stream entirely, and is not
# supported under ACL graph capture. Use 1 only for eager debugging, and never
# quote a latency number taken with it on.
export ASCEND_LAUNCH_BLOCKING=0

# The offload subsystem logs at INFO ([UPDATE-W], [PREFETCH], [multi_card_offload
# INIT], the interval cache-stats line); without this they are invisible. This is
# VISIBILITY only — the per-layer per-step trace is "moe_offload_debug": true in
# the additional-config, owned by the test script, which puts sorted() +
# .tolist() + a formatted log inside the host callback (D16). Keep that false for
# any timing run.
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
# export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"
# ---- 10. Deliberately absent -----------------------------------------------
# Do not re-add these; all four are dead in this stack and were verified so.
#   VLLM_USE_V1                  removed from vLLM 0.26.0 envs.py — V1 is the
#                                only engine; nothing reads it.
#   VLLM_RPC_TIMEOUT             also removed from vLLM 0.26.0 envs.py.
#   USE_MULTI_BLOCK_POOL         no os.getenv anywhere in vllm_ascend/. The
#   USE_MULTI_GROUPS_KV_CACHE    multi-group path is now a module constant
#                                (= True) in patch_kv_cache_coordinator.py. The
#                                fork's own script/*.sh still export them; they
#                                are vestigial.

# ---- 11. Distributed comms (auto-detected to THIS box) ---------------------
# GLOO_SOCKET_IFNAME is load-bearing for the MULTI-CARD cases: the offload
# planner all-reduces global expert counts over get_ep_group().cpu_group (gloo)
# from INSIDE the host callback. A wrong interface does not fail fast — it hangs
# the cluster mid-decode.
export NIC_NAME="$(ip -o -4 addr show scope global up 2>/dev/null | \
  awk '$2 !~ /^(veth|docker|br-|virbr|tun|tap|cni|flannel|lo|dummy)/ {print $2; exit}')"
export LOCAL_IP="$(ip -o -4 addr show dev "${NIC_NAME}" scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1)"
export HCCL_IF_IP="${LOCAL_IP}"
export GLOO_SOCKET_IFNAME="${NIC_NAME}"
export TP_SOCKET_IFNAME="${NIC_NAME}"
export HCCL_SOCKET_IFNAME="${NIC_NAME}"

# ---- 12. Library paths -----------------------------------------------------
# (conda's lib was prepended in section 1, before CANN, on purpose.)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

_jemalloc="$(find /usr/lib64 /usr/lib -name 'libjemalloc.so.2' 2>/dev/null | head -n1)"
[[ -n "${_jemalloc}" ]] && export LD_PRELOAD="${_jemalloc}:${LD_PRELOAD}" \
  || echo "WARN: libjemalloc.so.2 not found (yum install jemalloc)"

# ---- 13. Weights / multiproc -----------------------------------------------
export VLLM_USE_MODELSCOPE="True"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
# export MODELSCOPE_CACHE="/home/${USER}/.cache/modelscope"

# ---- 14. Per-run: owned by the TEST script ---------------------------------
# Unset here on purpose; named so both scripts agree.
#
#   ASCEND_RT_VISIBLE_DEVICES  one device for single-card configs, N for
#       multi-card. EP world size == the count listed here, and it must move
#       together with the expert_offload_config key "enable_multi_card":
#       neither direction of mismatch is caught by the engine (D14/D15), and
#       EP>1 without enable_multi_card is silently incoherent.
#   OMP_NUM_THREADS            re-export as ${VA_OMP_MULTI} for multi-card runs.
#
# Everything else separating the eight configs is a CLI argument, not an env var:
#   --compilation-config   cudagraph_mode NONE (eager) | FULL_DECODE_ONLY (graph)
#                          + cudagraph_capture_sizes [D2: max size must be
#                          <= num_device_experts // topk on single card]
#   --speculative-config   present (MTP on) | absent (MTP off)
#   --tensor-parallel-size / --enable-expert-parallel
#   --additional-config    the expert_offload_config block
#                          [D1: expert_prefetch_num must be 1 or topk]
#
# Host tuning (needs root) also belongs to the test script:
#   echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
#   sysctl -w vm.swappiness=0 ; sysctl -w kernel.numa_balancing=0
#   sysctl -w kernel.sched_migration_cost_ns=50000
#   rm -rf ~/ascend/log/debug/plog/*

# ---- 15. Summary -----------------------------------------------------------
echo "[env] ${VA_CONDA_ENV} | ${HOST_ARCH} | SOC=${SOC_VERSION}"
echo "[env] CANN=${_CANN_V:-UNKNOWN} (want 9.0.1) root=${VA_ASCEND_ROOT}"
echo "[env] vllm=${VLLM_TAG} vllm-ascend=${VLLM_ASCEND_BRANCH} torch=${TORCH_VERSION} torch-npu=${TORCH_NPU_VERSION}"
echo "[env] tools: cmake=${_CMAKE_V:-none} gcc=$(gcc -dumpversion 2>/dev/null || echo none) triton-c++=${_CXX} (${_CXX_V:-?})"
echo "[env] pip: primary=$(pip config get global.index-url 2>/dev/null || echo 'pypi.org') extra=${PIP_EXTRA_INDEX_URL} torch_idx='${TORCH_IDX_ARG:-none}'"
echo "[env] net: NIC=${NIC_NAME} IP=${LOCAL_IP} | TORCH_NPU_PATH='${TORCH_NPU_PATH:-unset, good}'"
echo "[env] build: KERNELS=${COMPILE_CUSTOM_KERNELS} MAX_JOBS=${MAX_JOBS} BATCH_INVARIANT=${VLLM_BATCH_INVARIANT}"
echo "[env] perf: NZ=${VLLM_ASCEND_ENABLE_NZ} MLAPO=${VLLM_ASCEND_ENABLE_MLAPO} HCCL=${HCCL_OP_EXPANSION_MODE} OMP=${OMP_NUM_THREADS} (single/multi ${VA_OMP_SINGLE}/${VA_OMP_MULTI})"
echo "[env] offload invariants: EPLB=${DYNAMIC_EPLB} MC2=${VLLM_ASCEND_ENABLE_FUSED_MC2} V2_RUNNER=${VLLM_USE_V2_MODEL_RUNNER} BLOCKING=${ASCEND_LAUNCH_BLOCKING} LOG=${VLLM_LOGGING_LEVEL}"
echo "[env] per-run (test script): ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-unset}"
command -v npu-smi >/dev/null 2>&1 || echo "[env] WARN: npu-smi not on PATH"

unset _f _lib _stale _jemalloc _CMAKE_V _CXX _CXX_V _CANN_V