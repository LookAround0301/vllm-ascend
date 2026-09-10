#!/usr/bin/env bash
# =============================================================================
#  v4_target_aisbench.sh — serve DeepSeek-V4-Flash and measure ACCURACY / TPOT
#  on gsm8k / gpqa_diamond / mmlu_pro via AISBench.
#
#    source env_a5_0260.sh
#    CARD=2 PORT=1161 TASKS=gpqa RUN=perf ./v4_target_aisbench.sh
#
#    EXPERT_PREDICTOR=mode2_har \
#      EXPERT_PREDICTOR_CKPT=/path/har_prompt_std_dist_lowrank_w=1280_miss=0.pt \
#      CARD=2 PORT=1161 TASKS=gpqa RUN=perf ./v4_target_aisbench.sh
#
#    EXPERT_PREDICTOR=mode2_prevhfr \
#      EXPERT_PREDICTOR_CKPT=/path/prevhfr_prompt_std_dist_lowrank_w=1280_miss=0.pt \
#      EXPERT_PREFETCH_WAIT_TIMING=1 CARD=2 PORT=1161 TASKS=gpqa RUN=perf ./v4_target_aisbench.sh
#
#    DRY_RUN=1 print the plan only   LIST=1 list dataset configs
#    MTP=0|1|2  spec decode: off | mtp | dspark
#
#  OUT_DIR/: run.log (this script) · serve.log (the engine) · acc/ perf/ (AISBench)
#            configs/ (generated + symlinked) · stats/ (decode-stats .txt and .csv)
#
#  The stats artefacts are written DURING the run by the collector's watchdog thread
#  (every STATS_FLUSH_EVERY steps, STATS_FLUSH_SECONDS seconds, or ~3s after decoding
#  stops). [EXPERT-OFFLOAD-FINAL] in serve.log is best effort on top of that.
#
#  FINDINGS — things that changed a number — are at the bottom. Read them once.
# =============================================================================
set -uo pipefail

# ── server ───────────────────────────────────────────────────────────────────
MODEL="${MODEL:-}"
SERVED_NAME="${SERVED_NAME:-}"
TOKENIZER="${TOKENIZER:-}"
AIS_BENCH_DATASETS_CACHE="${AIS_BENCH_DATASETS_CACHE-}"
REMOE_GATE="${REMOE_GATE:-}"                # fine-tuned MoE router gates, swapped in at load
REMOE_GATE_ROT="${REMOE_GATE_ROT:-quarot}"  # the rotation the REMOE_GATE checkpoints were fine-tuned against
CARD="${CARD:-6}"; PORT="${PORT:-7001}"
TP="${TP:-1}"; DP="${DP:-1}"; PP="${PP:-1}"   # EP spans TP*DP; the card list must be TP*DP*PP
SEED="${SEED:-1024}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"       # empty = the checkpoint's own
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
QUANTIZATION="${QUANTIZATION-ascend}"          # non-empty -> --quantization <v>; empty -> flag
                                         # omitted, engine resolves it from the checkpoint.
                                         # That is ALL this variable does.
API_SERVER_COUNT="${API_SERVER_COUNT:-1}"
PREFIX_CACHING="${PREFIX_CACHING:-1}"    # 0 -> --no-enable-prefix-caching

# ── graph mode ───────────────────────────────────────────────────────────────
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
# TOKEN counts. "auto" = 1..decode_tokens. "auto-safe" = truncated at the offload-safe
# bound (decode then runs eager). "1,2,3" = explicit; pin this when comparing runs.
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-auto}"
ENABLE_NPUGRAPH_EX="${ENABLE_NPUGRAPH_EX:-}"     # empty = engine default
ENABLE_STATIC_KERNEL="${ENABLE_STATIC_KERNEL:-}" # empty = engine default

# ── speculative decoding ─────────────────────────────────────────────────────
# MTP=0 off | 1 mtp (the serial drafter) | 2 dspark. Draft MoE layers never offload,
# so this changes only the decode TOKEN COUNT — and therefore whether the offload
# threshold is still cleared. See MOE_ROWS below.
MTP="${MTP:-0}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-2}"

# ── MoE offload ──────────────────────────────────────────────────────────────
OFFLOAD="${OFFLOAD:-1}"
NUM_DEVICE_EXPERTS="${NUM_DEVICE_EXPERTS:-36}"   # int, or a JSON list "[60,60,...]"
NUM_DEVICE_LAYERS="${NUM_DEVICE_LAYERS:-1}"
TOPK="${TOPK:-6}"                        # overridden from the checkpoint config below
EXPERT_MAP_PATH="${EXPERT_MAP_PATH:-}"
CACHE_POLICY="${CACHE_POLICY:-1}"        # 1 = LRC, required by prefetch
PREFETCH="${PREFETCH:-1}"
EXPERT_PREFETCH_NUM="${EXPERT_PREFETCH_NUM:-2}"     # TRANSFER budget: predicted misses
                                         # loaded per layer per step. Single card only.
EXPERT_PREFETCH_TOKENS="${EXPERT_PREFETCH_TOKENS:-1}"  # PREDICTION width: token rows the
                                         # predictor runs on. Orthogonal to _NUM.
# Which prefetch METHOD drives the targets a trained head COVERS. "fate" is the engine's
# built-in hash/gate next-layer predictor and stays the default; it also keeps every target
# a trained head does not own, in every configuration.
#   fate           targets 1..42, predicted from the post-prepare() MoE input of ell-1
#   mode2_har      targets 3..42 from each layer's OWN pre-attention residual; fate keeps 1,2
#   mode2_prevhfr  targets 4..42 from the PREVIOUS layer's post-attention residual, which
#                  buys a full extra layer of transfer window; fate keeps 1,2,3 — layer 3's
#                  predecessor was zero-filled in the study's dump, so that head never saw a
#                  real input and must not be used.
# The hash-routed targets stay on fate always: their experts come from the tid2eid table
# indexed by token id, which fate predicts exactly with one index_select and no matmul.
EXPERT_PREDICTOR="${EXPERT_PREDICTOR:-fate}"        # fate | mode2_har | mode2_prevhfr
EXPERT_PREDICTOR_CKPT="${EXPERT_PREDICTOR_CKPT:-}"  # e.g. .../har_prompt_std_dist_lowrank_w=1280_miss=0.pt
# Measure the latency the prefetch path ADDS to the critical path: milliseconds the main
# compute stream spends blocked at the prefetch join before the on-demand load can start.
# Device events bracket the join, so the number is correct under graph replay, where no
# Python runs. Costs two event records per covered layer per step plus 2*num_moe_layers
# timing events — so a run with this on is NOT a TPOT baseline. Off by default.
EXPERT_PREFETCH_WAIT_TIMING="${EXPERT_PREFETCH_WAIT_TIMING:-0}"
CACHE_STATS_LOG_INTERVAL="${CACHE_STATS_LOG_INTERVAL:-}"
MOE_OFFLOAD_DEBUG="${MOE_OFFLOAD_DEBUG:-0}"      # per-layer trace INSIDE the host
                                         # callback — invalid for any timing run
EXPERT_SUBSTITUTION="${EXPERT_SUBSTITUTION:-0}"  # CHANGES ROUTING
EXPERT_SUBSTITUTION_THRESHOLD="${EXPERT_SUBSTITUTION_THRESHOLD:-}"
HOT_PRELOAD="${HOT_PRELOAD:-0}"; HOT_EXPERTS_FILE="${HOT_EXPERTS_FILE:-}"
ENABLE_MULTI_CARD="${ENABLE_MULTI_CARD:-0}"      # must move together with EP world size
SHARD_PER_RANK="${SHARD_PER_RANK:-}"     # empty = engine default (true)
# Expert-weight H2D backend. "torch" is pinned CPU + copy_(), the engine default and what
# every number so far was measured with. "memfabric" swaps the host allocator and the copy
# for MemFabric Hybrid 1.2 — LOCAL DRAM on one card, SHARED DRAM across the EP group —
# leaving routing, eviction and placement untouched. SHARED publishes every rank's shard
# pointers, which RAISES the MC2 admission bound from one rank's slots to the whole pool
# (see the mc line below). Needs the MemFabric runtime sourced before this script runs.
H2D_BACKEND="${H2D_BACKEND:-torch}"              # torch | memfabric
MEMFABRIC_POOL_GIB="${MEMFABRIC_POOL_GIB:-0}"    # PER RANK, not the total. >0 for memfabric
MEMFABRIC_LOG_LEVEL="${MEMFABRIC_LOG_LEVEL:-}"   # empty = engine default (3)
CACHE_RECENT_WINDOW="${CACHE_RECENT_WINDOW:-}"   # empty = engine defaults for all six
CACHE_EMA_BETA="${CACHE_EMA_BETA:-}"
CACHE_RECENT_WEIGHT="${CACHE_RECENT_WEIGHT:-}"
CACHE_EMA_WEIGHT="${CACHE_EMA_WEIGHT:-}"
CACHE_ROUTER_WEIGHT="${CACHE_ROUTER_WEIGHT:-}"
CACHE_AGE_WEIGHT="${CACHE_AGE_WEIGHT:-}"

# ── decode statistics (TOP-LEVEL additional_config keys) ─────────────────────
DECODE_STATS="${DECODE_STATS:-1}"
CSV="${CSV:-1}"
CSV_PATH="${CSV_PATH:-}"                 # empty = OUT_DIR/stats
STATS_FLUSH_EVERY="${STATS_FLUSH_EVERY:-100}"
STATS_FLUSH_SECONDS="${STATS_FLUSH_SECONDS:-30}"
STATS_QUIESCE="${STATS_QUIESCE:-8}"      # idle seconds before stopping the server, so the
                                         # collector's quiet snapshot lands first

# ── other additional_config ──────────────────────────────────────────────────
MULTISTREAM_OVERLAP_SHARED_EXPERT="${MULTISTREAM_OVERLAP_SHARED_EXPERT:-0}"
ENABLE_CPU_BINDING="${ENABLE_CPU_BINDING:-0}"

# ── client ───────────────────────────────────────────────────────────────────
TASKS="${TASKS:-gsm8k gpqa mmlu_pro}"
RUN="${RUN:-both}"                       # acc | perf | both
THINK="${THINK:-}"                       # ""=presets, 0=non-think, 1=thinking
THINK_EFFORT="${THINK_EFFORT:-}"         # ""=preset, high | max
NUM_PROMPTS="${NUM_PROMPTS-unset}"       # acc pass; empty = FULL dataset
PERF_PROMPTS="${PERF_PROMPTS-30}"
AVG_N="${AVG_N:-}"                       # repeats per question; 4 to report
TEMP="${TEMP:-}"; TOPP="${TOPP-unset}"
PERF_TEMP="${PERF_TEMP:-0.0}"            # pinned so TPOT stays comparable
EXTRACT_RATE="${EXTRACT_RATE:-1}"

# ── run control ──────────────────────────────────────────────────────────────
OUT_DIR="${OUT_DIR:-$PWD/aisbench_results/$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"; LIST="${LIST:-0}"; SKIP_SERVE="${SKIP_SERVE:-0}"
PROBE="${PROBE:-1}"                      # verify the served reasoning mode
WAIT="${WAIT:-1000}"                     # seconds to wait for /health
HEALTH_GRACE="${HEALTH_GRACE:-120}"      # once the engine LOGS that it is serving,
                                         # /health has this long before the run fails
ENGINE_GRACE="${ENGINE_GRACE:-45}"       # EngineCore gets this alone before the group
REAP_WAIT="${REAP_WAIT:-180}"            # group grace before SIGKILL
HEARTBEAT="${HEARTBEAT:-300}"            # liveness line interval; 0 = off
ALLOW_UNSAFE="${ALLOW_UNSAFE:-0}"        # guards -> warnings

# presets: task | dataset | acc mode | perf mode | temp | top_p | max_out | prompts | avg_n
PRESETS="
gsm8k    | gsm8k_gen_0_shot_cot_chat_prompt | non-think | non-think | 0.0 |      | 32768 |     | 1
gpqa     | gpqa_gen_0_shot_cot_chat_prompt  | non-think | non-think | 0.6 | 0.95 | 65536 |     | 1
mmlu_pro | mmlu_pro_gen_0_shot_str          | non-think | non-think | 0.6 | 0.95 | 65536 |  20 | 1
"
ref() {   # vendor's published number. $1=task $2=mode
  case "$1:$2" in
    gpqa:non-think) echo "71.2 (GPQA-D Pass@1)" ;;      gpqa:high) echo "87.4" ;;
    gpqa:max) echo "88.1" ;;
    mmlu_pro:non-think) echo "83.0 (MMLU-Pro EM)" ;;    mmlu_pro:high) echo "86.4" ;;
    mmlu_pro:max) echo "86.2" ;;
    gsm8k:non-think) echo "90.8 (base 8-shot EM)" ;;    gsm8k:*) echo "none published" ;;
  esac
}

# #############################################################################
#                        DO NOT EDIT BELOW THIS LINE
# #############################################################################

die()   { echo "ERROR: $*" >&2; exit 1; }
guard() { [[ "${ALLOW_UNSAFE}" == "1" ]] && echo "  warn    : [unsafe] $*" || die "$*"; }
bool()  { [[ "$1" == "1" ]] && echo true || echo false; }

# the rotation a model build declares, as a canonical id. Echoes exactly one of:
#   absent      no quant_model_description.json -> the build ships no quantisation metadata
#               at all. The fp8/ue8m0 DeepSeek release is like this.
#   none        the file exists but carries no `optional` block
#   unreadable  the file exists but is not parseable JSON
#   <ids>       comma-separated sorted keys of `optional`, e.g. "quarot"
# Never fails: a bad path or a corrupt file resolves to a value the caller reports, rather
# than to a stack trace the caller has to interpret. The 8.5 MB / 136973-key file parses in
# ~94 ms, so this is cheap enough for preflight without a grep fast-path.
model_rotation() {   # $1 = model dir
  local desc="$1/quant_model_description.json"
  [[ -f "${desc}" ]] || { echo absent; return 0; }
  python3 - "${desc}" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("unreadable"); sys.exit(0)
o = d.get("optional")
print(",".join(sorted(map(str, o))) if isinstance(o, dict) and o else "none")
PY
}

# the predictor checkpoint's own `meta`, so a geometry mismatch is caught in seconds
# instead of after a two-minute model load. mmap keeps the ~460 MiB of bf16 head weights
# out of RAM — only the small `meta` dict is touched. Echoes
#   arch|in_dim|L|E|top_k|width   or   unreadable
# and never fails, for the same reason model_rotation() never fails.
predictor_meta() {   # $1 = .pt path
  python3 - "$1" <<'PY'
import sys
try:
    import torch
    try:
        s = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)
    except TypeError:                     # torch predates mmap=
        s = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    m = s["meta"]
    print("|".join(str(m[k]) for k in ("arch", "in_dim", "L", "E", "top_k", "hidden")))
except Exception:
    print("unreadable")
PY
}

# one failure path for the rotation gate, so the WHY/SYMPTOM/FIX explanation is written
# once instead of three times. $1 is the specific finding. guard() rather than die() to match
# every other safety check here — ALLOW_UNSAFE=1 downgrades it to a warning.
gate_rot_fail() {   # $1 = what was found
  guard "$1
       WHY   : a router gate is only meaningful in the hidden-space basis it was trained in.
               QuaRot's R1 acts on the residual stream and is absorbed into every input
               projection INCLUDING mlp.gate (https://arxiv.org/abs/2404.00456), so a rotated
               build stores W@R1 where an unrotated one stores W. The two are identical in
               shape, in every row norm and in their singular values, so the engine's shape
               check, its layer-index checks and its fp32 verifier all pass on the wrong one.
       SYMPTOM: no crash and no error. The forward pass runs at full speed; attention, the
               shared expert and the three hash-routed layers are untouched. The model simply
               stops emitting EOS, so every request runs to max_out_len — at MAX_NUM_SEQS=1
               that is ~15 min per prompt and AISBench reads as hung.
       FIX   : serve ${REMOE_GATE} on the build it was fine-tuned from; or set REMOE_GATE_ROT
               to match this build if that is genuinely correct; or unset REMOE_GATE to
               measure the base router. ALLOW_UNSAFE=1 to override."
}

export ASCEND_RT_VISIBLE_DEVICES="${CARD}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"; export NO_PROXY="${no_proxy}"
[[ -n "${AIS_BENCH_DATASETS_CACHE}" ]] && export AIS_BENCH_DATASETS_CACHE \
                                       || unset AIS_BENCH_DATASETS_CACHE
# env_a5_0260.sh owns VLLM_BATCH_INVARIANT: it is read at BUILD time too. Default it
# only if nothing set it — do not contradict the binaries.
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-0}"

RUN_LOG="${OUT_DIR}/run.log"; SERVE_LOG="${OUT_DIR}/serve.log"
CFG_ROOT="${OUT_DIR}/configs"; CFG_DIR="${CFG_ROOT}/models"
STATS_DIR="${CSV_PATH:-${OUT_DIR}/stats}"
SERVE_PID=""; SERVE_PGID=""; SERVE_SID=""; HB_PID=""; TAIL_PID=""; MIRROR_AT=0
STATS_REPORTED=0

# =============================================================================
#  derived values and validation
# =============================================================================

case "${MTP}" in
  0) SPEC_METHOD="" ;;
  1) SPEC_METHOD="mtp" ;;
  2) SPEC_METHOD="dspark" ;;
  *) die "MTP must be 0 (off), 1 (mtp) or 2 (dspark)" ;;
esac
DECODE_TOK_PER_REQ=1
[[ -n "${SPEC_METHOD}" ]] && DECODE_TOK_PER_REQ=$(( 1 + NUM_SPEC_TOKENS ))
DECODE_TOKENS=$(( MAX_NUM_SEQS * DECODE_TOK_PER_REQ ))    # per DP rank / per capture
GRAPH_ON=0
[[ "${ENFORCE_EAGER}" != "1" && "${CUDAGRAPH_MODE}" != "NONE" ]] && GRAPH_ON=1
NCARDS="$(awk -F',' '{print NF}' <<<"${CARD}")"
EP_SIZE=$(( TP * DP ))
WORLD_SIZE=$(( TP * DP * PP ))   # cards; EP does not span pipeline stages
# env_a5_0260.sh sets the single-card OMP value and leaves the multi-card re-export here.
(( EP_SIZE > 1 )) && [[ -n "${VA_OMP_MULTI:-}" ]] && export OMP_NUM_THREADS="${VA_OMP_MULTI}"

# MOE ROWS is what the engine compares against offload_threshold, and it is NOT the
# per-rank decode token count: AllGatherCommImpl.prepare() all-gathers across the DP
# group, so the MoE layer sees DP x decode_tokens rows.
MOE_ROWS=$(( DECODE_TOKENS * DP ))

NDE_MIN="${NUM_DEVICE_EXPERTS}"
[[ "${NUM_DEVICE_EXPERTS}" == \[* ]] && \
  NDE_MIN="$(tr -d '[] ' <<<"${NUM_DEVICE_EXPERTS}" | tr ',' '\n' | sort -n | head -1)"

[[ -z "${THINK}" || "${THINK}" =~ ^[01]$ ]]                       || die "THINK must be empty, 0 or 1"
[[ -z "${THINK_EFFORT}" || "${THINK_EFFORT}" =~ ^(high|max)$ ]]   || die "THINK_EFFORT must be empty, high or max"
[[ "${RUN}" =~ ^(acc|perf|both)$ ]]                               || die "RUN must be acc, perf or both"
[[ "${NUM_SPEC_TOKENS}" =~ ^[1-9][0-9]*$ ]]                       || die "NUM_SPEC_TOKENS must be a positive integer"
[[ "${NDE_MIN}" =~ ^[1-9][0-9]*$ ]]                               || die "NUM_DEVICE_EXPERTS must be a positive integer or a JSON list"
[[ -z "${SHARD_PER_RANK}" || "${SHARD_PER_RANK}" =~ ^(true|false)$ ]] || die "SHARD_PER_RANK must be empty, true or false"
[[ -z "${CUDAGRAPH_CAPTURE_SIZES}" || "${CUDAGRAPH_CAPTURE_SIZES}" =~ ^(auto|auto-safe|[0-9]+(,[0-9]+)*)$ ]] \
  || die "CUDAGRAPH_CAPTURE_SIZES must be empty, auto, auto-safe, or a comma-separated list"
[[ "${HOT_PRELOAD}" != "1" || -n "${HOT_EXPERTS_FILE}" ]]         || die "HOT_PRELOAD=1 needs HOT_EXPERTS_FILE (a .json)"
[[ "${PP}" =~ ^[1-9][0-9]*$ ]]                                    || die "PP must be a positive integer"
[[ "${NCARDS}" == "${WORLD_SIZE}" ]] || die "CARD lists ${NCARDS} device(s) but TP*DP*PP=${WORLD_SIZE}"
# These mirror ExpertOffloadConfig._validate_config exactly. The nested key set is
# CLOSED, so the engine would raise on them anyway — but two minutes into a model load.
[[ "${H2D_BACKEND}" =~ ^(torch|memfabric)$ ]]                     || die "H2D_BACKEND must be torch or memfabric"
[[ "${MEMFABRIC_POOL_GIB}" =~ ^[0-9]+$ ]]                         || die "MEMFABRIC_POOL_GIB must be a non-negative integer"
[[ -z "${MEMFABRIC_LOG_LEVEL}" || "${MEMFABRIC_LOG_LEVEL}" =~ ^[0-9]+$ ]] || die "MEMFABRIC_LOG_LEVEL must be empty or an integer"
if [[ "${H2D_BACKEND}" == memfabric ]]; then
  [[ "${OFFLOAD}" == "1" ]] || die "H2D_BACKEND=memfabric with OFFLOAD=0: the backend only moves expert weights."
  (( MEMFABRIC_POOL_GIB > 0 )) || die "H2D_BACKEND=memfabric needs MEMFABRIC_POOL_GIB > 0. It is the PER-RANK
       DRAM contribution, so the aggregate SHARED pool is EP_SIZE x that value — size it to hold this
       rank's expert shard and its quantisation metadata."
  [[ "${ENABLE_MULTI_CARD}" != "1" || "${SHARD_PER_RANK}" != false ]] \
    || die "multi-card MemFabric requires SHARD_PER_RANK=true: SHARED mode keeps the host weights sharded
       and shares the POINTERS. Replicated host weights have nothing to publish."
fi
# Same contract as the block above: expert_predictor / expert_predictor_ckpt /
# expert_prefetch_wait_timing are nested keys in the CLOSED expert_offload_config set, and
# the engine raises on every one of these — but only after the model has loaded.
[[ "${EXPERT_PREDICTOR}" =~ ^(fate|mode2_har|mode2_prevhfr)$ ]] || die "EXPERT_PREDICTOR must be fate, mode2_har or mode2_prevhfr"
if [[ "${EXPERT_PREDICTOR}" != fate ]]; then
  [[ "${OFFLOAD}" == "1" ]] || die "EXPERT_PREDICTOR=${EXPERT_PREDICTOR} with OFFLOAD=0: the whole
       expert_offload_config block is only emitted when OFFLOAD=1, so the key would never reach the
       engine and the run would silently measure nothing."
  [[ "${PREFETCH}" == "1" ]] || die "EXPERT_PREDICTOR=${EXPERT_PREDICTOR} needs PREFETCH=1: a trained
       predictor drives the prefetch pipeline, and with prefetch disabled there is nothing to drive."
  [[ -n "${EXPERT_PREDICTOR_CKPT}" ]] || die "EXPERT_PREDICTOR=${EXPERT_PREDICTOR} needs
       EXPERT_PREDICTOR_CKPT (the trained head, e.g. har_prompt_std_dist_lowrank_w=1280_miss=0.pt)."
  [[ -r "${EXPERT_PREDICTOR_CKPT}" ]] || die "EXPERT_PREDICTOR_CKPT not found or unreadable:
       ${EXPERT_PREDICTOR_CKPT}"
fi
[[ "${EXPERT_PREFETCH_WAIT_TIMING}" =~ ^[01]$ ]] || die "EXPERT_PREFETCH_WAIT_TIMING must be 0 or 1"
if [[ "${EXPERT_PREFETCH_WAIT_TIMING}" == "1" ]]; then
  [[ "${OFFLOAD}" == "1" && "${PREFETCH}" == "1" ]] || die "EXPERT_PREFETCH_WAIT_TIMING=1 needs
       OFFLOAD=1 and PREFETCH=1: with no prefetch there is no join for the compute stream to
       block on, so there is nothing to measure."
fi
[[ -z "${REMOE_GATE}" || -e "${REMOE_GATE}" ]]                    || die "REMOE_GATE not found: ${REMOE_GATE}"
# the two DeepSeek-V4-Flash builds have byte-identical config.json apart from
# quantization_config vs expert_dtype, and every routing-relevant field (num_hash_layers=3,
# n_routed_experts=256, hidden_size=4096) is the same in both — so config.json cannot tell
# them apart. quant_model_description.json can: only the w4a8 build ships one, and only it
# declares optional.quarot. Checked here, at ~0.1s, rather than after a two-minute load.
# MODEL_ROT is seeded so the plan echo below can read it whether or not the gate is in use.
MODEL_ROT="not checked (no REMOE_GATE)"
if [[ -n "${REMOE_GATE}" ]]; then
  [[ -d "${MODEL}" ]] || die "MODEL is not a directory: ${MODEL}"
  [[ -n "${REMOE_GATE_ROT}" ]] || die "REMOE_GATE is set but REMOE_GATE_ROT is empty. Set it to the
       rotation the gates were fine-tuned against ('quarot' for the w4a8 build), or unset REMOE_GATE."
  MODEL_ROT="$(model_rotation "${MODEL}")"
  if [[ "${MODEL_ROT}" == "${REMOE_GATE_ROT}" ]]; then
    :   # the build declares the rotation these gates belong to
  elif [[ "${MODEL_ROT}" == absent ]]; then
    gate_rot_fail "REMOE_GATE is set, but ${MODEL} has no quant_model_description.json — this build
       declares no rotation at all, and REMOE_GATE_ROT=${REMOE_GATE_ROT} was required."
  elif [[ "${MODEL_ROT}" == unreadable ]]; then
    gate_rot_fail "REMOE_GATE is set, but ${MODEL}/quant_model_description.json could not be parsed
       as JSON, so this build's rotation cannot be established (required: ${REMOE_GATE_ROT})."
  else
    gate_rot_fail "REMOE_GATE is set, but ${MODEL} declares rotation='${MODEL_ROT}' and
       REMOE_GATE_ROT='${REMOE_GATE_ROT}' was required."
  fi
fi

# TOPK from the checkpoint — the engine reads the config, not this variable. MUST resolve
# before the threshold and capture arithmetic.
_CFG_TOPK="$(python3 - "${MODEL}/config.json" 2>/dev/null <<'PY'
import json, sys
try: c = json.load(open(sys.argv[1]))
except Exception: sys.exit(0)
t = c.get("text_config") if isinstance(c.get("text_config"), dict) else {}
for k in ("num_experts_per_tok", "num_experts_per_token"):
    if c.get(k): print(int(c[k])); break
    if t.get(k): print(int(t[k])); break
PY
)"
[[ -n "${_CFG_TOPK}" ]] && TOPK="${_CFG_TOPK}"
[[ "${TOPK}" =~ ^[1-9][0-9]*$ ]] || die "TOPK must be a positive integer (got '${TOPK}')"

# Predictor geometry gate. Every failure mode of a mismatched head is SILENT — plausible
# expert ids, a degraded hit rate, no error anywhere — so establish the mapping here, at a
# few seconds, rather than reading it out of pred_acc after a benchmark. Only runs for a
# trained predictor; a fate run does no extra work.
PREDICTOR_NOTE="off (fate: built-in hash/gate next-layer predictor)"
FATE_KEEPS=""
if [[ "${EXPERT_PREDICTOR}" != fate ]]; then
  _geom="$(python3 - "${MODEL}/config.json" 2>/dev/null <<'PY'
import json, sys
try: c = json.load(open(sys.argv[1]))
except Exception: sys.exit(0)
t = c.get("text_config") if isinstance(c.get("text_config"), dict) else {}
g = lambda k: c.get(k, t.get(k))
print("%s|%s|%s|%s" % (g("hc_mult"), g("hidden_size"),
                       g("num_hidden_layers"), g("num_hash_layers")))
PY
)"
  _pm="$(predictor_meta "${EXPERT_PREDICTOR_CKPT}")"
  if [[ "${_pm}" == unreadable || -z "${_geom}" ]]; then
    guard "cannot read the predictor checkpoint's meta or the model's geometry
       (ckpt='${EXPERT_PREDICTOR_CKPT}' meta='${_pm}' model_geom='${_geom}'). The engine would
       still validate at load time; ALLOW_UNSAFE=1 to proceed on its checks alone."
    PREDICTOR_NOTE="${EXPERT_PREDICTOR} (meta unreadable — engine will validate)"
  else
    IFS='|' read -r _p_arch _p_in _p_L _p_E _p_k _p_w <<<"${_pm}"
    IFS='|' read -r _hc _hid _nhl _nhash <<<"${_geom}"
    [[ "${_p_arch}" == lowrank ]] || die "predictor checkpoint arch='${_p_arch}': the port implements
       'lowrank' only (expert_predictor.py::_ARCH_HEADS). Retrain or extend the registry."
    (( _p_in == _hc * _hid )) || die "predictor in_dim=${_p_in} but the model has hc_mult=${_hc} x
       hidden_size=${_hid} = $(( _hc * _hid )). Both the 'har' and 'hfr' atoms are the FULL
       hyper-connection residual, so in_dim must equal hc_mult*hidden_size — this checkpoint was
       trained on a different model."
    # The dump either drops the leading hash-routed layers (L = 40) or covers all of them
    # (L = 43). Both are handled at runtime; anything else silently shifts every head onto
    # the wrong layer, which is the one failure this gate exists to catch.
    if   (( _p_L == _nhl - _nhash )); then _map="layer_offset=${_nhash} (hash layers not in the dump)"
    elif (( _p_L == _nhl ));          then _map="layer_offset=0 (dump covers all layers; hash skipped at runtime)"
    else guard "predictor covers L=${_p_L} layers, but this model has num_hidden_layers=${_nhl} and
       num_hash_layers=${_nhash}, so the only self-consistent values are $(( _nhl - _nhash )) or ${_nhl}.
       Every head would be mapped onto the wrong layer, with no error and no crash — only a poor
       pred_acc. Check metadata.json's dumped_layer_indices: they must be a contiguous TRAILING block."
         _map="layer_offset=$(( _nhl - _p_L )) UNVERIFIED"
    fi
    (( _p_k == TOPK )) || echo "  warn    : predictor top_k=${_p_k} but the model routes top_k=${TOPK} —
            the prediction width differs from the router's, which moves pred_acc and changes how many
            candidates EXPERT_PREFETCH_NUM ranks."
    # r = max(MIN_RANK=64, width//LOWRANK_DIV=4), per the study's _rank_of(). Head params are
    # bf16 (the dtype the study saves); mu/sd stay fp32. This is HBM taken from the same
    # budget the expert cache lives in, so it is worth seeing before the run, not after an OOM.
    _r=$(( _p_w / 4 )); (( _r < 64 )) && _r=64
    _mib=$(( (_p_L * _p_in * _r * 2 + _p_L * _r * _p_w * 2 + _p_L * _p_w * _p_E * 2 \
              + 2 * _p_L * _p_in * 4) / 1048576 ))
    PREDICTOR_NOTE="${EXPERT_PREDICTOR} arch=${_p_arch} L=${_p_L} ${_map} in_dim=${_p_in} width=${_p_w} rank=${_r} E=${_p_E} top_k=${_p_k} head~${_mib}MiB/rank"
    # A layer-shifted atom cannot cover the first dumped layer — the study zero-filled it
    # during training — so fate keeps one more target than it does under mode2_har.
    if [[ "${EXPERT_PREDICTOR}" == mode2_prevhfr ]]; then
      FATE_KEEPS="targets 1..${_nhash} (hash, exact tid2eid) and target ${_nhash} + 1 (first MoE layer: no predecessor in the dump)"
    else
      FATE_KEEPS="targets 1..$(( _nhash - 1 )) (hash, exact tid2eid)"
    fi
  fi
fi

THR=$(( NDE_MIN / TOPK ))          # offload_threshold, in MoE token ROWS
NDE_FLOOR=$(( MOE_ROWS * TOPK ))   # smallest pool that keeps decode on the offload path
THR_PER_RANK=$(( THR / DP ))       # largest safe per-rank capture size
# fate predicts from the POST-prepare() MoE input, so its row ceiling is MOE_ROWS. A trained
# predictor captures at the decoder layer, BEFORE prepare()'s DP all-gather, so its ceiling
# is this rank's own decode token count. Same knob, different denominator.
PF_ROWS_AVAIL="${MOE_ROWS}"
[[ "${EXPERT_PREDICTOR}" != fate ]] && PF_ROWS_AVAIL="${DECODE_TOKENS}"
PF_TOK_EFF=$(( EXPERT_PREFETCH_TOKENS < PF_ROWS_AVAIL ? EXPERT_PREFETCH_TOKENS : PF_ROWS_AVAIL ))

CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES}"; AUTO_NOTE=""
if [[ "${CAPTURE_SIZES}" == auto || "${CAPTURE_SIZES}" == auto-safe ]]; then
  _mode="${CAPTURE_SIZES}"; _cap="${DECODE_TOKENS}"
  [[ "${_mode}" == auto-safe && "${OFFLOAD}" == "1" ]] && (( THR_PER_RANK < _cap )) && _cap="${THR_PER_RANK}"
  (( _cap > MAX_NUM_BATCHED_TOKENS )) && _cap="${MAX_NUM_BATCHED_TOKENS}"
  (( _cap < 1 )) && _cap=1          # an empty list silently falls back to vLLM defaults
  _sizes=(); for (( _s=1; _s<=_cap; _s++ )); do _sizes+=( "${_s}" ); done
  IFS=','; CAPTURE_SIZES="${_sizes[*]}"; unset IFS
  AUTO_NOTE="${_mode} -> [${CAPTURE_SIZES}]  (decode_tokens=${DECODE_TOKENS})"
  (( _cap < DECODE_TOKENS )) && AUTO_NOTE+="  TRUNCATED: the real decode shape is NOT captured, decode runs EAGER"
  unset _mode _cap _s _sizes
fi

TASK_LIST="$(tr ',' ' ' <<<"${TASKS}")"
PKG_CFG="$(python3 -c 'import os,ais_bench.benchmark as b;print(os.path.join(os.path.dirname(b.__file__),"configs"))' 2>/dev/null)"
DATA_ROOT="${AIS_BENCH_DATASETS_CACHE:-$(python3 -c 'import os,ais_bench;print(os.path.dirname(os.path.dirname(os.path.abspath(ais_bench.__file__))))' 2>/dev/null)}"

# =============================================================================
#  helpers
# =============================================================================

# Sets P_DS P_ACC_MODE P_PERF_MODE P_TEMP P_TOPP P_MAXOUT P_NUM P_AVGN for one task.
preset() {
  local line acc perf m
  line=$(awk -F'|' -v t="$1" '
    { for (i=1;i<=NF;i++) gsub(/^[ \t]+|[ \t]+$/, "", $i) }
    $1 == t { print $2"|"$3"|"$4"|"$5"|"$6"|"$7"|"$8"|"$9; f=1 } END { exit !f }' <<<"${PRESETS}") \
    || die "unknown task '$1' — valid: $(awk -F'|' 'NF>1{gsub(/ /,"",$1);printf "%s ",$1}' <<<"${PRESETS}")"
  IFS='|' read -r P_DS acc perf P_TEMP P_TOPP P_MAXOUT P_NUM P_AVGN <<<"${line}"
  for m in acc perf; do
    local v="${!m}"
    [[ "${THINK}" == "0" ]] && v="non-think"
    [[ "${THINK}" == "1" && "${v}" == "non-think" ]] && v="${THINK_EFFORT:-high}"
    [[ -n "${THINK_EFFORT}" && "${v}" != "non-think" ]] && v="${THINK_EFFORT}"
    [[ "${m}" == acc ]] && P_ACC_MODE="${v}" || P_PERF_MODE="${v}"
  done
  [[ "${NUM_PROMPTS}" != "unset" ]] && P_NUM="${NUM_PROMPTS}"
  [[ -n "${AVG_N}" ]] && P_AVGN="${AVG_N}"
  [[ -n "${TEMP}"  ]] && P_TEMP="${TEMP}"
  [[ "${TOPP}" != "unset" ]] && P_TOPP="${TOPP}"
  return 0
}

list_cfgs() {
  local d
  for d in gsm8k gpqa mmlu_pro; do
    echo "  [${d}]"
    find "${PKG_CFG}/datasets/${d}" -maxdepth 1 -name '*.py' 2>/dev/null \
      | sed 's|.*/||;s|\.py$||;s|^|    |' | sort
  done
}

# --config-dir REPLACES ais_bench's config root; it is NOT additive. --datasets resolves
# against <config-dir>/datasets and --summarizer against <config-dir>/summarizers, so a
# config dir holding only our generated models/ makes EVERY pass die at config
# resolution — before one request reaches the server, which reads like a hang.
# SYMLINKS, not copies: the shipped configs use read_base() with relative imports and
# only resolve at their real path. models/ is deliberately NOT linked — ours must win.
link_pkg_cfg() {
  local src base
  [[ -n "${PKG_CFG}" && -d "${PKG_CFG}" ]] || return 0
  for src in "${PKG_CFG}"/*; do
    [[ -d "${src}" ]] || continue
    base="$(basename "${src}")"
    [[ "${base}" == models || "${base}" == __pycache__ ]] && continue
    ln -sfn "${src}" "${CFG_ROOT}/${base}" 2>/dev/null \
      || die "cannot link ${src} -> ${CFG_ROOT}/${base} (ais_bench would not find its datasets)"
  done
  return 0
}

ensure_dirs() {
  local d dirs=( "${OUT_DIR}" "${CFG_DIR}" )
  for d in acc perf; do [[ "${RUN}" == "both" || "${RUN}" == "${d}" ]] && dirs+=( "${OUT_DIR}/${d}" ); done
  [[ "${DECODE_STATS}" == "1" ]] && dirs+=( "${STATS_DIR}" )
  for d in "${dirs[@]}"; do
    mkdir -p "${d}" 2>/dev/null || die "cannot create directory: ${d}"
    [[ -w "${d}" ]] || die "directory not writable: ${d}"
  done
  link_pkg_cfg
  return 0
}

# Each source is newline-TERMINATED: lsof -t can omit the trailing newline and the next
# source then lands on the same line, fusing two pids into one that matches nothing.
port_pids() {
  { lsof -t -i "TCP:${PORT}" -sTCP:LISTEN 2>/dev/null; echo
    fuser -n tcp "${PORT}" 2>/dev/null | tr -s ' ' '\n'; echo
    ss -lptnH "sport = :${PORT}" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2; echo
  } | grep -E '^[0-9]+$' | sort -u | tr '\n' ' '
}
kill_tree() { local k; for k in $(pgrep -P "$1" 2>/dev/null); do kill_tree "${k}" "$2"; done
              kill "-$2" "$1" 2>/dev/null; return 0; }
proc_tree() { local pid="$1" kid; echo "${pid}"
              for kid in $(pgrep -P "${pid}" 2>/dev/null); do proc_tree "${kid}"; done; }
pid_sid()   { local s; s="$(ps -o sid= -p "$1" 2>/dev/null | tr -d ' ')"
              [[ -z "${s}" ]] && s="$(ps -o sess= -p "$1" 2>/dev/null | tr -d ' ')"; echo "${s}"; }

# Every process in the SERVER's session. pgrep -P cannot see a worker that has been
# re-parented to init — the parent link is gone the moment the top process dies, while
# the worker still holds NPU memory. The session id survives re-parenting. Armed only
# when the server is CONFIRMED to be in a session of its own, so it can never name this
# script. Zombies are skipped: already dead, impossible to signal.
session_pids() {
  [[ -z "${SERVE_SID}" ]] && { echo ""; return 0; }
  { ps -eo sid=,pid=,stat= 2>/dev/null || ps -eo sess=,pid=,state= 2>/dev/null; } \
    | awk -v s="${SERVE_SID}" -v me="$$" '$1==s && $2!=me && $3 !~ /^Z/ {printf "%s ", $2}'
}
serve_alive() {
  if [[ -n "${SERVE_SID}" ]]; then [[ -n "$(session_pids)" ]] && return 0; return 1; fi
  [[ -n "${SERVE_PGID}" ]] && kill -0 -"${SERVE_PGID}" 2>/dev/null && return 0
  [[ -n "${SERVE_PID}"  ]] && kill -0  "${SERVE_PID}"  2>/dev/null && return 0
  return 1
}
# reap() SIGTERMs whatever this returns, so it must never name a process this script did
# not start. No host-wide fallback: a `pgrep -f EngineCore` sweep once TERMed a
# concurrent benchmark's engine mid-run.
engine_pids() {
  [[ -z "${SERVE_PID}" ]] && { echo ""; return 0; }
  local pid out=""
  for pid in $(proc_tree "${SERVE_PID}"); do
    [[ "${pid}" == "${SERVE_PID}" ]] && continue
    ps -o args= -p "${pid}" 2>/dev/null | grep -q 'EngineCore' && out+="${pid} "
  done
  echo "${out}"
}

HEALTH_URL=""      # set once PORT is final
# NOT curl: env_a5_0260.sh prepends ${CONDA_PREFIX}/lib to LD_LIBRARY_PATH, so
# /usr/bin/curl resolves the system libldap against conda's libcrypto and dies in the
# loader (exit 127, nothing sent) — indistinguishable from "the server is not ready".
# python3 IS the conda interpreter, so its libraries match by construction. Proxies are
# disabled explicitly: urllib honours http_proxy by default and this box sets it.
#   http_req <timeout> <url> [json-body] -> 0 ok | 7 unreachable | 22 http>=400 | 28 timeout
http_req() {
  python3 - "${1}" "${2}" "${3-}" <<'REQ'
import sys, urllib.error, urllib.request
timeout, url, body = float(sys.argv[1]), sys.argv[2], sys.argv[3]
req = urllib.request.Request(url)
if body:
    req.data = body.encode(); req.add_header("Content-Type", "application/json")
try:
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout) as r:
        sys.stdout.write(r.read().decode("utf-8", "replace"))
except urllib.error.HTTPError as e:
    sys.stderr.write("HTTP %s %s\n" % (e.code, e.reason)); sys.exit(22)
except urllib.error.URLError as e:
    sys.stderr.write("%s\n" % (e.reason,)); sys.exit(28 if isinstance(e.reason, TimeoutError) else 7)
except TimeoutError:
    sys.stderr.write("timed out\n"); sys.exit(28)
except Exception as e:
    sys.stderr.write("%s\n" % (e,)); sys.exit(7)
REQ
}

# Printed once, when the engine has logged that it is serving but /health still will not
# answer. At that point "still loading" is no longer a possible explanation.
health_diag() {
  local rc out l
  out="$(http_req 5 "${HEALTH_URL}" 2>&1 >/dev/null)"; rc=$?
  echo "[!!] the engine says it is serving, but ${HEALTH_URL} does not answer."
  grep -m2 'Starting vLLM server on\|Application startup complete' "${SERVE_LOG}" 2>/dev/null | sed 's/^/     /'
  echo "     http_req exit=${rc} ${out}"
  case "${rc}" in
    0)  echo "     -> it answers NOW; a startup race, the poll will pick it up." ;;
    7)  echo "     -> unreachable/refused. Compare the port with the engine's own line above." ;;
    22) echo "     -> it ANSWERED with an error status. Read the engine log, not the network." ;;
    28) echo "     -> timed out. Proxies are disabled here; look at a transparent proxy or iptables." ;;
    *)  echo "     -> python3/urllib itself failed — read the message above." ;;
  esac
  l="$(port_pids)"; echo "     listeners on :${PORT} -> ${l:-none}"
  command -v ss >/dev/null 2>&1 && ss -ltnp "sport = :${PORT}" 2>/dev/null | sed 's/^/     /'
  return 0
}

mirror_start() { tail -n "+$(( MIRROR_AT + 1 ))" -f "${SERVE_LOG}" >&3 2>/dev/null & TAIL_PID=$!; return 0; }
mirror_stop()  { [[ -n "${TAIL_PID}" ]] && kill "${TAIL_PID}" 2>/dev/null; TAIL_PID=""
                 MIRROR_AT=$(wc -l < "${SERVE_LOG}" 2>/dev/null || echo 0); return 0; }

# The mirror reaches the terminal only, so without this the engine's own DECODE-STATS
# lines never land in run.log.
stats_tail() {
  [[ -s "${SERVE_LOG}" ]] || return 0
  local n; n="$(grep -cE '\[DECODE-STATS\]|\[EXPERT-OFFLOAD' "${SERVE_LOG}" 2>/dev/null)"; n="${n:-0}"
  (( n == 0 )) && return 0
  echo "[stats]  engine lines so far (${n}); last 10:"
  grep -E '\[DECODE-STATS\]|\[EXPERT-OFFLOAD' "${SERVE_LOG}" | tail -10 | sed 's/^/           /'
  return 0
}

# ais_bench exits 0 on HTTP 500s, so its return code cannot see a dead engine. Must be
# called BEFORE reap(), which writes the same SIGTERM line.
engine_died_early() {
  grep -qE 'engine core exited unexpectedly|EngineDeadError|EngineCore: trigger received signal' \
       "${SERVE_LOG}" 2>/dev/null
}

install_traps() { trap 'on_signal INT' INT; trap 'on_signal TERM' TERM
                  trap 'on_signal HUP' HUP; trap on_exit EXIT; return 0; }

# 1. TERM the EngineCore alone (it logs [EXPERT-OFFLOAD-FINAL]) and wait ENGINE_GRACE —
#    signalling the group at once lets the API server force-kill it at timeout=0s first.
# 2. TERM the group and the tree, wait REAP_WAIT.  3. KILL group, tree, session, port.
# Idempotent, and NOT interruptible: a second ^C during the waits used to kill the script
# after the TERM but before the KILL escalation, which is how a setsid'd server outlives
# its own script.
reap() {
  trap '' INT TERM HUP QUIT
  [[ -n "${HB_PID}" ]] && kill "${HB_PID}" 2>/dev/null; HB_PID=""
  [[ -z "${SERVE_PID}${SERVE_PGID}${SERVE_SID}" ]] && { mirror_stop; return 0; }
  local i epids left
  epids="$(engine_pids)"
  if [[ -n "${epids// /}" && "${ENGINE_GRACE}" != "0" ]]; then
    echo "[reap] SIGTERM EngineCore first (pids=${epids% }); up to ${ENGINE_GRACE}s to flush" >&2
    kill -TERM ${epids} 2>/dev/null
    for ((i=0; i<ENGINE_GRACE; i++)); do kill -0 ${epids} 2>/dev/null || break; sleep 1; done
  else
    echo "[reap] no EngineCore in this run's tree (already gone, or outside it)" >&2
  fi
  echo "[reap] stopping server (pid=${SERVE_PID:-none} pgid=${SERVE_PGID:-none} sid=${SERVE_SID:-none})" >&2
  [[ -n "${SERVE_PGID}" ]] && kill -TERM -"${SERVE_PGID}" 2>/dev/null
  [[ -n "${SERVE_PID}"  ]] && kill_tree "${SERVE_PID}" TERM
  for ((i=0; i<REAP_WAIT; i++)); do serve_alive || break; sleep 1; done
  serve_alive && echo "[reap] no clean exit in ${REAP_WAIT}s — SIGKILL. The log block is lost;" \
                      "the artefacts on disk are the last snapshot." >&2
  sleep 2                     # tail -f is async; cutting it now truncates the tail
  mirror_stop
  [[ -n "${SERVE_PGID}" ]] && kill -KILL -"${SERVE_PGID}" 2>/dev/null
  [[ -n "${SERVE_PID}"  ]] && kill_tree "${SERVE_PID}" KILL
  left="$(session_pids)"; [[ -n "${left// /}" ]] && { kill -KILL ${left} 2>/dev/null; sleep 1; }
  left="$(port_pids)"
  [[ -n "${left// /}" ]] && { kill -KILL ${left} 2>/dev/null; sleep 2; left="$(port_pids)"; }
  [[ -n "${left// /}" ]] \
    && echo "[reap] WARNING: still listening on :${PORT} -> ${left}; check npu-smi info" >&2 \
    || echo "[reap] port ${PORT} released" >&2
  SERVE_PID=""; SERVE_PGID=""; SERVE_SID=""; return 0
}

# Called from the happy path AND the traps, so an interrupted run still surfaces its
# artefacts instead of leaving them on disk unmentioned. Idempotent.
report_stats() {
  (( STATS_REPORTED )) && return 0
  STATS_REPORTED=1
  [[ "${DECODE_STATS}" == "1" ]] || return 0
  echo "────────────────────────────────────────────────────────────"
  local pat _n _f
  for pat in 'config' 'topology' 'armed' 'first sample' 'heartbeat' 'flush' 'model runner shutdown' 'callback failed'; do
    _n="$(grep -c "DECODE-STATS. ${pat}\|EXPERT-OFFLOAD. .*${pat}" "${SERVE_LOG}" 2>/dev/null)"
    printf '[stats]  trace %-24s %s\n' "${pat}" "${_n:-0}"
  done
  # Newest by MTIME: the filename is rank<R>_<timestamp>, so a lexical sort picks the
  # highest RANK, not the latest file.
  _f="$(find "${STATS_DIR}" -maxdepth 1 -name 'decode_stats_summary_*.txt' -printf '%T@ %p\n' 2>/dev/null \
        | sort -n | tail -1 | cut -d' ' -f2-)"
  if [[ -n "${_f}" && -s "${_f}" ]]; then
    echo "[stats]  artefact: ${_f}"
    grep -q 'decode steps=0 ' "${_f}" && \
      echo "[!!]     decode steps=0 — this artefact is EMPTY. The run armed but never reached
         the paging decode path. Not a result."
    sed 's/^/  /' "${_f}"
  else
    echo "[stats]  WARNING: no summary under ${STATS_DIR}. Read the traces above:"
    echo "         config=0 keys swallowed | topology=0 offload never registered"
    echo "         armed=0 warmup hook missing | first sample=0 decode never hit the paging path"
  fi
  grep -q '\[EXPERT-OFFLOAD-FINAL\]' "${SERVE_LOG}" 2>/dev/null && {
    echo "[stats]  [EXPERT-OFFLOAD-FINAL] from serve.log:"
    awk '/\[EXPERT-OFFLOAD-FINAL\]/{f=1} f{print} f&&/^=+$/{r++; if(r>=2) exit}' \
      "${SERVE_LOG}" | head -60 | sed 's/^/  /'; }
  grep -q 'host callback failed' "${SERVE_LOG}" 2>/dev/null && {
    echo "[stats]  WARNING: expert-offload host callback failures — paging did not complete"
    grep -m3 'host callback failed' "${SERVE_LOG}" | sed 's/^/           /'; }
  # pf_wait degrades to "no samples" rather than raising, and decode_stats omits
  # a metric with an empty series entirely — so an absent row looks identical to
  # a key that never took. These two checks are the only thing that tells them
  # apart. Runtime signals, so they belong here, after decoding.
  if [[ "${EXPERT_PREFETCH_WAIT_TIMING}" == "1" ]]; then
    grep -qE 'elapsed_time (failed|is not usable)' "${SERVE_LOG}" 2>/dev/null && {
      echo "[stats]  WARNING: prefetch-stall timing failed at runtime — the pf_wait rows are"
      echo "           ABSENT, not zero. Traceback in serve.log."
      grep -m2 -E 'elapsed_time (failed|is not usable)' "${SERVE_LOG}" | sed 's/^/           /'; }
    grep -q 'PREFETCH-WAIT. first sample' "${SERVE_LOG}" 2>/dev/null || {
      echo "[stats]  WARNING: timing armed but produced NO samples — the pf_wait rows are ABSENT,"
      echo "           not zero. Read the '[PREFETCH-WAIT] first read' line below:"
      echo "             armed=0   no layer had a wait node when the first sample was read."
      echo "                       A prefill between capture and decode can clear every flag."
      echo "             armed>0   elapsed_time is failing inside the captured graph."
      grep -m3 'PREFETCH-WAIT' "${SERVE_LOG}" | sed 's/^/           /'; }
  fi
  return 0
}

on_signal() {
  trap '' PIPE           # a dead tee must not kill this shell before reap() has run
  echo >&2; echo "[signal] caught SIG${1:-INT} — stopping the server. Further ^C is IGNORED" \
                 "until the tree is down." >&2
  # A ^C is not an orderly end-of-run flush. Clamped, not zeroed: 20s still lets the
  # EngineCore write its summary.
  (( ENGINE_GRACE > 20 )) && ENGINE_GRACE=20
  (( REAP_WAIT   > 45 )) && REAP_WAIT=45
  reap; report_stats; exit 130
}
on_exit() { reap; report_stats; }

# Sets MODEL_TASK, CFG_FILE, P_PASS_MODE, CMD and the P_* globals.
build_cmd() {   # $1=task  $2=acc|perf
  local task="$1" pass="$2" gk n
  preset "${task}"
  [[ "${pass}" == perf ]] && P_PASS_MODE="${P_PERF_MODE}" || P_PASS_MODE="${P_ACC_MODE}"
  MODEL_TASK="v4_${task}_${pass}"; CFG_FILE="${CFG_DIR}/${MODEL_TASK}.py"

  if [[ "${DRY_RUN}" != "1" ]]; then
    if [[ "${pass}" == perf ]]; then
      gk="            temperature=${PERF_TEMP},"
    else
      gk="            temperature=${P_TEMP},"
      [[ -n "${P_TOPP}" ]] && gk+=$'\n'"            top_p=${P_TOPP},"
    fi
    # The mode is always stated explicitly, including thinking=False, so a run can never
    # silently sit in the wrong one.
    if [[ "${P_PASS_MODE}" == non-think ]]; then
      gk+=$'\n'"            chat_template_kwargs=dict(thinking=False),"
    else
      gk+=$'\n'"            chat_template_kwargs=dict(thinking=True, reasoning_effort=\"${P_PASS_MODE}\"),"
    fi
    [[ "${pass}" == acc && "${P_AVGN}" -gt 1 ]] && gk+=$'\n'"            num_return_sequences=${P_AVGN},"
    cat > "${CFG_FILE}" <<PYCFG
# GENERATED $(date -Is) — task=${task} pass=${pass} mode=${P_PASS_MODE}
from ais_bench.benchmark.models import VLLMCustomAPIChat
try:  # this path moved between releases
    from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content
except ImportError:
    from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service", type=VLLMCustomAPIChat, abbr="v4-${task}-${pass}",
        path="${TOKENIZER}",          # the perf summarizer loads this; acc never does
        model="${SERVED_NAME}",
        stream=$([[ "${pass}" == perf ]] && echo True || echo False),
        request_rate=0, retry=5, api_key="",
        host_ip="127.0.0.1", host_port=${PORT},
        max_out_len=${P_MAXOUT}, batch_size=${MAX_NUM_SEQS},
        trust_remote_code=True,
        generation_kwargs=dict(
${gk}
        ),
        # Runs before the dataset's answer regex: strips any inline <think> block.
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
PYCFG
  fi

  CMD=( ais_bench --models "${MODEL_TASK}" --datasets "${P_DS}"
        --config-dir "${CFG_ROOT}" --work-dir "${OUT_DIR}/${pass}" )
  if [[ "${pass}" == acc ]]; then
    CMD+=( --mode all --dump-eval-details --merge-ds )
    [[ "${EXTRACT_RATE}" == "1" ]] && CMD+=( --dump-extract-rate )
    n="${P_NUM}"
  else
    CMD+=( --mode perf --summarizer default_perf )     # --merge-ds is accuracy-only
    n="${PERF_PROMPTS}"
  fi
  [[ -n "${n}" ]] && CMD+=( --num-prompts "${n}" )
  return 0
}

report_step() {   # $1=task $2=acc|perf $3=rc
  local t="$1" p="$2" rc="$3" dir f found=0 pat
  (( rc != 0 )) && { echo "[result] ${t} ${p}: ais_bench FAILED rc=${rc}" >&2; return 0; }
  dir="$(find "${OUT_DIR}/${p}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)"
  [[ -z "${dir}" ]] && { echo "[result] ${t} ${p}: no run dir"; return 0; }
  while IFS= read -r f; do [[ -n "${f}" ]] && echo "[log]    ${t} ${p}: ${f}"; done \
    < <(find "${dir}/logs" -name '*.out' 2>/dev/null | sort)
  [[ "${p}" == acc ]] && pat=( -name 'summary_*.txt' ) || pat=( -path '*performances*' -name '*.csv' )
  while IFS= read -r f; do
    [[ -z "${f}" ]] && continue
    found=1; echo "[result] ${t} ${p} (${f}):"; sed 's/^/           /' "${f}"
  done < <(find "${dir}" "${pat[@]}" 2>/dev/null | sort)
  (( found == 0 )) && echo "[result] ${t} ${p}: nothing to report under ${dir}"
  [[ "${p}" == acc && "${P_AVGN:-1}" -gt 1 ]] && \
    echo "[warn]   read avg@${P_AVGN} ONLY — pass@n / cons@n are broken (see FINDINGS)"
  return 0
}

# A chat-template kwarg the template does not declare is dropped SILENTLY, so a run can
# sit in the wrong mode — worth ~16 points on gpqa. One 64-token request settles it.
probe_mode() {   # $1 = non-think | high | max
  local kw want got resp
  if [[ "$1" == non-think ]]; then want=off; kw='{"thinking":false}'
  else want=on; kw="{\"thinking\":true,\"reasoning_effort\":\"$1\"}"; fi
  resp=$(http_req 600 "http://127.0.0.1:${PORT}/v1/chat/completions" \
    "{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2?\"}],\"max_tokens\":64,\"chat_template_kwargs\":${kw}}") \
    || { echo "[probe] WARN: ${1} request failed; skipping" >&2; return 0; }
  got=$(python3 -c "
import json,sys
m = json.loads(sys.argv[1])['choices'][0]['message']
print('on' if (m.get('reasoning_content') or m.get('reasoning') or '').strip() else 'off')" "${resp}") \
    || { echo "[probe] WARN: unparsable response" >&2; return 0; }
  echo "[probe] ${1}: thinking=${got} (wanted ${want})"
  [[ "${got}" == "${want}" ]] || die "server is not honouring chat_template_kwargs for '${1}'.
       The template may use a different key (edit build_cmd: thinking -> enable_thinking).
       PROBE=0 to override."
  return 0
}

# =============================================================================
#  build the serve argv
# =============================================================================
# expert_offload_config is a CLOSED key set — an unknown key raises at startup, which is
# the good failure mode. TOP-LEVEL keys use .get(), so a typo there is ignored SILENTLY;
# hence the post-health greps. Every key is emitted only when non-empty, so an untouched
# knob keeps the engine's default rather than this script's opinion of it.

[[ "${LIST}" == "1" ]] && { list_cfgs; exit 0; }

PLAN=(); MODES=""
for t in ${TASK_LIST}; do
  preset "${t}"
  for p in acc perf; do
    [[ "${RUN}" == both || "${RUN}" == "${p}" ]] || continue
    PLAN+=( "${t}:${p}" )
    [[ "${p}" == perf ]] && MODES+=" ${P_PERF_MODE}" || MODES+=" ${P_ACC_MODE}"
  done
done
MODES="$(tr ' ' '\n' <<<"${MODES}" | grep -v '^$' | sort -u | tr '\n' ' ')"

ADDL_PARTS=()
if [[ "${OFFLOAD}" == "1" ]]; then
  p="\"expert_offload\":true,\"num_device_experts\":${NUM_DEVICE_EXPERTS}"
  p+=",\"num_device_layers\":${NUM_DEVICE_LAYERS},\"cache_policy_enabled\":$(bool "${CACHE_POLICY}")"
  [[ -n "${EXPERT_MAP_PATH}" ]] && p+=",\"expert_map_path\":\"${EXPERT_MAP_PATH}\""
  [[ "${PREFETCH}" == "1" ]] && \
    p+=",\"expert_prefetch_enabled\":true,\"expert_prefetch_num\":${EXPERT_PREFETCH_NUM}\
,\"expert_prefetch_tokens\":${EXPERT_PREFETCH_TOKENS}"
  # Emitted only when non-default, like every other knob here: a fate run sends exactly the
  # JSON it sent before these keys existed, so old and new runs stay comparable.
  if [[ "${EXPERT_PREDICTOR}" != fate ]]; then
    p+=",\"expert_predictor\":\"${EXPERT_PREDICTOR}\""
    p+=",\"expert_predictor_ckpt\":\"${EXPERT_PREDICTOR_CKPT}\""
  fi
  [[ "${EXPERT_PREFETCH_WAIT_TIMING}" == "1" ]] && p+=",\"expert_prefetch_wait_timing\":true"
  if [[ "${EXPERT_SUBSTITUTION}" == "1" ]]; then
    p+=",\"expert_substitution_enabled\":true"
    [[ -n "${EXPERT_SUBSTITUTION_THRESHOLD}" ]] && p+=",\"expert_substitution_threshold\":${EXPERT_SUBSTITUTION_THRESHOLD}"
  fi
  [[ "${HOT_PRELOAD}" == "1" ]] && p+=",\"hot_expert_preload\":true,\"hot_experts_file\":\"${HOT_EXPERTS_FILE}\""
  [[ "${ENABLE_MULTI_CARD}" == "1" ]] && p+=",\"enable_multi_card\":true"
  [[ -n "${SHARD_PER_RANK}"           ]] && p+=",\"shard_per_rank\":${SHARD_PER_RANK}"
  # Emitted only when non-default, like every other knob here: a torch-backend run sends
  # exactly the JSON it sent before these keys existed, so old and new runs stay comparable.
  if [[ "${H2D_BACKEND}" != torch ]]; then
    p+=",\"h2d_backend\":\"${H2D_BACKEND}\",\"memfabric_pool_size_gib\":${MEMFABRIC_POOL_GIB}"
    [[ -n "${MEMFABRIC_LOG_LEVEL}" ]] && p+=",\"memfabric_log_level\":${MEMFABRIC_LOG_LEVEL}"
  fi
  [[ -n "${CACHE_RECENT_WINDOW}"      ]] && p+=",\"cache_recent_window\":${CACHE_RECENT_WINDOW}"
  [[ -n "${CACHE_EMA_BETA}"           ]] && p+=",\"cache_ema_beta\":${CACHE_EMA_BETA}"
  [[ -n "${CACHE_RECENT_WEIGHT}"      ]] && p+=",\"cache_recent_weight\":${CACHE_RECENT_WEIGHT}"
  [[ -n "${CACHE_EMA_WEIGHT}"         ]] && p+=",\"cache_ema_weight\":${CACHE_EMA_WEIGHT}"
  [[ -n "${CACHE_ROUTER_WEIGHT}"      ]] && p+=",\"cache_router_weight\":${CACHE_ROUTER_WEIGHT}"
  [[ -n "${CACHE_AGE_WEIGHT}"         ]] && p+=",\"cache_age_weight\":${CACHE_AGE_WEIGHT}"
  [[ -n "${CACHE_STATS_LOG_INTERVAL}" ]] && p+=",\"cache_stats_log_interval\":${CACHE_STATS_LOG_INTERVAL}"
  p+=",\"moe_offload_debug\":$(bool "${MOE_OFFLOAD_DEBUG}")"
  ADDL_PARTS+=( "\"expert_offload_config\":{${p}}" )
fi
[[ -n "${REMOE_GATE}" ]] && ADDL_PARTS+=( "\"moe_gate_override_path\":\"${REMOE_GATE}\"" )
if [[ "${DECODE_STATS}" == "1" ]]; then
  ADDL_PARTS+=( "\"decode_stats_path\":\"${STATS_DIR}\"" )
  ADDL_PARTS+=( "\"decode_stats_flush_every\":${STATS_FLUSH_EVERY}" )
  ADDL_PARTS+=( "\"decode_stats_flush_seconds\":${STATS_FLUSH_SECONDS}" )
  [[ "${CSV}" == "1" ]] && ADDL_PARTS+=( "\"decode_stats_csv\":true" )
else
  ADDL_PARTS+=( "\"decode_stats_enabled\":false" )
fi
ADDL_PARTS+=( "\"enable_cpu_binding\":$(bool "${ENABLE_CPU_BINDING}")" )
ADDL_PARTS+=( "\"multistream_overlap_shared_expert\":$(bool "${MULTISTREAM_OVERLAP_SHARED_EXPERT}")" )
ACC_PARTS=()
[[ -n "${ENABLE_NPUGRAPH_EX}"   ]] && ACC_PARTS+=( "\"enable_npugraph_ex\":${ENABLE_NPUGRAPH_EX}" )
[[ -n "${ENABLE_STATIC_KERNEL}" ]] && ACC_PARTS+=( "\"enable_static_kernel\":${ENABLE_STATIC_KERNEL}" )
if (( ${#ACC_PARTS[@]} )); then IFS=','; ADDL_PARTS+=( "\"ascend_compilation_config\":{${ACC_PARTS[*]}}" ); unset IFS; fi
IFS=','; ADDL="{${ADDL_PARTS[*]}}"; unset IFS

COMPILE_CFG=""
if [[ "${GRAPH_ON}" == "1" ]]; then
  COMPILE_CFG="{\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\""
  [[ -n "${CAPTURE_SIZES}" ]] && COMPILE_CFG+=",\"cudagraph_capture_sizes\":[${CAPTURE_SIZES}]"
  COMPILE_CFG+="}"
fi
# Keys are the SpeculativeConfig field names: method, num_speculative_tokens,
# enforce_eager. DSpark's proposer sets use_cuda_graph=False regardless; enforce_eager
# is sent because the reference config sends it, and it makes the draft regime explicit.
SPEC_CFG=""
if [[ -n "${SPEC_METHOD}" ]]; then
  SPEC_CFG="{\"method\":\"${SPEC_METHOD}\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS}"
  SPEC_CFG+="}"
fi

# --override-generation-config only sets a server-side default; every AISBench request
# carries its own sampling and thinking mode, so the client wins.
SERVE=( vllm serve "${MODEL}" --host 0.0.0.0 --port "${PORT}"
        --served-model-name "${SERVED_NAME}"
        --tensor-parallel-size "${TP}" --data-parallel-size "${DP}"
        --max-num-seqs "${MAX_NUM_SEQS}" --seed "${SEED}"
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
        --gpu-memory-utilization "${GPU_MEM_UTIL}" --trust-remote-code
        --generation-config vllm
        --override-generation-config '{"temperature":0.0,"top_p":1.0}'
        --enable-expert-parallel
        --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4
        --enable-auto-tool-choice --reasoning-parser deepseek_v4
        --enable-chunked-prefill --aggregate-engine-logging
        --safetensors-load-strategy prefetch --api-server-count "${API_SERVER_COUNT}" )
[[ "${PREFIX_CACHING}" == "1" ]] && SERVE+=( --enable-prefix-caching ) \
                                 || SERVE+=( --no-enable-prefix-caching )
[[ -n "${QUANTIZATION}"      ]] && SERVE+=( --quantization "${QUANTIZATION}" )
[[ -n "${MAX_MODEL_LEN}"     ]] && SERVE+=( --max-model-len "${MAX_MODEL_LEN}" )
[[ "${PP}" != "1"            ]] && SERVE+=( --pipeline-parallel-size "${PP}" )
[[ "${ENFORCE_EAGER}" == "1" ]] && SERVE+=( --enforce-eager )
[[ -n "${COMPILE_CFG}"       ]] && SERVE+=( --compilation-config "${COMPILE_CFG}" )
[[ -n "${SPEC_CFG}"          ]] && SERVE+=( --speculative-config "${SPEC_CFG}" )
SERVE+=( --additional-config "${ADDL}" )

HEALTH_URL="http://127.0.0.1:${PORT}/health"

# =============================================================================
#  print the plan; check the guards
# =============================================================================

exec 3>&2                                          # fd3 = the real terminal
if [[ "${DRY_RUN}" != "1" ]]; then
  ensure_dirs
  # The tee sits in this script's process group, so a ^C would kill it and the script's
  # next write would take SIGPIPE and die mid-reap, with the log gone too.
  exec > >(trap '' INT TERM HUP QUIT; exec tee -a "${RUN_LOG}") 2>&1
fi

echo "────────────────────────────────────────────────────────────"
echo "  started : $(date '+%F %T')  host $(hostname)  pid $$"
echo "  serve   : ${SERVED_NAME} on :${PORT} (card ${CARD}) tp=${TP} dp=${DP} pp=${PP} ep=${EP_SIZE} max_num_seqs=${MAX_NUM_SEQS}"
[[ "${GRAPH_ON}" == "1" ]] \
  && echo "  graph   : ON ${CUDAGRAPH_MODE} sizes=[${CAPTURE_SIZES:-vllm-default}] ${AUTO_NOTE}" \
  || echo "  graph   : OFF (eager) — prefetch cannot overlap; TPOT is a debug number, accuracy is not"
echo "  spec    : ${SPEC_METHOD:-off} num_spec_tokens=${NUM_SPEC_TOKENS} decode_tokens=${DECODE_TOKENS} moe_rows=${MOE_ROWS} (x dp=${DP})"
echo "  offload : ${OFFLOAD} experts=${NUM_DEVICE_EXPERTS} layers=${NUM_DEVICE_LAYERS} lrc=${CACHE_POLICY} prefetch=${PREFETCH}/${EXPERT_PREFETCH_NUM} tokens=${EXPERT_PREFETCH_TOKENS}(eff ${PF_TOK_EFF}) subst=${EXPERT_SUBSTITUTION} multi_card=${ENABLE_MULTI_CARD} debug=${MOE_OFFLOAD_DEBUG}"
echo "  predict : ${PREDICTOR_NOTE}"
[[ "${EXPERT_PREDICTOR}" != fate ]] && echo "            ckpt=${EXPERT_PREDICTOR_CKPT}"
[[ -n "${FATE_KEEPS}" ]] && echo "            fate keeps ${FATE_KEEPS}; the head owns the rest"
echo "  wait-t  : ${EXPERT_PREFETCH_WAIT_TIMING} (prefetch stall on the compute stream -> pf_wait rows)"
[[ "${EXPERT_PREFETCH_WAIT_TIMING}" == "1" ]] && \
  echo "  warn    : EXPERT_PREFETCH_WAIT_TIMING=1 adds two device-event records per covered layer per
            step at the prefetch join — TPOT from this run is NOT a baseline. Use it to attribute
            latency, then turn it off to report."
echo "  h2d     : ${H2D_BACKEND}$([[ "${H2D_BACKEND}" == memfabric ]] && echo " pool=${MEMFABRIC_POOL_GIB}GiB/rank (aggregate $(( MEMFABRIC_POOL_GIB * EP_SIZE ))GiB) mode=$([[ "${ENABLE_MULTI_CARD}" == "1" ]] && echo shared || echo local)")"
echo "  gate    : ${REMOE_GATE:-off (base router)}"
[[ -n "${REMOE_GATE}" ]] && echo "            build rotation=${MODEL_ROT} (required=${REMOE_GATE_ROT}) — matched"
echo "  stats   : ${DECODE_STATS} csv=${CSV} -> ${STATS_DIR}  (quiesce=${STATS_QUIESCE}s engine_grace=${ENGINE_GRACE}s reap_wait=${REAP_WAIT}s)"
echo "  health  : ${HEALTH_URL}  (wait=${WAIT}s, grace_after_listening=${HEALTH_GRACE}s)"
echo "  env     : conda=${VA_CONDA_ENV:-unset} cann=${VA_CANN_SOURCED:-unset} omp=${OMP_NUM_THREADS:-unset} log=${VLLM_LOGGING_LEVEL:-unset} v2_runner=${VLLM_USE_V2_MODEL_RUNNER:-unset}"
[[ -n "${SPEC_CFG}"    ]] && echo "  spec-cfg: ${SPEC_CFG}"
[[ -n "${COMPILE_CFG}" ]] && echo "  compile : ${COMPILE_CFG}"
echo "  addl-cfg: ${ADDL}"
echo "  client  : run=${RUN} perf_prompts=${PERF_PROMPTS:-full} perf_temp=${PERF_TEMP}"

# Environment invariants. Assert rather than re-export, so a mismatch is visible instead
# of silently corrected.
if [[ "${OFFLOAD}" == "1" ]]; then
  [[ "${DYNAMIC_EPLB:-false}" == "false" ]] || guard "DYNAMIC_EPLB=${DYNAMIC_EPLB} with expert offload:
       process_weights_after_loading deletes the very tensors the paging primitive writes into."
  [[ "${VLLM_ASCEND_ENABLE_FUSED_MC2:-0}" == "0" ]] || guard "VLLM_ASCEND_ENABLE_FUSED_MC2=1 with
       expert offload: W8A8 fused scales go stale after paging, and FUSED_MC2 applies log2phy unclamped."
fi
[[ "${VLLM_USE_V2_MODEL_RUNNER:-0}" == "0" ]] || guard "VLLM_USE_V2_MODEL_RUNNER=1: every statistics
       hook lives in model_runner_v1.py. This run would produce NO summary and NO CSV."
[[ "${VLLM_LOGGING_LEVEL:-INFO}" =~ ^(INFO|DEBUG)$ ]] || guard "VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL}:
       every line this script greps for is logged at INFO. Below it the post-health checks fail and
       the tracer table reads all zeros while the run is fine."
[[ "${GRAPH_ON}" == "1" && "${ASCEND_LAUNCH_BLOCKING:-0}" != "0" ]] && \
  echo "  warn    : ASCEND_LAUNCH_BLOCKING=1 in graph mode — the prefetch stream cannot overlap anything."
[[ "${EXPERT_PREFETCH_WAIT_TIMING}" == "1" && "${GRAPH_ON}" != "1" ]] && \
  echo "  note    : wait timing in EAGER measures a stream the prefetch never overlaps, so pf_wait
            will read near zero for a reason that has nothing to do with the prefetch's cost.
            The number is only meaningful under graph capture."
[[ "${OFFLOAD}" == "1" && -z "${QUANTIZATION}" ]] && \
  echo "  note    : QUANTIZATION='' — the engine resolves it from the checkpoint. Only a genuinely
            unquantized model (quant_type NONE) makes OFFLOAD=1 a no-op; the post-health
            '[DECODE-STATS] topology:' line settles which happened."

if [[ "${OFFLOAD}" == "1" ]]; then
  echo "  thresh  : offload_threshold=${THR} MoE rows (${NDE_MIN}/${TOPK}); floor = moe_rows*topk = ${NDE_FLOOR}"
  # Above the threshold decode leaves the LRC path entirely: no paging, no prefetch, no
  # statistics — the run completes and measures nothing.
  (( MOE_ROWS > THR )) && guard "moe_rows=${MOE_ROWS} > offload_threshold=${THR}: every decode step takes
       the prefill-pool path, so there is NO paging, NO prefetch and NO cache statistics. Raise
       NUM_DEVICE_EXPERTS to >= ${NDE_FLOOR}, or lower MAX_NUM_SEQS / NUM_SPEC_TOKENS / DP."
  (( EP_SIZE > 1 )) && [[ "${ENABLE_MULTI_CARD}" != "1" ]] && guard "EP=${EP_SIZE} with ENABLE_MULTI_CARD=0
       and offload on: every rank loads the SAME experts. Set ENABLE_MULTI_CARD=1, or run one card."
  (( EP_SIZE == 1 )) && [[ "${ENABLE_MULTI_CARD}" == "1" ]] && guard "ENABLE_MULTI_CARD=1 at EP=1:
       comm selection takes the multi-card branch while the layers stay single-card."
  if [[ "${ENABLE_MULTI_CARD}" == "1" ]]; then
    # num_device_experts_for_rank RAISES at FORWARD time when capacity % ep_size != 0 —
    # an indivisible capacity starts the engine and dies mid-decode.
    for _e in $(tr -d '[] ' <<<"${NUM_DEVICE_EXPERTS}" | tr ',' ' '); do
      (( _e % EP_SIZE == 0 )) || die "num_device_experts entry ${_e} is not divisible by EP size ${EP_SIZE}"
    done
    _slots="${NDE_MIN}"; _scope=global
    [[ "${SHARD_PER_RANK}" != false ]] && { _slots=$(( NDE_MIN / EP_SIZE )); _scope=owner; }
    # MemFabric SHARED publishes every rank's shard POINTERS after weight conversion, so a
    # target-model expert can be placed on any rank and admission is bounded by the whole
    # pool again — select_moe_comm_method takes _supports_global_placement. Draft (MTP /
    # DSpark) layers register AFTER that publication and stay owner-constrained, so a
    # draft forward still sees the per-rank bound.
    [[ "${H2D_BACKEND}" == memfabric ]] && { _slots="${NDE_MIN}"; _scope="global (memfabric shared)"; }
    # In multi-card the decode/prefill split is the COMM TYPE: MC2 is admitted only when
    # num_tokens*topk <= admission slots. Miss it and every step goes ALLTOALL -> shard
    # prefill pool -> no paging, no stats, no prefetch, and the run looks normal.
    (( DECODE_TOKENS * TOPK > _slots )) && guard "multi-card decode never reaches MC2:
       decode_tokens*topk=$(( DECODE_TOKENS * TOPK )) > admission slots=${_slots} (${_scope})."
    echo "  mc      : admission slots=${_slots} (${_scope}); records NO per-layer cache statistics"
    echo "            (accounting lives in the single-card host callback). EXPERT_PREFETCH_NUM does not reach it."
  fi
  [[ "${EXPERT_SUBSTITUTION}" == "1" ]] && \
    echo "  note    : substitution ON — it CHANGES ROUTING. Accuracy is not comparable to a run with it off."
  [[ "${MOE_OFFLOAD_DEBUG}" == "1" ]] && \
    echo "  warn    : MOE_OFFLOAD_DEBUG=1 puts logging inside the host callback — TPOT here is not a timing number."
  # _prefill_load_layer runs in the outer Python of update_weights, so it executes once at
  # capture and never on replay: a captured size above the safe bound replays against
  # stale pool weights and records no prefetch nodes.
  if [[ "${GRAPH_ON}" == "1" && -n "${CAPTURE_SIZES}" ]]; then
    _unsafe=""; for _s in ${CAPTURE_SIZES//,/ }; do (( _s * DP > THR )) && _unsafe+="${_s} "; done
    [[ -n "${_unsafe}" ]] && echo "  warn    : capture size(s) ${_unsafe%% } exceed the safe bound (size x dp > ${THR}) —
            those graphs replay against stale prefill-pool weights. Raise NUM_DEVICE_EXPERTS to >= ${NDE_FLOOR}."
  fi
fi

for t in ${TASK_LIST}; do
  preset "${t}"
  echo "     - ${t}  ${P_DS}  acc:[${P_ACC_MODE}] temp=${P_TEMP} max_out=${P_MAXOUT} prompts=${P_NUM:-full} avg@n=${P_AVGN}  target: $(ref "${t}" "${P_ACC_MODE}")"
done
echo "  out     : ${OUT_DIR}"
echo "  repro   : $(for v in CARD PORT TASKS RUN THINK THINK_EFFORT NUM_PROMPTS PERF_PROMPTS AVG_N TEMP TOPP \
      PERF_TEMP MAX_NUM_SEQS MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS SEED GPU_MEM_UTIL TP DP PP QUANTIZATION PREFIX_CACHING \
      ENFORCE_EAGER CUDAGRAPH_MODE CUDAGRAPH_CAPTURE_SIZES ENABLE_NPUGRAPH_EX ENABLE_STATIC_KERNEL \
      MTP NUM_SPEC_TOKENS OFFLOAD NUM_DEVICE_EXPERTS NUM_DEVICE_LAYERS TOPK EXPERT_MAP_PATH \
      CACHE_POLICY PREFETCH EXPERT_PREFETCH_NUM EXPERT_PREFETCH_TOKENS EXPERT_PREDICTOR EXPERT_PREDICTOR_CKPT \
      EXPERT_PREFETCH_WAIT_TIMING CACHE_STATS_LOG_INTERVAL MOE_OFFLOAD_DEBUG REMOE_GATE REMOE_GATE_ROT \
      DECODE_STATS CSV CSV_PATH STATS_FLUSH_EVERY STATS_FLUSH_SECONDS STATS_QUIESCE WAIT HEALTH_GRACE ENGINE_GRACE REAP_WAIT \
      EXPERT_SUBSTITUTION EXPERT_SUBSTITUTION_THRESHOLD HOT_PRELOAD HOT_EXPERTS_FILE \
      ENABLE_MULTI_CARD SHARD_PER_RANK H2D_BACKEND MEMFABRIC_POOL_GIB MEMFABRIC_LOG_LEVEL \
      CACHE_RECENT_WINDOW CACHE_EMA_BETA CACHE_RECENT_WEIGHT \
      CACHE_EMA_WEIGHT CACHE_ROUTER_WEIGHT CACHE_AGE_WEIGHT MULTISTREAM_OVERLAP_SHARED_EXPERT \
      ENABLE_CPU_BINDING API_SERVER_COUNT PROBE MODEL SERVED_NAME TOKENIZER AIS_BENCH_DATASETS_CACHE
   do printf '%s=%q ' "${v}" "${!v-}"; done)$0"
echo "────────────────────────────────────────────────────────────"
printf '  $ '; printf '%q ' "${SERVE[@]}"; echo
for step in "${PLAN[@]}"; do build_cmd "${step%:*}" "${step#*:}"; printf '  $ '; printf '%q ' "${CMD[@]}"; echo; done
[[ "${DRY_RUN}" == "1" ]] && { echo "[DRY_RUN] nothing launched."; exit 0; }

# =============================================================================
#  preflight
# =============================================================================

# vLLM resolves its platform from an entry-point group at import time. Without
# vllm-ascend, DeviceConfig raises "Failed to infer device type" while BUILDING THE ARG
# PARSER — before any flag here is read. Name the cause, not the symptom.
python3 -c "
import sys
from importlib.metadata import entry_points
if not list(entry_points(group='vllm.platform_plugins')): sys.exit(1)
for m in ('vllm_ascend','torch_npu'): __import__(m)
" 2>/dev/null || die "vllm-ascend / torch_npu will not import, or no vllm.platform_plugins entry
       point is registered. Did you 'source env_a5_0260.sh' in this shell?"

# MemFabric is imported lazily, at FIRST LAYER ALLOCATION — i.e. minutes into the load —
# and offload.initialize() then dlopens libmf_hybm_accoffload.so from the runtime's own
# path. Without set_env.sh that surfaces as "offload launch load library failed" ->
# RuntimeError: MemFabric LOCAL initialization failed (ret=-1). Settle it here instead.
[[ "${H2D_BACKEND}" != memfabric ]] || python3 -c "
import memfabric_hybrid
from memfabric_hybrid import offload" 2>/dev/null \
  || die "H2D_BACKEND=memfabric but memfabric_hybrid will not import in this shell.
       Install MemFabric Hybrid 1.2 and source its runtime env — typically
       'source /usr/local/memfabric_hybrid/set_env.sh' — BEFORE this script."

command -v ais_bench >/dev/null 2>&1 || die "ais_bench not on PATH"
[[ -n "${PKG_CFG}" && -d "${PKG_CFG}" ]] || die "cannot locate the ais_bench configs dir"
ensure_dirs
# --config-dir is a REPLACEMENT root; if these two are not reachable under it, every pass
# dies at config resolution and the symptom looks like a hung benchmark.
for d in datasets summarizers; do
  [[ -d "${CFG_ROOT}/${d}" ]] || die "${CFG_ROOT}/${d} is not reachable — link_pkg_cfg failed.
       --config-dir REPLACES ais_bench's config root, so --datasets and --summarizer would
       resolve against a directory that does not exist."
done

_pids="$(port_pids)"
if [[ "${SKIP_SERVE}" == "1" ]]; then
  echo "[port ] :${PORT} pids=${_pids:-none}"
elif (exec 4<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
  die "port ${PORT} already in use${_pids:+ by PID(s) ${_pids}}"
else
  echo "[port ] :${PORT} is free"
  # Prove the HTTP path works BEFORE the model spends four minutes loading. The port is
  # free, so a real request to it must come back unreachable. Anything else means the
  # health poll would fail for a reason it cannot report.
  _err="$(http_req 3 "http://127.0.0.1:${PORT}/" 2>&1 >/dev/null)"; _rc=$?
  case "${_rc}" in
    7|28) echo "[http ] localhost round-trip OK (exit ${_rc}, as expected)" ;;
    0|22) die "something is already ANSWERING on :${PORT} that the port scan did not see" ;;
    *)    die "python3 cannot make an HTTP request on this box (exit ${_rc}): ${_err}" ;;
  esac
fi

# The perf summarizer loads the tokenizer only AFTER every request is sent, so a bad path
# otherwise costs a whole pass.
[[ "${RUN}" == acc ]] || python3 -c "
from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('${TOKENIZER}', trust_remote_code=True)" \
  2>/dev/null || die "the perf summary would fail on TOKENIZER='${TOKENIZER}'"

# AIS_BENCH_DATASETS_CACHE is a ROOT that AISBench appends 'ais_bench/datasets/<name>' to.
_miss=0
for t in ${TASK_LIST}; do
  preset "${t}"
  _f="$(find "${PKG_CFG}/datasets" -name "${P_DS}.py" -print -quit 2>/dev/null)"
  [[ -n "${_f}" ]] || { echo "[!!] dataset config '${P_DS}' not installed" >&2; _miss=1; continue; }
  _rel="$(grep -oE "path='[^']*'" "${_f}" | head -1 | sed "s/^path='//;s/'$//")"
  [[ "${_rel}" == /* ]] && _abs="${_rel}" || _abs="${DATA_ROOT}/${_rel}"
  [[ -d "${_abs}" ]] && echo "[data ] ${t}: ${_abs}" || {
    echo "[!!] ${t}: no data at ${_abs}" >&2
    echo "     cd '${DATA_ROOT}/ais_bench/datasets' && wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/${t}.zip && unzip ${t}.zip" >&2
    _miss=1; }
done
(( _miss )) && die "missing dataset config or data"

# =============================================================================
#  launch
# =============================================================================

if [[ "${SKIP_SERVE}" != "1" ]]; then
  install_traps
  echo "[serve] launching — engine output streams below and into ${SERVE_LOG}"
  : > "${SERVE_LOG}"
  if command -v setsid >/dev/null 2>&1; then setsid "${SERVE[@]}" >> "${SERVE_LOG}" 2>&1 &
  else "${SERVE[@]}" >> "${SERVE_LOG}" 2>&1 & fi
  SERVE_PID=$!
  # setsid() gives the server sid == pgid == pid, but the exec'd `setsid` has not
  # necessarily reached that call when bash returns from `&`. Reading ps ONCE races it:
  # we would see the SCRIPT's pgid, blank SERVE_PGID, and silently demote reap to a
  # pgrep -P walk that misses everything re-parented. Poll for the real answer.
  for ((i=0; i<50; i++)); do
    _sid="$(pid_sid "${SERVE_PID}")"
    [[ -z "${_sid}" ]] && break
    if [[ "${_sid}" == "${SERVE_PID}" ]]; then
      SERVE_SID="${_sid}"
      SERVE_PGID="$(ps -o pgid= -p "${SERVE_PID}" 2>/dev/null | tr -d ' ')"
      break
    fi
    sleep 0.1
  done
  # No setsid, or it did not take: the server shares this script's group, so signalling
  # the group would signal the script too. Leave it empty and let reap walk the tree.
  [[ "${SERVE_PGID}" == "$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')" ]] && SERVE_PGID=""

  mirror_start
  echo "[serve] waiting up to ${WAIT}s for ${HEALTH_URL} ..."
  ok=0; _listening=0; _since=0; _diagged=0
  for ((i=0; i<WAIT/2; i++)); do
    kill -0 "${SERVE_PID}" 2>/dev/null || die "serve exited during startup (see ${SERVE_LOG})"
    http_req 5 "${HEALTH_URL}" >/dev/null 2>&1 && { ok=1; break; }
    if (( ! _listening )) \
       && grep -q 'Application startup complete\|Starting vLLM server on' "${SERVE_LOG}" 2>/dev/null; then
      _listening=1
    fi
    if (( _listening )); then
      (( _since += 2 ))
      (( ! _diagged && _since >= 10 )) && { _diagged=1; health_diag; }
      (( _since > HEALTH_GRACE )) && die "the engine is serving on :${PORT} but ${HEALTH_URL} has not
       answered in ${HEALTH_GRACE}s. See the [!!] block above — this is connectivity, not a slow load."
    fi
    (( i % 30 == 29 )) && echo "[serve] still waiting ($(( (i+1) * 2 ))s of ${WAIT}s) ..."
    sleep 2
  done
  (( ok )) || die "server not ready in ${WAIT}s (see ${SERVE_LOG})"
  echo "[serve] healthy"

  # TOP-LEVEL additional_config keys are read with .get(), so an unpatched engine or a
  # typo swallows them SILENTLY. The engine's own echoed lines turn that back into a
  # failure. (The nested expert_offload_config set is closed and raises at startup.)
  [[ -n "${REMOE_GATE}" ]] && { grep -q '\[GATE-OVERRIDE\] applied' "${SERVE_LOG}" \
    || guard "REMOE_GATE is set but the engine never logged '[GATE-OVERRIDE] applied' — this run
       would measure the BASE router."; }
  # The manager states its backend once, at construction. A memfabric run that fell back
  # — or a config whose nested keys never reached ExpertOffloadConfig — reads torch here.
  if [[ "${OFFLOAD}" == "1" ]]; then
    grep -o '\[EXPERT-OFFLOAD-H2D\].*$' "${SERVE_LOG}" | sort -u | sed 's/^/  h2d     : /'
    [[ "${H2D_BACKEND}" == memfabric ]] \
      && ! grep -q '\[EXPERT-OFFLOAD-H2D\] backend=memfabric' "${SERVE_LOG}" \
      && guard "H2D_BACKEND=memfabric but the engine did not log 'backend=memfabric' — this run
       is measuring the torch copy path under a memfabric label."
  fi
  # A trained predictor announces its geometry once, at offload finalize, and then how many
  # TARGETS it resolved. A count that is not (non-hash layers - layer_delta) means the head
  # mapped onto the wrong layers, or the decoder-layer hook never fired — both silent at runtime.
  if [[ "${EXPERT_PREDICTOR}" != fate ]]; then
    grep -o '\[PREFETCH-AI\].*$' "${SERVE_LOG}" | sort -u | sed 's/^/  predict : /'
    grep -q '\[PREFETCH-AI\] armed' "${SERVE_LOG}" \
      || guard "EXPERT_PREDICTOR=${EXPERT_PREDICTOR} but the engine never logged '[PREFETCH-AI] armed' —
       the head did not load, or no decoder layer resolved a capture site. Every prefetch this run
       makes would come from the built-in fate predictor on its own targets alone."
  fi
  # The stall timer arms once, at offload finalize — that IS a startup signal.
  # Whether it produced samples is a RUNTIME question and is checked in
  # report_stats, after decoding; testing it here fires on every run.
  if [[ "${EXPERT_PREFETCH_WAIT_TIMING}" == "1" ]]; then
    grep -o '\[PREFETCH-WAIT\].*$' "${SERVE_LOG}" | sort -u | sed 's/^/  wait-t  : /'
    grep -q '\[PREFETCH-WAIT\] timing enabled' "${SERVE_LOG}" \
      || guard "EXPERT_PREFETCH_WAIT_TIMING=1 but the engine never logged
       '[PREFETCH-WAIT] timing enabled' — the key was not applied and this run measures no stall."
  fi
  if grep -q '\[DECODE-STATS\] config:' "${SERVE_LOG}"; then
    grep -o '\[DECODE-STATS\] \(config\|topology\|armed\|decode\):.*$' "${SERVE_LOG}" | sort -u | sed 's/^/  stats   : /'
    grep -q '\[DECODE-STATS\] armed' "${SERVE_LOG}" || \
      echo "  warn    : the engine never logged '[DECODE-STATS] armed' — collection is NOT live."
    # Draft MoE layers are excluded by a LAYER-NAME test; the registered count settles
    # whether it held for THIS drafter.
    _nml="$(grep -o 'topology: moe_layers=[0-9]*' "${SERVE_LOG}" | head -1 | grep -o '[0-9]*$')"
    _nhl2="$(python3 -c "
import json;print(json.load(open('${MODEL}/config.json')).get('num_hidden_layers',''))" 2>/dev/null)"
    [[ -n "${_nml}" && -n "${_nhl2}" && "${_nml}" != "${_nhl2}" ]] && \
      echo "  warn    : registered ${_nml} MoE layers but num_hidden_layers=${_nhl2} — the drafter's MoE
            layers are being OFFLOADED, which the sizing model assumes never happens."
  elif [[ "${DECODE_STATS}" == "1" ]]; then
    guard "the engine never logged '[DECODE-STATS] config:' — the statistics keys were swallowed,
       so this run produces NO summary and NO CSV."
  fi
else
  echo "[serve] SKIP_SERVE=1 — using whatever is on :${PORT}"
  http_req 5 "${HEALTH_URL}" >/dev/null 2>&1 || die "nothing healthy on :${PORT}"
fi

[[ "${PROBE}" == "1" ]] && for m in ${MODES}; do probe_mode "${m}"; done

# =============================================================================
#  run
# =============================================================================

rc=0
for step in "${PLAN[@]}"; do
  t="${step%:*}"; p="${step#*:}"
  build_cmd "${t}" "${p}"
  echo "[${p}] ${t} [${P_PASS_MODE}] ${P_DS}"
  printf '  $ '; printf '%q ' "${CMD[@]}"; echo
  # On a terminal, hand AISBench the real one: its progress table needs a TTY. The table
  # is also the liveness signal, so the heartbeat only runs when there is no table.
  if [[ -t 3 ]]; then
    mirror_stop
    "${CMD[@]}" >&3 2>&3; src=$?
    mirror_start
  else
    if (( HEARTBEAT > 0 )); then
      ( t0=${SECONDS}
        while sleep "${HEARTBEAT}"; do
          el=$(( SECONDS - t0 ))
          n=$(http_req 5 "http://127.0.0.1:${PORT}/metrics" 2>/dev/null \
              | awk '$1 ~ /^vllm:request_success_total/ {s+=$2} END {printf "%d", s+0}')
          printf '[hb] %s %s  elapsed %02d:%02d:%02d  server_completed=%s\n' \
                 "${t}" "${p}" $((el/3600)) $((el%3600/60)) $((el%60)) "${n:-?}"
        done ) &
      HB_PID=$!
    fi
    "${CMD[@]}"; src=$?
    [[ -n "${HB_PID}" ]] && kill "${HB_PID}" 2>/dev/null; HB_PID=""
  fi
  (( src != 0 )) && rc=${src}
  report_step "${t}" "${p}" "${src}"
  stats_tail
  if engine_died_early; then
    echo "[!!] ENGINE DIED DURING THIS PASS — the numbers above are NOT a measurement. The API"
    echo "     server stayed up and answered the rest with 500s, which is why ais_bench exited 0."
    grep -nE 'engine core exited unexpectedly|EngineCore: trigger received signal' \
         "${SERVE_LOG}" | head -3 | sed 's/^/     /'
    rc=1; break
  fi
done

# =============================================================================
#  quiesce, shut down, then report — in that order
# =============================================================================
# The collector snapshots ~3s after decoding stops, from its own watchdog thread, and
# that idle snapshot is what makes the artefact contain the WHOLE run. It needs the
# server alive and idle. So: wait, then stop, then read a settled STATS_DIR.

if [[ "${SKIP_SERVE}" != "1" ]]; then
  [[ "${DECODE_STATS}" == "1" ]] && (( STATS_QUIESCE > 0 )) && {
    echo "[stats]  quiescing ${STATS_QUIESCE}s so the idle snapshot lands ..."; sleep "${STATS_QUIESCE}"; }
  reap
fi
report_stats

echo "────────────────────────────────────────────────────────────"
echo "[done] rc=${rc}   finished $(date '+%F %T')   everything under ${OUT_DIR}"
echo "────────────────────────────────────────────────────────────"
sleep 1     # the tee behind the process substitution is asynchronous
exit "${rc}"

# =============================================================================
#  FINDINGS — things that changed a number
# =============================================================================
#
#  --config-dir REPLACES ais_bench's CONFIG ROOT; IT IS NOT ADDITIVE. --datasets resolves
#  against <config-dir>/datasets and --summarizer against <config-dir>/summarizers. A
#  config dir holding only our generated models/ makes EVERY pass die at config
#  resolution — before one request reaches the server — and the symptom is a healthy,
#  idle server with a benchmark that "never starts". link_pkg_cfg() symlinks every other
#  config subdir of the installed package in beside models/. SYMLINKS, not copies: the
#  shipped configs use read_base() with relative imports and only resolve at their real
#  path. `find` does not descend a symlink, so checks that traverse them need -L.
#
#  curl DOES NOT WORK IN THIS ENVIRONMENT. env_a5_0260.sh §1 does
#  `export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"`, which also puts
#  conda ahead of the SYSTEM for every binary started from that shell:
#      curl: symbol lookup error: /usr/lib64/libldap.so.2: undefined symbol: EVP_md2
#  curl exits 127 having sent nothing — indistinguishable from "the server is not ready
#  yet" to a poll that discards stderr. Every HTTP call goes through http_req() (python3
#  urllib; python3 IS the conda interpreter, so its libraries match) with proxies
#  disabled explicitly. Do not reintroduce curl. The real fix is to delete that env line.
#
#  ^C MUST NOT BE ABLE TO ABANDON THE CLEANUP. setsid puts the server in a session of its
#  own, so the terminal's SIGINT never reaches it and this script's trap is the only thing
#  that stops it. Four defects conspired: the trap missed HUP; reap() was interruptible,
#  so a second ^C killed the script after the SIGTERM but before the SIGKILL; SERVE_PGID
#  was read with one `ps` racing setsid()'s own call, blanking it and demoting reap to a
#  tree walk; and the tee sits in this script's group, so ^C killed it and the next write
#  took SIGPIPE. All four are fixed above.
#
#  pgrep -P CANNOT SEE A RE-PARENTED WORKER. The moment the top vllm process dies its
#  children go to init and the parent link is gone, while they still hold NPU memory. The
#  session id survives, so session_pids() is the sweep that removes them.
#
#  reap() MUST NOT SEARCH THE WHOLE HOST for EngineCore processes. A `pgrep -f EngineCore`
#  fallback once TERMed a concurrent benchmark's engine mid-run, leaving that run with an
#  API server answering 500s and rc=0.
#
#  THE MoE ROW COUNT IS NOT THE DECODE TOKEN COUNT WHEN DP > 1. AllGatherCommImpl.prepare()
#  pads to max_tokens_across_dp then ALL-GATHERS across the DP group, so the MoE layer sees
#  DP x decode_tokens rows and that is what is compared against offload_threshold. Above it
#  every decode step takes the prefill pool: no paging, no prefetch, no statistics.
#
#  THE EXPERT-POOL FLOOR SCALES WITH NUM_SPEC_TOKENS *AND* DP: moe_rows*topk. Each
#  increment of NUM_SPEC_TOKENS costs topk*dp slots. DSpark's reference config uses k=7,
#  i.e. 8 decode tokens per request — at topk=6 that is a floor of 48 on one card.
#
#  CAPTURED SIZES ABOVE THE SAFE BOUND ARE NOT SAFE. _prefill_load_layer runs in the outer
#  Python of update_weights, so it executes once at capture and never on replay: a captured
#  size whose MoE rows exceed offload_threshold replays against whatever was left in the
#  pool and records no prefetch nodes. Neither symptom raises.
#
#  THE TWO PREFETCH KNOBS ARE ORTHOGONAL. EXPERT_PREFETCH_TOKENS sets how many token ROWS
#  the predictor runs on (how many candidates exist); EXPERT_PREFETCH_NUM caps how many are
#  TRANSFERRED, and reaches the single-card path only. The prediction is topk-WIDE
#  regardless, so _NUM=1 does not pin |P| to 1 — it caps |N|.
#
#  THE PREFETCH METHOD IS CHOSEN PER TARGET LAYER, NOT PER RUN. EXPERT_PREDICTOR selects
#  who drives the targets the trained head COVERS; fate keeps every other target, in every
#  configuration. That is layers 1-2 under mode2_har, and layers 1-3 under mode2_prevhfr —
#  a layer-shifted atom has no predecessor at the first covered layer, and the study
#  zero-filled it during training, so that head never saw a real input. Either way 40
#  layers contribute, so pf_loads_step / pf_loads stays 40.0000 and remains the cheapest
#  proof that coverage did not shift. [PREFETCH-W] should appear for l=1,2 (har) or
#  l=1,2,3 (prevhfr) plus the head's range, and never for l=0: nothing precedes layer 0.
#
#  A MISMATCHED PREDICTOR CHECKPOINT FAILS SILENTLY. The head maps checkpoint index h to
#  MoE layer h + (num_moe_layers - L). If L is neither num_hidden_layers - num_hash_layers
#  nor num_hidden_layers, every head lands on the wrong layer: no crash, no error, just a
#  poor pred_acc that reads like a weak model. The geometry gate above settles arch, in_dim
#  (which must equal hc_mult x hidden_size for both the 'har' and 'hfr' atoms), L and top_k
#  in a few seconds. It cannot detect a BASIS mismatch — a rotated build preserves every
#  shape and norm — so serve the head on the build its residuals were dumped from.
#
#  THE PREDICTOR HEAD IS ~460 MiB PER RANK, out of the same HBM the expert cache lives in
#  (bf16 Wd dominates: L x hc_mult*hidden x rank x 2 bytes). The plan echo prints the exact
#  figure. Re-check NUM_DEVICE_EXPERTS after enabling it — a head that costs six experts'
#  worth of slots can cost more hit rate than it wins.
#
#  pf_wait IS THE ONLY NUMBER THAT MEASURES WHAT THE PREFETCH COSTS. Every other prefetch
#  metric describes what it BUYS (|N&G| avoided, |A|/|P| resident, pred_acc). pf_wait is
#  the milliseconds the compute stream is blocked at the join before the on-demand load can
#  start, so pf_wait_step (ms/step) is the additive half of the trade and is directly
#  comparable to a TPOT delta. It is measured with device events bracketing the join, which
#  is the only instrument that works under graph replay — no Python runs there, so a host
#  timer would measure nothing. Two consequences. First, EXPERT_PREFETCH_WAIT_TIMING=1 adds
#  two event records per covered layer per step, so that run is not a TPOT baseline: read
#  pf_wait from it, then turn it off and measure TPOT. Second, in EAGER the prefetch never
#  overlaps anything, so pf_wait reads near zero for a reason unrelated to its cost — the
#  number is only meaningful under graph capture.
#
#  IF pf_wait IS ALL ZEROS OR ALL IDENTICAL, DO NOT BELIEVE IT. Timing events recorded
#  inside a captured ACL graph are not exercised anywhere else in this tree, so the
#  mechanism can silently return a constant instead of a per-replay timestamp. The engine
#  latches the feature off and logs 'elapsed_time unavailable' if the call raises, and
#  report_stats surfaces that — but a constant is not an error. Confirm by running two
#  EXPERT_PREFETCH_NUM values: pf_wait must move with the transfer volume. If it does not,
#  the numbers are not per-replay and the feature is measuring nothing.
#
#  READ avg@n ONLY. AISBench de-interleaves the avg@n prediction list with the wrong
#  stride; avg@n survives as a flat mean, pass@n and cons@n come out as 1-(1-p)^n and p^n.
#  Pass@1 IS avg@n. At n=1 the 198-row gpqa set carries ~3.2 points of binomial noise.
#
#  GPQA DATASET CONFIG. gpqa_gen_0_shot_cot_chat_prompt is the simple-evals protocol behind
#  the published numbers. gpqa_gen_0_shot_str asks for the letter with no CoT and measured
#  ~8 points lower, because first_option_postprocess takes the FIRST regex match and a
#  self-correction scores at its pre-correction letter. Keep EXTRACT_RATE=1 and read the
#  rate BEFORE the score.
#
#  EAGER TPOT IS NOT A BASELINE FOR GRAPH TPOT. In eager the prefetch callback runs inline
#  and ends in load_stream.synchronize(), so PREFETCH=1 is SLOWER than PREFETCH=0. Use
#  eager for accuracy parity — scores must match graph mode exactly — never for latency.
#  This applies to both trained methods too, and more so: their heads run on the prefetch
#  stream, which only overlaps anything under graph capture.
#
#  ais_bench EXITS 0 WHEN THE ENGINE IS DEAD: HTTP 500s are responses. engine_died_early()
#  turns that back into a non-zero rc.
#
#  MULTI-CARD RECORDS NO PER-LAYER STATISTICS. The accounting lives in the single-card host
#  callback. The summary prints with those rows empty; the spec-decode rows are unaffected.
#
#  MEMFABRIC SHARED MOVES THE MC2 ADMISSION BOUND. With h2d_backend=torch and
#  shard_per_rank=true a rank can only load experts from its own host shard, so MC2 is
#  admitted on ONE RANK'S slots (num_device_experts / ep_size). SHARED all-gathers every
#  rank's shard pointers once after weight conversion, so placement is global again and the
#  bound becomes the WHOLE pool — the same NUM_DEVICE_EXPERTS now admits ep_size x the
#  batch. Draft (MTP / DSpark) layers register after that publication and keep the per-rank
#  bound. This is the one behavioural difference between the two backends; everything else
#  — routing, eviction, placement, NPU weight layout — is byte-identical, which is what
#  makes a torch-vs-memfabric A/B a pure transfer-path measurement.
#
#  DRAFT MoE LAYERS DO NOT OFFLOAD — the exclusion is a layer-NAME test ("mtp" as a path
#  component). The post-health moe_layers check proves it held for the drafter in front of
#  you; DSpark builds its layers with the same prefix, but verify rather than assume. The
#  trained predictors are excluded from the drafters twice over: by that same test, and by
#  an is_draft_layer check at decoder-layer construction.
#
#  MODES ARE PER-REQUEST. chat_template_kwargs rides on each request, which is what lets acc
#  and perf run different modes against one server. A kwarg the template does not declare is
#  dropped silently — hence probe_mode().
# =============================================================================