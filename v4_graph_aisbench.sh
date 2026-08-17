#!/usr/bin/env bash
# =============================================================================
#  v4_target_aisbench.sh — serve DeepSeek-V4-Flash on the TARGET codebase and measure
#  ACCURACY and TPOT on gsm8k / gpqa_diamond / mmlu_pro via AISBench.
#
#    source <your_v4_env>.sh
#    CARD=12 PORT=6061 OUT_DIR=./results/run1 TASKS=gpqa ./v4_target_aisbench.sh
#
#    DRY_RUN=1 ...   print the plan, launch nothing      LIST=1 ...  list configs
#    RUN=acc|perf    one pass only (default: both)       THINK=1     thinking on
#    ENFORCE_EAGER=1 eager debug path (default: graph)   MTP=1       speculative decode
#
#  DIFFERENCE FROM v4_aisbench.sh (the SOURCE script)
#    Every knob belonging to an UNPORTED feature is GONE, not defaulted-off:
#      EXPERT_PREFETCH_MAX, PREDICTOR, PREDICTOR_CKPT,
#      SEQ_STATS, STATS_WINDOW, TIMING, PF_WAIT, CSV, CSV_PATH.
#    TARGET's ExpertOffloadConfig REJECTS unknown keys with
#      ValueError: Config has no attribute '<key>'
#    so emitting any of them stops the engine at startup. They come back one at a
#    time as each feature lands. REMOE_GATE (F1) has landed; its key
#    moe_gate_override_path is TOP-LEVEL, so an unpatched engine swallows it
#    silently — the post-health check below turns that back into a failure.
#
#    Knobs that are NATIVE on TARGET and therefore exposed here rather than removed:
#      EXPERT_SUBSTITUTION / EXPERT_SUBSTITUTION_THRESHOLD   (F4 landed upstream)
#      HOT_PRELOAD / HOT_EXPERTS_FILE                        (offline hot preload)
#      ENABLE_MULTI_CARD / SHARD_PER_RANK                    (multi-card EP offload)
#      EXPERT_MAP_PATH, and the six LRC weight knobs
#
#    Added instead: graph mode (ACL graph) and MTP. Graph is now the default; the
#    SOURCE script hardcoded --enforce-eager.
#
#  Output, all under OUT_DIR. One copy of each thing; everything is on the terminal
#  too, and the engine's own logging is left verbatim rather than filtered:
#    run.log     what this script prints: config, commands, AISBench's scores
#    serve.log   what the engine prints, including its offload cache stats
#    acc/ perf/  AISBench work-dirs — it writes its own per-task logs there
#    configs/    the model configs actually sent
# =============================================================================
set -uo pipefail

# #############################################################################
#                            EDIT BELOW THIS LINE
# #############################################################################

# ── server ───────────────────────────────────────────────────────────────────
MODEL="${MODEL:-/home/yangliu/llms/DeepSeek-V4-Flash-w4a8-mtp}"
SERVED_NAME="${SERVED_NAME:-A3-deepseekv4-flash-w4a8}"
TOKENIZER="${TOKENIZER:-/home/yangliu/keyi/code/benchmarks/dsv4-tokenizer}"
AIS_BENCH_DATASETS_CACHE="${AIS_BENCH_DATASETS_CACHE-/home/yangliu/keyi/code/benchmarks}"
# ── ReMoE router-gate override ─────────────────────────────────
# Fine-tuned MoE router gates, swapped in at load time
REMOE_GATE="${REMOE_GATE:-}"
CARD="${CARD:-6}"                        # one id for single-card; "6,7" for multi-card,
                                         # in which case TP must match the count
PORT="${PORT:-7001}"
TP="${TP:-1}"; DP="${DP:-1}"
SEED="${SEED:-1024}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"        # with MTP the offload threshold is compared
                                         # against MAX_NUM_SEQS*(1+NUM_SPEC_TOKENS)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"       # empty = the checkpoint's own (1048576)
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
QUANTIZATION="${QUANTIZATION-ascend}"    # w8a8/w4a8 -> ascend; bf16 -> QUANTIZATION=
                                         # (`-` not `:-`, so an explicit empty value is
                                         # honoured; `:-` silently fell back to ascend)
                                         # NOTE: bf16 has NO offload on TARGET at all
API_SERVER_COUNT="${API_SERVER_COUNT:-1}"

# ── graph mode (ACL graph) ───────────────────────────────────────────────────
# Three gates must all pass for aclgraph to be live (model_runner_v1.py ::
# _use_aclgraph): cudagraph_mode != NONE, compilation mode == VLLM_COMPILE, and
# not enforce_eager. --enforce-eager silently disables everything below it.
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"      # 1 = eager debug path
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
# A list of TOKEN counts, not request counts. One graph per entry; a batch is padded
# up to the smallest entry >= its token count, and anything larger falls back to
# eager. Ceiling is MAX_NUM_SEQS*(1+NUM_SPEC_TOKENS), but smaller counts are real:
# with spec decode a request can be scheduled short (no drafts on the first decode
# step after prefill), and _dummy_run's uniform_decode branch builds those shapes.
#   ""      -> vLLM's own default list (warned against; see the threshold check)
#   "auto"  -> 1..cap, cap = min(decode tokens, offload_threshold)
#   "1,2,3" -> explicit. Pin this when comparing runs.
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-auto}"
ENABLE_NPUGRAPH_EX="${ENABLE_NPUGRAPH_EX:-}"     # empty = TARGET default (true)
ENABLE_STATIC_KERNEL="${ENABLE_STATIC_KERNEL:-}" # empty = TARGET default (false); true
                                         # runs the forward twice at warmup

# ── MTP / speculative decoding ───────────────────────────────────────────────
# DEFAULT OFF: on the previous snapshot MTP + W4A8 failed to load (see FINDINGS);
# re-verify on this one. num_nextn_predict_layers=1, so NUM_SPEC_TOKENS>1 re-runs the
# one draft layer via spec_step_idx rather than adding layers. Draft MoE layers do NOT
# participate in expert offload on TARGET, so MTP changes the TOKEN COUNT and nothing
# else about the offload path.
MTP="${MTP:-0}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-2}"  # decode_token_per_req = 1 + this

# ── MoE offload (native TARGET keys only — no ported features) ───────────────
OFFLOAD="${OFFLOAD:-1}"
NUM_DEVICE_EXPERTS="${NUM_DEVICE_EXPERTS:-24}"   # int, or a JSON list "[36,36,...]"
                                         # per MoE layer (TARGET accepts both)
NUM_DEVICE_LAYERS="${NUM_DEVICE_LAYERS:-1}"
TOPK="${TOPK:-6}"                        # model top_k; drives the threshold warning
                                         # AND the prefetch-num check below
EXPERT_MAP_PATH="${EXPERT_MAP_PATH:-}"   # empty = TARGET default (None)
CACHE_POLICY="${CACHE_POLICY:-1}"        # 1 = LRC, required by prefetch
PREFETCH="${PREFETCH:-1}"
EXPERT_PREFETCH_NUM="${EXPERT_PREFETCH_NUM:-1}"   # PIN THIS when comparing runs.
                                         # TARGET-native; NOT the same key as SOURCE's
                                         # expert_prefetch_max. Validator: integer >= 1.
                                         # MUST be 1 or >= TOPK on this snapshot (the
                                         # engine clamps to TOPK); 1 < n < TOPK crashes
                                         # — see the guard below and FINDINGS.
CACHE_STATS_LOG_INTERVAL="${CACHE_STATS_LOG_INTERVAL:-}"  # empty = TARGET default 1000
MOE_OFFLOAD_DEBUG="${MOE_OFFLOAD_DEBUG:-0}"       # per-layer per-step trace INSIDE the
                                         # host callback — invalid for any timing run

# ── expert substitution (NATIVE on TARGET — F4 is already upstream) ──────────
# Re-routes a token away from a non-resident expert to a resident one whose router
# score is within THRESHOLD of it, instead of paging. Skipped on hash-routed layers
# and whenever router_logits width != num_total_experts.
EXPERT_SUBSTITUTION="${EXPERT_SUBSTITUTION:-0}"
EXPERT_SUBSTITUTION_THRESHOLD="${EXPERT_SUBSTITUTION_THRESHOLD:-}"  # empty = default 0.01

# ── offline hot-expert preload (NATIVE) ──────────────────────────────────────
# Loads the top-N offline-ranked experts per layer at startup and seeds LRC hotness
# so they survive the first evictions. A RELATIVE filename resolves against
# vllm_ascend/expert_offload/ — put the json there, or give an absolute path.
HOT_PRELOAD="${HOT_PRELOAD:-0}"
HOT_EXPERTS_FILE="${HOT_EXPERTS_FILE:-}"          # e.g. expert_rank_gsm8k.json

# ── multi-card EP offload (NATIVE, opt-in) ───────────────────────────────────
# MUST move together with EP world size (TP*DP). TARGET does not check either
# direction: EP>1 without this leaves a standard-EP expert_map next to an offload
# num_local_experts and every rank loads the SAME experts; this without EP>1 makes
# comm selection take the multi-card branch while the layers stay single-card.
ENABLE_MULTI_CARD="${ENABLE_MULTI_CARD:-0}"
SHARD_PER_RANK="${SHARD_PER_RANK:-}"     # empty = TARGET default (true)

# ── LRC policy weights (NATIVE; empty = TARGET defaults) ─────────────────────
CACHE_RECENT_WINDOW="${CACHE_RECENT_WINDOW:-}"   # default 32
CACHE_EMA_BETA="${CACHE_EMA_BETA:-}"             # default 0.9,  must be [0,1)
CACHE_RECENT_WEIGHT="${CACHE_RECENT_WEIGHT:-}"   # default 1.0
CACHE_EMA_WEIGHT="${CACHE_EMA_WEIGHT:-}"         # default 0.5
CACHE_ROUTER_WEIGHT="${CACHE_ROUTER_WEIGHT:-}"   # default 0.3; 0 skips the
                                                 # topk_weights staging copy entirely
CACHE_AGE_WEIGHT="${CACHE_AGE_WEIGHT:-}"         # default 0.01

# ── other TARGET additional_config knobs ─────────────────────────────────────
MULTISTREAM_OVERLAP_SHARED_EXPERT="${MULTISTREAM_OVERLAP_SHARED_EXPERT:-0}"
ENABLE_CPU_BINDING="${ENABLE_CPU_BINDING:-0}"   # TARGET default is true; SOURCE sent
                                                # false, so false is kept here

# ── client ───────────────────────────────────────────────────────────────────
TASKS="${TASKS:-gsm8k gpqa mmlu_pro}"    # space- or comma-separated
RUN="${RUN:-both}"                       # acc | perf | both
THINK="${THINK:-}"                       # ""=presets, 0=force non-think, 1=force thinking
THINK_EFFORT="${THINK_EFFORT:-}"         # ""=preset, high | max
NUM_PROMPTS="${NUM_PROMPTS-unset}"       # acc pass; NUM_PROMPTS= (empty) = FULL dataset
PERF_PROMPTS="${PERF_PROMPTS-30}"        # perf pass; empty = full dataset
AVG_N="${AVG_N:-}"                       # repeats per question; 4 for a reportable number
TEMP="${TEMP:-}"; TOPP="${TOPP-unset}"   # acc sampling override; TOPP= omits top_p
PERF_TEMP="${PERF_TEMP:-0.0}"            # perf sampling, pinned so TPOT stays comparable
EXTRACT_RATE="${EXTRACT_RATE:-1}"        # --dump-extract-rate: read it BEFORE the score
# Use an AISBench-shipped model config instead of the generated one. Bypasses every
# preset (mode, sampling, avg@n) — reproduction only, never a model-card comparison.
# A copy is retargeted to this run's port, so any PORT works and runs can be parallel.
STOCK_MODEL="${STOCK_MODEL:-}"

# ── run control ──────────────────────────────────────────────────────────────
OUT_DIR="${OUT_DIR:-$PWD/aisbench_results/$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"; LIST="${LIST:-0}"; SKIP_SERVE="${SKIP_SERVE:-0}"
PROBE="${PROBE:-1}"                      # verify the served reasoning mode before running
WAIT="${WAIT:-1000}"                     # seconds to wait for /health
REAP_WAIT="${REAP_WAIT:-60}"             # graceful-shutdown grace before SIGKILL
HEARTBEAT="${HEARTBEAT:-300}"            # liveness line interval; 0 = off
ALLOW_UNSAFE="${ALLOW_UNSAFE:-0}"        # downgrade the offload guards to warnings.
                                         # Use it to REPRODUCE a known defect, not to
                                         # get past one.

# ── presets:  task | dataset config | acc mode | perf mode | temp | top_p | max_out | prompts | avg_n
#
#  acc/perf mode   non-think | high | max. THINK / THINK_EFFORT override BOTH columns.
#                  Both default non-think: for acc it is the comparable baseline, for
#                  perf a thinking pass runs for hours. Set perf to `high` when you
#                  want the thinking latency and have the wall clock.
#  temp/top_p      ACCURACY PASS ONLY; perf uses PERF_TEMP and omits top_p.
#  prompts         PER SUB-DATASET — mmlu_pro fans out to 14 categories, so 20 = ~280
#                  docs. Blank = full set (gsm8k 1319, gpqa 198, mmlu_pro 12032).
#  avg_n           DeepSeek's Pass@1 = avg@n. 1 for a quick pass, AVG_N=4 to report.
#                  Read avg@n ONLY — pass@n / cons@n are broken, see FINDINGS.
PRESETS="
gsm8k    | gsm8k_gen_0_shot_cot_chat_prompt | non-think | non-think | 0.0 |      | 32768 |     | 1
gpqa     | gpqa_gen_0_shot_cot_chat_prompt  | non-think | non-think | 0.6 | 0.95 | 65536 |     | 1
mmlu_pro | mmlu_pro_gen_0_shot_str          | non-think | non-think | 0.6 | 0.95 | 65536 |  20 | 1
"

# Vendor's published number for a task+mode, printed as the target. $1=task $2=mode
ref() {
  case "$1:$2" in
    gpqa:non-think)     echo "71.2 (GPQA-D Pass@1, non-think)" ;;
    gpqa:high)          echo "87.4 (GPQA-D Pass@1, Think High)" ;;
    gpqa:max)           echo "88.1 (GPQA-D Pass@1, Think Max)" ;;
    mmlu_pro:non-think) echo "83.0 (MMLU-Pro EM, non-think)" ;;
    mmlu_pro:high)      echo "86.4 (MMLU-Pro EM, Think High)" ;;
    mmlu_pro:max)       echo "86.2 (MMLU-Pro EM, Think Max)" ;;
    gsm8k:non-think)    echo "90.8 (base 8-shot EM); no instruct table" ;;
    gsm8k:*)            echo "none published with thinking on" ;;
  esac
}

# #############################################################################
#                        DO NOT EDIT BELOW THIS LINE
# #############################################################################
#
#  WORKFLOW
#    1. resolve presets for every task; die on a bad task name
#    2. build the plan: one (task, pass) pair per requested RUN mode
#    3. check the offload guards; resolve the graph capture sizes
#    4. build the `vllm serve` argv; print config + commands; exit if DRY_RUN
#    5. preflight: stock config, port, host RAM, tokenizer, datasets
#    6. launch the server (output tees to serve.log + terminal), wait, probe modes
#    7. per pair: generate a model config, run ais_bench, report its scores
#    8. shut the server down — its output has been streaming the whole time
#
#  Serve flags that never vary are inlined below rather than exposed as knobs:
#  --enable-expert-parallel, --trust-remote-code, --enable-chunked-prefill,
#  --enable-prefix-caching, --safetensors-load-strategy prefetch and the four
#  deepseek_v4 parser flags. Edit build_serve.
#
#  FINDINGS worth knowing before you trust a number are at the bottom.
# =============================================================================

die()  { echo "ERROR: $*" >&2; exit 1; }
bool() { [[ "$1" == "1" ]] && echo true || echo false; }
# Guards downgrade to warnings under ALLOW_UNSAFE=1, so a defect can be reproduced
# on purpose without editing the script.
guard() { if [[ "${ALLOW_UNSAFE}" == "1" ]]; then echo "  warn    : [unsafe] $*"; else die "$*"; fi; }

export ASCEND_RT_VISIBLE_DEVICES="${CARD}"
export VLLM_BATCH_INVARIANT=0
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"; export NO_PROXY="${no_proxy}"
[[ -n "${AIS_BENCH_DATASETS_CACHE}" ]] && export AIS_BENCH_DATASETS_CACHE \
                                       || unset AIS_BENCH_DATASETS_CACHE

RUN_LOG="${OUT_DIR}/run.log"; SERVE_LOG="${OUT_DIR}/serve.log"
CFG_DIR="${OUT_DIR}/configs/models"
STOCK_EFFECTIVE="${STOCK_MODEL:+v4_stock_${STOCK_MODEL}}"
SERVE_PID=""; SERVE_PGID=""; HB_PID=""; TAIL_PID=""

# decode tokens per step; the offload threshold is compared against this, not seqs
DECODE_TOK_PER_REQ=1
[[ "${MTP}" == "1" ]] && DECODE_TOK_PER_REQ=$(( 1 + NUM_SPEC_TOKENS ))
DECODE_TOKENS=$(( MAX_NUM_SEQS * DECODE_TOK_PER_REQ ))
GRAPH_ON=0
[[ "${ENFORCE_EAGER}" != "1" && "${CUDAGRAPH_MODE}" != "NONE" ]] && GRAPH_ON=1
NCARDS="$(awk -F',' '{print NF}' <<<"${CARD}")"
EP_SIZE=$(( TP * DP ))

# The threshold uses the SMALLEST per-layer entry when num_device_experts is a list.
NDE_MIN="${NUM_DEVICE_EXPERTS}"
if [[ "${NUM_DEVICE_EXPERTS}" == \[* ]]; then
  NDE_MIN="$(tr -d '[] ' <<<"${NUM_DEVICE_EXPERTS}" | tr ',' '\n' | sort -n | head -1)"
fi

# auto capture sizes: 1..cap, capped by the decode token count and — while
# _prefill_load_layer is still capture-only — by offload_threshold.
CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES}"; AUTO_NOTE=""
if [[ "${CAPTURE_SIZES}" == "auto" ]]; then
  _cap="${DECODE_TOKENS}"
  if [[ "${OFFLOAD}" == "1" ]]; then
    _thr=$(( NDE_MIN / TOPK ))
    [[ "${_thr}" -lt "${_cap}" ]] && _cap="${_thr}"
  fi
  [[ "${_cap}" -gt "${MAX_NUM_BATCHED_TOKENS}" ]] && _cap="${MAX_NUM_BATCHED_TOKENS}"
  _sizes=()
  for (( _s=1; _s<=_cap; _s++ )); do _sizes+=( "${_s}" ); done
  if [[ ${#_sizes[@]} -eq 0 ]]; then
    CAPTURE_SIZES=""; AUTO_NOTE="auto produced nothing: cap=${_cap}"
  else
    IFS=','; CAPTURE_SIZES="${_sizes[*]}"; unset IFS
    AUTO_NOTE="auto -> ${CAPTURE_SIZES} (1..${_cap}; cap = min(decode_tokens=${DECODE_TOKENS}, threshold))"
  fi
fi

[[ -z "${THINK}" || "${THINK}" =~ ^[01]$ ]]                     || die "THINK must be empty, 0 or 1"
[[ -z "${THINK_EFFORT}" || "${THINK_EFFORT}" =~ ^(high|max)$ ]] || die "THINK_EFFORT must be empty, high or max"
[[ "${RUN}" =~ ^(acc|perf|both)$ ]]                             || die "RUN must be acc, perf or both"
[[ -z "${AVG_N}" || "${AVG_N}" =~ ^[1-9][0-9]*$ ]]              || die "AVG_N must be a positive integer"
[[ "${NUM_SPEC_TOKENS}" =~ ^[1-9][0-9]*$ ]]                     || die "NUM_SPEC_TOKENS must be a positive integer"
[[ "${EXPERT_PREFETCH_NUM}" =~ ^[1-9][0-9]*$ ]]                 || die "EXPERT_PREFETCH_NUM must be a positive integer"
[[ -z "${SHARD_PER_RANK}" || "${SHARD_PER_RANK}" =~ ^(true|false)$ ]] \
                                                                || die "SHARD_PER_RANK must be empty, true or false (raw JSON, not 0/1)"
[[ "${HOT_PRELOAD}" != "1" || -n "${HOT_EXPERTS_FILE}" ]]       || die "HOT_PRELOAD=1 needs HOT_EXPERTS_FILE (a .json)"
[[ -z "${HOT_EXPERTS_FILE}" || "${HOT_EXPERTS_FILE}" == *.json ]] || die "HOT_EXPERTS_FILE must end in .json"
[[ "${NCARDS}" == "${EP_SIZE}" ]] \
  || die "CARD lists ${NCARDS} device(s) but TP*DP=${EP_SIZE}. They must match."
[[ -z "${REMOE_GATE}" || -e "${REMOE_GATE}" ]]                  || die "REMOE_GATE not found: ${REMOE_GATE}"
# TOPK drives the offload threshold and the prefetch-shape check, so take it from
# the model's own config.json when we can read it — the manager does the same
# (num_experts_per_tok / num_experts_per_token, hf_config or text_config). The env
# var stays as an override for a model whose config we cannot read.
_CFG_TOPK="$(python3 - "${MODEL}/config.json" 2>/dev/null <<'PY'
import json, sys
try:
    c = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
t = c.get("text_config", {}) if isinstance(c.get("text_config"), dict) else {}
for k in ("num_experts_per_tok", "num_experts_per_token"):
    if c.get(k):  print(int(c[k])); break
    if t.get(k):  print(int(t[k])); break
PY
)"
if [[ -n "${_CFG_TOPK}" ]]; then
  if [[ "${TOPK}" != "${_CFG_TOPK}" ]]; then
    echo "  warn    : TOPK=${TOPK} but ${MODEL}/config.json says num_experts_per_tok=${_CFG_TOPK};" >&2
    echo "            using ${_CFG_TOPK} (the engine reads the config, not this variable)." >&2
  fi
  TOPK="${_CFG_TOPK}"
fi

TASK_LIST="$(tr ',' ' ' <<<"${TASKS}")"
PKG_CFG="$(python3 -c 'import os,ais_bench.benchmark as b;print(os.path.join(os.path.dirname(b.__file__),"configs"))' 2>/dev/null)"
DATA_ROOT="${AIS_BENCH_DATASETS_CACHE:-$(python3 -c 'import os,ais_bench;print(os.path.dirname(os.path.dirname(os.path.abspath(ais_bench.__file__))))' 2>/dev/null)}"
STOCK_SRC=""
[[ -n "${STOCK_MODEL}" && -n "${PKG_CFG}" ]] && \
  STOCK_SRC="$(find "${PKG_CFG}/models" -name "${STOCK_MODEL}.py" -print -quit 2>/dev/null)"

# =============================================================================
#  presets
# =============================================================================

# Apply the THINK / THINK_EFFORT overrides to one raw preset mode.
resolve_mode() {
  local m="$1"
  [[ "${THINK}" == "0" ]] && m="non-think"
  [[ "${THINK}" == "1" && "${m}" == "non-think" ]] && m="${THINK_EFFORT:-high}"
  [[ -n "${THINK_EFFORT}" && "${m}" != "non-think" ]] && m="${THINK_EFFORT}"
  printf '%s' "${m}"
}
mode_for() { [[ "$1" == "perf" ]] && printf '%s' "${P_PERF_MODE}" || printf '%s' "${P_ACC_MODE}"; }

# Sets P_DS P_ACC_MODE P_PERF_MODE P_TEMP P_TOPP P_MAXOUT P_NUM P_AVGN for one task.
# Split on '|' not tab: tab is IFS whitespace, so `read` would collapse empty cells.
preset() {
  local line acc perf
  line=$(awk -F'|' -v t="$1" '
    { for (i=1;i<=NF;i++) gsub(/^[ \t]+|[ \t]+$/, "", $i) }
    $1 == t { print $2"|"$3"|"$4"|"$5"|"$6"|"$7"|"$8"|"$9; f=1 } END { exit !f }' <<<"${PRESETS}") \
    || die "unknown task '$1' — valid: $(awk -F'|' 'NF>1{gsub(/ /,"",$1);printf "%s ",$1}' <<<"${PRESETS}")"
  IFS='|' read -r P_DS acc perf P_TEMP P_TOPP P_MAXOUT P_NUM P_AVGN <<<"${line}"
  P_ACC_MODE="$(resolve_mode "${acc}")"; P_PERF_MODE="$(resolve_mode "${perf}")"
  [[ "${NUM_PROMPTS}" != "unset" ]] && P_NUM="${NUM_PROMPTS}"
  [[ -n "${AVG_N}" ]] && P_AVGN="${AVG_N}"
  [[ -n "${TEMP}"  ]] && P_TEMP="${TEMP}"
  [[ "${TOPP}" != "unset" ]] && P_TOPP="${TOPP}"
  return 0
}

# Full dataset sizes, and how many sub-datasets --num-prompts applies to.
dataset_rows()    { case "$1" in gsm8k) echo 1319;; gpqa) echo 198;; mmlu_pro) echo 12032;; *) echo 0;; esac; }
dataset_subsets() { [[ "$1" == "mmlu_pro" ]] && echo 14 || echo 1; }

# =============================================================================
#  server: argv, lifecycle, teardown
# =============================================================================

# expert_offload_config is a CLOSED key set (ascend_config.py ::
# ExpertOffloadConfig._defaults) — an unknown key raises at startup. Top-level keys
# use .get(), so an unknown one there is IGNORED silently, which is worse: a typo
# costs a run rather than failing fast. Read the engine's echoed config, not this.
# Every key below is emitted ONLY when it is non-empty, so an untouched knob keeps
# TARGET's own default rather than this script's opinion of it.
offload_json() {
  local parts=() p
  if [[ "${OFFLOAD}" == "1" ]]; then
    p="\"expert_offload\":true,\"num_device_experts\":${NUM_DEVICE_EXPERTS}"
    p+=",\"num_device_layers\":${NUM_DEVICE_LAYERS},\"cache_policy_enabled\":$(bool "${CACHE_POLICY}")"
    [[ -n "${EXPERT_MAP_PATH}" ]] && p+=",\"expert_map_path\":\"${EXPERT_MAP_PATH}\""
    if [[ "${PREFETCH}" == "1" ]]; then
      p+=",\"expert_prefetch_enabled\":true,\"expert_prefetch_num\":${EXPERT_PREFETCH_NUM}"
    fi
    if [[ "${EXPERT_SUBSTITUTION}" == "1" ]]; then
      p+=",\"expert_substitution_enabled\":true"
      [[ -n "${EXPERT_SUBSTITUTION_THRESHOLD}" ]] && \
        p+=",\"expert_substitution_threshold\":${EXPERT_SUBSTITUTION_THRESHOLD}"
    fi
    if [[ "${HOT_PRELOAD}" == "1" ]]; then
      p+=",\"hot_expert_preload\":true,\"hot_experts_file\":\"${HOT_EXPERTS_FILE}\""
    fi
    [[ "${ENABLE_MULTI_CARD}" == "1" ]] && p+=",\"enable_multi_card\":true"
    [[ -n "${SHARD_PER_RANK}"          ]] && p+=",\"shard_per_rank\":${SHARD_PER_RANK}"
    [[ -n "${CACHE_RECENT_WINDOW}"     ]] && p+=",\"cache_recent_window\":${CACHE_RECENT_WINDOW}"
    [[ -n "${CACHE_EMA_BETA}"          ]] && p+=",\"cache_ema_beta\":${CACHE_EMA_BETA}"
    [[ -n "${CACHE_RECENT_WEIGHT}"     ]] && p+=",\"cache_recent_weight\":${CACHE_RECENT_WEIGHT}"
    [[ -n "${CACHE_EMA_WEIGHT}"        ]] && p+=",\"cache_ema_weight\":${CACHE_EMA_WEIGHT}"
    [[ -n "${CACHE_ROUTER_WEIGHT}"     ]] && p+=",\"cache_router_weight\":${CACHE_ROUTER_WEIGHT}"
    [[ -n "${CACHE_AGE_WEIGHT}"        ]] && p+=",\"cache_age_weight\":${CACHE_AGE_WEIGHT}"
    [[ -n "${CACHE_STATS_LOG_INTERVAL}" ]] && p+=",\"cache_stats_log_interval\":${CACHE_STATS_LOG_INTERVAL}"
    p+=",\"moe_offload_debug\":$(bool "${MOE_OFFLOAD_DEBUG}")"
    parts+=( "\"expert_offload_config\":{${p}}" )
  fi
  [[ -n "${REMOE_GATE}" ]] && parts+=( "\"moe_gate_override_path\":\"${REMOE_GATE}\"" )
  parts+=( "\"enable_cpu_binding\":$(bool "${ENABLE_CPU_BINDING}")" )
  parts+=( "\"multistream_overlap_shared_expert\":$(bool "${MULTISTREAM_OVERLAP_SHARED_EXPERT}")" )
  local acc=()
  [[ -n "${ENABLE_NPUGRAPH_EX}"   ]] && acc+=( "\"enable_npugraph_ex\":${ENABLE_NPUGRAPH_EX}" )
  [[ -n "${ENABLE_STATIC_KERNEL}" ]] && acc+=( "\"enable_static_kernel\":${ENABLE_STATIC_KERNEL}" )
  if [[ ${#acc[@]} -gt 0 ]]; then
    local IFS=','; parts+=( "\"ascend_compilation_config\":{${acc[*]}}" )
  fi
  [[ ${#parts[@]} -eq 0 ]] && return 0
  local IFS=','; printf '{%s}' "${parts[*]}"
  return 0
}

compilation_json() {
  [[ "${GRAPH_ON}" != "1" ]] && return 0
  local p="\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\""
  [[ -n "${CAPTURE_SIZES}" ]] && p+=",\"cudagraph_capture_sizes\":[${CAPTURE_SIZES}]"
  printf '{%s}' "${p}"
  return 0
}

speculative_json() {
  [[ "${MTP}" != "1" ]] && return 0
  printf '{"num_speculative_tokens":%s,"method":"mtp"}' "${NUM_SPEC_TOKENS}"
  return 0
}

# The generation-config override only sets a server-side default; every AISBench
# request carries its own sampling, so the client wins. Thinking is per-request too,
# which is what lets the acc and perf passes run different modes on one server.
build_serve() {
  ADDL="$(offload_json)"
  COMPILE_CFG="$(compilation_json)"
  SPEC_CFG="$(speculative_json)"
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
          --enable-chunked-prefill --enable-prefix-caching --aggregate-engine-logging
          --safetensors-load-strategy prefetch --api-server-count "${API_SERVER_COUNT}" )
  [[ -n "${QUANTIZATION}"  ]] && SERVE+=( --quantization "${QUANTIZATION}" )
  [[ -n "${MAX_MODEL_LEN}" ]] && SERVE+=( --max-model-len "${MAX_MODEL_LEN}" )
  [[ "${ENFORCE_EAGER}" == "1" ]] && SERVE+=( --enforce-eager )
  [[ -n "${COMPILE_CFG}" ]] && SERVE+=( --compilation-config "${COMPILE_CFG}" )
  [[ -n "${SPEC_CFG}"    ]] && SERVE+=( --speculative-config "${SPEC_CFG}" )
  [[ -n "${ADDL}"        ]] && SERVE+=( --additional-config "${ADDL}" )
  return 0
}

# Anything listening on our port. /dev/tcp needs no external tool; port_pids only
# names the offenders when lsof/fuser/ss happens to exist.
port_in_use() { (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null && { exec 3<&- 3>&-; return 0; }; return 1; }
port_pids() {
  { lsof -t -i "TCP:${PORT}" -sTCP:LISTEN 2>/dev/null
    fuser -n tcp "${PORT}" 2>/dev/null | tr -s ' ' '\n'
    ss -lptnH "sport = :${PORT}" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2
  } | grep -E '^[0-9]+$' | sort -u | tr '\n' ' '
}

kill_tree() {   # $1=pid $2=signal — deepest first, so children die before parents
  local k; for k in $(pgrep -P "$1" 2>/dev/null); do kill_tree "${k}" "$2"; done
  kill "-$2" "$1" 2>/dev/null; return 0
}

# `vllm serve` forks an APIServer and an EngineCore per engine; they hold NPU and
# host memory until they actually exit, and matching on process names is fragile
# (the names changed). Signal the whole process group, then sweep the port.
reap() {
  [[ -n "${HB_PID}" ]] && kill "${HB_PID}" 2>/dev/null; HB_PID=""
  mirror_stop
  [[ -z "${SERVE_PID}${SERVE_PGID}" ]] && return 0
  local i left
  echo "[reap] stopping server (pgid=${SERVE_PGID:-none} pid=${SERVE_PID:-none})" >&2
  [[ -n "${SERVE_PGID}" ]] && kill -TERM -"${SERVE_PGID}" 2>/dev/null || kill_tree "${SERVE_PID}" TERM
  for ((i=0; i<REAP_WAIT; i++)); do    # a clean exit is what flushes the final stats
    { [[ -n "${SERVE_PGID}" ]] && kill -0 -"${SERVE_PGID}" 2>/dev/null; } || \
    { [[ -z "${SERVE_PGID}" ]] && kill -0 "${SERVE_PID}" 2>/dev/null; } || break
    sleep 1
  done
  [[ -n "${SERVE_PGID}" ]] && kill -KILL -"${SERVE_PGID}" 2>/dev/null || kill_tree "${SERVE_PID}" KILL
  left="$(port_pids)"
  [[ -n "${left// /}" ]] && { kill -KILL ${left} 2>/dev/null; sleep 2; left="$(port_pids)"; }
  [[ -n "${left// /}" ]] \
    && echo "[reap] WARNING: still listening on :${PORT} -> ${left}; check npu-smi info" >&2 \
    || echo "[reap] port ${PORT} released" >&2
  SERVE_PID=""; SERVE_PGID=""; return 0
}

# Show serve.log on the terminal. Off during a client run so AISBench's progress
# table owns the screen, on either side of it otherwise. It RESUMES from where it
# stopped rather than from the end, so whatever the engine logged while the table was
# up — the cache stats among it — still reaches the terminal. That means a burst of
# buffered engine lines when a pass finishes; all of it is in serve.log either way.
MIRROR_AT=0
mirror_start() { tail -n "+$(( MIRROR_AT + 1 ))" -f "${SERVE_LOG}" >&3 2>/dev/null & TAIL_PID=$!; return 0; }
mirror_stop()  {
  [[ -n "${TAIL_PID}" ]] && kill "${TAIL_PID}" 2>/dev/null; TAIL_PID=""
  MIRROR_AT=$(wc -l < "${SERVE_LOG}" 2>/dev/null || echo 0); return 0
}

# A full pass at batch_size=1 with offload runs for hours with no visible change.
# Counts come from the SERVER, so they are independent of what the client draws.
heartbeat_start() {   # $1=label
  [[ "${HEARTBEAT}" -gt 0 ]] 2>/dev/null || return 0
  ( local t0=${SECONDS} el n
    while sleep "${HEARTBEAT}"; do
      el=$(( SECONDS - t0 ))
      n=$(curl -sf -m 5 "http://127.0.0.1:${PORT}/metrics" 2>/dev/null \
          | awk '$1 ~ /^vllm:request_success_total/ {s+=$2} END {printf "%d", s+0}')
      printf '[hb] %s  elapsed %02d:%02d:%02d  server_completed=%s\n' \
             "$1" $((el/3600)) $((el%3600/60)) $((el%60)) "${n:-?}"
    done ) &
  HB_PID=$!; return 0
}
heartbeat_stop() { [[ -n "${HB_PID}" ]] && kill "${HB_PID}" 2>/dev/null; HB_PID=""; return 0; }

# =============================================================================
#  client: model configs and the ais_bench command
# =============================================================================

# Stock configs hardcode their address (vllm_api_general_chat: localhost:8080) and
# ship path="", which kills the perf summarizer after every request has been sent.
# Copy with only those rewritten, so any PORT works and parallel runs cannot collide.
write_stock_cfg() {
  [[ -n "${STOCK_SRC}" ]] || die "STOCK_MODEL='${STOCK_MODEL}' not installed; try LIST=1 $0"
  python3 - "${STOCK_SRC}" "${CFG_FILE}" "${PORT}" "${TOKENIZER}" \
           "$([[ "${RUN}" != "acc" ]] && echo 1 || echo 0)" <<'PY' || die "could not retarget the stock config"
import re, sys
src, dst, port, tok, patch = sys.argv[1:6]
s = open(src).read()
s, n_ip   = re.subn(r'(host_ip\s*=\s*)([\'"][^\'"]*[\'"])', r'\g<1>"127.0.0.1"', s)
s, n_port = re.subn(r'(host_port\s*=\s*)([\'"]?\d+[\'"]?)', r'\g<1>' + port, s)
if not n_port:
    sys.exit("[stock] no host_port in %s — cannot retarget it" % src)
s, n_path = re.subn(r'(path\s*=\s*)[\'"][\'"]', r'\g<1>"' + tok + '"', s) if patch == "1" else (s, 0)
open(dst, "w").write("# RETARGETED COPY of %s — host_ip->127.0.0.1 (x%d) host_port->%s (x%d) "
                     "path->TOKENIZER (x%d). Nothing else changed.\n" % (src, n_ip, port, n_port, n_path) + s)
print("[stock] retargeted -> %s" % dst)
PY
  return 0
}

# One generated config per task+pass, written under OUT_DIR and loaded with
# --config-dir (which supplements, never replaces, the built-ins). The file is the
# record of what was actually sent.
gen_cfg() {
  local task="$1" pass="$2" gk topp
  preset "${task}"
  MODEL_TASK="v4_${task}_${pass}"; CFG_FILE="${CFG_DIR}/${MODEL_TASK}.py"
  P_PASS_MODE="$(mode_for "${pass}")"

  if [[ -n "${STOCK_MODEL}" ]]; then      # one shared config; nothing in it varies by task
    MODEL_TASK="${STOCK_EFFECTIVE}"; CFG_FILE="${CFG_DIR}/${MODEL_TASK}.py"
    [[ "${DRY_RUN}" == "1" || -f "${CFG_FILE}" ]] || write_stock_cfg
    return 0
  fi
  [[ "${DRY_RUN}" == "1" ]] && return 0

  # perf is pinned to PERF_TEMP with no top_p, so TPOT stays comparable across tasks.
  if [[ "${pass}" == "perf" ]]; then gk="            temperature=${PERF_TEMP},"
  else
    gk="            temperature=${P_TEMP},"
    [[ -n "${P_TOPP}" ]] && gk+=$'\n'"            top_p=${P_TOPP},"
  fi
  # The mode is always stated explicitly, including thinking=False, so a run can
  # never silently sit in the wrong one.
  if [[ "${P_PASS_MODE}" == "non-think" ]]; then
    gk+=$'\n'"            chat_template_kwargs=dict(thinking=False),"
  else
    gk+=$'\n'"            chat_template_kwargs=dict(thinking=True, reasoning_effort=\"${P_PASS_MODE}\"),"
  fi
  # avg@n on the accuracy pass only: repeats would inflate the perf request count
  # without changing the latency distribution.
  [[ "${pass}" == "acc" && "${P_AVGN}" -gt 1 ]] && gk+=$'\n'"            num_return_sequences=${P_AVGN},"

  cat > "${CFG_FILE}" <<PYCFG
# GENERATED by v4_target_aisbench.sh $(date -Is) — task=${task} pass=${pass} mode=${P_PASS_MODE}
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
        stream=$([[ "${pass}" == "perf" ]] && echo True || echo False),
        request_rate=0, retry=5, api_key="",
        host_ip="127.0.0.1", host_port=${PORT},
        max_out_len=${P_MAXOUT}, batch_size=${MAX_NUM_SEQS},
        trust_remote_code=True,
        generation_kwargs=dict(
${gk}
        ),
        # Runs before the dataset's answer regex: strips any inline <think> block
        # so the pattern matches the answer, not the trace.
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
PYCFG
  return 0
}

build_cmd() {   # $1=task $2=acc|perf -> CMD, plus the P_* globals
  gen_cfg "$1" "$2"
  CMD=( ais_bench --models "${MODEL_TASK}" --datasets "${P_DS}"
        --config-dir "${OUT_DIR}/configs" --work-dir "${OUT_DIR}/$2" )
  local n
  if [[ "$2" == "acc" ]]; then
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

# Surface AISBench's numbers into run.log, and point at the logs it wrote itself
# rather than copying them.
report_step() {   # $1=task $2=acc|perf $3=rc
  local t="$1" p="$2" rc="$3" dir f found=0 pat
  [[ "${rc}" -ne 0 ]] && { echo "[result] ${t} ${p}: ais_bench FAILED rc=${rc}" >&2; return 0; }
  # Each invocation makes a fresh timestamped dir whose name sorts chronologically.
  dir="$(find "${OUT_DIR}/${p}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)"
  [[ -z "${dir}" ]] && { echo "[result] ${t} ${p}: no run dir under ${OUT_DIR}/${p}"; return 0; }
  while IFS= read -r f; do [[ -n "${f}" ]] && echo "[log]    ${t} ${p}: ${f}"; done \
    < <(find "${dir}/logs" -name '*.out' 2>/dev/null | sort)
  [[ "${p}" == "acc" ]] && pat=( -name 'summary_*.txt' ) || pat=( -path '*performances*' -name '*.csv' )
  while IFS= read -r f; do
    [[ -z "${f}" ]] && continue
    found=1; echo "[result] ${t} ${p} (${f}):"; sed 's/^/           /' "${f}"
  done < <(find "${dir}" "${pat[@]}" 2>/dev/null | sort)
  [[ "${found}" == "0" ]] && echo "[result] ${t} ${p}: nothing to report under ${dir}"
  [[ "${p}" == "acc" && "${P_AVGN:-1}" -gt 1 ]] && \
    echo "[warn]   read avg@${P_AVGN} ONLY — pass@n / cons@n are broken (see FINDINGS)"
  return 0
}

# =============================================================================
#  preflight
# =============================================================================

list_cfgs() {
  local d
  echo "Dataset configs (${PKG_CFG}/datasets):"
  for d in gsm8k gpqa mmlu_pro; do
    echo "  [${d}]"
    find "${PKG_CFG}/datasets/${d}" -maxdepth 1 -name '*.py' 2>/dev/null | sed 's|.*/||;s|\.py$||;s|^|    |' | sort
  done
  echo "  [stock model configs — valid for STOCK_MODEL=<name>]"
  find "${PKG_CFG}/models" -maxdepth 1 -name '*.py' 2>/dev/null | sed 's|.*/||;s|\.py$||;s|^|    |' | sort
}

ensure_dirs() {
  local d dirs=( "${OUT_DIR}" "${CFG_DIR}" )
  for d in acc perf; do [[ "${RUN}" == "both" || "${RUN}" == "${d}" ]] && dirs+=( "${OUT_DIR}/${d}" ); done
  for d in "${dirs[@]}"; do
    mkdir -p "${d}" 2>/dev/null || die "cannot create directory: ${d}"
    [[ -w "${d}" ]] || die "directory not writable: ${d}"
  done
  return 0
}

# A stale server on the port makes serve fail to bind — or worse, the health poll
# passes against an orphan and the run measures that instead.
check_port() {
  local pids; pids="$(port_pids)"
  if [[ "${SKIP_SERVE}" == "1" ]]; then
    echo "[port ] :${PORT} in_use=$(port_in_use && echo yes || echo no) pids=${pids:-unknown}"
    return 0
  fi
  port_in_use || { echo "[port ] :${PORT} is free"; return 0; }
  die "port ${PORT} already in use${pids:+ by PID(s) ${pids}}. Kill them, pick another PORT,
       or SKIP_SERVE=1 to deliberately evaluate the server already there."
}

# vLLM resolves its platform from the `vllm.platform_plugins` entry-point group at
# import time. If vllm-ascend is not installed in THIS env, the group is empty, vLLM
# falls through TPU/CUDA/ROCm/XPU/CPU to UnspecifiedPlatform, and DeviceConfig raises
#   RuntimeError: Failed to infer device type
# while BUILDING THE ARGUMENT PARSER — before any flag here is read. Catch it up front
# so the message names the cause instead of the symptom.
check_platform() {
  python3 - <<'PY'
import sys
try:
    from importlib.metadata import entry_points
except Exception as e:
    sys.exit(0)                      # too old to check; let vLLM speak for itself
eps = list(entry_points(group="vllm.platform_plugins"))
if not eps:
    sys.stderr.write(
        "[!!] no 'vllm.platform_plugins' entry point is registered in this env.\n"
        "     vLLM will fall through to UnspecifiedPlatform and DeviceConfig will\n"
        "     raise 'Failed to infer device type' while building the arg parser.\n"
        "     Cause: vllm-ascend is not installed here. Check with:\n"
        "       pip show vllm-ascend\n"
        "       python -c \"import vllm_ascend; print(vllm_ascend.__file__)\"\n")
    sys.exit(1)
try:
    import vllm_ascend  # noqa: F401
except Exception as e:
    sys.stderr.write("[!!] the ascend platform plugin is registered but will not "
                     "import: %s\n" % e)
    sys.exit(1)
try:
    import torch_npu  # noqa: F401
except Exception as e:
    sys.stderr.write("[!!] torch_npu will not import: %s\n"
                     "     Did you `source` the env file (CANN set_env.sh)?\n" % e)
    sys.exit(1)
print("[plat ] vllm.platform_plugins: %s" % ", ".join(ep.name for ep in eps))
PY
  [[ $? -eq 1 ]] && die "fix the environment first — no flag in this script is reached
       until vLLM can resolve a platform. Build and install vllm-ascend into
       '${CONDA_DEFAULT_ENV:-this env}' (source the env file first: setup.py reads
       ASCEND_TOOLKIT_HOME / SOC_VERSION / COMPILE_CUSTOM_KERNELS at build time)."

  # ASCEND_RT_VISIBLE_DEVICES is already exported, so device_count() reports the
  # MASKED set. Advisory only: a wrong CARD otherwise surfaces much later.
  python3 -c "
import torch, torch_npu
n = torch.npu.device_count()
print('[plat ] visible NPUs after ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}: %d' % n)
if n != ${NCARDS}:
    print('[warn ] expected ${NCARDS} (from CARD=${CARD}) — check npu-smi info')
" 2>/dev/null || echo "[plat ] could not count NPUs (advisory only)"
  return 0
}

# _stage_predicted_topk stages the learned branch's [1, prefetch_topk] prediction
# into topk_ids_h[:1], which is [1, topk], where
#   prefetch_topk = max(1, min(topk, expert_prefetch_num)).
# On the snapshot this script was written against the destination is NOT sliced, so
# any 1 < expert_prefetch_num < topk fails the copy. Rather than assert that from a
# code reading, verify it against the INSTALLED vllm_ascend and the INSTALLED torch:
# read the staging line out of the module source, and run the real copy_. A build
# that slices the destination passes and the run proceeds.
check_prefetch_shape() {
  [[ "${OFFLOAD}" == "1" && "${PREFETCH}" == "1" ]] || return 0
  (( EXPERT_PREFETCH_NUM > 1 && EXPERT_PREFETCH_NUM < TOPK )) || return 0
  [[ "${ALLOW_PREFETCH_NUM:-0}" == "1" ]] && {
    echo "[pfetch] ALLOW_PREFETCH_NUM=1 — skipping the staging-shape probe"; return 0; }

  python3 - "${TOPK}" "${EXPERT_PREFETCH_NUM}" <<'PY'
import sys, torch, inspect, ast, textwrap
topk, n = int(sys.argv[1]), int(sys.argv[2])
pf = max(1, min(topk, n))

# Is the staging destination narrowed to the prediction width? Two equivalent
# forms both count, and the destination of `.copy_` is a BARE NAME in the first:
#   (a) topk_ids_h = self.topk_ids_h[:num_tokens, :k] ; topk_ids_h.copy_(...)
#   (b) topk_ids_h = self.topk_ids_h[:num_tokens]     ; topk_ids_h[:, :k].copy_(...)
# Anything else means the destination is the full [1, topk] row.
narrowed, why, path = None, "", "?"
try:
    from vllm_ascend.expert_offload import expert_offload_manager as mod
    path = getattr(mod, "__file__", "?")
    tree = ast.parse(textwrap.dedent(
        inspect.getsource(mod.ExpertOffloadManager._stage_predicted_topk)))
    narrowed = False
    for node in ast.walk(tree):
        # (a) self.topk_ids_h[ <2+ subscript dims> ]
        if isinstance(node, ast.Subscript):
            base = node.value
            if isinstance(base, ast.Attribute) and base.attr == "topk_ids_h":
                if isinstance(node.slice, ast.Tuple) and len(node.slice.elts) >= 2:
                    narrowed, why = True, "self.topk_ids_h[...] has %d subscript dims" % \
                                          len(node.slice.elts)
        # (b) topk_ids_h[...].copy_(...)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "copy_"
                and isinstance(node.func.value, ast.Subscript)):
            b = node.func.value.value
            if getattr(b, "id", None) == "topk_ids_h" or getattr(b, "attr", None) == "topk_ids_h":
                narrowed, why = True, "copy_ destination is subscripted"
    if not narrowed:
        why = "self.topk_ids_h sliced on rows only, copy_ destination is the full row"
    print("[pfetch] %s" % path)
    print("[pfetch] staging destination narrowed to the prediction width: %s (%s)"
          % (narrowed, why))
except Exception as e:
    print("[pfetch] could not analyse the installed manager (%s) — not blocking; "
          "the engine will raise on the main thread if the shape is wrong." % e)

if narrowed is None:
    sys.exit(0)                        # unknown: only stop on positive evidence

# Simulate at the EFFECTIVE destination width, not a hardcoded one.
dst_w = pf if narrowed else topk
try:
    torch.zeros(max(1, topk), topk, dtype=torch.int32)[:1, :dst_w].copy_(
        torch.zeros(1, pf, dtype=torch.int32))
    print("[pfetch] copy [1,%d] -> [1,%d] OK — expert_prefetch_num=%d is supported"
          % (pf, dst_w, n))
    sys.exit(0)
except RuntimeError as e:
    sys.stderr.write(
        "[!!] expert_prefetch_num=%d with topk=%d: the prefetch staging copy\n"
        "     [1,%d] -> [1,%d] raises on the INSTALLED build at\n"
        "       %s\n"
        "     %s\n"
        "     _stage_predicted_topk runs on the MAIN thread (not the host callback),\n"
        "     so this kills the engine on the first learned-layer prefetch — with\n"
        "     num_hash_layers=3 that is MoE layer 2 predicting layer 3.\n"
        "     If you patched a source tree, check the path above is the copy python\n"
        "     actually imports (a non-editable install shadows an edited checkout).\n"
        "     Otherwise use EXPERT_PREFETCH_NUM=1 or >=%d, or ALLOW_PREFETCH_NUM=1.\n"
        % (n, topk, pf, dst_w, path, e, topk))
    sys.exit(1)
PY
  [[ $? -eq 1 ]] && die "prefetch staging shape rejected — see above.
       If you believe this config works, your vllm_ascend tree differs from the
       snapshot this script was written against; send expert_offload_manager.py."
  return 0
}

preflight() {
  check_platform
  check_prefetch_shape
  command -v ais_bench >/dev/null 2>&1 || die "ais_bench not on PATH"
  [[ -n "${PKG_CFG}" && -d "${PKG_CFG}" ]] || die "cannot locate the ais_bench configs dir"
  [[ -n "${STOCK_MODEL}" && -z "${STOCK_SRC}" ]] && die "STOCK_MODEL='${STOCK_MODEL}' not installed; try LIST=1 $0"
  check_port

  # AISBench aborts its infer task above 80% host RAM regardless of how much it
  # needs, and it checks AFTER the model has loaded. Expert offload is a host-memory
  # design, so the server itself pushes the box over. 78 here: our server adds more.
  python3 -c "
import sys
try: import psutil
except ImportError: sys.exit(0)
v = psutil.virtual_memory()
if v.percent > 78:
    sys.stderr.write('[!!] host RAM %.1f%% used (%.0f/%.0f GB); AISBench aborts above 80%%.\n'
                     '     Check for orphans: pgrep -af \'vllm|EngineCore\'\n'
                     % (v.percent, v.used/2**30, v.total/2**30))
    sys.exit(1)
"
  [[ $? -eq 1 ]] && die "free host memory, or raise threshold_percent in
       ais_bench/benchmark/tasks/openicl_api_infer.py (95 is safe for these datasets)"

  # The perf summarizer loads the tokenizer only AFTER every request is sent, so a
  # bad path otherwise costs a whole pass. Check it the same way AISBench will.
  if [[ "${RUN}" != "acc" ]]; then
    [[ -e "${TOKENIZER}" ]] || die "TOKENIZER not found: '${TOKENIZER}' (RUN=acc skips the perf pass)"
    python3 -c "
import sys
try: from transformers import AutoTokenizer
except Exception as e: sys.stderr.write('[warn] no transformers (%s); skipping\n' % e); sys.exit(2)
try: AutoTokenizer.from_pretrained('${TOKENIZER}', trust_remote_code=True)
except Exception as e: sys.stderr.write('[!!] tokenizer failed to load: %s\n' % e); sys.exit(1)
"
    [[ $? -eq 1 ]] && die "the perf summary would fail on this tokenizer — fix TOKENIZER"
  fi

  # AIS_BENCH_DATASETS_CACHE is a ROOT that AISBench appends 'ais_bench/datasets/<name>'
  # to; it is NOT the HF cache and NOT a folder of data files.
  local t miss=0 f rel abs
  echo "[data] root: ${DATA_ROOT}"
  for t in ${TASK_LIST}; do
    preset "${t}"
    f="$(find "${PKG_CFG}/datasets" -name "${P_DS}.py" -print -quit 2>/dev/null)"
    [[ -n "${f}" ]] || { echo "[!!] dataset config '${P_DS}' (${t}) not installed" >&2; miss=1; continue; }
    rel="$(grep -oE "path='[^']*'" "${f}" | head -1 | sed "s/^path='//;s/'$//")"
    [[ "${rel}" == /* ]] && abs="${rel}" || abs="${DATA_ROOT}/${rel}"
    if [[ -d "${abs}" ]]; then echo "[data] ${t}: ${abs}"
    else
      echo "[!!] ${t}: no data at ${abs}" >&2
      echo "     cd '${DATA_ROOT}/ais_bench/datasets' && wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/${t}.zip && unzip ${t}.zip" >&2
      miss=1
    fi
  done
  [[ "${miss}" == "1" ]] && { list_cfgs >&2; die "missing dataset config or data"; }
  return 0
}

# A chat-template kwarg the template does not declare is dropped SILENTLY, so a run
# can sit in the wrong mode — worth ~16 points on gpqa. One 64-token request settles it.
probe_mode() {   # $1 = non-think | high | max
  local kw want got resp
  if [[ "$1" == "non-think" ]]; then want=off; kw='{"thinking":false}'
  else want=on; kw="{\"thinking\":true,\"reasoning_effort\":\"$1\"}"; fi
  resp=$(curl -sf -m 600 "http://127.0.0.1:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2?\"}],\"max_tokens\":64,\"chat_template_kwargs\":${kw}}") \
    || { echo "[probe] WARN: ${1} request failed; skipping" >&2; return 0; }
  got=$(python3 -c "
import json,sys
m = json.loads(sys.argv[1])['choices'][0]['message']
print('on' if (m.get('reasoning_content') or m.get('reasoning') or '').strip() else 'off')" "${resp}") \
    || { echo "[probe] WARN: unparsable response" >&2; return 0; }
  echo "[probe] ${1}: thinking=${got} (wanted ${want})"
  [[ "${got}" == "${want}" ]] || die "server is not honouring chat_template_kwargs for '${1}'.
       The measured mode would not be the requested one. The template may use a
       different key (edit gen_cfg: thinking -> enable_thinking). PROBE=0 to override."
  return 0
}

# =============================================================================
#  main
# =============================================================================
[[ "${LIST}" == "1" ]] && { list_cfgs; exit 0; }
for t in ${TASK_LIST}; do preset "${t}"; done     # a bad task name dies before anything loads

PLAN=(); MODES=""
for t in ${TASK_LIST}; do
  preset "${t}"
  for p in acc perf; do
    [[ "${RUN}" == "both" || "${RUN}" == "${p}" ]] || continue
    PLAN+=( "${t}:${p}" ); MODES+=" $(mode_for "${p}")"
  done
done
MODES="$(tr ' ' '\n' <<<"${MODES}" | grep -v '^$' | sort -u | tr '\n' ' ')"

build_serve

exec 3>&2                                         # fd3 = the real terminal
if [[ "${DRY_RUN}" != "1" ]]; then
  ensure_dirs                                     # before the tee: run.log needs a parent
  exec > >(tee -a "${RUN_LOG}") 2>&1
fi

echo "────────────────────────────────────────────────────────────"
echo "  started : $(date '+%F %T')  host $(hostname)  pid $$"
echo "  serve   : ${SERVED_NAME} on :${PORT} (card ${CARD})  tp=${TP} dp=${DP} ep=${EP_SIZE} util=${GPU_MEM_UTIL} max_num_seqs=${MAX_NUM_SEQS}"
if [[ "${GRAPH_ON}" == "1" ]]; then
  echo "  graph   : ON  mode=${CUDAGRAPH_MODE} capture_sizes=${CAPTURE_SIZES:-vllm-default} npugraph_ex=${ENABLE_NPUGRAPH_EX:-default} static_kernel=${ENABLE_STATIC_KERNEL:-default}"
  [[ -n "${AUTO_NOTE}" ]] && echo "            ${AUTO_NOTE}"
else
  echo "  graph   : OFF (eager debug path)  enforce_eager=${ENFORCE_EAGER} cudagraph_mode=${CUDAGRAPH_MODE}"
fi
echo "  mtp     : ${MTP} num_spec_tokens=${NUM_SPEC_TOKENS} decode_tok_per_req=${DECODE_TOK_PER_REQ} decode_tokens=${DECODE_TOKENS}"
echo "  offload : ${OFFLOAD} experts=${NUM_DEVICE_EXPERTS} layers=${NUM_DEVICE_LAYERS} lrc=${CACHE_POLICY} prefetch=${PREFETCH} prefetch_num=${EXPERT_PREFETCH_NUM} debug=${MOE_OFFLOAD_DEBUG}"
echo "  offload+: subst=${EXPERT_SUBSTITUTION}${EXPERT_SUBSTITUTION_THRESHOLD:+/${EXPERT_SUBSTITUTION_THRESHOLD}} hot_preload=${HOT_PRELOAD}${HOT_EXPERTS_FILE:+/${HOT_EXPERTS_FILE}} multi_card=${ENABLE_MULTI_CARD} shard_per_rank=${SHARD_PER_RANK:-default}"
echo "  gate    : ${REMOE_GATE:-off (base router)}"
[[ -n "${COMPILE_CFG}" ]] && echo "  compile : ${COMPILE_CFG}"
[[ -n "${SPEC_CFG}"    ]] && echo "  spec    : ${SPEC_CFG}"
[[ -n "${ADDL}" ]] && echo "  addl-cfg: ${ADDL}"
echo "  client  : run=${RUN} perf_prompts=${PERF_PROMPTS:-full} perf_temp=${PERF_TEMP} extract_rate=${EXTRACT_RATE}"
[[ -n "${STOCK_MODEL}" ]] && echo "  !! STOCK_MODEL=${STOCK_MODEL}: presets NOT applied. Not model-card comparable."

# ── environment invariants the offload subsystem depends on ──────────────────
# These belong to the env file; assert rather than re-export, so a mismatch is
# visible instead of silently corrected here. Both combinations are legitimate
# WITHOUT expert offload (dynamic EPLB is a real feature), so gate on OFFLOAD=1.
if [[ "${OFFLOAD}" == "1" ]]; then
  [[ "${DYNAMIC_EPLB:-false}" == "false" ]] || \
    guard "DYNAMIC_EPLB=${DYNAMIC_EPLB} with expert offload: process_weights_after_loading
       deletes the very tensors the paging primitive writes into. Export DYNAMIC_EPLB=false
       or run with OFFLOAD=0."
  [[ "${VLLM_ASCEND_ENABLE_FUSED_MC2:-0}" == "0" ]] || \
    guard "VLLM_ASCEND_ENABLE_FUSED_MC2=${VLLM_ASCEND_ENABLE_FUSED_MC2} with expert offload:
       W8A8 fused scales go stale after paging and FUSED_MC2 applies log2phy unclamped."
fi
[[ "${GRAPH_ON}" != "1" || "${ASCEND_LAUNCH_BLOCKING:-0}" == "0" ]] && : || \
  echo "  warn    : ASCEND_LAUNCH_BLOCKING=${ASCEND_LAUNCH_BLOCKING} in graph mode — every launch is
            serialised on the host; the prefetch stream cannot overlap anything."

# ── offload consistency checks ───────────────────────────────────────────────
if [[ "${OFFLOAD}" == "1" && -z "${QUANTIZATION}" ]]; then
  echo "  warn    : QUANTIZATION='' (bf16) — TARGET drops expert offload for unquantized MoE"
  echo "            layers (quant_type == NONE), so OFFLOAD=1 is silently a no-op here."
fi
# offload_threshold = experts/topk, compared against the DECODE TOKEN count. Above
# it, decode leaves the LRC path, so prefetch never runs and nothing is recorded.
if [[ "${OFFLOAD}" == "1" && -n "${QUANTIZATION}" ]]; then
  THR=$(( NDE_MIN / TOPK ))
  echo "  thresh  : offload_threshold=${THR} (${NDE_MIN}/${TOPK}); decode_tokens=${DECODE_TOKENS}"
  [[ "${DECODE_TOKENS}" -gt "${THR}" ]] && \
    echo "  warn    : decode_tokens=${DECODE_TOKENS} > offload_threshold=${THR} — decode takes the prefill-pool path; no paging, no cache stats"
  [[ "${PREFETCH}" == "1" && "${CACHE_POLICY}" != "1" ]] && \
    echo "  warn    : PREFETCH=1 with CACHE_POLICY=0 — eviction is arbitrary, and cache stats are not recorded at all"

  # The prefetch staging copy writes a [1, prefetch_topk] prediction into a
  # [1, TOPK] pinned buffer, with prefetch_topk = max(1, min(TOPK, EXPERT_PREFETCH_NUM))
  # (expert_offload_manager.py:149). copy_ broadcasts only from a singleton dim,
  # so 1 works (id replicated, then set()-reduced) and >= TOPK works (clamped to
  # TOPK, exact copy); strictly between raises. This is only a heads-up — the
  # binding check is check_prefetch_shape() in preflight, which runs the real
  # copy_ against the installed torch instead of trusting this arithmetic.
  if [[ "${PREFETCH}" == "1" ]] && (( EXPERT_PREFETCH_NUM > 1 && EXPERT_PREFETCH_NUM < TOPK )); then
    echo "  note    : EXPERT_PREFETCH_NUM=${EXPERT_PREFETCH_NUM} is between 1 and TOPK=${TOPK} —"
    echo "            preflight will verify whether _stage_predicted_topk's staging copy"
    echo "            accepts that shape on this build (see D1_stage_predicted_topk.patched.py)."
  fi
  if [[ "${PREFETCH}" == "1" && "${EXPERT_PREFETCH_NUM}" == "1" ]]; then
    echo "  note    : EXPERT_PREFETCH_NUM=1 is the only value that works UNPATCHED, and it is"
    echo "            the degenerate case (one predicted id). The engine's own default is 2;"
    echo "            apply the D1 fix and set 2 for a real prefetch baseline."
  fi
  [[ "${PREFETCH}" == "1" && "${EXPERT_PREFETCH_NUM}" -gt "${TOPK}" ]] && \
    echo "  note    : EXPERT_PREFETCH_NUM=${EXPERT_PREFETCH_NUM} > TOPK — clamped to ${TOPK} by the engine"

  # enable_multi_card and EP size must move together; TARGET checks neither direction.
  if [[ "${EP_SIZE}" -gt 1 && "${ENABLE_MULTI_CARD}" != "1" ]]; then
    guard "EP=${EP_SIZE} with ENABLE_MULTI_CARD=0 and offload on: every rank keeps a
       standard-EP expert_map next to an offload num_local_experts, mgr.ep_size stays 1,
       and every rank loads the SAME experts. Set ENABLE_MULTI_CARD=1 or run one card."
  fi
  if [[ "${EP_SIZE}" == "1" && "${ENABLE_MULTI_CARD}" == "1" ]]; then
    guard "ENABLE_MULTI_CARD=1 at EP=1: comm selection takes the multi-card admission
       branch while the layers stay single-card. Set ENABLE_MULTI_CARD=0."
  fi
  if [[ "${ENABLE_MULTI_CARD}" == "1" ]]; then
    # num_device_experts_for_rank RAISES when capacity % ep_size != 0 — and it is
    # called from comm selection and the prefetch path, i.e. at FORWARD time, not
    # at config validation. An indivisible capacity starts the engine and dies
    # mid-decode. Check every per-layer entry here instead.
    for _e in $(tr -d '[] ' <<<"${NUM_DEVICE_EXPERTS}" | tr ',' ' '); do
      (( _e % EP_SIZE == 0 )) || \
        die "num_device_experts entry ${_e} is not divisible by EP size ${EP_SIZE}:
       num_device_experts_for_rank raises ValueError at forward time, after the
       engine has loaded. Pick a multiple of ${EP_SIZE}."
    done
    _slots="${NDE_MIN}"
    [[ "${SHARD_PER_RANK}" != "false" ]] && _slots=$(( NDE_MIN / EP_SIZE ))
    _mc_tok=$(( _slots / TOPK ))
    echo "  mc      : per-rank slots=${_slots}; MC2 needs decode_tokens*${TOPK} <= ${_slots}"
    echo "            (i.e. decode_tokens <= ${_mc_tok}); decode_tokens=${DECODE_TOKENS}"
    # In multi-card the DECODE/PREFILL split is the comm type, not the token
    # threshold: select_moe_comm_method admits MC2 only when
    # num_tokens*topk <= per-rank slots. Miss it and EVERY step goes ALLTOALL ->
    # the EP-shard prefill pool -> update_weights_multi_card returns early and
    # trigger_next_layer_prefetch is never reached. The run completes and looks
    # normal while measuring no paging and no prefetch at all.
    if (( DECODE_TOKENS > _mc_tok )); then
      _need=$(( DECODE_TOKENS * TOPK * EP_SIZE ))
      [[ "${SHARD_PER_RANK}" == "false" ]] && _need=$(( DECODE_TOKENS * TOPK ))
      guard "multi-card decode never reaches MC2: decode_tokens=${DECODE_TOKENS} > ${_mc_tok}.
       Every step would take ALLTOALL + the shard prefill pool, so there is no paging,
       no cache stats and NO PREFETCH — the run would measure nothing.
       Raise NUM_DEVICE_EXPERTS to >= ${_need} (a multiple of ${EP_SIZE}), or lower
       MAX_NUM_SEQS / NUM_SPEC_TOKENS so decode_tokens <= ${_mc_tok}."
    fi
    # Placement overflow is a hard RuntimeError mid-decode, not a fallback: with
    # shard_per_rank each expert is pinned to rank e//(num_experts/ep), so a layer
    # whose active experts crowd into one shard needs that many slots on one rank.
    # At most decode_tokens*topk experts can be active, so per-rank slots >= that
    # makes overflow impossible.
    (( _slots >= DECODE_TOKENS * TOPK )) || \
      echo "  warn    : per-rank slots=${_slots} < worst-case active experts $(( DECODE_TOKENS * TOPK )) —"
    (( _slots > DECODE_TOKENS * TOPK )) && \
      echo "            headroom above the worst case: $(( _slots - DECODE_TOKENS * TOPK )) slots"
    echo "  note    : multi-card records NO cache statistics — only the interval debug lines."
    echo "  note    : if this checkpoint is W8A8: multi-card PREFILL is not wired for it —"
    echo "            w8a8_dynamic.py falls through to the single-card full-expert pool and"
    echo "            never patches the All2All dispatcher (scope guide D8). W4A8 is wired."
  fi

  [[ "${EXPERT_SUBSTITUTION}" == "1" ]] && \
    echo "  note    : substitution ON — it CHANGES ROUTING. Accuracy is not comparable to a run with it off."
  [[ "${MOE_OFFLOAD_DEBUG}" == "1" ]] && \
    echo "  warn    : MOE_OFFLOAD_DEBUG=1 puts sorted()+tolist()+logging inside the host callback, per layer per step — TPOT here is not a timing number"

  # _prefill_load_layer runs in the outer Python of update_weights, so it is captured
  # once and never replays. A captured size above the threshold replays against stale
  # pool weights.
  if [[ "${GRAPH_ON}" == "1" ]]; then
    if [[ -z "${CAPTURE_SIZES}" ]]; then
      echo "  warn    : graph ON with vLLM's default capture sizes. Any captured size > ${THR}"
      echo "            replays against stale prefill-pool weights. Pin CUDAGRAPH_CAPTURE_SIZES"
      echo "            (or leave it at 'auto') until that is resolved."
    else
      for _s in ${CAPTURE_SIZES//,/ }; do
        [[ "${_s}" -gt "${THR}" ]] && \
          echo "  warn    : capture size ${_s} > offload_threshold=${THR} — that graph replays against stale prefill-pool weights"
      done
    fi
  fi
fi
if [[ "${GRAPH_ON}" != "1" ]]; then
  echo "  note    : eager path — prefetch cannot overlap. TPOT here is a debug number,"
  echo "            not comparable to graph mode. Accuracy IS comparable, and must match."
fi
[[ "${MTP}" == "1" && "${QUANTIZATION}" == "ascend" ]] && \
  echo "  note    : MTP=1 — the previous snapshot failed to load W4A8 drafts (see FINDINGS); re-verify"
for t in ${TASK_LIST}; do
  preset "${t}"
  echo "     - ${t}  ${P_DS}"
  echo "         acc : [${P_ACC_MODE}] temp=${P_TEMP} top_p=${P_TOPP:-server} max_out=${P_MAXOUT} prompts=${P_NUM:-full} avg@n=${P_AVGN}   target: $(ref "${t}" "${P_ACC_MODE}")"
  echo "         perf: [${P_PERF_MODE}] temp=${PERF_TEMP} prompts=${PERF_PROMPTS:-full}"
done
echo "  out     : ${OUT_DIR}"
echo "  repro   : $(for v in CARD PORT TASKS RUN THINK THINK_EFFORT NUM_PROMPTS PERF_PROMPTS AVG_N TEMP TOPP \
      PERF_TEMP MAX_NUM_SEQS MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS SEED GPU_MEM_UTIL TP DP QUANTIZATION \
      ENFORCE_EAGER CUDAGRAPH_MODE CUDAGRAPH_CAPTURE_SIZES ENABLE_NPUGRAPH_EX ENABLE_STATIC_KERNEL \
      MTP NUM_SPEC_TOKENS OFFLOAD NUM_DEVICE_EXPERTS NUM_DEVICE_LAYERS TOPK EXPERT_MAP_PATH \
      CACHE_POLICY PREFETCH EXPERT_PREFETCH_NUM CACHE_STATS_LOG_INTERVAL MOE_OFFLOAD_DEBUG REMOE_GATE \
      EXPERT_SUBSTITUTION EXPERT_SUBSTITUTION_THRESHOLD HOT_PRELOAD HOT_EXPERTS_FILE \
      ENABLE_MULTI_CARD SHARD_PER_RANK CACHE_RECENT_WINDOW CACHE_EMA_BETA CACHE_RECENT_WEIGHT \
      CACHE_EMA_WEIGHT CACHE_ROUTER_WEIGHT CACHE_AGE_WEIGHT \
      MULTISTREAM_OVERLAP_SHARED_EXPERT ENABLE_CPU_BINDING API_SERVER_COUNT \
      PROBE STOCK_MODEL MODEL SERVED_NAME TOKENIZER AIS_BENCH_DATASETS_CACHE
   do printf '%s=%q ' "${v}" "${!v-}"; done)$0"
echo "────────────────────────────────────────────────────────────"
printf '  $ '; printf '%q ' "${SERVE[@]}"; echo
for step in "${PLAN[@]}"; do build_cmd "${step%:*}" "${step#*:}"; printf '  $ '; printf '%q ' "${CMD[@]}"; echo; done
[[ "${DRY_RUN}" == "1" ]] && { echo "[DRY_RUN] nothing launched."; exit 0; }

preflight
ensure_dirs                                       # again: the engine writes CSVs at startup

if [[ "${SKIP_SERVE}" != "1" ]]; then
  trap 'reap; exit 130' INT TERM
  trap reap EXIT
  echo "[serve] launching — engine output streams below and into ${SERVE_LOG}"
  : > "${SERVE_LOG}"
  # Engine output goes to serve.log, and mirror_start puts it on the terminal too.
  # The mirror is toggled off around each client run because AISBench's progress
  # table repaints the same rows — two streams there turn into soup [see mirror_*].
  # setsid gives the server its own process group, so reap can signal the whole tree
  # without also signalling this script.
  if command -v setsid >/dev/null 2>&1; then
    setsid "${SERVE[@]}" >> "${SERVE_LOG}" 2>&1 &
  else
    "${SERVE[@]}" >> "${SERVE_LOG}" 2>&1 &
  fi
  SERVE_PID=$!
  SERVE_PGID="$(ps -o pgid= -p "${SERVE_PID}" 2>/dev/null | tr -d ' ')"
  [[ "${SERVE_PGID}" == "$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')" ]] && SERVE_PGID=""
  # Immediately, not after /health: weight loading takes minutes and this is the only
  # thing on screen during it. Starting it later made a healthy run look frozen.
  mirror_start
  echo "[serve] waiting up to ${WAIT}s for /health ..."
  ok=0
  for ((i=0; i<WAIT/2; i++)); do
    kill -0 "${SERVE_PID}" 2>/dev/null || die "serve exited during startup (see ${SERVE_LOG})"
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { ok=1; break; }
    # A liveness line every 60s, so a quiet engine still cannot look hung.
    (( i % 30 == 29 )) && echo "[serve] still waiting ($(( (i+1) * 2 ))s of ${WAIT}s) ..."
    sleep 2
  done
  [[ "${ok}" == "1" ]] || die "server not ready in ${WAIT}s (see ${SERVE_LOG})"
  echo "[serve] healthy"
else
  echo "[serve] SKIP_SERVE=1 — using whatever is on :${PORT}"
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || die "nothing healthy on :${PORT}"
fi

# ReMoE: moe_gate_override_path is a TOP-LEVEL additional_config key read with
# .get(), so an engine without the F1 patch — or a typo — is ignored SILENTLY and the
# run measures the BASE router while everything else says otherwise. The engine logs
# "[GATE-OVERRIDE] applied" once per rank during weight loading, well before /health.
# The second line prints the deduplicated gate fingerprint: more than one line means
# ranks read different files, which under EP is inconsistent dispatch with no error.
if [[ -n "${REMOE_GATE}" && "${SKIP_SERVE}" != "1" ]]; then
  grep -q '\[GATE-OVERRIDE\] applied' "${SERVE_LOG}" || guard "REMOE_GATE is set but the
       engine never logged '[GATE-OVERRIDE] applied' — this run would measure the BASE
       router. Apply Groups 1-3, or unset REMOE_GATE."
  grep -o 'sha256_16=.*$' "${SERVE_LOG}" | sort -u | sed 's/^/  gate    : /'
fi

[[ "${PROBE}" == "1" ]] && for m in ${MODES}; do probe_mode "${m}"; done

rc=0
for step in "${PLAN[@]}"; do
  t="${step%:*}"; p="${step#*:}"
  build_cmd "${t}" "${p}"
  [[ "${p}" == "acc" ]] \
    && echo "[acc] ${t} [${P_PASS_MODE}] ${P_DS}   target: $(ref "${t}" "${P_PASS_MODE}")" \
    || echo "[perf] ${t} [${P_PASS_MODE}] ${P_DS}   ${PERF_PROMPTS:-full} prompts (latency, not a score)"
  printf '  $ '; printf '%q ' "${CMD[@]}"; echo
  # On a terminal, hand AISBench the real one: its progress table needs a TTY, and
  # piping the output would drop it to a one-line fallback ("running in background
  # mode"). run.log loses nothing — AISBench writes its own per-task log under acc/
  # or perf/, and report_step prints the path. The table is also the liveness signal
  # there, so the heartbeat only runs when there is no table.
  if [[ -t 3 ]]; then
    mirror_stop                                   # the table owns the terminal
    "${CMD[@]}" >&3 2>&3
    src=$?
    mirror_start
  else
    heartbeat_start "${t} ${p}"
    "${CMD[@]}"
    src=$?; heartbeat_stop
  fi
  [[ ${src} -ne 0 ]] && rc=${src}
  report_step "${t}" "${p}" "${src}"
done

echo "────────────────────────────────────────────────────────────"
echo "[done] rc=${rc}   finished $(date '+%F %T')   everything under ${OUT_DIR}"
echo "────────────────────────────────────────────────────────────"
exit "${rc}"

# =============================================================================
#  FINDINGS — things that changed a number, kept so they are not rediscovered
# =============================================================================
#
#  READ avg@n ONLY. AISBench de-interleaves the avg@n prediction list with the
#  wrong stride, pairing each question's sample with a DIFFERENT question's. avg@n
#  is a flat mean and survives; pass@n and cons@n come out as exactly 1-(1-p)^n and
#  p^n. Measured on a 2-sample run: reported 83.33 / 34.85 against a true
#  71.21 / 46.97, and the independent-draw prediction was 83.26 / 34.92.
#
#  Pass@1 IS avg@n. DeepSeek samples k responses at non-zero temperature and
#  averages. AVG_N=4 is the reportable setting; at n=1 the 198-row gpqa set carries
#  ~3.2 points of binomial noise.
#
#  GPQA DATASET CONFIG. gpqa_gen_0_shot_cot_chat_prompt is the simple-evals protocol
#  behind the published 71.2/87.4/88.1. The other config (gpqa_gen_0_shot_str) asks
#  for "The correct answer is (X)" with no CoT instruction and measured 62.63
#  non-think greedy. Two structural causes: ~2/3 of answers open with the letter and
#  rationalise after (answer-first ~55% correct, reason-first ~78%), and
#  first_option_postprocess takes the FIRST regex match, so a self-correction is
#  scored at its pre-correction letter (5/198 of that run). Keep EXTRACT_RATE=1 and
#  read the rate before the score: a failed extraction scores as wrong, so a format
#  regression is indistinguishable from a capability one.
#
#  EXPERT_PREFETCH_NUM: THE CRASH RANGE IS STRICTLY BETWEEN 1 AND TOPK. The engine
#  computes prefetch_topk = max(1, min(topk, expert_prefetch_num))
#  (expert_offload_manager.py:149), and _stage_predicted_topk copies the learned
#  branch's [1, prefetch_topk] prediction into topk_ids_h[:1], which is [1, topk].
#  torch's copy_ broadcasts only from a SINGLETON dim, so 1 works (the id is
#  replicated across all columns and the consumer set()-reduces it, so the behaviour
#  is still "prefetch exactly one expert"), and any value >= topk works (clamped to
#  topk, exact copy). Anything strictly between raises RuntimeError on the first
#  non-hash-layer prefetch. The hash branch (tid2eid) always produces [1, topk] and
#  is unaffected. This is why the default here is 1, not TARGET's own 2.
#
#  BF16 HAS NO EXPERT OFFLOAD. AscendMoERunner drops enable_expert_offload when the
#  resolved quant_type is NONE, and _get_quant_type returns NONE for the unquantized
#  MoE method. The offload block inside AscendUnquantizedFusedMoEMethod.apply is
#  unreachable for a bf16 model. OFFLOAD=1 with QUANTIZATION="" is a silent no-op —
#  prefetch, substitution, cache stats, all of it.
#
#  SUBSTITUTION IS NATIVE NOW, AND IT CHANGES ROUTING. plan_expert_substitutions /
#  commit_expert_substitutions live in experts_selector.py upstream, single- and
#  multi-card. It re-routes a token to a resident expert whose score is within
#  expert_substitution_threshold of the intended one. Accuracy with it on is NOT
#  comparable to accuracy with it off — run it as its own arm.
#
#  DRAFT MoE LAYERS DO NOT OFFLOAD. AscendMoERunner skips any layer whose name
#  contains an "mtp" path component, which covers both the serial MTP drafter and
#  DSpark (its layers are built with prefix "mtp.<i>"). So MTP does not change the
#  registered layer count, does not change prefetch coverage, and leaves the draft
#  layer's experts fully resident in HBM. What MTP DOES change is the decode token
#  count, and therefore whether the offload threshold is cleared.
#
#  THERE IS NO STATS WINDOW ON TARGET. seq_stats_num_seqs does not exist here, so
#  STATS_WINDOW and planned_requests are gone. TARGET emits a per-layer cache line
#  every cache_stats_log_interval CALLS (default 1000, per layer, not per request)
#  and has no end-of-run summary at all. Read serve.log for the interval lines.
#  Three caveats while reading them: the counters are allocated and recorded ONLY
#  when cache_policy_enabled is true; _record_cache_stats runs on the PREFETCH pass
#  as well as the on-demand one — so with PREFETCH=1 the reported hit rate blends
#  actual routing with predicted routing, and misses are undercounted by the
#  prefetch cap; and the MULTI-CARD path records no counters at all.
#
#  expert_prefetch_num IS NOT expert_prefetch_max. SOURCE's key does not exist here.
#  TARGET's expert_prefetch_num (int >= 1, default 2) already caps the prefetch to the
#  top-N highest-scoring predicted misses — same intent, same need to pin it.
#
#  THE LAST MoE LAYER IS NEVER PREFETCHED. _stage_predicted_topk returns early on
#  next_idx >= len(moe_layers) - 1, so coverage is next_idx in [1, L-2]. Layer 0 has
#  nothing before it and layer L-1 is skipped. Coverage does not move with MTP.
#
#  THE OFFLOAD THRESHOLD IS ABOUT TOKENS, NOT SEQUENCES. offload_threshold =
#  min(num_device_experts) // topk, compared against the decode token count, which
#  with MTP is max_num_seqs * (1 + num_speculative_tokens). The SOURCE script compared
#  it against MAX_NUM_SEQS, which is wrong by a factor of (1+N) once MTP is on. The
#  min() matters because num_device_experts may be a per-layer list on TARGET.
#
#  MULTI-CARD OFFLOAD IS OPT-IN AND UNGUARDED. enable_multi_card must be set AND EP
#  world size must be > 1; TARGET checks neither direction, and the EP>1-without-the-
#  flag combination silently gives every rank the same experts. When it IS on, the
#  decision unit changes from per-expert eviction to a whole-layer global placement
#  agreed by a gloo all-reduce inside the host callback, MC2 admission
#  (decode_tokens*topk <= per-rank slots) decides decode vs prefill instead of the
#  token threshold, and shard_per_rank (default true) pins each expert to one rank.
#
#  GRAPH MODE NEEDS THREE GATES. model_runner_v1.py :: _use_aclgraph requires
#  cudagraph_mode != NONE, compilation mode == VLLM_COMPILE, and not enforce_eager.
#  --enforce-eager silently disables everything under --compilation-config. The SOURCE
#  script hardcoded it, so nothing here inherits it.
#
#  cudagraph_capture_sizes IS IN TOKENS, AND SUB-FULL SIZES ARE REAL. The ceiling is
#  max_num_seqs * (1 + num_speculative_tokens), but with spec decode a request can be
#  scheduled with fewer tokens — no drafts on the first decode step after prefill, or a
#  short proposal. _dummy_run's uniform_decode branch shortens the last request for a
#  non-multiple num_tokens, so 1 and 2 (with query_len 3) are valid 1-request shapes,
#  not dead entries. What you pass is also only an INPUT: the effective list comes from
#  cudagraph_dispatcher.get_capture_descs(), and the same list feeds
#  set_draft_graph_params() for the MTP draft graphs. Confirm against the engine log.
#
#  CAPTURED SIZES ABOVE THE THRESHOLD ARE NOT SAFE. _prefill_load_layer is called from
#  the outer Python of update_weights, so it executes once at capture and never on
#  replay; a captured size above offload_threshold replays against whatever two layers'
#  weights were left in the pool, AND that graph records no prefetch nodes at all
#  because the trigger is guarded on the same threshold. capture_model() does not set
#  _skip_prefill, so this is reachable, not hypothetical.
#  CUDAGRAPH_CAPTURE_SIZES=auto keeps every entry under the threshold for this reason.
#
#  EAGER TPOT IS NOT A BASELINE FOR GRAPH TPOT. In eager the prefetch callback runs
#  inline on the main thread and ends in load_stream.synchronize(), so prefetch cannot
#  overlap and TPOT with PREFETCH=1 is worse than with PREFETCH=0. In replay the same
#  callback is a host-func node on the prefetch stream and does overlap. Use eager for
#  accuracy parity — the scores must match graph mode exactly — never for latency.
#
#  DYNAMIC_EPLB AND EXPERT OFFLOAD ARE MUTUALLY DESTRUCTIVE. w4a8.py / w8a8_dynamic.py
#  process_weights_after_loading delete w13_weight, w2_weight, the weight scales and
#  the scale biases under dynamic_eplb, replacing them with per-expert _list clones —
#  the exact attributes _load_expert_weights_into_slot writes into — and skip
#  pack_to_int32, so the byte layout diverges too. Nothing in TARGET guards it. Same
#  story for VLLM_ASCEND_ENABLE_FUSED_MC2=1 on W8A8: fused_w1_scale / fused_w2_scale
#  are derived once at load and never refreshed by paging. Both are asserted above.
#
#  MTP + W4A8 DID NOT LOAD ON THE PREVIOUS SNAPSHOT. The draft model raised
#    KeyError: 'model.layers.0.mtp_block.mlp.experts.w13_weight_scale_bias'
#  in deepseek_v4_mtp.py :: load_weights. w4a8.py :: get_weight creates w13_scale_bias
#  / w2_scale_bias only when new_quant_version (quant_description["version"]=="1.0.0"),
#  read from get_current_vllm_config() at method construction — so the draft model's
#  experts lacked a parameter the checkpoint supplies. This is a LOAD-time issue, not
#  an offload one (draft layers never offload). Re-verify on this snapshot; MTP still
#  defaults to 0 here. MTP with W8A8/bf16 is the isolation test.
#
#  ENGINE-CORE ORPHANS. `vllm serve` forks workers that hold NPU and host memory
#  until they exit; a name-matching reaper missed them (the process names changed),
#  so they accumulated across runs until AISBench refused to start on host RAM
#  (TINFER-PARAM-005, which triggers above 80% regardless of what it needs). reap()
#  signals the process group and then sweeps the port.
#
#  --num-prompts IS PER SUB-DATASET. mmlu_pro fans out to 14 categories, so 20 gives
#  ~280 docs and 300 would give 4200. It applies to the perf pass too.
#
#  MODES ARE PER-REQUEST. chat_template_kwargs rides on each request, which is what
#  lets the acc and perf passes run different modes against one server. A kwarg the
#  template does not declare is dropped silently — hence probe_mode().
#
#  STOCK CONFIGS HARDCODE THEIR ADDRESS. vllm_api_general_chat points at
#  localhost:8080 and carries no chat_template_kwargs, so it dials a dead port after
#  the model has loaded unless retargeted, and even then runs in whatever mode the
#  template defaults to. Reproduction only.
# =============================================================================