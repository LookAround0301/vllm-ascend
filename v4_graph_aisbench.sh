#!/usr/bin/env bash
# =============================================================================
#  v4_target_aisbench.sh — serve DeepSeek-V4-Flash on the TARGET codebase and
#  measure ACCURACY and TPOT on gsm8k / gpqa_diamond / mmlu_pro via AISBench.
#
#    source <your_v4_env>.sh
#    CARD=12 PORT=6061 TASKS=gpqa ./v4_target_aisbench.sh
#
#    DRY_RUN=1  print the plan, launch nothing     LIST=1  list dataset configs
#    RUN=acc|perf  one pass only (default both)    THINK=1 thinking on
#    ENFORCE_EAGER=1  eager debug path             MTP=1   speculative decode
#
#  OUTPUT, all under OUT_DIR:
#    run.log    this script: config, commands, AISBench scores, engine summary
#    serve.log  the engine, verbatim
#    acc/ perf/ AISBench work-dirs      configs/ the model configs actually sent
#    stats/     decode-stats summary .txt and per-layer .csv
#
#  THE ENGINE'S SUMMARY IS A SHUTDOWN ARTEFACT. [EXPERT-OFFLOAD-FINAL] is emitted
#  from NPUModelRunner.shutdown(), after the last request, on SIGTERM only. So:
#  reap() keeps the terminal mirror live across the whole grace window; the server
#  is stopped BEFORE the final banner rather than from the EXIT trap; and the block
#  is lifted out of serve.log into run.log afterwards. A SIGKILL loses it — hence
#  REAP_WAIT=180.
#
#  FINDINGS that change how you read a number are at the bottom.
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
REMOE_GATE="${REMOE_GATE:-}"             # fine-tuned MoE router gates, swapped in at load
CARD="${CARD:-6}"                        # "6" single-card, "6,7" multi-card (TP*DP must match)
PORT="${PORT:-7001}"
TP="${TP:-1}"; DP="${DP:-1}"
SEED="${SEED:-1024}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"       # empty = the checkpoint's own (1048576)
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
QUANTIZATION="${QUANTIZATION-ascend}"    # w8a8/w4a8 -> ascend; bf16 -> empty.
                                         # `-` not `:-`, so an explicit empty is honoured.
                                         # bf16 has NO offload on TARGET at all.
API_SERVER_COUNT="${API_SERVER_COUNT:-1}"

# ── graph mode (ACL graph) ───────────────────────────────────────────────────
# aclgraph needs all three: cudagraph_mode != NONE, compilation mode VLLM_COMPILE,
# not enforce_eager. --enforce-eager silently disables everything below it.
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
# TOKEN counts, not request counts. One graph per entry; a batch pads up to the
# smallest entry >= its token count, anything larger falls back to eager.
#   ""          vLLM's own defaults (warned against)
#   "auto"      1..decode_tokens = 1..MAX_NUM_SEQS*(1+NUM_SPEC_TOKENS). Never
#               truncated; entries above offload_threshold are WARNED about.
#   "auto-safe" same, truncated at offload_threshold — never captures an unsafe
#               size, at the cost of decode falling back to eager. Opt in.
#   "1,2,3"     explicit. Pin this when comparing runs.
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-auto}"
ENABLE_NPUGRAPH_EX="${ENABLE_NPUGRAPH_EX:-}"     # empty = TARGET default (true)
ENABLE_STATIC_KERNEL="${ENABLE_STATIC_KERNEL:-}" # empty = TARGET default (false)

# ── MTP / speculative decoding ───────────────────────────────────────────────
# Draft MoE layers never offload, so MTP changes only the decode TOKEN COUNT —
# and therefore whether the offload threshold is still cleared. Raising
# NUM_SPEC_TOKENS raises the NUM_DEVICE_EXPERTS floor by TOPK per increment.
MTP="${MTP:-0}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-2}"

# ── MoE offload (native TARGET keys only) ────────────────────────────────────
OFFLOAD="${OFFLOAD:-1}"
NUM_DEVICE_EXPERTS="${NUM_DEVICE_EXPERTS:-36}"   # int, or a JSON list "[36,36,...]".
                                         # FLOOR: MAX_NUM_SEQS*(1+NUM_SPEC_TOKENS)*TOPK.
                                         # 48 covers spec<=7 at seqs=1, topk=6.
NUM_DEVICE_LAYERS="${NUM_DEVICE_LAYERS:-1}"
TOPK="${TOPK:-6}"                        # overridden from the checkpoint config below
EXPERT_MAP_PATH="${EXPERT_MAP_PATH:-}"
CACHE_POLICY="${CACHE_POLICY:-1}"        # 1 = LRC, required by prefetch
PREFETCH="${PREFETCH:-1}"
EXPERT_PREFETCH_NUM="${EXPERT_PREFETCH_NUM:-1}"  # PIN when comparing. 1 or >=TOPK is
                                         # always safe; 1<n<TOPK depends on the build
                                         # (preflight probes the installed manager).
CACHE_STATS_LOG_INTERVAL="${CACHE_STATS_LOG_INTERVAL:-}"   # empty = default 1000
MOE_OFFLOAD_DEBUG="${MOE_OFFLOAD_DEBUG:-0}"      # per-layer trace INSIDE the host
                                         # callback — invalid for any timing run

# ── decode statistics (TOP-LEVEL keys, not expert_offload_config) ────────────
# One end-of-run summary covering both MoE cache/prefetch and spec-decode
# acceptance. Top level because the spec-decode half survives OFFLOAD=0.
# Numbers are per-step cross-layer means over NON-HASH MoE layers, summarised
# avg/median/min/max over steps — min/max mean worst/best STEP, not layer.
DECODE_STATS="${DECODE_STATS:-1}"
CSV="${CSV:-1}"                          # per-layer file ON TOP OF the summary
CSV_PATH="${CSV_PATH:-}"                 # empty = OUT_DIR/stats
STATS_FLUSH_EVERY="${STATS_FLUSH_EVERY:-100}"
STATS_FLUSH_SECONDS="${STATS_FLUSH_SECONDS:-30}"

# ── expert substitution (native; CHANGES ROUTING) ────────────────────────────
EXPERT_SUBSTITUTION="${EXPERT_SUBSTITUTION:-0}"
EXPERT_SUBSTITUTION_THRESHOLD="${EXPERT_SUBSTITUTION_THRESHOLD:-}"   # default 0.01

# ── offline hot-expert preload (native) ──────────────────────────────────────
# A relative filename resolves against vllm_ascend/expert_offload/.
HOT_PRELOAD="${HOT_PRELOAD:-0}"
HOT_EXPERTS_FILE="${HOT_EXPERTS_FILE:-}"

# ── multi-card EP offload (native, opt-in) ───────────────────────────────────
# MUST move together with EP world size. TARGET checks neither direction.
ENABLE_MULTI_CARD="${ENABLE_MULTI_CARD:-0}"
SHARD_PER_RANK="${SHARD_PER_RANK:-}"     # empty = TARGET default (true)

# ── LRC policy weights (empty = TARGET defaults) ─────────────────────────────
CACHE_RECENT_WINDOW="${CACHE_RECENT_WINDOW:-}"   # 32
CACHE_EMA_BETA="${CACHE_EMA_BETA:-}"             # 0.9, must be [0,1)
CACHE_RECENT_WEIGHT="${CACHE_RECENT_WEIGHT:-}"   # 1.0
CACHE_EMA_WEIGHT="${CACHE_EMA_WEIGHT:-}"         # 0.5
CACHE_ROUTER_WEIGHT="${CACHE_ROUTER_WEIGHT:-}"   # 0.3; 0 skips the staging copy
CACHE_AGE_WEIGHT="${CACHE_AGE_WEIGHT:-}"         # 0.01

# ── other additional_config knobs ────────────────────────────────────────────
MULTISTREAM_OVERLAP_SHARED_EXPERT="${MULTISTREAM_OVERLAP_SHARED_EXPERT:-0}"
ENABLE_CPU_BINDING="${ENABLE_CPU_BINDING:-0}"    # TARGET default is true

# ── client ───────────────────────────────────────────────────────────────────
TASKS="${TASKS:-gsm8k gpqa mmlu_pro}"
RUN="${RUN:-both}"                       # acc | perf | both
THINK="${THINK:-}"                       # ""=presets, 0=non-think, 1=thinking
THINK_EFFORT="${THINK_EFFORT:-}"         # ""=preset, high | max
NUM_PROMPTS="${NUM_PROMPTS-unset}"       # acc pass; empty = FULL dataset
PERF_PROMPTS="${PERF_PROMPTS-30}"        # perf pass; empty = full dataset
AVG_N="${AVG_N:-}"                       # repeats per question; 4 to report
TEMP="${TEMP:-}"; TOPP="${TOPP-unset}"   # acc sampling override
PERF_TEMP="${PERF_TEMP:-0.0}"            # pinned so TPOT stays comparable
EXTRACT_RATE="${EXTRACT_RATE:-1}"        # read the rate BEFORE the score
STOCK_MODEL="${STOCK_MODEL:-}"           # AISBench-shipped config; bypasses presets

# ── run control ──────────────────────────────────────────────────────────────
OUT_DIR="${OUT_DIR:-$PWD/aisbench_results/$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"; LIST="${LIST:-0}"; SKIP_SERVE="${SKIP_SERVE:-0}"
PROBE="${PROBE:-1}"                      # verify the served reasoning mode
WAIT="${WAIT:-1000}"                     # seconds to wait for /health
REAP_WAIT="${REAP_WAIT:-180}"            # grace before SIGKILL. The summary is
                                         # printed during it; SIGKILL destroys it.
HEARTBEAT="${HEARTBEAT:-300}"            # liveness line interval; 0 = off
ALLOW_UNSAFE="${ALLOW_UNSAFE:-0}"        # guards -> warnings. To REPRODUCE a
                                         # defect, not to get past one.

# ── presets: task | dataset | acc mode | perf mode | temp | top_p | max_out | prompts | avg_n
#  temp/top_p are ACCURACY ONLY; perf uses PERF_TEMP and omits top_p.
#  prompts is PER SUB-DATASET (mmlu_pro fans out to 14). Blank = full set.
PRESETS="
gsm8k    | gsm8k_gen_0_shot_cot_chat_prompt | non-think | non-think | 0.0 |      | 32768 |     | 1
gpqa     | gpqa_gen_0_shot_cot_chat_prompt  | non-think | non-think | 0.6 | 0.95 | 65536 |     | 1
mmlu_pro | mmlu_pro_gen_0_shot_str          | non-think | non-think | 0.6 | 0.95 | 65536 |  20 | 1
"

ref() {   # vendor's published number. $1=task $2=mode
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

die()   { echo "ERROR: $*" >&2; exit 1; }
guard() { [[ "${ALLOW_UNSAFE}" == "1" ]] && echo "  warn    : [unsafe] $*" || die "$*"; }
bool()  { [[ "$1" == "1" ]] && echo true || echo false; }

export ASCEND_RT_VISIBLE_DEVICES="${CARD}"
export VLLM_BATCH_INVARIANT=0
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"; export NO_PROXY="${no_proxy}"
[[ -n "${AIS_BENCH_DATASETS_CACHE}" ]] && export AIS_BENCH_DATASETS_CACHE \
                                       || unset AIS_BENCH_DATASETS_CACHE

RUN_LOG="${OUT_DIR}/run.log"; SERVE_LOG="${OUT_DIR}/serve.log"
CFG_DIR="${OUT_DIR}/configs/models"; STATS_DIR="${CSV_PATH:-${OUT_DIR}/stats}"
SERVE_PID=""; SERVE_PGID=""; HB_PID=""; TAIL_PID=""; MIRROR_AT=0

# =============================================================================
#  derived values, validation, capture sizes
# =============================================================================

DECODE_TOK_PER_REQ=1
[[ "${MTP}" == "1" ]] && DECODE_TOK_PER_REQ=$(( 1 + NUM_SPEC_TOKENS ))
DECODE_TOKENS=$(( MAX_NUM_SEQS * DECODE_TOK_PER_REQ ))
GRAPH_ON=0
[[ "${ENFORCE_EAGER}" != "1" && "${CUDAGRAPH_MODE}" != "NONE" ]] && GRAPH_ON=1
NCARDS="$(awk -F',' '{print NF}' <<<"${CARD}")"
EP_SIZE=$(( TP * DP ))

# The threshold uses the SMALLEST per-layer entry when num_device_experts is a list.
NDE_MIN="${NUM_DEVICE_EXPERTS}"
[[ "${NUM_DEVICE_EXPERTS}" == \[* ]] && \
  NDE_MIN="$(tr -d '[] ' <<<"${NUM_DEVICE_EXPERTS}" | tr ',' '\n' | sort -n | head -1)"

[[ -z "${THINK}" || "${THINK}" =~ ^[01]$ ]]                       || die "THINK must be empty, 0 or 1"
[[ -z "${THINK_EFFORT}" || "${THINK_EFFORT}" =~ ^(high|max)$ ]]   || die "THINK_EFFORT must be empty, high or max"
[[ "${RUN}" =~ ^(acc|perf|both)$ ]]                               || die "RUN must be acc, perf or both"
[[ -z "${AVG_N}" || "${AVG_N}" =~ ^[1-9][0-9]*$ ]]                || die "AVG_N must be a positive integer"
[[ "${NUM_SPEC_TOKENS}" =~ ^[1-9][0-9]*$ ]]                       || die "NUM_SPEC_TOKENS must be a positive integer"
[[ "${EXPERT_PREFETCH_NUM}" =~ ^[1-9][0-9]*$ ]]                   || die "EXPERT_PREFETCH_NUM must be a positive integer"
[[ "${NDE_MIN}" =~ ^[1-9][0-9]*$ ]]                               || die "NUM_DEVICE_EXPERTS must be a positive integer or a JSON list of them"
[[ -z "${SHARD_PER_RANK}" || "${SHARD_PER_RANK}" =~ ^(true|false)$ ]] || die "SHARD_PER_RANK must be empty, true or false (raw JSON)"
[[ -z "${CUDAGRAPH_CAPTURE_SIZES}" || "${CUDAGRAPH_CAPTURE_SIZES}" =~ ^(auto|auto-safe|[0-9]+(,[0-9]+)*)$ ]] \
  || die "CUDAGRAPH_CAPTURE_SIZES must be empty, auto, auto-safe, or a comma-separated integer list"
[[ "${HOT_PRELOAD}" != "1" || -n "${HOT_EXPERTS_FILE}" ]]         || die "HOT_PRELOAD=1 needs HOT_EXPERTS_FILE (a .json)"
[[ -z "${HOT_EXPERTS_FILE}" || "${HOT_EXPERTS_FILE}" == *.json ]] || die "HOT_EXPERTS_FILE must end in .json"
[[ "${NCARDS}" == "${EP_SIZE}" ]] || die "CARD lists ${NCARDS} device(s) but TP*DP=${EP_SIZE}. They must match."
[[ -z "${REMOE_GATE}" || -e "${REMOE_GATE}" ]]                    || die "REMOE_GATE not found: ${REMOE_GATE}"

# TOPK comes from the checkpoint when we can read it — the engine reads the config,
# not this variable. MUST resolve before the threshold and capture arithmetic below.
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
  [[ "${TOPK}" != "${_CFG_TOPK}" ]] && \
    echo "  warn    : TOPK=${TOPK} but ${MODEL}/config.json says ${_CFG_TOPK}; using ${_CFG_TOPK}" >&2
  TOPK="${_CFG_TOPK}"
fi
[[ "${TOPK}" =~ ^[1-9][0-9]*$ ]] || die "TOPK must be a positive integer (got '${TOPK}')"

THR=$(( NDE_MIN / TOPK ))            # offload_threshold, in decode TOKENS
NDE_FLOOR=$(( DECODE_TOKENS * TOPK )) # smallest pool that keeps decode on the offload path

# auto: 1..decode_tokens, always reaching the real decode shape. It used to clamp at
# THR, which did not make the run safe — it made the decode shape uncaptured, so every
# step fell back to eager while the run reported graph mode. Unsafe entries are warned
# about further down instead; auto-safe restores the clamp for anyone who wants it.
CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES}"; AUTO_NOTE=""; AUTO_TRUNCATED=0
if [[ "${CAPTURE_SIZES}" == auto || "${CAPTURE_SIZES}" == auto-safe ]]; then
  _mode="${CAPTURE_SIZES}"; _cap="${DECODE_TOKENS}"
  if [[ "${_mode}" == auto-safe && "${OFFLOAD}" == "1" ]] && (( THR < _cap )); then
    _cap="${THR}"; AUTO_TRUNCATED=1
  fi
  (( _cap > MAX_NUM_BATCHED_TOKENS )) && _cap="${MAX_NUM_BATCHED_TOKENS}"
  # An empty list silently falls back to vLLM's defaults, the one outcome nobody wants.
  (( _cap < 1 )) && { _cap=1; AUTO_NOTE="${_mode}: offload_threshold=${THR} < 1 — floored to [1]"; }
  _sizes=(); for (( _s=1; _s<=_cap; _s++ )); do _sizes+=( "${_s}" ); done
  IFS=','; CAPTURE_SIZES="${_sizes[*]}"; unset IFS
  if [[ -z "${AUTO_NOTE}" ]]; then
    AUTO_NOTE="${_mode} -> ${CAPTURE_SIZES}   (1..${_cap}; decode_tokens=${DECODE_TOKENS} = ${MAX_NUM_SEQS}x(1+${NUM_SPEC_TOKENS}))"
    (( AUTO_TRUNCATED )) && AUTO_NOTE+=$'\n            '"TRUNCATED at offload_threshold=${THR}: decode shape ${DECODE_TOKENS} is NOT captured — every decode step runs EAGER"
  fi
  (( ${#_sizes[@]} > 16 )) && AUTO_NOTE+=$'\n            '"${#_sizes[@]} graphs to capture — warmup time and graph memory scale with this"
  unset _mode _cap _s _sizes
fi

TASK_LIST="$(tr ',' ' ' <<<"${TASKS}")"
PKG_CFG="$(python3 -c 'import os,ais_bench.benchmark as b;print(os.path.join(os.path.dirname(b.__file__),"configs"))' 2>/dev/null)"
DATA_ROOT="${AIS_BENCH_DATASETS_CACHE:-$(python3 -c 'import os,ais_bench;print(os.path.dirname(os.path.dirname(os.path.abspath(ais_bench.__file__))))' 2>/dev/null)}"
STOCK_SRC=""
[[ -n "${STOCK_MODEL}" && -n "${PKG_CFG}" ]] && \
  STOCK_SRC="$(find "${PKG_CFG}/models" -name "${STOCK_MODEL}.py" -print -quit 2>/dev/null)"

# =============================================================================
#  helpers
# =============================================================================

# Sets P_DS P_ACC_MODE P_PERF_MODE P_TEMP P_TOPP P_MAXOUT P_NUM P_AVGN for one task.
# Split on '|' not tab: tab is IFS whitespace, so `read` would collapse empty cells.
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
  echo "Dataset configs (${PKG_CFG}/datasets):"
  for d in gsm8k gpqa mmlu_pro; do
    echo "  [${d}]"
    find "${PKG_CFG}/datasets/${d}" -maxdepth 1 -name '*.py' 2>/dev/null | sed 's|.*/||;s|\.py$||;s|^|    |' | sort
  done
  echo "  [stock model configs — valid for STOCK_MODEL=<name>]"
  find "${PKG_CFG}/models" -maxdepth 1 -name '*.py' 2>/dev/null | sed 's|.*/||;s|\.py$||;s|^|    |' | sort
}

# Called twice: before the tee (run.log needs a parent) and after preflight.
ensure_dirs() {
  local d dirs=( "${OUT_DIR}" "${CFG_DIR}" )
  for d in acc perf; do [[ "${RUN}" == "both" || "${RUN}" == "${d}" ]] && dirs+=( "${OUT_DIR}/${d}" ); done
  # The engine makes the stats dir itself, but only at shutdown — by which point an
  # unwritable path costs the whole run's table. Create and write-test it up front.
  [[ "${DECODE_STATS}" == "1" ]] && dirs+=( "${STATS_DIR}" )
  for d in "${dirs[@]}"; do
    mkdir -p "${d}" 2>/dev/null || die "cannot create directory: ${d}"
    [[ -w "${d}" ]] || die "directory not writable: ${d}"
  done
  return 0
}

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

# serve.log -> terminal (fd3). Off during a client run so AISBench's progress table
# owns the screen. RESUMES from where it stopped, so nothing logged behind the table
# is lost — expect a burst when a pass finishes.
mirror_start() { tail -n "+$(( MIRROR_AT + 1 ))" -f "${SERVE_LOG}" >&3 2>/dev/null & TAIL_PID=$!; return 0; }
mirror_stop()  { [[ -n "${TAIL_PID}" ]] && kill "${TAIL_PID}" 2>/dev/null; TAIL_PID=""
                 MIRROR_AT=$(wc -l < "${SERVE_LOG}" 2>/dev/null || echo 0); return 0; }

# TERM the process group, watch it exit WITH THE MIRROR STILL LIVE (the summary is
# printed during shutdown), drain, then stop the mirror and KILL any survivors.
# Idempotent: clears SERVE_PID/SERVE_PGID, so the EXIT trap stays a safe net.
reap() {
  [[ -n "${HB_PID}" ]] && kill "${HB_PID}" 2>/dev/null; HB_PID=""
  [[ -z "${SERVE_PID}${SERVE_PGID}" ]] && { mirror_stop; return 0; }
  local i=0 left
  echo "[reap] stopping server (pgid=${SERVE_PGID:-none} pid=${SERVE_PID:-none}); up to" \
       "${REAP_WAIT}s for a clean exit — the statistics summary is printed during it" >&2
  [[ -n "${SERVE_PGID}" ]] && kill -TERM -"${SERVE_PGID}" 2>/dev/null || kill_tree "${SERVE_PID}" TERM
  for ((i=0; i<REAP_WAIT; i++)); do
    { [[ -n "${SERVE_PGID}" ]] && kill -0 -"${SERVE_PGID}" 2>/dev/null; } || \
    { [[ -z "${SERVE_PGID}" ]] && kill -0 "${SERVE_PID}" 2>/dev/null; } || break
    sleep 1
  done
  (( i >= REAP_WAIT )) && echo "[reap] WARNING: no clean exit in ${REAP_WAIT}s — SIGKILL." \
       "Anything unflushed, including the end-of-run summary, is LOST. Raise REAP_WAIT." >&2
  sleep 2                     # tail -f is async; cutting it now truncates the tail
  mirror_stop
  [[ -n "${SERVE_PGID}" ]] && kill -KILL -"${SERVE_PGID}" 2>/dev/null || kill_tree "${SERVE_PID}" KILL
  left="$(port_pids)"
  [[ -n "${left// /}" ]] && { kill -KILL ${left} 2>/dev/null; sleep 2; left="$(port_pids)"; }
  [[ -n "${left// /}" ]] \
    && echo "[reap] WARNING: still listening on :${PORT} -> ${left}; check npu-smi info" >&2 \
    || echo "[reap] port ${PORT} released" >&2
  SERVE_PID=""; SERVE_PGID=""; return 0
}

# One generated config per task+pass, plus the ais_bench argv. Sets MODEL_TASK,
# CFG_FILE, P_PASS_MODE, CMD and the P_* globals.
build_cmd() {   # $1=task  $2=acc|perf
  local task="$1" pass="$2" gk n
  preset "${task}"
  [[ "${pass}" == perf ]] && P_PASS_MODE="${P_PERF_MODE}" || P_PASS_MODE="${P_ACC_MODE}"
  MODEL_TASK="v4_${task}_${pass}"; CFG_FILE="${CFG_DIR}/${MODEL_TASK}.py"

  if [[ -n "${STOCK_MODEL}" ]]; then
    # Stock configs hardcode localhost:8080 and ship path="", which kills the perf
    # summarizer after every request has been sent. Copy with only those rewritten.
    MODEL_TASK="v4_stock_${STOCK_MODEL}"; CFG_FILE="${CFG_DIR}/${MODEL_TASK}.py"
    if [[ "${DRY_RUN}" != "1" && ! -f "${CFG_FILE}" ]]; then
      [[ -n "${STOCK_SRC}" ]] || die "STOCK_MODEL='${STOCK_MODEL}' not installed; try LIST=1 $0"
      python3 - "${STOCK_SRC}" "${CFG_FILE}" "${PORT}" "${TOKENIZER}" \
               "$([[ "${RUN}" != acc ]] && echo 1 || echo 0)" <<'PY' || die "could not retarget the stock config"
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
    fi
  elif [[ "${DRY_RUN}" != "1" ]]; then
    if [[ "${pass}" == perf ]]; then
      gk="            temperature=${PERF_TEMP},"
    else
      gk="            temperature=${P_TEMP},"
      [[ -n "${P_TOPP}" ]] && gk+=$'\n'"            top_p=${P_TOPP},"
    fi
    # The mode is always stated explicitly, including thinking=False, so a run can
    # never silently sit in the wrong one.
    if [[ "${P_PASS_MODE}" == non-think ]]; then
      gk+=$'\n'"            chat_template_kwargs=dict(thinking=False),"
    else
      gk+=$'\n'"            chat_template_kwargs=dict(thinking=True, reasoning_effort=\"${P_PASS_MODE}\"),"
    fi
    # avg@n on the accuracy pass only: repeats would inflate the perf request count
    # without changing the latency distribution.
    [[ "${pass}" == acc && "${P_AVGN}" -gt 1 ]] && gk+=$'\n'"            num_return_sequences=${P_AVGN},"
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
        stream=$([[ "${pass}" == perf ]] && echo True || echo False),
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
  fi

  CMD=( ais_bench --models "${MODEL_TASK}" --datasets "${P_DS}"
        --config-dir "${OUT_DIR}/configs" --work-dir "${OUT_DIR}/${pass}" )
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

# Lift AISBench's numbers into run.log; point at the logs it wrote itself.
report_step() {   # $1=task $2=acc|perf $3=rc
  local t="$1" p="$2" rc="$3" dir f found=0 pat
  (( rc != 0 )) && { echo "[result] ${t} ${p}: ais_bench FAILED rc=${rc}" >&2; return 0; }
  dir="$(find "${OUT_DIR}/${p}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)"
  [[ -z "${dir}" ]] && { echo "[result] ${t} ${p}: no run dir under ${OUT_DIR}/${p}"; return 0; }
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

# A chat-template kwarg the template does not declare is dropped SILENTLY, so a run
# can sit in the wrong mode — worth ~16 points on gpqa. One 64-token request settles it.
probe_mode() {   # $1 = non-think | high | max
  local kw want got resp
  if [[ "$1" == non-think ]]; then want=off; kw='{"thinking":false}'
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
       different key (edit build_cmd: thinking -> enable_thinking). PROBE=0 to override."
  return 0
}

# =============================================================================
#  build the serve argv
# =============================================================================
# expert_offload_config is a CLOSED key set — an unknown key raises at startup.
# TOP-LEVEL keys use .get(), so an unknown one there is IGNORED silently, which is
# worse: a typo costs a run rather than failing fast. Hence the post-health greps.
# Every key is emitted ONLY when non-empty, so an untouched knob keeps TARGET's
# default rather than this script's opinion of it.

[[ "${LIST}" == "1" ]] && { list_cfgs; exit 0; }
for t in ${TASK_LIST}; do preset "${t}"; done      # a bad task name dies before anything loads

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
    p+=",\"expert_prefetch_enabled\":true,\"expert_prefetch_num\":${EXPERT_PREFETCH_NUM}"
  if [[ "${EXPERT_SUBSTITUTION}" == "1" ]]; then
    p+=",\"expert_substitution_enabled\":true"
    [[ -n "${EXPERT_SUBSTITUTION_THRESHOLD}" ]] && p+=",\"expert_substitution_threshold\":${EXPERT_SUBSTITUTION_THRESHOLD}"
  fi
  [[ "${HOT_PRELOAD}" == "1" ]] && p+=",\"hot_expert_preload\":true,\"hot_experts_file\":\"${HOT_EXPERTS_FILE}\""
  [[ "${ENABLE_MULTI_CARD}" == "1" ]] && p+=",\"enable_multi_card\":true"
  [[ -n "${SHARD_PER_RANK}"           ]] && p+=",\"shard_per_rank\":${SHARD_PER_RANK}"
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
SPEC_CFG=""
[[ "${MTP}" == "1" ]] && SPEC_CFG="{\"num_speculative_tokens\":${NUM_SPEC_TOKENS},\"method\":\"mtp\"}"

# --override-generation-config only sets a server-side default; every AISBench
# request carries its own sampling and thinking mode, so the client wins — which is
# what lets the acc and perf passes run different modes against one server.
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
[[ -n "${QUANTIZATION}"      ]] && SERVE+=( --quantization "${QUANTIZATION}" )
[[ -n "${MAX_MODEL_LEN}"     ]] && SERVE+=( --max-model-len "${MAX_MODEL_LEN}" )
[[ "${ENFORCE_EAGER}" == "1" ]] && SERVE+=( --enforce-eager )
[[ -n "${COMPILE_CFG}"       ]] && SERVE+=( --compilation-config "${COMPILE_CFG}" )
[[ -n "${SPEC_CFG}"          ]] && SERVE+=( --speculative-config "${SPEC_CFG}" )
SERVE+=( --additional-config "${ADDL}" )

# =============================================================================
#  print the plan; check the guards
# =============================================================================

exec 3>&2                                          # fd3 = the real terminal
if [[ "${DRY_RUN}" != "1" ]]; then
  ensure_dirs                                      # before the tee: run.log needs a parent
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
echo "  stats   : ${DECODE_STATS} csv=${CSV} -> ${STATS_DIR} (summary at graceful shutdown; reap_wait=${REAP_WAIT}s)"
[[ -n "${COMPILE_CFG}" ]] && echo "  compile : ${COMPILE_CFG}"
[[ -n "${SPEC_CFG}"    ]] && echo "  spec    : ${SPEC_CFG}"
echo "  addl-cfg: ${ADDL}"
echo "  client  : run=${RUN} perf_prompts=${PERF_PROMPTS:-full} perf_temp=${PERF_TEMP} extract_rate=${EXTRACT_RATE}"
[[ -n "${STOCK_MODEL}" ]] && echo "  !! STOCK_MODEL=${STOCK_MODEL}: presets NOT applied. Not model-card comparable."

# Environment invariants the offload subsystem depends on. These belong to the env
# file; assert rather than re-export, so a mismatch is visible instead of silently
# corrected. Both are legitimate WITHOUT offload, so gate on OFFLOAD=1.
if [[ "${OFFLOAD}" == "1" ]]; then
  [[ "${DYNAMIC_EPLB:-false}" == "false" ]] || guard "DYNAMIC_EPLB=${DYNAMIC_EPLB} with expert offload:
       process_weights_after_loading deletes the very tensors the paging primitive writes
       into. Export DYNAMIC_EPLB=false, or run with OFFLOAD=0."
  [[ "${VLLM_ASCEND_ENABLE_FUSED_MC2:-0}" == "0" ]] || guard "VLLM_ASCEND_ENABLE_FUSED_MC2=${VLLM_ASCEND_ENABLE_FUSED_MC2}
       with expert offload: W8A8 fused scales go stale after paging, and FUSED_MC2 applies
       log2phy unclamped."
fi
[[ "${GRAPH_ON}" == "1" && "${ASCEND_LAUNCH_BLOCKING:-0}" != "0" ]] && \
  echo "  warn    : ASCEND_LAUNCH_BLOCKING=${ASCEND_LAUNCH_BLOCKING} in graph mode — every launch is
            serialised on the host; the prefetch stream cannot overlap anything."

if [[ "${OFFLOAD}" == "1" && -z "${QUANTIZATION}" ]]; then
  echo "  warn    : QUANTIZATION='' (bf16) — TARGET drops expert offload for unquantized MoE"
  echo "            layers (quant_type == NONE), so OFFLOAD=1 is silently a no-op. The summary"
  echo "            still prints, with only the spec-decode rows populated."
fi
[[ "${OFFLOAD}" != "1" && "${DECODE_STATS}" == "1" ]] && \
  echo "  note    : OFFLOAD=0 — the summary reports spec-decode acceptance only; CSV=1 would write an empty table."

if [[ "${OFFLOAD}" == "1" && -n "${QUANTIZATION}" ]]; then
  echo "  thresh  : offload_threshold=${THR} (${NDE_MIN}/${TOPK}); decode_tokens=${DECODE_TOKENS}"
  echo "  sizing  : NUM_DEVICE_EXPERTS floor = decode_tokens*topk = ${DECODE_TOKENS}*${TOPK} = ${NDE_FLOOR}"
  # Above the threshold, decode leaves the LRC path entirely: no paging, no prefetch,
  # no statistics. The floor is the actionable form of the same statement.
  (( DECODE_TOKENS > THR )) && guard "decode_tokens=${DECODE_TOKENS} > offload_threshold=${THR}: decode takes
       the prefill-pool path, so there is NO paging, NO prefetch and NO cache statistics — the
       offload half of the summary would be empty and the run would measure nothing about
       offload. Raise NUM_DEVICE_EXPERTS to >= ${NDE_FLOOR}, or lower MAX_NUM_SEQS / NUM_SPEC_TOKENS."

  # CACHE_POLICY=0 no longer costs the measurement: [EXPERT-OFFLOAD-FINAL] is gated on
  # the offload decode path being active, not on the eviction policy. Only the old
  # per-layer interval counters disappear.
  [[ "${PREFETCH}" == "1" && "${CACHE_POLICY}" != "1" ]] && \
    echo "  warn    : PREFETCH=1 with CACHE_POLICY=0 — eviction is arbitrary (the summary is still collected; the old interval counters are not)"
  [[ "${PREFETCH}" == "1" ]] && (( EXPERT_PREFETCH_NUM > 1 && EXPERT_PREFETCH_NUM < TOPK )) && \
    echo "  note    : EXPERT_PREFETCH_NUM=${EXPERT_PREFETCH_NUM} is between 1 and TOPK=${TOPK} — preflight probes the installed _stage_predicted_topk for this shape."
  [[ "${PREFETCH}" == "1" && "${EXPERT_PREFETCH_NUM}" == "1" ]] && \
    echo "  note    : EXPERT_PREFETCH_NUM=1 is degenerate — one predicted id, so 'predicted resident |A|/|P|' can only read 0.0 or 1.0. The engine's default is 2."
  [[ "${PREFETCH}" == "1" && "${EXPERT_PREFETCH_NUM}" -gt "${TOPK}" ]] && \
    echo "  note    : EXPERT_PREFETCH_NUM=${EXPERT_PREFETCH_NUM} > TOPK — clamped to ${TOPK} by the engine"

  # enable_multi_card and EP size must move together; TARGET checks neither direction.
  (( EP_SIZE > 1 )) && [[ "${ENABLE_MULTI_CARD}" != "1" ]] && guard "EP=${EP_SIZE} with ENABLE_MULTI_CARD=0
       and offload on: every rank keeps a standard-EP expert_map next to an offload
       num_local_experts, mgr.ep_size stays 1, and every rank loads the SAME experts.
       Set ENABLE_MULTI_CARD=1, or run one card."
  (( EP_SIZE == 1 )) && [[ "${ENABLE_MULTI_CARD}" == "1" ]] && guard "ENABLE_MULTI_CARD=1 at EP=1: comm
       selection takes the multi-card admission branch while the layers stay single-card."

  if [[ "${ENABLE_MULTI_CARD}" == "1" ]]; then
    # num_device_experts_for_rank RAISES when capacity % ep_size != 0, and it is called
    # at FORWARD time — an indivisible capacity starts the engine and dies mid-decode.
    for _e in $(tr -d '[] ' <<<"${NUM_DEVICE_EXPERTS}" | tr ',' ' '); do
      (( _e % EP_SIZE == 0 )) || die "num_device_experts entry ${_e} is not divisible by EP size ${EP_SIZE}:
       num_device_experts_for_rank raises ValueError at forward time, after the engine has
       loaded. Pick a multiple of ${EP_SIZE}."
    done
    _slots="${NDE_MIN}"
    [[ "${SHARD_PER_RANK}" != false ]] && _slots=$(( NDE_MIN / EP_SIZE ))
    _mc_tok=$(( _slots / TOPK ))
    echo "  mc      : per-rank slots=${_slots}; MC2 needs decode_tokens*${TOPK} <= ${_slots} (i.e. decode_tokens <= ${_mc_tok}); decode_tokens=${DECODE_TOKENS}"
    # In multi-card the decode/prefill split is the COMM TYPE, not the token threshold:
    # MC2 is admitted only when num_tokens*topk <= per-rank slots. Miss it and every step
    # goes ALLTOALL -> shard prefill pool -> update_weights_multi_card returns early and
    # prefetch is never triggered. The run looks normal and measures nothing.
    if (( DECODE_TOKENS > _mc_tok )); then
      _need=$(( DECODE_TOKENS * TOPK * EP_SIZE ))
      [[ "${SHARD_PER_RANK}" == false ]] && _need=$(( DECODE_TOKENS * TOPK ))
      guard "multi-card decode never reaches MC2: decode_tokens=${DECODE_TOKENS} > ${_mc_tok}. Every step
       would take ALLTOALL + the shard prefill pool, so no paging, no cache stats and NO
       PREFETCH. Raise NUM_DEVICE_EXPERTS to >= ${_need} (a multiple of ${EP_SIZE}), or lower
       MAX_NUM_SEQS / NUM_SPEC_TOKENS so decode_tokens <= ${_mc_tok}."
    fi
    # With shard_per_rank each expert is pinned to one rank, so a layer whose active
    # experts crowd into one shard needs that many slots there. At most decode_tokens*topk
    # can be active, so slots >= that makes overflow (a hard mid-decode RuntimeError)
    # impossible.
    (( _slots >= DECODE_TOKENS * TOPK )) \
      && echo "            headroom above the worst case: $(( _slots - DECODE_TOKENS * TOPK )) slots" \
      || echo "  warn    : per-rank slots=${_slots} < worst-case active experts $(( DECODE_TOKENS * TOPK ))"
    echo "  note    : multi-card records NO per-layer cache statistics (the accounting lives in the"
    echo "            single-card host callback). The summary prints with those rows empty; the"
    echo "            spec-decode rows are unaffected. CSV=1 yields empty metric columns."
    echo "  note    : if this checkpoint is W8A8, multi-card PREFILL is not wired for it — w8a8_dynamic.py"
    echo "            falls through to the single-card pool and never patches the All2All dispatcher (D8)."
  fi

  [[ "${EXPERT_SUBSTITUTION}" == "1" ]] && \
    echo "  note    : substitution ON — it CHANGES ROUTING. Accuracy is not comparable to a run with it off."
  [[ "${MOE_OFFLOAD_DEBUG}" == "1" ]] && \
    echo "  warn    : MOE_OFFLOAD_DEBUG=1 puts sorted()+tolist()+logging inside the host callback, per layer per step — TPOT here is not a timing number, and serve.log will be enormous"

  # _prefill_load_layer runs in the outer Python of update_weights, so it executes once
  # at capture and never on replay: a captured size above the threshold replays against
  # stale pool weights AND records no prefetch nodes. `auto` warns rather than truncating,
  # because a truncated list means decode runs eager — a quieter failure.
  if [[ "${GRAPH_ON}" == "1" ]]; then
    if [[ -z "${CAPTURE_SIZES}" ]]; then
      echo "  warn    : graph ON with vLLM's default capture sizes. Any size > ${THR} replays against stale"
      echo "            prefill-pool weights. Set CUDAGRAPH_CAPTURE_SIZES=auto, or pin a list."
    else
      _unsafe=""
      for _s in ${CAPTURE_SIZES//,/ }; do (( _s > THR )) && _unsafe+="${_s} "; done
      if [[ -n "${_unsafe}" ]]; then
        echo "  warn    : capture size(s) ${_unsafe%% } > offload_threshold=${THR} — those graphs replay against"
        echo "            stale prefill-pool weights. Raise NUM_DEVICE_EXPERTS to >= ${NDE_FLOOR} to make every"
        echo "            captured size safe, or use auto-safe to drop them (decode then runs eager)."
      fi
    fi
  fi
fi
[[ "${GRAPH_ON}" != "1" ]] && \
  echo "  note    : eager path — prefetch cannot overlap. TPOT is a debug number, not comparable to graph mode. Accuracy IS comparable, and must match."
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
      DECODE_STATS CSV CSV_PATH REAP_WAIT EXPERT_SUBSTITUTION EXPERT_SUBSTITUTION_THRESHOLD \
      HOT_PRELOAD HOT_EXPERTS_FILE ENABLE_MULTI_CARD SHARD_PER_RANK CACHE_RECENT_WINDOW \
      CACHE_EMA_BETA CACHE_RECENT_WEIGHT CACHE_EMA_WEIGHT CACHE_ROUTER_WEIGHT CACHE_AGE_WEIGHT \
      MULTISTREAM_OVERLAP_SHARED_EXPERT ENABLE_CPU_BINDING API_SERVER_COUNT \
      PROBE STOCK_MODEL MODEL SERVED_NAME TOKENIZER AIS_BENCH_DATASETS_CACHE
   do printf '%s=%q ' "${v}" "${!v-}"; done)$0"
echo "────────────────────────────────────────────────────────────"
printf '  $ '; printf '%q ' "${SERVE[@]}"; echo
for step in "${PLAN[@]}"; do build_cmd "${step%:*}" "${step#*:}"; printf '  $ '; printf '%q ' "${CMD[@]}"; echo; done
[[ "${DRY_RUN}" == "1" ]] && { echo "[DRY_RUN] nothing launched."; exit 0; }

# =============================================================================
#  preflight
# =============================================================================

# vLLM resolves its platform from the vllm.platform_plugins entry-point group at
# import time. Without vllm-ascend the group is empty and DeviceConfig raises
# "Failed to infer device type" while BUILDING THE ARG PARSER — before any flag here
# is read. Catch it up front so the message names the cause, not the symptom.
python3 - <<'PY'
import sys
try:
    from importlib.metadata import entry_points
except Exception:
    sys.exit(0)
eps = list(entry_points(group="vllm.platform_plugins"))
if not eps:
    sys.stderr.write("[!!] no 'vllm.platform_plugins' entry point in this env — vLLM will fall\n"
                     "     through to UnspecifiedPlatform. vllm-ascend is not installed here.\n"
                     "       pip show vllm-ascend ; python -c \"import vllm_ascend; print(vllm_ascend.__file__)\"\n")
    sys.exit(1)
for mod in ("vllm_ascend", "torch_npu"):
    try:
        __import__(mod)
    except Exception as e:
        sys.stderr.write("[!!] %s will not import: %s\n"
                         "     Did you `source` the env file (CANN set_env.sh)?\n" % (mod, e))
        sys.exit(1)
print("[plat ] vllm.platform_plugins: %s" % ", ".join(ep.name for ep in eps))
PY
[[ $? -eq 1 ]] && die "fix the environment first — no flag in this script is reached until vLLM
       can resolve a platform. Build and install vllm-ascend into '${CONDA_DEFAULT_ENV:-this env}'
       (source the env file first: setup.py reads ASCEND_TOOLKIT_HOME / SOC_VERSION /
       COMPILE_CUSTOM_KERNELS at build time)."

# ASCEND_RT_VISIBLE_DEVICES is already exported, so this counts the MASKED set.
python3 -c "
import torch, torch_npu
n = torch.npu.device_count()
print('[plat ] visible NPUs after ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}: %d' % n)
if n != ${NCARDS}: print('[warn ] expected ${NCARDS} (from CARD=${CARD}) — check npu-smi info')
" 2>/dev/null || echo "[plat ] could not count NPUs (advisory only)"

# _stage_predicted_topk copies a [1, prefetch_topk] prediction into topk_ids_h[:1].
# Whether 1 < EXPERT_PREFETCH_NUM < TOPK is legal depends on whether that build
# narrows the DESTINATION to the prediction width. Rather than assert it from a code
# reading, inspect the INSTALLED module and run the real copy_.
if [[ "${OFFLOAD}" == "1" && "${PREFETCH}" == "1" && "${ALLOW_PREFETCH_NUM:-0}" != "1" ]] \
   && (( EXPERT_PREFETCH_NUM > 1 && EXPERT_PREFETCH_NUM < TOPK )); then
  python3 - "${TOPK}" "${EXPERT_PREFETCH_NUM}" <<'PY'
import sys, torch, inspect, ast, textwrap
topk, n = int(sys.argv[1]), int(sys.argv[2])
pf = max(1, min(topk, n))
narrowed, why, path = None, "", "?"
try:
    from vllm_ascend.expert_offload import expert_offload_manager as mod
    path = getattr(mod, "__file__", "?")
    tree = ast.parse(textwrap.dedent(
        inspect.getsource(mod.ExpertOffloadManager._stage_predicted_topk)))
    narrowed = False
    for node in ast.walk(tree):
        # (a) self.topk_ids_h[:num_tokens, :k]  — destination is a bare name later
        if isinstance(node, ast.Subscript):
            b = node.value
            if isinstance(b, ast.Attribute) and b.attr == "topk_ids_h" \
               and isinstance(node.slice, ast.Tuple) and len(node.slice.elts) >= 2:
                narrowed, why = True, "self.topk_ids_h[...] has %d subscript dims" % len(node.slice.elts)
        # (b) topk_ids_h[:, :k].copy_(...)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "copy_" and isinstance(node.func.value, ast.Subscript)):
            b = node.func.value.value
            if getattr(b, "id", None) == "topk_ids_h" or getattr(b, "attr", None) == "topk_ids_h":
                narrowed, why = True, "copy_ destination is subscripted"
    if not narrowed:
        why = "sliced on rows only; copy_ destination is the full [1, topk] row"
    print("[pfetch] %s\n[pfetch] destination narrowed to the prediction width: %s (%s)"
          % (path, narrowed, why))
except Exception as e:
    print("[pfetch] could not analyse the installed manager (%s) — not blocking." % e)
if narrowed is None:
    sys.exit(0)                       # unknown: only stop on positive evidence
dst_w = pf if narrowed else topk      # simulate at the EFFECTIVE width
try:
    torch.zeros(max(1, topk), topk, dtype=torch.int32)[:1, :dst_w].copy_(
        torch.zeros(1, pf, dtype=torch.int32))
    print("[pfetch] copy [1,%d] -> [1,%d] OK — expert_prefetch_num=%d is supported" % (pf, dst_w, n))
except RuntimeError as e:
    sys.stderr.write(
        "[!!] expert_prefetch_num=%d with topk=%d: the staging copy [1,%d] -> [1,%d] raises on\n"
        "     the INSTALLED build at %s\n     %s\n"
        "     _stage_predicted_topk runs on the MAIN thread, so this kills the engine on the\n"
        "     first learned-layer prefetch. Use EXPERT_PREFETCH_NUM=1 or >=%d, or ALLOW_PREFETCH_NUM=1.\n"
        % (n, topk, pf, dst_w, path, e, topk))
    sys.exit(1)
PY
  [[ $? -eq 1 ]] && die "prefetch staging shape rejected — see above. If you believe this config
       works, your vllm_ascend tree differs from the snapshot this script was written against."
fi

command -v ais_bench >/dev/null 2>&1 || die "ais_bench not on PATH"
[[ -n "${PKG_CFG}" && -d "${PKG_CFG}" ]] || die "cannot locate the ais_bench configs dir"
[[ -n "${STOCK_MODEL}" && -z "${STOCK_SRC}" ]] && die "STOCK_MODEL='${STOCK_MODEL}' not installed; try LIST=1 $0"

# A stale server on the port makes serve fail to bind — or worse, the health poll
# passes against an orphan and the run measures that instead.
_pids="$(port_pids)"
if [[ "${SKIP_SERVE}" == "1" ]]; then
  echo "[port ] :${PORT} pids=${_pids:-none}"
elif (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
  exec 3<&- 3>&- 2>/dev/null
  die "port ${PORT} already in use${_pids:+ by PID(s) ${_pids}}. Kill them, pick another PORT,
       or SKIP_SERVE=1 to deliberately evaluate the server already there."
else
  echo "[port ] :${PORT} is free"
fi

# AISBench aborts its infer task above 80% host RAM regardless of how much it needs,
# and it checks AFTER the model has loaded. Expert offload is a host-memory design,
# so the server itself pushes the box over. 78 here: our server adds more.
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

# The perf summarizer loads the tokenizer only AFTER every request is sent, so a bad
# path otherwise costs a whole pass. Check it the same way AISBench will.
if [[ "${RUN}" != acc ]]; then
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
_miss=0
echo "[data] root: ${DATA_ROOT}"
for t in ${TASK_LIST}; do
  preset "${t}"
  _f="$(find "${PKG_CFG}/datasets" -name "${P_DS}.py" -print -quit 2>/dev/null)"
  [[ -n "${_f}" ]] || { echo "[!!] dataset config '${P_DS}' (${t}) not installed" >&2; _miss=1; continue; }
  _rel="$(grep -oE "path='[^']*'" "${_f}" | head -1 | sed "s/^path='//;s/'$//")"
  [[ "${_rel}" == /* ]] && _abs="${_rel}" || _abs="${DATA_ROOT}/${_rel}"
  if [[ -d "${_abs}" ]]; then echo "[data] ${t}: ${_abs}"
  else
    echo "[!!] ${t}: no data at ${_abs}" >&2
    echo "     cd '${DATA_ROOT}/ais_bench/datasets' && wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/${t}.zip && unzip ${t}.zip" >&2
    _miss=1
  fi
done
(( _miss )) && { list_cfgs >&2; die "missing dataset config or data"; }

ensure_dirs                                        # again: the engine writes CSVs at startup

# =============================================================================
#  launch
# =============================================================================

if [[ "${SKIP_SERVE}" != "1" ]]; then
  trap 'reap; exit 130' INT TERM
  trap reap EXIT
  echo "[serve] launching — engine output streams below and into ${SERVE_LOG}"
  : > "${SERVE_LOG}"
  # setsid gives the server its own process group, so reap can signal the whole tree
  # without also signalling this script.
  if command -v setsid >/dev/null 2>&1; then setsid "${SERVE[@]}" >> "${SERVE_LOG}" 2>&1 &
  else "${SERVE[@]}" >> "${SERVE_LOG}" 2>&1 & fi
  SERVE_PID=$!
  SERVE_PGID="$(ps -o pgid= -p "${SERVE_PID}" 2>/dev/null | tr -d ' ')"
  [[ "${SERVE_PGID}" == "$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')" ]] && SERVE_PGID=""
  # Immediately, not after /health: weight loading takes minutes and this is the only
  # thing on screen during it.
  mirror_start
  echo "[serve] waiting up to ${WAIT}s for /health ..."
  ok=0
  for ((i=0; i<WAIT/2; i++)); do
    kill -0 "${SERVE_PID}" 2>/dev/null || die "serve exited during startup (see ${SERVE_LOG})"
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { ok=1; break; }
    (( i % 30 == 29 )) && echo "[serve] still waiting ($(( (i+1) * 2 ))s of ${WAIT}s) ..."
    sleep 2
  done
  (( ok )) || die "server not ready in ${WAIT}s (see ${SERVE_LOG})"
  echo "[serve] healthy"

  # The engine rewrites the capture list (cudagraph_dispatcher.get_capture_descs), so
  # print what it actually built rather than what we asked for.
  [[ "${GRAPH_ON}" == "1" ]] && \
    grep -o "cudagraph_capture_sizes': \[[^]]*\]" "${SERVE_LOG}" | sort -u | sed 's/^/  graph   : engine /'

  # TOP-LEVEL additional_config keys are read with .get(), so an unpatched engine — or
  # a typo — swallows them SILENTLY and the run measures something else. Both blocks
  # below turn that back into a failure, using the engine's own echoed config lines,
  # which appear during weight loading and so are already in serve.log by now.
  if [[ -n "${REMOE_GATE}" ]]; then
    grep -q '\[GATE-OVERRIDE\] applied' "${SERVE_LOG}" || guard "REMOE_GATE is set but the engine never
       logged '[GATE-OVERRIDE] applied' — this run would measure the BASE router."
    # More than one fingerprint means ranks read different files: under EP that is
    # inconsistent dispatch, with no error.
    grep -o 'sha256_16=.*$' "${SERVE_LOG}" | sort -u | sed 's/^/  gate    : /'
  fi
  if grep -q '\[DECODE-STATS\] config:' "${SERVE_LOG}"; then
    grep -o '\[DECODE-STATS\] \(config\|topology\):.*$' "${SERVE_LOG}" | sort -u | sed 's/^/  stats   : /'
  elif [[ "${DECODE_STATS}" == "1" ]]; then
    guard "the engine never logged '[DECODE-STATS] config:' — the statistics keys were swallowed,
       so this run will produce NO end-of-run summary and NO CSV. Apply the F5/F7 groups, or
       set DECODE_STATS=0 to acknowledge running without them."
  else
    echo "  stats   : DECODE_STATS=0 — no summary, no CSV; nothing to verify."
  fi
  [[ "${CSV}" == "1" ]] && ! grep -q 'decode_stats_csv.*[Tt]rue\|csv=True' "${SERVE_LOG}" && \
    echo "  warn    : CSV=1 but the engine's echoed config does not show csv enabled — check the key name."
else
  echo "[serve] SKIP_SERVE=1 — using whatever is on :${PORT}"
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || die "nothing healthy on :${PORT}"
fi

[[ "${PROBE}" == "1" ]] && for m in ${MODES}; do probe_mode "${m}"; done

# =============================================================================
#  run
# =============================================================================

rc=0
for step in "${PLAN[@]}"; do
  t="${step%:*}"; p="${step#*:}"
  build_cmd "${t}" "${p}"
  [[ "${p}" == acc ]] \
    && echo "[acc] ${t} [${P_PASS_MODE}] ${P_DS}   target: $(ref "${t}" "${P_PASS_MODE}")" \
    || echo "[perf] ${t} [${P_PASS_MODE}] ${P_DS}   ${PERF_PROMPTS:-full} prompts (latency, not a score)"
  printf '  $ '; printf '%q ' "${CMD[@]}"; echo
  # On a terminal, hand AISBench the real one: its progress table needs a TTY, and
  # piping drops it to a one-line fallback. run.log loses nothing — AISBench writes
  # its own per-task log and report_step prints the path. The table is also the
  # liveness signal, so the heartbeat only runs when there is no table.
  if [[ -t 3 ]]; then
    mirror_stop                                    # the table owns the terminal
    "${CMD[@]}" >&3 2>&3; src=$?
    mirror_start
  else
    if (( HEARTBEAT > 0 )); then
      ( t0=${SECONDS}
        while sleep "${HEARTBEAT}"; do
          el=$(( SECONDS - t0 ))
          n=$(curl -sf -m 5 "http://127.0.0.1:${PORT}/metrics" 2>/dev/null \
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
done

# =============================================================================
#  shut down, then report — in that order
# =============================================================================
# Stop the server HERE, not from the EXIT trap: the summary is emitted during its
# graceful shutdown, so that has to happen while the mirror is live and run.log is
# still being written in order; and the report below must read a complete serve.log.
# reap() is idempotent, so the EXIT trap remains a correct net for every other path.

if [[ "${SKIP_SERVE}" != "1" ]]; then
  reap
  if [[ "${DECODE_STATS}" == "1" ]]; then
    # The mirror wrote to the terminal only, so lift the summary into run.log.
    _f="$(find "${STATS_DIR}" -maxdepth 1 -name 'decode_stats_summary_*.txt' 2>/dev/null | sort | tail -1)"
    if [[ -n "${_f}" && -s "${_f}" ]]; then
      echo "────────────────────────────────────────────────────────────"
      sed 's/^/  /' "${_f}"
      echo "  [stats]  from ${_f}"
      find "${STATS_DIR}" -maxdepth 1 -name 'decode_stats_rank*.csv' 2>/dev/null | sed 's/^/  [stats]  per-layer CSV: /'
    elif grep -q '\[EXPERT-OFFLOAD-FINAL\]' "${SERVE_LOG}" 2>/dev/null; then
      echo "────────────────────────────────────────────────────────────"
      awk '/\[EXPERT-OFFLOAD-FINAL\]/{f=1} f{print} f&&/^=+$/{r++; if(r>=2) exit}' \
        "${SERVE_LOG}" | head -60 | sed 's/^/  /'
      echo "  [stats]  note: log only, no artifact in ${STATS_DIR} — check decode_stats_path."
    else
      echo "[stats]  WARNING: no summary in ${STATS_DIR} and no [EXPERT-OFFLOAD-FINAL] in ${SERVE_LOG}."
      echo "         Tracer counts localise the break:"
      for pat in 'config' 'topology' 'armed' 'first sample' 'flush' 'model runner shutdown'; do
        printf '         DECODE-STATS %-24s %s\n' "${pat}" \
          "$(grep -c "DECODE-STATS. ${pat}" "${SERVE_LOG}" 2>/dev/null || echo 0)"
      done
      echo "         config=0 -> keys swallowed (unpatched engine) | topology=0 -> _finalize_offload hook missing"
      echo "         armed=0  -> compile_or_warm_up_model hook missing | first sample=0 -> _update_weights never records"
      echo "         flush=0  -> ran for less than STATS_FLUSH_SECONDS | shutdown=0 -> EngineCore killed before shutdown()"
    fi
  fi
else
  echo "[stats]  SKIP_SERVE=1 — this script did not start or stop the server, so no end-of-run"
  echo "         summary is produced here. Stop that server with SIGTERM (not SIGKILL)."
fi

echo "────────────────────────────────────────────────────────────"
echo "[done] rc=${rc}   finished $(date '+%F %T')   everything under ${OUT_DIR}"
[[ "${CSV}" == "1" ]] && echo "[done] per-layer CSV under ${STATS_DIR}"
echo "────────────────────────────────────────────────────────────"
# stdout is a process substitution (tee -a run.log), which is asynchronous: the script
# can exit while tee still holds the last lines, and they never reach run.log.
sleep 1
exit "${rc}"

# =============================================================================
#  FINDINGS — things that changed a number, kept so they are not rediscovered
# =============================================================================
#
#  READ avg@n ONLY. AISBench de-interleaves the avg@n prediction list with the wrong
#  stride, pairing each question's sample with a DIFFERENT question's. avg@n is a flat
#  mean and survives; pass@n and cons@n come out as exactly 1-(1-p)^n and p^n.
#  Pass@1 IS avg@n — DeepSeek samples k at non-zero temperature and averages. AVG_N=4
#  to report; at n=1 the 198-row gpqa set carries ~3.2 points of binomial noise.
#
#  GPQA DATASET CONFIG. gpqa_gen_0_shot_cot_chat_prompt is the simple-evals protocol
#  behind the published 71.2/87.4/88.1. gpqa_gen_0_shot_str asks for "The correct
#  answer is (X)" with no CoT and measured 62.63 non-think greedy: ~2/3 of answers
#  open with the letter and rationalise after (answer-first ~55% correct vs
#  reason-first ~78%), and first_option_postprocess takes the FIRST regex match, so a
#  self-correction scores at its pre-correction letter. Keep EXTRACT_RATE=1 and read
#  the rate BEFORE the score — a failed extraction scores as wrong, so a format
#  regression is indistinguishable from a capability one.
#
#  AUTO CAPTURE SIZES MUST REACH decode_tokens, AND USED NOT TO. `auto` clamped its
#  ceiling to offload_threshold. With NUM_DEVICE_EXPERTS=24, TOPK=6 (threshold 4),
#  NUM_SPEC_TOKENS=2 gave a correct [1,2,3] but NUM_SPEC_TOKENS=4 gave [1,2,3,4] for a
#  decode shape of 5 — the real shape uncaptured, every decode step eager, the run
#  reporting graph mode throughout. `auto` now always emits 1..decode_tokens and warns
#  about unsafe entries; `auto-safe` is the old clamp, opt-in. The real fix in both
#  cases is to raise NUM_DEVICE_EXPERTS to decode_tokens*topk.
#
#  THE EXPERT-POOL FLOOR SCALES WITH NUM_SPEC_TOKENS. threshold = min(nde)//topk and
#  decode_tokens = max_num_seqs*(1+spec), so each increment of NUM_SPEC_TOKENS raises
#  the minimum pool by topk slots. At seqs=1, topk=6: spec=2 needs 18, spec=4 needs 30,
#  spec=7 needs 48. Below the floor decode leaves the offload branch entirely and the
#  offload half of the summary is empty — the run completes and measures nothing.
#
#  TOPK RESOLUTION MUST PRECEDE THE CAPTURE ARITHMETIC. _CFG_TOPK is read from the
#  checkpoint and overrides the env var, because the engine reads the config and not
#  this script. The capture block used to run BEFORE that override, so `auto` computed
#  its threshold from the guess — harmless when they agreed, silently wrong otherwise.
#
#  EXPERT_PREFETCH_NUM: THE CRASH RANGE IS STRICTLY BETWEEN 1 AND TOPK. The engine uses
#  prefetch_topk = max(1, min(topk, expert_prefetch_num)) and _stage_predicted_topk
#  copies [1, prefetch_topk] into topk_ids_h[:1] = [1, topk]. copy_ broadcasts only
#  from a SINGLETON dim, so 1 works (id replicated, consumer set()-reduces it, so the
#  behaviour is still "prefetch exactly one") and >= topk works (clamped, exact copy).
#  Strictly between raises on the first non-hash-layer prefetch. Preflight probes the
#  installed build rather than trusting this note.
#
#  BF16 HAS NO EXPERT OFFLOAD. AscendMoERunner drops enable_expert_offload when the
#  resolved quant_type is NONE, so the offload block in AscendUnquantizedFusedMoEMethod
#  .apply is unreachable. OFFLOAD=1 with QUANTIZATION="" is a silent no-op — prefetch,
#  substitution, cache stats, all of it.
#
#  W8A8 SINGLE-CARD group_list IS 1 UNLESS PATCHED. w8a8_dynamic.py's
#  build_fused_experts_input call omits num_local_experts, so moe_runtime_args.py
#  defaults it to len(w1) — correct only on the dynamic_eplb path where w1 is a list of
#  per-expert tensors, and 1 on the normal path where w1 is [layer.w13_weight], one
#  batched [E,...] tensor. The AllGather dispatcher (EP=1, expert_map None) then builds
#  a length-1 group_list against an E-row scale and aclnnGroupedMatmulWeightNz raises
#  EZ1001 / 161002 "scale batch size[E] should be equal with groupList length[1]".
#  w4a8.py passes num_local_experts=layer.moe_config.num_local_experts explicitly,
#  which is why W4A8 is unaffected; EP>1 hides it (that branch reads
#  self.num_experts_local). One-line fix, mirroring w4a8.py.
#
#  SUBSTITUTION IS NATIVE, AND IT CHANGES ROUTING. plan/commit_expert_substitutions
#  live in experts_selector.py upstream. Accuracy with it on is NOT comparable to
#  accuracy with it off — run it as its own arm.
#
#  DRAFT MoE LAYERS DO NOT OFFLOAD. AscendMoERunner skips any layer with an "mtp" path
#  component, covering both the serial drafter and DSpark. MTP changes the decode token
#  count and nothing else about the offload path.
#
#  THE OFFLOAD THRESHOLD IS ABOUT TOKENS, NOT SEQUENCES. min(num_device_experts)//topk
#  compared against max_num_seqs*(1+num_speculative_tokens). The SOURCE script compared
#  it against MAX_NUM_SEQS, wrong by (1+N) once MTP is on. min() matters because
#  num_device_experts may be a per-layer list.
#
#  DO NOT READ THE OLD INTERVAL LINES AS THE RESULT. [EXPERT-OFFLOAD-CACHE] fires every
#  cache_stats_log_interval CALLS (per layer, not per request) and keeps its defects:
#  counters exist only when cache_policy_enabled, and _record_cache_stats runs on the
#  PREFETCH pass too, so with PREFETCH=1 that hit rate blends actual with predicted
#  routing. [EXPERT-OFFLOAD-FINAL] separates the two passes and is collected either
#  way. When they disagree, trust the summary.
#
#  THE SUMMARY IS A SHUTDOWN ARTEFACT, and there is no window hierarchy. It comes from
#  NPUModelRunner.shutdown() on SIGTERM, plus an atexit backstop; SIGKILL produces
#  nothing. Aggregation is two levels: per-(layer, step) reduced across NON-HASH MoE
#  layers at step close, then avg/median/min/max over steps — so min/max mean worst and
#  best STEP. Per-layer detail goes to the CSV only, never the terminal.
#
#  MULTI-CARD REPORTS NOTHING and IS UNGUARDED. The per-layer accounting lives in the
#  single-card host callback; update_weights_multi_card writes no counters. And
#  enable_multi_card must be set AND EP>1 — TARGET checks neither direction, and EP>1
#  without the flag silently gives every rank the same experts. When on, MC2 admission
#  (decode_tokens*topk <= per-rank slots) decides decode vs prefill instead of the token
#  threshold, and shard_per_rank (default true) pins each expert to one rank.
#
#  GRAPH MODE NEEDS THREE GATES: cudagraph_mode != NONE, compilation mode VLLM_COMPILE,
#  and not enforce_eager. --enforce-eager silently disables everything under
#  --compilation-config.
#
#  cudagraph_capture_sizes IS IN TOKENS, AND SUB-FULL SIZES ARE REAL. With spec decode a
#  request can be scheduled short (no drafts on the first decode step after prefill), and
#  _dummy_run's uniform_decode branch builds those shapes. What you pass is only an
#  INPUT: the effective list comes from cudagraph_dispatcher.get_capture_descs() and also
#  feeds set_draft_graph_params(). The post-health grep prints the engine's own list.
#
#  CAPTURED SIZES ABOVE THE THRESHOLD ARE NOT SAFE. _prefill_load_layer is called from
#  the outer Python of update_weights, so it runs once at capture and never on replay: a
#  captured size above offload_threshold replays against whatever weights were left in
#  the pool, and records no prefetch nodes. capture_model() does not set _skip_prefill,
#  so this is reachable. `auto` warns rather than truncating — see above for why.
#
#  EAGER TPOT IS NOT A BASELINE FOR GRAPH TPOT. In eager the prefetch callback runs
#  inline and ends in load_stream.synchronize(), so PREFETCH=1 is SLOWER than
#  PREFETCH=0. In replay it is a host-func node on the prefetch stream and does overlap.
#  Use eager for accuracy parity — scores must match graph mode exactly — never latency.
#
#  DYNAMIC_EPLB AND EXPERT OFFLOAD ARE MUTUALLY DESTRUCTIVE. Under dynamic_eplb,
#  process_weights_after_loading deletes w13_weight / w2_weight / the scales / the scale
#  biases — the exact attributes _load_expert_weights_into_slot writes into — replacing
#  them with per-expert _list clones and skipping pack_to_int32, so the byte layout
#  diverges too. Same story for FUSED_MC2=1 on W8A8: fused_w1_scale / fused_w2_scale are
#  derived once at load and never refreshed by paging. Both are asserted above.
#
#  MTP + W4A8 DID NOT LOAD ON THE PREVIOUS SNAPSHOT: KeyError
#  'model.layers.0.mtp_block.mlp.experts.w13_weight_scale_bias' in
#  deepseek_v4_mtp.py::load_weights. w4a8.py::get_weight creates the scale biases only
#  when quant_description["version"]=="1.0.0", so the draft model's experts lacked a
#  parameter the checkpoint supplies. A LOAD-time issue, not an offload one. Re-verify;
#  MTP with W8A8/bf16 is the isolation test.
#
#  ENGINE-CORE ORPHANS. `vllm serve` forks workers holding NPU and host memory until
#  they exit, and a name-matching reaper missed them (the names changed), so they
#  accumulated until AISBench refused to start on host RAM. reap() signals the process
#  group and then sweeps the port.
#
#  --num-prompts IS PER SUB-DATASET. mmlu_pro fans out to 14 categories, so 20 gives
#  ~280 docs. It applies to the perf pass too.
#
#  MODES ARE PER-REQUEST. chat_template_kwargs rides on each request, which is what lets
#  acc and perf run different modes against one server. A kwarg the template does not
#  declare is dropped silently — hence probe_mode().
#
#  STOCK CONFIGS HARDCODE THEIR ADDRESS (localhost:8080) and carry no
#  chat_template_kwargs, so they dial a dead port after the model has loaded unless
#  retargeted, and even then run in whatever mode the template defaults to.
# =============================================================================