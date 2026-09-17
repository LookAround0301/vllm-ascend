#!/usr/bin/env bash
# =============================================================================
#  v4_target_aisbench.sh — serve DeepSeek-V4-Flash, measure ACCURACY / TPOT on
#  gsm8k / gpqa_diamond / mmlu_pro via AISBench, tear the server down cleanly.
#
#    source env_a5_0260.sh                        # REQUIRED, in this same shell
#    CARD=2 PORT=1161 TASKS=gpqa RUN=perf MTP=1 ./v4_target_aisbench.sh
#    ... EXPERT_PREFETCH_NUM=2 EXPERT_PREFETCH_WAIT_TIMING=1 ./...
#    ... EXPERT_PREDICTOR=mode2_har EXPERT_PREDICTOR_CKPT=/path/har_...pt ./...
#    CARD=12,13 DP=2 ENABLE_MULTI_CARD=1 PORT=7081 ./...
#    DRY_RUN=1 ...                                # print the plan, launch nothing
#
#  Every knob meant to be edited is above "DO NOT EDIT BELOW"; under it is frozen
#  config and machinery. Read FINDINGS at the bottom once — those are the things
#  that silently changed a number.
#
#  OUT_DIR/: run.log · serve.log · acc/ perf/ · configs/ · stats/. The stats
#  artefacts are written DURING the run by the collector's watchdog thread;
#  [EXPERT-OFFLOAD-FINAL] in serve.log is best effort on top of that.
# =============================================================================
set -uo pipefail

# ═════════════════════════════════ EDIT HERE ═════════════════════════════════

# ── what to serve ────────────────────────────────────────────────────────────
MODEL="${MODEL:-/home/yangliu/llms/DeepSeek-V4-Flash-DSpark-w4a8-test}"
SERVED_NAME="${SERVED_NAME:-A3-deepseekv4-flash-w4a8-dspark}"
TOKENIZER="${TOKENIZER:-/home/yangliu/keyi/code/benchmarks/dsv4-tokenizer}"
DATASETS_CACHE="${DATASETS_CACHE:-/home/yangliu/keyi/code/benchmarks}"   # AISBench data root

# ── context / generation length ──────────────────────────────────────────────
# These two interact, so they live together. MAX_MODEL_LEN is the served context
# (empty = whatever the checkpoint declares). MAX_OUTPUT_LEN overrides max_out in
# the presets for EVERY task (empty = each task's own preset value). Whichever
# wins is capped to CTX - PROMPT_RESERVE so a request cannot ask for more tokens
# than the context can hold — vLLM rejects that per request, mid-pass, long after
# the model has loaded.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_OUTPUT_LEN="${MAX_OUTPUT_LEN:-}"

# ── placement ────────────────────────────────────────────────────────────────
CARD="${CARD:-6}"                        # comma list; its length must equal DP
PORT="${PORT:-7001}"
DP="${DP:-1}"                            # EP spans DP (TP and PP are pinned to 1)

# ── speculative decoding ─────────────────────────────────────────────────────
# 0 off | 1 mtp (serial drafter) | 2 dspark. Draft MoE layers never offload, so
# this changes only the decode TOKEN COUNT — and thus whether the offload
# threshold is still cleared. Depth defaults per drafter; override it to sweep.
MTP="${MTP:-0}"
if [[ "${MTP}" == "2" ]]; then NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-5}"
else                           NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-2}"; fi

# ── MoE expert offload ───────────────────────────────────────────────────────
OFFLOAD="${OFFLOAD:-1}"
NUM_DEVICE_EXPERTS="${NUM_DEVICE_EXPERTS:-36}"    # int, or a JSON list "[60,60,...]"
NUM_DEVICE_LAYERS="${NUM_DEVICE_LAYERS:-1}"
ENABLE_MULTI_CARD="${ENABLE_MULTI_CARD:-0}"       # must move together with DP > 1
EXPERT_SUBSTITUTION="${EXPERT_SUBSTITUTION:-0}"   # CHANGES ROUTING — see FINDINGS
EXPERT_SUBSTITUTION_THRESHOLD="${EXPERT_SUBSTITUTION_THRESHOLD:-}"  # empty = engine default
REMOE_GATE="${REMOE_GATE:-}"             # fine-tuned router gates, swapped in at load

# ── prefetch ─────────────────────────────────────────────────────────────────
# The two budgets are ORTHOGONAL: _TOKENS is how many token rows the predictor
# runs on (how many candidates exist), _NUM caps how many are TRANSFERRED.
PREFETCH="${PREFETCH:-1}"
EXPERT_PREFETCH_NUM="${EXPERT_PREFETCH_NUM:-2}"
EXPERT_PREFETCH_TOKENS="${EXPERT_PREFETCH_TOKENS:-3}"
# fate | mode2_har | mode2_prevhfr. fate is the engine's built-in hash/gate
# predictor and always keeps the targets a trained head does not own.
EXPERT_PREDICTOR="${EXPERT_PREDICTOR:-fate}"
EXPERT_PREDICTOR_CKPT="${EXPERT_PREDICTOR_CKPT:-}"

# ── measurement ──────────────────────────────────────────────────────────────
# The per-step decode and cache statistics: a .txt summary plus a .csv under
# OUT_DIR/stats, written DURING the run by the collector's watchdog thread.
# DECODE_STATS=0 sends decode_stats_enabled:false — the run then produces no
# summary and no CSV, which is only what you want for a clean TPOT baseline.
DECODE_STATS="${DECODE_STATS:-1}"
CSV="${CSV:-1}"                          # also emit the per-step .csv
# Milliseconds the compute stream is BLOCKED at the prefetch join — the only
# number that measures what prefetch COSTS. Adds two device events per covered
# layer per step, so a run with this on is NOT a TPOT baseline. The pf_wait rows
# land in the decode-stats artefacts, so it needs DECODE_STATS=1.
EXPERT_PREFETCH_WAIT_TIMING="${EXPERT_PREFETCH_WAIT_TIMING:-0}"

# ── graph mode ───────────────────────────────────────────────────────────────
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"      # 1 = no capture. Accuracy parity only,
                                         # NEVER a latency baseline (see FINDINGS)

# ── client ───────────────────────────────────────────────────────────────────
TASKS="${TASKS:-gsm8k gpqa mmlu_pro}"    # space or comma separated
RUN="${RUN:-both}"                       # acc | perf | both
THINK="${THINK:-0}"                      # 0 non-think, 1 thinking
THINK_EFFORT="${THINK_EFFORT:-high}"     # high | max, only read when THINK=1
NUM_PROMPTS="${NUM_PROMPTS:-}"           # acc pass; empty = preset, 0 = full dataset
PERF_PROMPTS="${PERF_PROMPTS:-30}"
AVG_N="${AVG_N:-}"                       # repeats per question; 4-5 to report

# ── run control ──────────────────────────────────────────────────────────────
OUT_DIR="${OUT_DIR:-$PWD/aisbench_results/$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_SERVE="${SKIP_SERVE:-0}"            # reuse whatever already listens on PORT
PROBE="${PROBE:-1}"                      # verify the served reasoning mode
ALLOW_UNSAFE="${ALLOW_UNSAFE:-0}"        # every guard below becomes a warning

# ── task presets: task | dataset | temp | top_p | max_out | acc_prompts | avg_n | published
PRESETS="
gsm8k    | gsm8k_gen_0_shot_cot_chat_prompt | 0.0 |      | 32768 |    | 1 | 90.8 (base 8-shot EM)
gpqa     | gpqa_gen_0_shot_cot_chat_prompt  | 0.6 | 0.95 | 65536 |    | 1 | 71.2 (GPQA-D Pass@1)
mmlu_pro | mmlu_pro_gen_0_shot_str          | 0.6 | 0.95 | 65536 | 20 | 1 | 83.0 (MMLU-Pro EM)
"

# ══════════════════════════ DO NOT EDIT BELOW THIS LINE ══════════════════════

# FIXED — never varied across this campaign. Change one and no previous number is
# comparable, which is why they live here and not in the environment.
TP=1; PP=1; SEED=1024; GPU_MEM_UTIL=0.97; MAX_NUM_SEQS=1
PROMPT_RESERVE=4096                      # context held back for the prompt when
                                         # the max_out cap is computed below
MAX_NUM_BATCHED_TOKENS=8192; QUANTIZATION=ascend; API_SERVER_COUNT=1
CUDAGRAPH_MODE=FULL_DECODE_ONLY          # capture list is auto: 1..decode_tokens
PERF_TEMP=0.0                            # pinned so TPOT stays comparable
STATS_FLUSH_EVERY=100; STATS_FLUSH_SECONDS=30
STATS_QUIESCE=8                          # idle seconds before stopping the server,
                                         # so the collector's quiet snapshot lands
WAIT=1000                                # seconds to wait for /health
HEALTH_GRACE=120                         # grace once the engine LOGS it is serving
ENGINE_GRACE=45                          # EngineCore alone before the group
REAP_WAIT=180                            # group grace before SIGKILL
HEARTBEAT=300                            # liveness line when there is no TTY

die()   { echo "ERROR: $*" >&2; exit 1; }
guard() { [[ "${ALLOW_UNSAFE}" == "1" ]] && echo "  warn    : [unsafe] $*" || die "$*"; }
cfg()   { python3 -c "
import json,sys
c=json.load(open('${MODEL}/config.json')); t=c.get('text_config') or {}
print(c.get(sys.argv[1], t.get(sys.argv[1], '')))" "$1" 2>/dev/null; }

export ASCEND_RT_VISIBLE_DEVICES="${CARD}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"; export NO_PROXY="${no_proxy}"
export AIS_BENCH_DATASETS_CACHE="${DATASETS_CACHE}"
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-0}"   # env_a5_0260.sh owns this

RUN_LOG="${OUT_DIR}/run.log"; SERVE_LOG="${OUT_DIR}/serve.log"
CFG_ROOT="${OUT_DIR}/configs"; CFG_DIR="${CFG_ROOT}/models"; STATS_DIR="${OUT_DIR}/stats"
SERVE_PID=""; SERVE_PGID=""; SERVE_SID=""; HB_PID=""; TAIL_PID=""; MIRROR_AT=0
STATS_REPORTED=0

# ── derived values and guards ─────────────────────────────────────────────────

case "${MTP}" in
  0) SPEC_METHOD="" ;;
  1) SPEC_METHOD="mtp" ;;
  2) SPEC_METHOD="dspark" ;;
  *) die "MTP must be 0 (off), 1 (mtp) or 2 (dspark)" ;;
esac
[[ "${NUM_SPEC_TOKENS}" =~ ^[1-9][0-9]*$ ]] || die "NUM_SPEC_TOKENS must be a positive integer"
[[ "${RUN}" =~ ^(acc|perf|both)$ ]]         || die "RUN must be acc, perf or both"
[[ "${THINK}" =~ ^[01]$ ]]                  || die "THINK must be 0 or 1"
[[ "${THINK_EFFORT}" =~ ^(high|max)$ ]]     || die "THINK_EFFORT must be high or max"
[[ "${EXPERT_PREDICTOR}" =~ ^(fate|mode2_har|mode2_prevhfr)$ ]] \
  || die "EXPERT_PREDICTOR must be fate, mode2_har or mode2_prevhfr"
[[ "${EXPERT_PREFETCH_WAIT_TIMING}" =~ ^[01]$ ]] || die "EXPERT_PREFETCH_WAIT_TIMING must be 0 or 1"
[[ "${DECODE_STATS}" =~ ^[01]$ ]] || die "DECODE_STATS must be 0 or 1"
[[ "${CSV}" =~ ^[01]$ ]]          || die "CSV must be 0 or 1"
[[ -z "${MAX_MODEL_LEN}"  || "${MAX_MODEL_LEN}"  =~ ^[1-9][0-9]*$ ]] || die "MAX_MODEL_LEN must be empty or a positive integer"
[[ -z "${MAX_OUTPUT_LEN}" || "${MAX_OUTPUT_LEN}" =~ ^[1-9][0-9]*$ ]] || die "MAX_OUTPUT_LEN must be empty or a positive integer"
[[ -z "${REMOE_GATE}" || -e "${REMOE_GATE}" ]]   || die "REMOE_GATE not found: ${REMOE_GATE}"

MODE=non-think; [[ "${THINK}" == "1" ]] && MODE="${THINK_EFFORT}"
DECODE_TOK_PER_REQ=1
[[ -n "${SPEC_METHOD}" ]] && DECODE_TOK_PER_REQ=$(( 1 + NUM_SPEC_TOKENS ))
DECODE_TOKENS=$(( MAX_NUM_SEQS * DECODE_TOK_PER_REQ ))   # per rank / per capture
GRAPH_ON=1; [[ "${ENFORCE_EAGER}" == "1" ]] && GRAPH_ON=0
EP_SIZE=$(( TP * DP ))
NCARDS="$(awk -F',' '{print NF}' <<<"${CARD}")"
[[ "${NCARDS}" == "${EP_SIZE}" ]] || die "CARD lists ${NCARDS} device(s) but TP*DP=${EP_SIZE}"
(( EP_SIZE > 1 )) && [[ -n "${VA_OMP_MULTI:-}" ]] && export OMP_NUM_THREADS="${VA_OMP_MULTI}"

# The MoE layer sees DP x decode_tokens rows: prepare() all-gathers across the DP
# group, and THAT is what the engine compares against offload_threshold.
MOE_ROWS=$(( DECODE_TOKENS * DP ))

TOPK="$(cfg num_experts_per_tok)"
[[ "${TOPK}" =~ ^[1-9][0-9]*$ ]] || die "cannot read num_experts_per_tok from ${MODEL}/config.json"
NDE_MIN="${NUM_DEVICE_EXPERTS}"
[[ "${NUM_DEVICE_EXPERTS}" == \[* ]] && \
  NDE_MIN="$(tr -d '[] ' <<<"${NUM_DEVICE_EXPERTS}" | tr ',' '\n' | sort -n | head -1)"
[[ "${NDE_MIN}" =~ ^[1-9][0-9]*$ ]] || die "NUM_DEVICE_EXPERTS must be a positive int or a JSON list"
THR=$(( NDE_MIN / TOPK ))            # offload_threshold, in MoE token rows
NDE_FLOOR=$(( MOE_ROWS * TOPK ))     # smallest pool that keeps decode on the offload path

# The context this run actually serves, and the largest max_out that fits in it.
# CTX empty means the checkpoint does not declare max_position_embeddings and
# MAX_MODEL_LEN was not set: no cap is computed and the engine stays the only
# authority. An explicit MAX_OUTPUT_LEN at or above the context is a hard error —
# it can never be satisfied — while a preset that overruns is simply clamped.
CTX="${MAX_MODEL_LEN:-$(cfg max_position_embeddings)}"
[[ "${CTX}" =~ ^[1-9][0-9]*$ ]] || CTX=""
OUT_CAP=""
if [[ -n "${CTX}" ]]; then
  [[ -n "${MAX_OUTPUT_LEN}" ]] && (( MAX_OUTPUT_LEN >= CTX )) && die "MAX_OUTPUT_LEN=${MAX_OUTPUT_LEN}
       must be smaller than the served context (${CTX}${MAX_MODEL_LEN:+, from MAX_MODEL_LEN}): a
       request whose prompt + max_tokens exceeds max_model_len is rejected per request, mid-pass."
  OUT_CAP=$(( CTX - PROMPT_RESERVE ))
  (( OUT_CAP < 1 )) && OUT_CAP=$(( CTX / 2 ))   # tiny context: leave half for the prompt
fi

# A trained predictor and the stall timer both ride the prefetch pipeline; with
# OFFLOAD=0 the whole config block is never emitted and the run measures nothing.
if [[ "${EXPERT_PREDICTOR}" != fate ]]; then
  [[ "${OFFLOAD}" == "1" && "${PREFETCH}" == "1" ]] || die "EXPERT_PREDICTOR needs OFFLOAD=1 and PREFETCH=1"
  [[ -r "${EXPERT_PREDICTOR_CKPT}" ]] || die "EXPERT_PREDICTOR_CKPT unreadable: '${EXPERT_PREDICTOR_CKPT}'"
fi
[[ "${EXPERT_PREFETCH_WAIT_TIMING}" == "0" || ( "${OFFLOAD}" == "1" && "${PREFETCH}" == "1" ) ]] \
  || die "EXPERT_PREFETCH_WAIT_TIMING=1 needs OFFLOAD=1 and PREFETCH=1: no prefetch, no join to time"
[[ "${EXPERT_PREFETCH_WAIT_TIMING}" == "0" || "${DECODE_STATS}" == "1" ]] \
  || die "EXPERT_PREFETCH_WAIT_TIMING=1 needs DECODE_STATS=1: pf_wait is reported in the decode-stats
       summary and CSV, so with statistics off the run pays for the events and records nothing."
# A gate fine-tuned on the QuaRot build is silently wrong on any other basis: same
# shapes, same norms, every engine check passes, and the model just stops emitting
# EOS. Only the w4a8 build declares optional.quarot, so that file is the tell.
if [[ -n "${REMOE_GATE}" ]]; then
  MODEL_ROT="$(python3 - "${MODEL}/quant_model_description.json" 2>/dev/null <<'PY'
import json, sys
try: d = json.load(open(sys.argv[1]))
except Exception: print("absent"); sys.exit(0)
o = d.get("optional")
print(",".join(sorted(map(str, o))) if isinstance(o, dict) and o else "none")
PY
)"
  [[ "${MODEL_ROT}" == quarot ]] || guard "REMOE_GATE is set but ${MODEL} declares
       rotation='${MODEL_ROT:-absent}', not 'quarot'. A router gate is only meaningful in the basis it
       was trained in; on the wrong build there is NO crash and NO error — the model stops emitting
       EOS and every prompt runs to max_out_len, which reads as a hung benchmark. Serve the gate on
       the build it was fine-tuned from, or unset REMOE_GATE."
fi

# Capture 1..decode_tokens. An empty list silently falls back to vLLM's defaults.
CAPTURE_SIZES=""
if (( GRAPH_ON )); then
  _cap="${DECODE_TOKENS}"; (( _cap > MAX_NUM_BATCHED_TOKENS )) && _cap="${MAX_NUM_BATCHED_TOKENS}"
  for (( _s=1; _s<=_cap; _s++ )); do CAPTURE_SIZES+="${_s},"; done
  CAPTURE_SIZES="${CAPTURE_SIZES%,}"
fi

TASK_LIST="$(tr ',' ' ' <<<"${TASKS}")"
PKG_CFG="$(python3 -c 'import os,ais_bench.benchmark as b;print(os.path.join(os.path.dirname(b.__file__),"configs"))' 2>/dev/null)"

# ── helpers ───────────────────────────────────────────────────────────────────

# Sets P_DS P_TEMP P_TOPP P_MAXOUT P_NUM P_AVGN P_REF (and P_CLAMP) for one task.
preset() {
  local line
  line=$(awk -F'|' -v t="$1" '
    { for (i=1;i<=NF;i++) gsub(/^[ \t]+|[ \t]+$/, "", $i) }
    $1 == t { print $2"|"$3"|"$4"|"$5"|"$6"|"$7"|"$8; f=1 } END { exit !f }' <<<"${PRESETS}") \
    || die "unknown task '$1' — valid: $(awk -F'|' 'NF>1{gsub(/ /,"",$1);printf "%s ",$1}' <<<"${PRESETS}")"
  IFS='|' read -r P_DS P_TEMP P_TOPP P_MAXOUT P_NUM P_AVGN P_REF <<<"${line}"
  [[ -n "${NUM_PROMPTS}" ]] && { P_NUM="${NUM_PROMPTS}"; [[ "${P_NUM}" == 0 ]] && P_NUM=""; }
  [[ -n "${AVG_N}" ]] && P_AVGN="${AVG_N}"
  [[ -n "${MAX_OUTPUT_LEN}" ]] && P_MAXOUT="${MAX_OUTPUT_LEN}"
  P_CLAMP=""
  [[ -n "${OUT_CAP}" ]] && (( P_MAXOUT > OUT_CAP )) && \
    { P_CLAMP=" (clamped from ${P_MAXOUT} to fit ctx=${CTX})"; P_MAXOUT="${OUT_CAP}"; }
  return 0
}

# --config-dir REPLACES ais_bench's config root, so without these links every
# pass dies at config resolution before one request is sent. SYMLINKS, not copies
# (read_base() only resolves at the real path); models/ stays ours.
ensure_dirs() {
  local d src base dirs
  dirs=( "${OUT_DIR}" "${CFG_DIR}" "${OUT_DIR}/acc" "${OUT_DIR}/perf" )
  [[ "${DECODE_STATS}" == "1" ]] && dirs+=( "${STATS_DIR}" )
  for d in "${dirs[@]}"; do
    mkdir -p "${d}" 2>/dev/null || die "cannot create directory: ${d}"
    [[ -w "${d}" ]] || die "directory not writable: ${d}"
  done
  for src in "${PKG_CFG}"/*; do
    [[ -d "${src}" ]] || continue
    base="$(basename "${src}")"
    [[ "${base}" == models || "${base}" == __pycache__ ]] && continue
    ln -sfn "${src}" "${CFG_ROOT}/${base}" 2>/dev/null \
      || die "cannot link ${src} -> ${CFG_ROOT}/${base} (ais_bench would not find its datasets)"
  done
  for d in datasets summarizers; do
    [[ -d "${CFG_ROOT}/${d}" ]] || die "${CFG_ROOT}/${d} is not reachable — see ensure_dirs()"
  done
  return 0
}

# NOT curl — it dies in the loader here (see FINDINGS). python3 is the conda
# interpreter, so its libraries match by construction. Proxies off explicitly.
#   http_req <timeout> <url> [body] -> 0 ok | 7 unreachable | 22 http>=400 | 28 timeout
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

# ── process bookkeeping (see FINDINGS: ^C must not abandon the cleanup) ───────
# Each source is newline-TERMINATED: lsof -t can omit the trailing newline and
# fuse two pids into one that matches nothing.
port_pids() {
  { lsof -t -i "TCP:${PORT}" -sTCP:LISTEN 2>/dev/null; echo
    fuser -n tcp "${PORT}" 2>/dev/null | tr -s ' ' '\n'; echo
    ss -lptnH "sport = :${PORT}" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2; echo
  } | grep -E '^[0-9]+$' | sort -u | tr '\n' ' '
}
kill_tree() { local k; for k in $(pgrep -P "$1" 2>/dev/null); do kill_tree "${k}" "$2"; done
              kill "-$2" "$1" 2>/dev/null; return 0; }
proc_tree() { local kid; echo "$1"; for kid in $(pgrep -P "$1" 2>/dev/null); do proc_tree "${kid}"; done; }
# pgrep -P cannot see a worker re-parented to init, but the session id survives.
# Armed only when the server is CONFIRMED to be in its own session, so this can
# never name the script. Zombies are skipped: dead already, impossible to signal.
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
# reap() SIGTERMs whatever this returns, so it must never name a process this
# script did not start. NO host-wide fallback — see FINDINGS.
engine_pids() {
  [[ -z "${SERVE_PID}" ]] && { echo ""; return 0; }
  local pid out=""
  for pid in $(proc_tree "${SERVE_PID}"); do
    [[ "${pid}" == "${SERVE_PID}" ]] && continue
    ps -o args= -p "${pid}" 2>/dev/null | grep -q 'EngineCore' && out+="${pid} "
  done
  echo "${out}"
}

mirror_start() { tail -n "+$(( MIRROR_AT + 1 ))" -f "${SERVE_LOG}" >&3 2>/dev/null & TAIL_PID=$!; return 0; }
mirror_stop()  { [[ -n "${TAIL_PID}" ]] && kill "${TAIL_PID}" 2>/dev/null; TAIL_PID=""
                 MIRROR_AT=$(wc -l < "${SERVE_LOG}" 2>/dev/null || echo 0); return 0; }

# 1. TERM the EngineCore ALONE (it writes [EXPERT-OFFLOAD-FINAL]) and wait
#    ENGINE_GRACE; signalling the group at once loses that flush.
# 2. TERM the group and the tree, wait REAP_WAIT.
# 3. KILL group, tree, session, then anything still holding the port.
# Idempotent and NOT interruptible — see FINDINGS before touching any of it.
reap() {
  trap '' INT TERM HUP QUIT
  [[ -n "${HB_PID}" ]] && kill "${HB_PID}" 2>/dev/null; HB_PID=""
  [[ -z "${SERVE_PID}${SERVE_PGID}${SERVE_SID}" ]] && { mirror_stop; return 0; }
  local i epids left
  epids="$(engine_pids)"
  if [[ -n "${epids// /}" ]]; then
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

# Called from the happy path AND the traps, so an interrupted run still surfaces
# its artefacts. Idempotent.
report_stats() {
  (( STATS_REPORTED )) && return 0
  STATS_REPORTED=1
  [[ "${DECODE_STATS}" == "1" ]] || return 0
  echo "────────────────────────────────────────────────────────────"
  local pat _n _f
  for pat in config topology armed 'first sample' flush 'callback failed'; do
    _n="$(grep -c "DECODE-STATS. ${pat}\|EXPERT-OFFLOAD. .*${pat}" "${SERVE_LOG}" 2>/dev/null)"
    printf '[stats]  trace %-18s %s\n' "${pat}" "${_n:-0}"
  done
  # Newest by MTIME: the filename is rank<R>_<timestamp>, so a lexical sort picks
  # the highest RANK, not the latest file.
  _f="$(find "${STATS_DIR}" -maxdepth 1 -name 'decode_stats_summary_*.txt' -printf '%T@ %p\n' 2>/dev/null \
        | sort -n | tail -1 | cut -d' ' -f2-)"
  if [[ -n "${_f}" && -s "${_f}" ]]; then
    echo "[stats]  artefact: ${_f}"
    grep -q 'decode steps=0 ' "${_f}" && echo "[!!]     decode steps=0 — EMPTY: armed but never reached
         the paging decode path. Not a result."
    sed 's/^/  /' "${_f}"
  else
    echo "[stats]  WARNING: no summary under ${STATS_DIR}. Read the traces above: config=0 keys
         swallowed | topology=0 offload never registered | armed=0 warmup hook missing |
         first sample=0 decode never paged."
  fi
  grep -q '\[EXPERT-OFFLOAD-FINAL\]' "${SERVE_LOG}" 2>/dev/null && {
    echo "[stats]  [EXPERT-OFFLOAD-FINAL] from serve.log:"
    awk '/\[EXPERT-OFFLOAD-FINAL\]/{f=1} f{print} f&&/^=+$/{r++; if(r>=2) exit}' \
      "${SERVE_LOG}" | head -60 | sed 's/^/  /'; }
  grep -q 'host callback failed' "${SERVE_LOG}" 2>/dev/null && \
    echo "[stats]  WARNING: expert-offload host callback failures — paging did not complete"
  # The stall timer latches off rather than raising when the device cannot
  # timestamp inside a captured graph; without this the pf_wait rows would just be
  # absent and look like a key that never took.
  [[ "${EXPERT_PREFETCH_WAIT_TIMING}" == "1" ]] && \
    grep -q 'elapsed_time unavailable' "${SERVE_LOG}" 2>/dev/null && \
    echo "[stats]  WARNING: prefetch-stall timing was DISABLED at runtime — pf_wait rows are empty."
  return 0
}

on_signal() {
  trap '' PIPE           # a dead tee must not kill this shell before reap() runs
  echo >&2; echo "[signal] caught SIG${1} — stopping the server. Further ^C is IGNORED until" \
                 "the tree is down." >&2
  # A ^C is not an orderly flush. Clamped, not zeroed: 20s still lets the
  # EngineCore write its summary.
  (( ENGINE_GRACE > 20 )) && ENGINE_GRACE=20
  (( REAP_WAIT   > 45 )) && REAP_WAIT=45
  reap; report_stats; exit 130
}
on_exit() { reap; report_stats; }

# Sets P_*, MODEL_TASK and CMD, and writes the AISBench model config.
build_cmd() {   # $1=task  $2=acc|perf
  local task="$1" pass="$2" gk n
  preset "${task}"
  MODEL_TASK="v4_${task}_${pass}"

  if [[ "${DRY_RUN}" != "1" ]]; then
    if [[ "${pass}" == perf ]]; then
      gk="            temperature=${PERF_TEMP},"
    else
      gk="            temperature=${P_TEMP},"
      [[ -n "${P_TOPP}" ]] && gk+=$'\n'"            top_p=${P_TOPP},"
    fi
    # Always stated explicitly, including thinking=False, so a run can never
    # silently sit in the wrong mode.
    if [[ "${MODE}" == non-think ]]; then
      gk+=$'\n'"            chat_template_kwargs=dict(thinking=False),"
    else
      gk+=$'\n'"            chat_template_kwargs=dict(thinking=True, reasoning_effort=\"${MODE}\"),"
    fi
    [[ "${pass}" == acc && "${P_AVGN}" -gt 1 ]] && gk+=$'\n'"            num_return_sequences=${P_AVGN},"
    cat > "${CFG_DIR}/${MODEL_TASK}.py" <<PYCFG
# GENERATED $(date -Is) — task=${task} pass=${pass} mode=${MODE}
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
    CMD+=( --mode all --dump-eval-details --merge-ds --dump-extract-rate ); n="${P_NUM}"
  else
    CMD+=( --mode perf --summarizer default_perf ); n="${PERF_PROMPTS}"   # --merge-ds is acc-only
  fi
  [[ -n "${n}" ]] && CMD+=( --num-prompts "${n}" )
  return 0
}

report_step() {   # $1=task $2=acc|perf $3=rc
  local t="$1" p="$2" rc="$3" dir f found=0 pat
  (( rc != 0 )) && { echo "[result] ${t} ${p}: ais_bench FAILED rc=${rc}" >&2; return 0; }
  dir="$(find "${OUT_DIR}/${p}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)"
  [[ -z "${dir}" ]] && { echo "[result] ${t} ${p}: no run dir"; return 0; }
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

# A kwarg the template does not declare is dropped SILENTLY — worth ~16 points on
# gpqa. One request settles it.
probe_mode() {
  local kw want got resp
  if [[ "${MODE}" == non-think ]]; then want=off; kw='{"thinking":false}'
  else want=on; kw="{\"thinking\":true,\"reasoning_effort\":\"${MODE}\"}"; fi
  resp=$(http_req 600 "http://127.0.0.1:${PORT}/v1/chat/completions" \
    "{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2?\"}],\"max_tokens\":64,\"chat_template_kwargs\":${kw}}") \
    || { echo "[probe] WARN: request failed; skipping" >&2; return 0; }
  got=$(python3 -c "
import json,sys
m = json.loads(sys.argv[1])['choices'][0]['message']
print('on' if (m.get('reasoning_content') or m.get('reasoning') or '').strip() else 'off')" "${resp}") \
    || { echo "[probe] WARN: unparsable response" >&2; return 0; }
  echo "[probe] ${MODE}: thinking=${got} (wanted ${want})"
  [[ "${got}" == "${want}" ]] || die "server is not honouring chat_template_kwargs.
       The template may use a different key (edit build_cmd: thinking -> enable_thinking).
       PROBE=0 to override."
  return 0
}

# ── build the serve argv ──────────────────────────────────────────────────────
# expert_offload_config is a CLOSED key set: an unknown key raises at startup,
# the good failure mode. TOP-LEVEL keys use .get(), so a typo there is ignored
# SILENTLY — hence the post-health greps. Keys are emitted only when non-default,
# so an untouched knob keeps the engine's default and old runs stay comparable.

ADDL_PARTS=()
if [[ "${OFFLOAD}" == "1" ]]; then
  p="\"expert_offload\":true,\"num_device_experts\":${NUM_DEVICE_EXPERTS}"
  p+=",\"num_device_layers\":${NUM_DEVICE_LAYERS},\"cache_policy_enabled\":true"  # LRC, required by prefetch
  [[ "${PREFETCH}" == "1" ]] && p+=",\"expert_prefetch_enabled\":true\
,\"expert_prefetch_num\":${EXPERT_PREFETCH_NUM},\"expert_prefetch_tokens\":${EXPERT_PREFETCH_TOKENS}"
  [[ "${EXPERT_PREDICTOR}" != fate ]] && \
    p+=",\"expert_predictor\":\"${EXPERT_PREDICTOR}\",\"expert_predictor_ckpt\":\"${EXPERT_PREDICTOR_CKPT}\""
  [[ "${EXPERT_PREFETCH_WAIT_TIMING}" == "1" ]] && p+=",\"expert_prefetch_wait_timing\":true"
  if [[ "${EXPERT_SUBSTITUTION}" == "1" ]]; then
    p+=",\"expert_substitution_enabled\":true"
    [[ -n "${EXPERT_SUBSTITUTION_THRESHOLD}" ]] && \
      p+=",\"expert_substitution_threshold\":${EXPERT_SUBSTITUTION_THRESHOLD}"
  fi
  [[ "${ENABLE_MULTI_CARD}" == "1" ]] && p+=",\"enable_multi_card\":true"
  p+=",\"moe_offload_debug\":false"     # per-layer trace inside the host callback:
                                        # invalid for any timing run, so never on
  ADDL_PARTS+=( "\"expert_offload_config\":{${p}}" )
fi
[[ -n "${REMOE_GATE}" ]] && ADDL_PARTS+=( "\"moe_gate_override_path\":\"${REMOE_GATE}\"" )
if [[ "${DECODE_STATS}" == "1" ]]; then
  ADDL_PARTS+=( "\"decode_stats_path\":\"${STATS_DIR}\"" \
                "\"decode_stats_flush_every\":${STATS_FLUSH_EVERY}" \
                "\"decode_stats_flush_seconds\":${STATS_FLUSH_SECONDS}" )
  [[ "${CSV}" == "1" ]] && ADDL_PARTS+=( "\"decode_stats_csv\":true" )
else
  ADDL_PARTS+=( "\"decode_stats_enabled\":false" )
fi
ADDL_PARTS+=( "\"enable_cpu_binding\":false" \
              "\"multistream_overlap_shared_expert\":false" )
IFS=','; ADDL="{${ADDL_PARTS[*]}}"; unset IFS

SPEC_CFG=""
[[ -n "${SPEC_METHOD}" ]] && \
  SPEC_CFG="{\"method\":\"${SPEC_METHOD}\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS}}"

# --override-generation-config only sets a server-side default; every AISBench
# request carries its own sampling and thinking mode, so the client wins.
SERVE=( vllm serve "${MODEL}" --host 0.0.0.0 --port "${PORT}"
        --served-model-name "${SERVED_NAME}"
        --tensor-parallel-size "${TP}" --data-parallel-size "${DP}"
        --max-num-seqs "${MAX_NUM_SEQS}" --seed "${SEED}"
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
        --gpu-memory-utilization "${GPU_MEM_UTIL}" --trust-remote-code
        --quantization "${QUANTIZATION}"
        --generation-config vllm
        --override-generation-config '{"temperature":0.0,"top_p":1.0}'
        --enable-expert-parallel --enable-prefix-caching
        --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4
        --enable-auto-tool-choice --reasoning-parser deepseek_v4
        --enable-chunked-prefill --aggregate-engine-logging
        --safetensors-load-strategy prefetch --api-server-count "${API_SERVER_COUNT}" )
[[ -n "${MAX_MODEL_LEN}" ]] && SERVE+=( --max-model-len "${MAX_MODEL_LEN}" )
if (( GRAPH_ON )); then
  SERVE+=( --compilation-config \
    "{\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\",\"cudagraph_capture_sizes\":[${CAPTURE_SIZES}]}" )
else
  SERVE+=( --enforce-eager )
fi
[[ -n "${SPEC_CFG}" ]] && SERVE+=( --speculative-config "${SPEC_CFG}" )
SERVE+=( --additional-config "${ADDL}" )

HEALTH_URL="http://127.0.0.1:${PORT}/health"
PLAN=()
for t in ${TASK_LIST}; do preset "${t}"
  for p in acc perf; do [[ "${RUN}" == both || "${RUN}" == "${p}" ]] && PLAN+=( "${t}:${p}" ); done
done

# ── print the plan; check the guards ──────────────────────────────────────────

exec 3>&2                                          # fd3 = the real terminal
if [[ "${DRY_RUN}" != "1" ]]; then
  ensure_dirs
  # The tee sits in this script's process group, so a ^C would kill it and the
  # script's next write would take SIGPIPE and die mid-reap, with the log gone too.
  exec > >(trap '' INT TERM HUP QUIT; exec tee -a "${RUN_LOG}") 2>&1
fi

echo "────────────────────────────────────────────────────────────"
echo "  started : $(date '+%F %T')  host $(hostname)  pid $$"
echo "  serve   : ${SERVED_NAME} on :${PORT} (card ${CARD}) dp=${DP} ep=${EP_SIZE} max_num_seqs=${MAX_NUM_SEQS}"
(( GRAPH_ON )) \
  && echo "  graph   : ON ${CUDAGRAPH_MODE} sizes=[${CAPTURE_SIZES}]" \
  || echo "  graph   : OFF (eager) — prefetch cannot overlap; TPOT is a debug number, accuracy is not"
echo "  spec    : ${SPEC_METHOD:-off} num_spec_tokens=${NUM_SPEC_TOKENS} decode_tokens=${DECODE_TOKENS} moe_rows=${MOE_ROWS} (x dp=${DP})"
echo "  offload : ${OFFLOAD} experts=${NUM_DEVICE_EXPERTS} layers=${NUM_DEVICE_LAYERS} multi_card=${ENABLE_MULTI_CARD} subst=${EXPERT_SUBSTITUTION}${EXPERT_SUBSTITUTION_THRESHOLD:+/${EXPERT_SUBSTITUTION_THRESHOLD}}"
echo "  prefetch: ${PREFETCH} num=${EXPERT_PREFETCH_NUM} tokens=${EXPERT_PREFETCH_TOKENS} predictor=${EXPERT_PREDICTOR} wait_timing=${EXPERT_PREFETCH_WAIT_TIMING}"
[[ "${EXPERT_PREDICTOR}" != fate ]] && echo "            ckpt=${EXPERT_PREDICTOR_CKPT}"
echo "  gate    : ${REMOE_GATE:-off (base router)}"
echo "  client  : run=${RUN} mode=${MODE} tasks=${TASK_LIST} perf_prompts=${PERF_PROMPTS} avg_n=${AVG_N:-preset}"
echo "  length  : max_model_len=${MAX_MODEL_LEN:-checkpoint} ctx=${CTX:-unknown} max_out=${MAX_OUTPUT_LEN:-preset} cap=${OUT_CAP:-none}"
echo "  stats   : ${DECODE_STATS} csv=${CSV} -> ${STATS_DIR}  (quiesce=${STATS_QUIESCE}s engine_grace=${ENGINE_GRACE}s reap_wait=${REAP_WAIT}s)"
echo "  health  : ${HEALTH_URL} (wait=${WAIT}s)"
[[ -n "${SPEC_CFG}" ]] && echo "  spec-cfg: ${SPEC_CFG}"
echo "  addl-cfg: ${ADDL}"

# Environment invariants, asserted rather than re-exported so a mismatch is
# visible instead of silently corrected.
if [[ "${OFFLOAD}" == "1" ]]; then
  [[ "${DYNAMIC_EPLB:-false}" == "false" ]] || guard "DYNAMIC_EPLB with expert offload:
       process_weights_after_loading deletes the tensors the paging primitive writes into."
  [[ "${VLLM_ASCEND_ENABLE_FUSED_MC2:-0}" == "0" ]] || guard "FUSED_MC2 with expert offload: W8A8
       fused scales go stale after paging, and it applies log2phy unclamped."
fi
[[ "${VLLM_USE_V2_MODEL_RUNNER:-0}" == "0" ]] || guard "VLLM_USE_V2_MODEL_RUNNER=1: every statistics
       hook lives in model_runner_v1.py — NO summary and NO CSV from this run."
[[ "${VLLM_LOGGING_LEVEL:-INFO}" =~ ^(INFO|DEBUG)$ ]] || guard "VLLM_LOGGING_LEVEL below INFO: every
       line this script greps for is logged at INFO, so the checks below fail on a healthy run."
(( GRAPH_ON )) && [[ "${ASCEND_LAUNCH_BLOCKING:-0}" != "0" ]] && \
  echo "  warn    : ASCEND_LAUNCH_BLOCKING=1 in graph mode — the prefetch stream cannot overlap."
[[ "${EXPERT_PREFETCH_WAIT_TIMING}" == "1" ]] && {
  echo "  warn    : wait timing adds device events at the prefetch join — TPOT here is NOT a baseline."
  (( GRAPH_ON )) || echo "  warn    : in EAGER the prefetch overlaps nothing, so pf_wait reads near zero
            for reasons unrelated to its cost. Only meaningful under graph capture."; }
[[ "${EXPERT_SUBSTITUTION}" == "1" ]] && \
  echo "  note    : substitution ON — it CHANGES ROUTING. Accuracy is not comparable to a run without it."

if [[ "${OFFLOAD}" == "1" ]]; then
  echo "  thresh  : offload_threshold=${THR} MoE rows (${NDE_MIN}/${TOPK}); floor = moe_rows*topk = ${NDE_FLOOR}"
  # Above the threshold decode leaves the LRC path entirely: no paging, no
  # prefetch, no statistics — the run completes and measures nothing.
  (( MOE_ROWS > THR )) && guard "moe_rows=${MOE_ROWS} > offload_threshold=${THR}: every decode step takes
       the prefill-pool path — NO paging, NO prefetch, NO statistics. Raise NUM_DEVICE_EXPERTS to
       >= ${NDE_FLOOR}, or lower NUM_SPEC_TOKENS / DP."
  (( EP_SIZE > 1 )) && [[ "${ENABLE_MULTI_CARD}" != "1" ]] && \
    guard "EP=${EP_SIZE} with ENABLE_MULTI_CARD=0 and offload on: every rank loads the SAME experts."
  (( EP_SIZE == 1 )) && [[ "${ENABLE_MULTI_CARD}" == "1" ]] && \
    guard "ENABLE_MULTI_CARD=1 at EP=1: comm takes the multi-card branch, the layers stay single-card."
  if [[ "${ENABLE_MULTI_CARD}" == "1" ]]; then
    # num_device_experts_for_rank RAISES at FORWARD time when capacity % ep_size
    # != 0 — an indivisible capacity starts the engine and dies mid-decode.
    for _e in $(tr -d '[] ' <<<"${NUM_DEVICE_EXPERTS}" | tr ',' ' '); do
      (( _e % EP_SIZE == 0 )) || die "num_device_experts entry ${_e} not divisible by EP size ${EP_SIZE}"
    done
    # Multi-card splits decode from prefill by COMM TYPE: MC2 is admitted only when
    # num_tokens*topk <= one rank's slots. Miss it and every step goes ALLTOALL ->
    # shard prefill pool -> no paging, no stats, no prefetch, and it looks normal.
    _slots=$(( NDE_MIN / EP_SIZE ))
    (( DECODE_TOKENS * TOPK > _slots )) && guard "multi-card decode never reaches MC2:
       decode_tokens*topk=$(( DECODE_TOKENS * TOPK )) > per-rank admission slots=${_slots}."
    echo "  mc      : admission slots=${_slots}/rank; multi-card records NO per-layer cache stats
            (they live in the single-card host callback) and EXPERT_PREFETCH_NUM does not reach it."
  fi
  # _prefill_load_layer runs in the outer Python of update_weights: once at
  # capture, never on replay. A captured size above the safe bound therefore
  # replays against stale pool weights and records no prefetch nodes.
  if (( GRAPH_ON )); then
    _unsafe=""; for _s in ${CAPTURE_SIZES//,/ }; do (( _s * DP > THR )) && _unsafe+="${_s} "; done
    [[ -n "${_unsafe}" ]] && echo "  warn    : capture size(s) ${_unsafe%% } exceed the safe bound
            (size x dp > ${THR}) — those graphs replay against stale prefill-pool weights."
  fi
fi

for t in ${TASK_LIST}; do
  preset "${t}"
  echo "     - ${t}  ${P_DS}  temp=${P_TEMP} max_out=${P_MAXOUT}${P_CLAMP} acc_prompts=${P_NUM:-full} avg@n=${P_AVGN}  published(non-think): ${P_REF}"
done
echo "  out     : ${OUT_DIR}"
echo "  repro   : $(for v in CARD PORT DP MTP NUM_SPEC_TOKENS OFFLOAD NUM_DEVICE_EXPERTS NUM_DEVICE_LAYERS \
      ENABLE_MULTI_CARD PREFETCH EXPERT_PREFETCH_NUM EXPERT_PREFETCH_TOKENS EXPERT_PREFETCH_WAIT_TIMING \
      EXPERT_PREDICTOR EXPERT_PREDICTOR_CKPT EXPERT_SUBSTITUTION EXPERT_SUBSTITUTION_THRESHOLD REMOE_GATE \
      ENFORCE_EAGER MAX_MODEL_LEN MAX_OUTPUT_LEN DECODE_STATS CSV TASKS RUN THINK THINK_EFFORT NUM_PROMPTS PERF_PROMPTS AVG_N MODEL
   do printf '%s=%q ' "${v}" "${!v-}"; done)$0"
echo "────────────────────────────────────────────────────────────"
printf '  $ '; printf '%q ' "${SERVE[@]}"; echo
for step in "${PLAN[@]}"; do build_cmd "${step%:*}" "${step#*:}"; printf '  $ '; printf '%q ' "${CMD[@]}"; echo; done
[[ "${DRY_RUN}" == "1" ]] && { echo "[DRY_RUN] nothing launched."; exit 0; }

# ── preflight — everything that can fail in seconds, before a 4-minute model load ───

# Without vllm-ascend, DeviceConfig raises "Failed to infer device type" while
# BUILDING THE ARG PARSER, before any flag here is read. Name the cause.
python3 -c "
import sys
from importlib.metadata import entry_points
if not list(entry_points(group='vllm.platform_plugins')): sys.exit(1)
for m in ('vllm_ascend','torch_npu'): __import__(m)
" 2>/dev/null || die "vllm-ascend / torch_npu will not import, or no vllm.platform_plugins entry
       point is registered. Did you 'source env_a5_0260.sh' in this shell?"
command -v ais_bench >/dev/null 2>&1 || die "ais_bench not on PATH"
[[ -n "${PKG_CFG}" && -d "${PKG_CFG}" ]] || die "cannot locate the ais_bench configs dir"

# The perf summarizer loads the tokenizer only AFTER every request is sent, so a
# bad path otherwise costs a whole pass.
[[ "${RUN}" == acc ]] || python3 -c "
from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('${TOKENIZER}', trust_remote_code=True)" \
  2>/dev/null || die "the perf summary would fail on TOKENIZER='${TOKENIZER}'"

# AIS_BENCH_DATASETS_CACHE is a ROOT that AISBench appends the config's own
# relative path to.
_miss=0
for t in ${TASK_LIST}; do
  preset "${t}"
  _f="$(find "${PKG_CFG}/datasets" -name "${P_DS}.py" -print -quit 2>/dev/null)"
  [[ -n "${_f}" ]] || { echo "[!!] dataset config '${P_DS}' not installed" >&2; _miss=1; continue; }
  _rel="$(grep -oE "path='[^']*'" "${_f}" | head -1 | sed "s/^path='//;s/'$//")"
  [[ "${_rel}" == /* ]] && _abs="${_rel}" || _abs="${DATASETS_CACHE}/${_rel}"
  [[ -d "${_abs}" ]] && echo "[data ] ${t}: ${_abs}" || {
    echo "[!!] ${t}: no data at ${_abs}" >&2
    echo "     cd '${DATASETS_CACHE}/ais_bench/datasets' && wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/${t}.zip && unzip ${t}.zip" >&2
    _miss=1; }
done
(( _miss )) && die "missing dataset config or data"

_pids="$(port_pids)"
if [[ "${SKIP_SERVE}" == "1" ]]; then
  echo "[port ] :${PORT} pids=${_pids:-none}"
elif (exec 4<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
  die "port ${PORT} already in use${_pids:+ by PID(s) ${_pids}}"
else
  # Prove the HTTP path works BEFORE a four-minute model load. The port is free,
  # so a real request must come back unreachable; anything else means the health
  # poll would later fail for a reason it cannot report.
  _err="$(http_req 3 "http://127.0.0.1:${PORT}/" 2>&1 >/dev/null)"; _rc=$?
  case "${_rc}" in
    7|28) echo "[port ] :${PORT} free, localhost round-trip OK (exit ${_rc}, as expected)" ;;
    0|22) die "something is already ANSWERING on :${PORT} that the port scan did not see" ;;
    *)    die "python3 cannot make an HTTP request on this box (exit ${_rc}): ${_err}" ;;
  esac
fi

# ── launch ────────────────────────────────────────────────────────────────────

if [[ "${SKIP_SERVE}" != "1" ]]; then
  trap 'on_signal INT' INT; trap 'on_signal TERM' TERM
  trap 'on_signal HUP' HUP; trap on_exit EXIT
  echo "[serve] launching — engine output streams below and into ${SERVE_LOG}"
  : > "${SERVE_LOG}"
  if command -v setsid >/dev/null 2>&1; then setsid "${SERVE[@]}" >> "${SERVE_LOG}" 2>&1 &
  else "${SERVE[@]}" >> "${SERVE_LOG}" 2>&1 & fi
  SERVE_PID=$!
  # setsid() gives sid == pgid == pid, but the exec'd `setsid` has not
  # necessarily got there when bash returns from `&`. Reading ps ONCE races it,
  # blanks SERVE_PGID, and silently demotes reap to a tree walk. So poll.
  for ((i=0; i<50; i++)); do
    _sid="$(ps -o sid= -p "${SERVE_PID}" 2>/dev/null | tr -d ' ')"
    [[ -z "${_sid}" ]] && break
    if [[ "${_sid}" == "${SERVE_PID}" ]]; then
      SERVE_SID="${_sid}"
      SERVE_PGID="$(ps -o pgid= -p "${SERVE_PID}" 2>/dev/null | tr -d ' ')"
      break
    fi
    sleep 0.1
  done
  # No setsid, or it did not take: the server shares this script's group, so
  # signalling the group would hit the script too. Leave it empty; reap walks.
  [[ "${SERVE_PGID}" == "$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')" ]] && SERVE_PGID=""

  mirror_start
  echo "[serve] waiting up to ${WAIT}s for ${HEALTH_URL} ..."
  ok=0; _listening=0; _since=0
  for ((i=0; i<WAIT/2; i++)); do
    kill -0 "${SERVE_PID}" 2>/dev/null || die "serve exited during startup (see ${SERVE_LOG})"
    http_req 5 "${HEALTH_URL}" >/dev/null 2>&1 && { ok=1; break; }
    (( ! _listening )) && grep -q 'Application startup complete\|Starting vLLM server on' \
      "${SERVE_LOG}" 2>/dev/null && _listening=1
    if (( _listening )); then
      (( _since += 2 ))
      (( _since > HEALTH_GRACE )) && die "the engine is serving on :${PORT} but ${HEALTH_URL} has not
       answered in ${HEALTH_GRACE}s. This is connectivity, not a slow load: check the engine's own
       'Starting vLLM server on' line for the port it really bound, and 'ss -ltnp sport = :${PORT}'."
    fi
    (( i % 30 == 29 )) && echo "[serve] still waiting ($(( (i+1) * 2 ))s of ${WAIT}s) ..."
    sleep 2
  done
  (( ok )) || die "server not ready in ${WAIT}s (see ${SERVE_LOG})"
  echo "[serve] healthy"

  # TOP-LEVEL additional_config keys are read with .get(), so an unpatched engine
  # or a typo swallows them SILENTLY. The engine's own echoed lines turn that back
  # into a failure. (The nested expert_offload_config set is closed and raises.)
  [[ -n "${REMOE_GATE}" ]] && { grep -q '\[GATE-OVERRIDE\] applied' "${SERVE_LOG}" \
    || guard "no '[GATE-OVERRIDE] applied' in the log — REMOE_GATE was ignored and this run would
       measure the BASE router."; }
  if [[ "${EXPERT_PREDICTOR}" != fate ]]; then
    grep -o '\[PREFETCH-AI\].*$' "${SERVE_LOG}" | sort -u | sed 's/^/  predict : /'
    grep -q '\[PREFETCH-AI\] armed' "${SERVE_LOG}" || guard "no '[PREFETCH-AI] armed' in the log — the
       head did not load or no decoder layer resolved a capture site, so every prefetch this run
       makes comes from the built-in fate predictor."
  fi
  [[ "${EXPERT_PREFETCH_WAIT_TIMING}" == "1" ]] && \
    { grep -q '\[PREFETCH-WAIT\] timing enabled' "${SERVE_LOG}" \
      || guard "no '[PREFETCH-WAIT] timing enabled' in the log — the key was not applied and this
       run measures no stall."; }
  if grep -q '\[DECODE-STATS\] config:' "${SERVE_LOG}"; then
    grep -o '\[DECODE-STATS\] \(config\|topology\|armed\|decode\):.*$' "${SERVE_LOG}" | sort -u | sed 's/^/  stats   : /'
    grep -q '\[DECODE-STATS\] armed' "${SERVE_LOG}" || \
      echo "  warn    : no '[DECODE-STATS] armed' — collection is NOT live."
    # Draft MoE layers are excluded by a layer-NAME test; the registered count
    # settles whether that held for THIS drafter.
    _nml="$(grep -o 'topology: moe_layers=[0-9]*' "${SERVE_LOG}" | head -1 | grep -o '[0-9]*$')"
    _nhl="$(cfg num_hidden_layers)"
    [[ -n "${_nml}" && -n "${_nhl}" && "${_nml}" != "${_nhl}" ]] && \
      echo "  warn    : ${_nml} MoE layers registered but num_hidden_layers=${_nhl} — the drafter's MoE
            layers are being OFFLOADED, which the sizing model assumes never happens."
  elif [[ "${DECODE_STATS}" == "1" ]]; then
    guard "no '[DECODE-STATS] config:' in the log — the statistics keys were swallowed, so this run
       produces NO summary and NO CSV."
  fi
else
  echo "[serve] SKIP_SERVE=1 — using whatever is on :${PORT}"
  http_req 5 "${HEALTH_URL}" >/dev/null 2>&1 || die "nothing healthy on :${PORT}"
fi

[[ "${PROBE}" == "1" ]] && probe_mode

# ── run ───────────────────────────────────────────────────────────────────────

rc=0
for step in "${PLAN[@]}"; do
  t="${step%:*}"; p="${step#*:}"
  build_cmd "${t}" "${p}"
  echo "[${p}] ${t} [${MODE}] ${P_DS}"
  printf '  $ '; printf '%q ' "${CMD[@]}"; echo
  # AISBench's progress table needs a real TTY, and that table is also the
  # liveness signal — so the heartbeat only runs when there is none.
  if [[ -t 3 ]]; then
    mirror_stop; "${CMD[@]}" >&3 2>&3; src=$?; mirror_start
  else
    ( t0=${SECONDS}
      while sleep "${HEARTBEAT}"; do
        el=$(( SECONDS - t0 ))
        n=$(http_req 5 "http://127.0.0.1:${PORT}/metrics" 2>/dev/null \
            | awk '$1 ~ /^vllm:request_success_total/ {s+=$2} END {printf "%d", s+0}')
        printf '[hb] %s %s  elapsed %02d:%02d:%02d  server_completed=%s\n' \
               "${t}" "${p}" $((el/3600)) $((el%3600/60)) $((el%60)) "${n:-?}"
      done ) &
    HB_PID=$!
    "${CMD[@]}"; src=$?
    kill "${HB_PID}" 2>/dev/null; HB_PID=""
  fi
  (( src != 0 )) && rc=${src}
  report_step "${t}" "${p}" "${src}"
  # ais_bench exits 0 on HTTP 500s, so its rc cannot see a dead engine. Checked
  # BEFORE reap(), which writes the same SIGTERM line.
  if grep -qE 'engine core exited unexpectedly|EngineDeadError|EngineCore: trigger received signal' \
       "${SERVE_LOG}" 2>/dev/null; then
    echo "[!!] ENGINE DIED DURING THIS PASS — the numbers above are NOT a measurement. The API"
    echo "     server stayed up and answered the rest with 500s, which is why ais_bench exited 0."
    grep -nE 'engine core exited unexpectedly|EngineCore: trigger received signal' \
         "${SERVE_LOG}" | head -3 | sed 's/^/     /'
    rc=1; break
  fi
done

# ── quiesce, shut down, then report — IN THAT ORDER ───────────────────────────
# The collector snapshots ~3s after decoding stops, from its own watchdog thread,
# and that idle snapshot is what makes the artefact cover the WHOLE run — it needs
# the server alive and idle. Wait, then stop, then read a settled STATS_DIR.

if [[ "${SKIP_SERVE}" != "1" ]]; then
  [[ "${DECODE_STATS}" == "1" ]] && {
    echo "[stats]  quiescing ${STATS_QUIESCE}s so the idle snapshot lands ..."
    sleep "${STATS_QUIESCE}"; }
  reap
fi
report_stats

echo "────────────────────────────────────────────────────────────"
echo "[done] rc=${rc}   finished $(date '+%F %T')   everything under ${OUT_DIR}"
echo "────────────────────────────────────────────────────────────"
sleep 1     # the tee behind the process substitution is asynchronous
exit "${rc}"

# ── FINDINGS — what silently changed a number. Read once. ────────────────────
#
#  ^C MUST NOT ABANDON THE CLEANUP. setsid puts the server in its own session, so
#  the terminal's SIGINT never reaches it and this script's trap is the only thing
#  that stops it. Four defects conspired once: the trap missed HUP; reap() was
#  interruptible, so a second ^C killed the script between the TERM and the KILL;
#  SERVE_PGID was read with one `ps` racing setsid()'s own call; and the tee sits
#  in this script's group, so ^C killed it and the next write took SIGPIPE. Also:
#  pgrep -P cannot see a worker re-parented to init (hence session_pids), and
#  reap() must NEVER `pgrep -f EngineCore` host-wide — that once killed a
#  concurrent benchmark's engine. Do not "simplify" reap(), its traps, or the tee.
#
#  ORDER IS QUIESCE -> ENGINE -> GROUP -> KILL -> REPORT. The collector's idle
#  snapshot needs an alive, idle server, and [EXPERT-OFFLOAD-FINAL] needs the
#  EngineCore to get its own SIGTERM before the API server force-kills it at
#  timeout=0s. report_stats() runs after the reap, from the happy path and from
#  the traps alike, so an interrupted run still prints what it collected.
#
#  curl DOES NOT WORK HERE. env_a5_0260.sh puts ${CONDA_PREFIX}/lib ahead of the
#  system, so /usr/bin/curl fails in the loader (exit 127, nothing sent) — which a
#  poll cannot tell apart from "not ready". Everything goes through http_req().
#
#  --config-dir REPLACES ais_bench's config root, it is not additive. Without the
#  symlinks ensure_dirs() makes, every pass dies at config resolution before one
#  request is sent, and the symptom is a healthy idle server and a benchmark that
#  never starts.
#
#  MoE ROWS != DECODE TOKENS WHEN DP > 1. prepare() all-gathers across the DP
#  group, so the MoE layer sees DP x decode_tokens rows and that is what is
#  compared against offload_threshold. Above it every decode step takes the
#  prefill pool: no paging, no prefetch, no statistics, and the run looks normal.
#  The pool floor is moe_rows*topk, so each extra NUM_SPEC_TOKENS costs topk*dp
#  slots — dspark at k=5 is 6 decode tokens, i.e. a floor of 36 at topk=6. The
#  same bound gates MC2 in multi-card, where missing it drops to ALLTOALL just as
#  quietly. And a captured graph above that bound is not safe either:
#  _prefill_load_layer runs in the outer Python of update_weights, once at capture
#  and never on replay, so it replays against stale pool weights.
#
#  THE TWO PREFETCH KNOBS ARE ORTHOGONAL: _TOKENS is the prediction width (how
#  many candidates exist), _NUM the transfer budget, single-card only. And the
#  METHOD is per target layer, not per run — EXPERT_PREDICTOR drives only the
#  targets its head covers, fate keeps the rest (layers 1-2 under mode2_har, 1-3
#  under mode2_prevhfr). Either way 40 layers contribute, so pf_loads_step /
#  pf_loads staying at 40.0000 is the cheapest proof coverage did not shift.
#
#  A MISMATCHED PREDICTOR CHECKPOINT FAILS SILENTLY — plausible expert ids, a
#  degraded hit rate, no error anywhere. in_dim must equal hc_mult*hidden_size and
#  L must be num_hidden_layers or num_hidden_layers - num_hash_layers.
#  '[PREFETCH-AI] armed' is the runtime half of the check; the basis half is
#  undetectable, so serve the head on the build its residuals were dumped from.
#  It costs ~460 MiB/rank out of the same HBM the expert cache lives in.
#
#  pf_wait IS THE ONLY NUMBER THAT MEASURES WHAT PREFETCH COSTS — ms the compute
#  stream is blocked at the join, taken with device events, the only instrument
#  that works under graph replay. So a run with the timing on is not a TPOT
#  baseline, and in EAGER it reads near zero for reasons unrelated to cost. If it
#  comes out all zeros or all identical, confirm it moves with EXPERT_PREFETCH_NUM
#  before believing it. More generally: in eager the prefetch callback runs inline
#  and ends in load_stream.synchronize(), so PREFETCH=1 is SLOWER than PREFETCH=0.
#  Eager is for accuracy parity, never for latency.
#
#  READ avg@n ONLY. AISBench de-interleaves the list with the wrong stride: avg@n
#  survives as a flat mean, pass@n and cons@n come out as 1-(1-p)^n and p^n.
#  Pass@1 IS avg@n. At n=1 the 198-row gpqa set carries ~3.2 points of binomial
#  noise. Keep gpqa_gen_0_shot_cot_chat_prompt: the _str variant measured ~8
#  points lower because first_option_postprocess takes the FIRST regex match, so
#  a self-correction scores at its pre-correction letter.
#
#  A REMOE GATE ON THE WRONG BASIS DOES NOT CRASH. QuaRot's R1 is absorbed into
#  mlp.gate, so a rotated build stores W@R1 where an unrotated one stores W —
#  same shapes, same norms, same singular values, every engine check passes. The
#  model simply stops emitting EOS and every prompt runs to max_out_len.
#
#  ALSO SILENT: multi-card records no per-layer cache statistics (they live in the
#  single-card host callback); draft MoE layers are excluded from offload by a
#  layer-NAME test, so the moe_layers check proves it only for the drafter in
#  front of you; ais_bench exits 0 when the engine is dead, because HTTP 500s are
#  responses; and a chat_template kwarg the template does not declare is dropped
#  without a word, which is what probe_mode() exists for.
# ─────────────────────────────────────────────────────────────────────────────