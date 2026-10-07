#!/usr/bin/env bash
# =============================================================================
#  v4_target_aisbench.sh — serve DeepSeek-V4-Flash, measure ACCURACY / TPOT via
#  AISBench (gsm8k gpqa mmlu_pro aime2025 math500 lcb mgsm), tear down cleanly.
#
#    source env_a5_0260.sh                     # REQUIRED, in this same shell
#    CARD=2 PORT=1161 TASKS=gpqa RUN=perf MTP=1 ./v4_target_aisbench.sh
#    TASKS="lcb mgsm" RUN=acc EXPERT_SUBSTITUTION=1 ./...   # the substitution A/B
#    MTP=2 ACT_ROUTE=1 ./...                    # upstream anchor-union routing A/B
#    EXPERTS_PRUNING=1 EXPERTS_PRUNING_THRESHOLD='[0,0,0,0.16,0.165,0.17]' ./...
#    CARD=12,13 DP=2 ENABLE_MULTI_CARD=1 PORT=7081 ./...
#    DRY_RUN=1 ...                             # print the plan, launch nothing
#
#  Knobs live above "DO NOT EDIT BELOW"; below it is frozen config and machinery.
#  Read FINDINGS at the bottom once — each entry is something that silently
#  changed a number.  OUT_DIR/: run.log serve.log acc/ perf/ configs/ stats/.
# =============================================================================
set -uo pipefail

# ═════════════════════════════════ EDIT HERE ═════════════════════════════════

# ── what to serve ────────────────────────────────────────────────────────────
MODEL="${MODEL:-}"
SERVED_NAME="${SERVED_NAME:-}"
TOKENIZER="${TOKENIZER:-"${MODEL}"}"
DATASETS_CACHE="${DATASETS_CACHE:-}"

# ── context / generation length ──────────────────────────────────────────────
# MAX_MODEL_LEN = served context (empty = checkpoint's own); MAX_OUTPUT_LEN
# overrides every preset's max_out. Capped to CTX - PROMPT_RESERVE: a request
# over the context is rejected mid-pass.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_OUTPUT_LEN="${MAX_OUTPUT_LEN:-}"

# ── placement ────────────────────────────────────────────────────────────────
CARD="${CARD:-6}"                        # comma list; its length must equal DP
PORT="${PORT:-7001}"
DP="${DP:-1}"                            # EP spans DP (TP and PP are pinned to 1)

# ── speculative decoding ─────────────────────────────────────────────────────
# 0 off | 1 mtp | 2 dspark. Draft MoE layers never offload, so this changes only
# the decode TOKEN COUNT — i.e. whether the offload threshold is still cleared.
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

# ── expert pruning — UPSTREAM FEATURE, default OFF ───────────────────────────
# Drops a route that is weak AND non-resident: id -> -1, weight -> 0, and the
# remaining mixture weights are NOT renormalised. It runs inside update_weights
# before substitution, so it needs OFFLOAD=1, and it is SINGLE-CARD only
# (update_weights_multi_card never calls maybe_prune_topk_experts).
# THRESHOLD is RANK-INDEXED over top_k: entry i is tested against rank i of the
# weight-sorted row as `weight < row_sum * thr[i]`, so thr[0]=0 keeps the
# strongest expert unprunable. The engine default is all zeros, which prunes
# NOTHING — see the guard below.
EXPERTS_PRUNING="${EXPERTS_PRUNING:-0}"            # CHANGES ROUTING — see FINDINGS
EXPERTS_PRUNING_THRESHOLD="${EXPERTS_PRUNING_THRESHOLD:-[0.0, 0.01, 0.02, 0.05, 0.09, 0.14]}"  # JSON list, len == topk; empty = engine default
EXPERTS_PRUNING_DEBUG="${EXPERTS_PRUNING_DEBUG:-0}"  # per-layer-per-step JSON: NOT a timing baseline

# ── anchor-union activation routing — UPSTREAM FEATURE, default OFF ──────────
# The rows of a DSpark verify block share one expert pool: anchor rows keep
# their own top-k, suffix rows are masked to the batch-wide union of the pool.
# Needs MTP=2 (dspark; every other drafter is rejected at startup) and
# --no-async-scheduling, which is added to the serve argv automatically below.
# LEAVE THE TUNING KNOBS EMPTY. The engine derives verify_block_size from the
# verify query length, fused_rows from it plus the 512 prefill bucket, and
# expected_router_layers from the model config — and writing a value EXPLICITLY
# SUPPRESSES that derivation for that field (user_keys always wins), even when
# the value equals the default.
ACT_ROUTE="${ACT_ROUTE:-0}"                        # CHANGES ROUTING — see FINDINGS
ACT_ROUTE_BACKEND="${ACT_ROUTE_BACKEND:-anchor_union_fused_native}"  # or anchor_union_reference
ACT_ROUTE_VERIFY_BLOCK="${ACT_ROUTE_VERIFY_BLOCK:-}"       # empty = derived (must equal 1+NUM_SPEC_TOKENS)
ACT_ROUTE_PROTECTED_ROWS="${ACT_ROUTE_PROTECTED_ROWS:-}"   # empty = engine default 3
ACT_ROUTE_SUFFIX_POOL_TOP_K="${ACT_ROUTE_SUFFIX_POOL_TOP_K:-}"   # empty = engine default 2
ACT_ROUTE_FUSED_ROWS="${ACT_ROUTE_FUSED_ROWS:-}"           # JSON list; empty = derived [verify_block,512]

# ── prefetch ─────────────────────────────────────────────────────────────────
# ORTHOGONAL: _TOKENS = rows the predictor runs on (candidates), _NUM = how many
# are actually TRANSFERRED.
PREFETCH="${PREFETCH:-1}"
EXPERT_PREFETCH_NUM="${EXPERT_PREFETCH_NUM:-1}"
EXPERT_PREFETCH_TOKENS="${EXPERT_PREFETCH_TOKENS:-6}"
# fate (built-in) | mode2_har | mode2_prevhfr; a head drives only its own layers.
EXPERT_PREDICTOR="${EXPERT_PREDICTOR:-fate}"
EXPERT_PREDICTOR_CKPT="${EXPERT_PREDICTOR_CKPT:-}"

# ── measurement ──────────────────────────────────────────────────────────────
DECODE_STATS="${DECODE_STATS:-0}"        # 0 = no summary, no CSV: TPOT baselines only
CSV="${CSV:-0}"                          # also emit the per-step .csv
# ms the compute stream is BLOCKED at the prefetch join — what prefetch COSTS.
# Adds device events, so NOT a TPOT baseline.
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
# Both are TOTALS per task; a config that expands into sub-datasets gets
# total / subsets of each (mmlu_pro 14 categories, mgsm 11 languages).
AVG_N="${AVG_N:-}"                       # repeats per question; 4-5 to report

# ── run control ──────────────────────────────────────────────────────────────
OUT_DIR="${OUT_DIR:-$PWD/aisbench_results/$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_SERVE="${SKIP_SERVE:-0}"            # reuse whatever already listens on PORT
PROBE="${PROBE:-1}"                      # verify the served reasoning mode
ALLOW_UNSAFE="${ALLOW_UNSAFE:-0}"        # every guard below becomes a warning

# ── task presets: task | dataset | temp | top_p | max_out | acc_prompts | avg_n | get | published
#  The last five are the fast lossy A/B sets (acc under ~30 min on A3): greedy, so
#  two runs differ only by the method, and a max_out that bounds a run which stops
#  emitting EOS. acc_prompts is a TOTAL and the FIRST N items, the same ones every
#  run: mmlu_pro 98 = 7 per category, mgsm 132 = 12 per language.
#  Substitution replaces the COLD experts, so the tasks that show it are the ones
#  routing away from the hot set: lcb (code identifiers, long generations, one bad
#  token fails the tests) and mgsm (per-language experts; sw/te/bn/th go first).
#  mrcr is the long-context probe: it is PREFILL-dominated with ~1k-token answers,
#  so it exercises retrieval over a 32k context, not many decode steps — read it as
#  "long context still works", not as a substitution signal (see FINDINGS).
#  get: oss = OpenCompass zip named after the data dir; hf:<repo>[@<rev>] = a HF
#  clone, pinned when main no longer carries the files the loader reads;
#  hfcli:<repo>[:<subdir>] = a huggingface-cli download into a named local dir.
#  NOTE: lcb executes model-written code on THIS host to score it.
PRESETS="
gsm8k    | gsm8k_gen_0_shot_cot_chat_prompt   | 0.0 |      | 16384 |    | 1 | oss | 90.8 (base 8-shot EM)
gpqa     | gpqa_gen_0_shot_cot_chat_prompt    | 0.6 | 0.95 | 65536 |    | 1 | oss | 71.2 (GPQA-D Pass@1)
mmlu_pro | mmlu_pro_gen_0_shot_str            | 0.6 | 0.95 | 32768 | 98 | 1 | oss | 83.0 (MMLU-Pro EM)
aime2025 | aime2025_gen_0_shot_chat_prompt    | 1.0 |      | 16384 |    | 1 | oss | n/a (all 30, integer EM)
math500  | math500_gen_0_shot_cot_chat_prompt | 1.0 |      | 16384 | 80 | 1 | oss | n/a (MATH-500 pass@1)
lcb      | livecodebench_0_shot_chat_v6       | 1.0 |      | 16384 | 30 | 1 | hf:livecodebench/code_generation_lite | n/a (LCB v6 pass@1)
mgsm     | mgsm_gen_0_shot_cot_chat_prompt    | 1.0 |      | 16384 |132 | 1 | hf:juletxara/mgsm@f52417ca77bd71e9888ddc29f92587660725d2b4 | n/a (12 x 11 languages)
mrcr     | mrcr_32k_gen                       | 1.0 |      | 8192  | 40 | 1 | hfcli:openai/mrcr:MRCR | n/a (8needle, 32k bin; official temp is 1.0)
"

# ══════════════════════════ DO NOT EDIT BELOW THIS LINE ══════════════════════

# FIXED — change one and no previous number is comparable.
TP=1; PP=1; SEED=1024; GPU_MEM_UTIL=0.95; MAX_NUM_SEQS=1
PROMPT_RESERVE=4096                      # context held back for the prompt
MAX_NUM_BATCHED_TOKENS=8192; QUANTIZATION=ascend; API_SERVER_COUNT=1
CUDAGRAPH_MODE=FULL_DECODE_ONLY          # capture list is auto: 1..decode_tokens
PERF_TEMP=0.0                            # pinned so TPOT stays comparable
STATS_FLUSH_EVERY=100; STATS_FLUSH_SECONDS=30
STATS_QUIESCE=8                          # idle seconds before stopping, so the
                                         # collector's quiet snapshot lands
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
export AIS_BENCH_DATASETS_CACHE="${DATASETS_CACHE}"   # where the data is READ from
# lcb is additionally BUILT into HF's own cache, which defaults to the ROOT
# filesystem. Keep it on the data volume; HF_HOME moves the modules cache too.
export HF_HOME="${HF_HOME:-${DATASETS_CACHE}/.hf_home}"
# mrcr bins by o200k_base token count, and tiktoken FETCHES that BPE file at
# dataset-load time. Its default cache is under $TMPDIR, which is wiped, so a
# seeded copy would be lost; keep it on the data volume instead.
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-${DATASETS_CACHE}/.tiktoken_cache}"
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-0}"   # env_a5_0260.sh owns this

RUN_LOG="${OUT_DIR}/run.log"; SERVE_LOG="${OUT_DIR}/serve.log"
CFG_ROOT="${OUT_DIR}/configs"; CFG_DIR="${CFG_ROOT}/models"; STATS_DIR="${OUT_DIR}/stats"
SERVE_PID=""; SERVE_PGID=""; SERVE_SID=""; HB_PID=""; TAIL_PID=""; MIRROR_AT=0
STATS_REPORTED=0
declare -A DS_PARTS=()                   # dataset config -> sub-dataset count (0 = unknown)

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
[[ -z "${NUM_PROMPTS}" || "${NUM_PROMPTS}" =~ ^[0-9]+$ ]] || die "NUM_PROMPTS must be empty or a non-negative integer"
[[ "${PERF_PROMPTS}" =~ ^[1-9][0-9]*$ ]]                 || die "PERF_PROMPTS must be a positive integer"
[[ -z "${REMOE_GATE}" || -e "${REMOE_GATE}" ]]           || die "REMOE_GATE not found: ${REMOE_GATE}"

MODE=non-think; [[ "${THINK}" == "1" ]] && MODE="${THINK_EFFORT}"
DECODE_TOK_PER_REQ=1
[[ -n "${SPEC_METHOD}" ]] && DECODE_TOK_PER_REQ=$(( 1 + NUM_SPEC_TOKENS ))
DECODE_TOKENS=$(( MAX_NUM_SEQS * DECODE_TOK_PER_REQ ))   # per rank / per capture
GRAPH_ON=1; [[ "${ENFORCE_EAGER}" == "1" ]] && GRAPH_ON=0
EP_SIZE=$(( TP * DP ))
NCARDS="$(awk -F',' '{print NF}' <<<"${CARD}")"
[[ "${NCARDS}" == "${EP_SIZE}" ]] || die "CARD lists ${NCARDS} device(s) but TP*DP=${EP_SIZE}"
(( EP_SIZE > 1 )) && [[ -n "${VA_OMP_MULTI:-}" ]] && export OMP_NUM_THREADS="${VA_OMP_MULTI}"

# prepare() all-gathers across the DP group, so the MoE layer sees DP x
# decode_tokens rows and THAT is what is compared against offload_threshold.
MOE_ROWS=$(( DECODE_TOKENS * DP ))

TOPK="$(cfg num_experts_per_tok)"
[[ "${TOPK}" =~ ^[1-9][0-9]*$ ]] || die "cannot read num_experts_per_tok from ${MODEL}/config.json"
NDE_MIN="${NUM_DEVICE_EXPERTS}"
[[ "${NUM_DEVICE_EXPERTS}" == \[* ]] && \
  NDE_MIN="$(tr -d '[] ' <<<"${NUM_DEVICE_EXPERTS}" | tr ',' '\n' | sort -n | head -1)"
[[ "${NDE_MIN}" =~ ^[1-9][0-9]*$ ]] || die "NUM_DEVICE_EXPERTS must be a positive int or a JSON list"
THR=$(( NDE_MIN / TOPK ))            # offload_threshold, in MoE token rows
NDE_FLOOR=$(( MOE_ROWS * TOPK ))     # smallest pool that keeps decode on the offload path

# CTX empty = no cap computed, the engine stays the only authority. An explicit
# MAX_OUTPUT_LEN at or above the context can never be satisfied, so it is fatal;
# a preset that overruns is simply clamped.
CTX="${MAX_MODEL_LEN:-$(cfg max_position_embeddings)}"
[[ "${CTX}" =~ ^[1-9][0-9]*$ ]] || CTX=""
OUT_CAP=""
if [[ -n "${CTX}" ]]; then
  [[ -n "${MAX_OUTPUT_LEN}" ]] && (( MAX_OUTPUT_LEN >= CTX )) && \
    die "MAX_OUTPUT_LEN=${MAX_OUTPUT_LEN} must be < ctx=${CTX}: rejected per request, mid-pass."
  OUT_CAP=$(( CTX - PROMPT_RESERVE ))
  (( OUT_CAP < 1 )) && OUT_CAP=$(( CTX / 2 ))   # tiny context: leave half for the prompt
fi

# A trained predictor and the stall timer both ride the prefetch pipeline; with
# OFFLOAD=0 the config block is never emitted and the run measures nothing.
if [[ "${EXPERT_PREDICTOR}" != fate ]]; then
  [[ "${OFFLOAD}" == "1" && "${PREFETCH}" == "1" ]] || die "EXPERT_PREDICTOR needs OFFLOAD=1 and PREFETCH=1"
  [[ -r "${EXPERT_PREDICTOR_CKPT}" ]] || die "EXPERT_PREDICTOR_CKPT unreadable: '${EXPERT_PREDICTOR_CKPT}'"
fi
[[ "${EXPERT_PREFETCH_WAIT_TIMING}" == "0" || ( "${OFFLOAD}" == "1" && "${PREFETCH}" == "1" ) ]] \
  || die "EXPERT_PREFETCH_WAIT_TIMING=1 needs OFFLOAD=1 and PREFETCH=1: no prefetch, no join to time"
[[ "${EXPERT_PREFETCH_WAIT_TIMING}" == "0" || "${DECODE_STATS}" == "1" ]] \
  || die "EXPERT_PREFETCH_WAIT_TIMING=1 needs DECODE_STATS=1: pf_wait is reported there."

# ── expert pruning (upstream) ────────────────────────────────────────────────
[[ "${EXPERTS_PRUNING}" =~ ^[01]$ ]]       || die "EXPERTS_PRUNING must be 0 or 1"
[[ "${EXPERTS_PRUNING_DEBUG}" =~ ^[01]$ ]] || die "EXPERTS_PRUNING_DEBUG must be 0 or 1"
PRUNE_THR_LEN=0; PRUNE_THR_ALLZERO=1; PRUNE_THR_OK=ok
if [[ -n "${EXPERTS_PRUNING_THRESHOLD}" ]]; then
  read -r PRUNE_THR_LEN PRUNE_THR_ALLZERO PRUNE_THR_OK <<<"$(python3 - "${EXPERTS_PRUNING_THRESHOLD}" <<'PY'
import json, sys
try:
    v = json.loads(sys.argv[1])
except Exception:
    print("0 1 not-json"); raise SystemExit
bad = (not isinstance(v, list) or not v
       or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in v))
if bad:
    print("0 1 not-a-number-list"); raise SystemExit
if any(x < 0 for x in v):
    print(f"{len(v)} 0 negative"); raise SystemExit
print(f"{len(v)} {int(all(x == 0 for x in v))} ok")
PY
)"
  [[ "${PRUNE_THR_OK}" == ok ]] || die "EXPERTS_PRUNING_THRESHOLD invalid (${PRUNE_THR_OK}): a non-empty
       JSON list of numbers >= 0, e.g. '[0,0,0,0.16,0.165,0.17]'. AscendConfig rejects anything else."
  (( PRUNE_THR_LEN == TOPK )) || die "EXPERTS_PRUNING_THRESHOLD has ${PRUNE_THR_LEN} entries but
       top_k=${TOPK}: dynamic_pruning_unsorted raises on the FIRST decode forward, i.e. after the
       full model load and health check, mid-pass."
fi
if [[ "${EXPERTS_PRUNING}" == "1" ]]; then
  [[ "${OFFLOAD}" == "1" ]] || die "EXPERTS_PRUNING=1 needs OFFLOAD=1: pruning rides inside
       update_weights, and with OFFLOAD=0 that block is never emitted."
  [[ "${ENABLE_MULTI_CARD}" != "1" ]] || guard "EXPERTS_PRUNING=1 with ENABLE_MULTI_CARD=1:
       update_weights_multi_card never calls maybe_prune_topk_experts, so pruning is a NO-OP on the
       multi-card path and this run is identical to EXPERTS_PRUNING=0."
  if [[ -z "${EXPERTS_PRUNING_THRESHOLD}" ]]; then
    guard "EXPERTS_PRUNING=1 with no EXPERTS_PRUNING_THRESHOLD: the engine default is [0,...] and
       weak_mask is 'weight < row_sum * 0', which is never true — this run prunes NOTHING and is
       identical to EXPERTS_PRUNING=0. Set a rank-indexed list, e.g. '[0,0,0,0.16,0.165,0.17]'."
  elif (( PRUNE_THR_ALLZERO )); then
    guard "EXPERTS_PRUNING_THRESHOLD is all zeros: weak_mask is never true, so this run prunes
       NOTHING and is identical to EXPERTS_PRUNING=0."
  fi
fi

# ── anchor-union activation routing (upstream) ───────────────────────────────
[[ "${ACT_ROUTE}" =~ ^[01]$ ]] || die "ACT_ROUTE must be 0 or 1"
[[ "${ACT_ROUTE_BACKEND}" =~ ^(anchor_union_fused_native|anchor_union_reference)$ ]] \
  || die "ACT_ROUTE_BACKEND must be anchor_union_fused_native or anchor_union_reference
       (baseline routing is ACT_ROUTE=0; 'anchor_union_topm' passes AscendConfig validation and is
       then REJECTED by DFlashTopMState as an unvalidated prototype)"
VERIFY_BLOCK=$(( 1 + NUM_SPEC_TOKENS ))     # == the engine's uniform_decode_query_len under dspark
if [[ "${ACT_ROUTE}" == "1" ]]; then
  [[ "${MTP}" == "2" ]] || die "ACT_ROUTE=1 needs MTP=2 (dspark): validate_activation_routing accepts
       only dflash/dspark and raises at startup for anything else, MTP=0 and MTP=1 included."
  [[ -z "${ACT_ROUTE_VERIFY_BLOCK}" || "${ACT_ROUTE_VERIFY_BLOCK}" == "${VERIFY_BLOCK}" ]] \
    || die "ACT_ROUTE_VERIFY_BLOCK=${ACT_ROUTE_VERIFY_BLOCK} != 1+NUM_SPEC_TOKENS=${VERIFY_BLOCK}:
       validate_activation_routing forces verify_block_size to equal the target query length. Leave
       it empty and the engine derives it."
  (( VERIFY_BLOCK >= 2 )) || die "ACT_ROUTE=1 needs verify_block_size >= 2, i.e. NUM_SPEC_TOKENS >= 1"
  if [[ -n "${ACT_ROUTE_PROTECTED_ROWS}" ]]; then
    [[ "${ACT_ROUTE_PROTECTED_ROWS}" =~ ^[1-9][0-9]*$ ]] \
      || die "ACT_ROUTE_PROTECTED_ROWS must be a positive integer"
    (( ACT_ROUTE_PROTECTED_ROWS <= VERIFY_BLOCK )) \
      || die "ACT_ROUTE_PROTECTED_ROWS=${ACT_ROUTE_PROTECTED_ROWS} > verify_block_size=${VERIFY_BLOCK}:
       DFlashTopMState requires protected_rows in [1, verify_block_size] and raises at startup."
  else
    (( VERIFY_BLOCK >= 3 )) || die "the engine default protected_rows=3 exceeds the derived
       verify_block_size=${VERIFY_BLOCK}, and DFlashTopMState raises at startup. Set
       ACT_ROUTE_PROTECTED_ROWS <= ${VERIFY_BLOCK}, or raise NUM_SPEC_TOKENS."
  fi
  if [[ -n "${ACT_ROUTE_SUFFIX_POOL_TOP_K}" ]]; then
    [[ "${ACT_ROUTE_SUFFIX_POOL_TOP_K}" =~ ^[1-9][0-9]*$ ]] \
      || die "ACT_ROUTE_SUFFIX_POOL_TOP_K must be a positive integer"
    (( ACT_ROUTE_SUFFIX_POOL_TOP_K <= TOPK )) \
      || die "ACT_ROUTE_SUFFIX_POOL_TOP_K=${ACT_ROUTE_SUFFIX_POOL_TOP_K} > top_k=${TOPK}:
       DFlashTopMState requires suffix_pool_top_k in [1, route_top_k]."
  fi
  [[ -z "${ACT_ROUTE_FUSED_ROWS}" || "${ACT_ROUTE_FUSED_ROWS}" == \[*\] ]] \
    || die "ACT_ROUTE_FUSED_ROWS must be a JSON list of positive ints, e.g. '[1,2,3,4,5,6,512]'"
fi
# A gate fine-tuned on the QuaRot build is silently wrong on any other basis;
# only the w4a8 build declares optional.quarot, so that file is the tell.
if [[ -n "${REMOE_GATE}" ]]; then
  MODEL_ROT="$(python3 - "${MODEL}/quant_model_description.json" 2>/dev/null <<'PY'
import json, sys
try: d = json.load(open(sys.argv[1]))
except Exception: print("absent"); sys.exit(0)
o = d.get("optional")
print(",".join(sorted(map(str, o))) if isinstance(o, dict) and o else "none")
PY
)"
  [[ "${MODEL_ROT}" == quarot ]] || guard "REMOE_GATE set but the model declares
       rotation='${MODEL_ROT:-absent}', not 'quarot'. Wrong basis = no crash, no EOS, every prompt
       runs to max_out_len. Serve the gate on the build it was fine-tuned from."
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

# --num-prompts keeps the first N of EACH sub-dataset a config expands into, so
# the budgets above are totals and this is the divisor. Counted from the config
# (mmengine loads it lazily). Sets P_PARTS (>= 1) and P_PARTS_OK (0 = count failed).
ds_parts() {   # $1=dataset config name
  local f n=""
  if [[ -z "${DS_PARTS[$1]:-}" ]]; then
    f="$(find "${PKG_CFG:-/nonexistent}/datasets" -name "$1.py" -print -quit 2>/dev/null)"
    [[ -n "${f}" ]] && n="$(python3 - "${f}" 2>/dev/null <<'PY'
import sys
from mmengine.config import Config
c = Config.fromfile(sys.argv[1])
print(sum(len(v) for k, v in c.items() if k.endswith('_datasets') and isinstance(v, list)))
PY
)"
    [[ "${n}" =~ ^[1-9][0-9]*$ ]] || n=0
    DS_PARTS[$1]="${n}"
  fi
  P_PARTS="${DS_PARTS[$1]}"; P_PARTS_OK=1
  (( P_PARTS == 0 )) && { P_PARTS=1; P_PARTS_OK=0; }
  return 0
}
per_part() { local n="$1"; [[ -z "${n}" ]] && return 0
             n=$(( n / P_PARTS )); (( n < 1 )) && n=1; echo "${n}"; }

# Sets P_DS P_TEMP P_TOPP P_MAXOUT P_NUM P_AVGN P_GET P_REF P_CLAMP P_PERF
# P_PARTS P_ACC. P_NUM / P_PERF are PER SUB-DATASET (what --num-prompts takes);
# P_ACC describes the resulting acc total.
preset() {
  local line
  line=$(awk -F'|' -v t="$1" '
    { for (i=1;i<=NF;i++) gsub(/^[ \t]+|[ \t]+$/, "", $i) }
    $1 == t { print $2"|"$3"|"$4"|"$5"|"$6"|"$7"|"$8"|"$9; f=1 } END { exit !f }' <<<"${PRESETS}") \
    || die "unknown task '$1' — valid: $(awk -F'|' 'NF>1{gsub(/ /,"",$1);printf "%s ",$1}' <<<"${PRESETS}")"
  IFS='|' read -r P_DS P_TEMP P_TOPP P_MAXOUT P_NUM P_AVGN P_GET P_REF <<<"${line}"
  [[ -n "${NUM_PROMPTS}" ]] && { P_NUM="${NUM_PROMPTS}"; [[ "${P_NUM}" == 0 ]] && P_NUM=""; }
  ds_parts "${P_DS}"
  P_NUM="$(per_part "${P_NUM}")"; P_PERF="$(per_part "${PERF_PROMPTS}")"
  P_ACC="full"; [[ -n "${P_NUM}" ]] && P_ACC="$(( P_NUM * P_PARTS ))"
  (( P_PARTS > 1 )) && P_ACC+=" (${P_NUM:-all} x ${P_PARTS} sub-datasets)"
  (( P_PARTS_OK )) || P_ACC+=" (sub-dataset count UNKNOWN, taken as 1)"
  [[ -n "${AVG_N}" ]] && P_AVGN="${AVG_N}"
  [[ -n "${MAX_OUTPUT_LEN}" ]] && P_MAXOUT="${MAX_OUTPUT_LEN}"
  P_CLAMP=""
  [[ -n "${OUT_CAP}" ]] && (( P_MAXOUT > OUT_CAP )) && \
    { P_CLAMP=" (clamped from ${P_MAXOUT} to fit ctx=${CTX})"; P_MAXOUT="${OUT_CAP}"; }
  return 0
}

# AISBench resolves --datasets by WALKING <config-dir>/datasets, and os.walk does
# not follow symlinked sub-directories, so a dataset config has to sit inside the
# package's own tree. mrcr ships only mrcr_1m_gen (all bins, up to 1M tokens =
# rejected at this context), so the bin-limited variant this script's preset names
# is written there once, next to it. Nothing is overwritten.
ensure_dataset_cfg() {
  local dir f shipped
  [[ " ${TASK_LIST} " == *" mrcr "* ]] || return 0
  # Match by FILE name, anywhere under datasets/: the folder holding the mrcr
  # configs is named mrcr in some releases and MRCR in others, and AISBench's own
  # resolver walks the tree by file name, so the folder name does not matter.
  f="$(find "${PKG_CFG:-/nonexistent}/datasets" -name 'mrcr_32k_gen.py' -print -quit 2>/dev/null)"
  [[ -n "${f}" ]] && return 0
  # Write it beside the config AISBench ships, whatever that folder is called.
  shipped="$(find "${PKG_CFG:-/nonexistent}/datasets" -name 'mrcr_1m_gen.py' -print -quit 2>/dev/null)"
  [[ -n "${shipped}" ]] && dir="$(dirname "${shipped}")" || dir="${PKG_CFG}/datasets/mrcr"
  f="${dir}/mrcr_32k_gen.py"
  [[ "${DRY_RUN}" == "1" ]] && { echo "  note    : mrcr config ${f} is missing; a real run writes it"; return 0; }
  # The config this writes imports the mrcr loader, so a failure here would
  # otherwise surface later as a config-import error. Report what python said:
  # a missing ais_bench...datasets.mrcr means the install predates MRCR support,
  # anything else names a package to install.
  local err
  if ! err="$(python3 -c "import ais_bench.benchmark.datasets.mrcr" 2>&1)"; then
    die "the mrcr dataset module will not import, so the mrcr preset cannot run here:
         $(tail -1 <<<"${err}")
       If that names ais_bench.benchmark.datasets.mrcr, this ais_bench predates MRCR support (it
       ships configs/datasets/*/mrcr_1m_gen.py and datasets/mrcr.py) — update the install. If it
       names any other module, pip install that one."
  fi
  mkdir -p "${dir}" 2>/dev/null
  [[ -d "${dir}" && -w "${dir}" ]] || die "cannot write ${f}: ${dir} is missing or not writable.
       Create the file by hand, or run as a user that can write to the ais_bench install."
  cat > "${f}" <<'MRCRCFG'
# GENERATED by v4_target_aisbench.sh — MRCR 8needle, 32k bin only.
# The shipped mrcr_1m_gen uses length_bin=None (all bins, up to 1M tokens), which
# no request can satisfy at max_model_len=70000. This slice keeps prompt+output
# inside the served context. Delete it to regenerate.
from ais_bench.benchmark.datasets.mrcr import (
    MRCRDataset, MRCREvaluator, MRCRPromptTemplate,
)
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever

mrcr_32k_datasets = [
    dict(
        abbr='mrcr_32k',
        type=MRCRDataset,
        path='ais_bench/datasets/MRCR',
        subset='8needle',
        length_bin='32k',
        reader_cfg=dict(input_columns=['prompt'], output_column='answer'),
        infer_cfg=dict(
            prompt_template=dict(type=MRCRPromptTemplate),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        ),
        eval_cfg=dict(
            evaluator=dict(type=MRCREvaluator),
            pred_postprocessor=dict(type='mrcr_postprocess'),
        ),
    )
]
MRCRCFG
  echo "[cfg  ] wrote ${f} (mrcr 8needle, 32k bin)"
  return 0
}

# --config-dir REPLACES ais_bench's config root, so without these links every
# pass dies at config resolution. SYMLINKS, not copies (read_base() resolves at
# the real path only); models/ stays ours.
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

# NOT curl — it dies in the loader here (FINDINGS). Proxies off explicitly.
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

# ── process bookkeeping (FINDINGS: ^C must not abandon the cleanup) ───────────
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
# Armed only when the server is CONFIRMED to be in its own session, so it can
# never name this script. Zombies are skipped: impossible to signal.
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

# 1. TERM the EngineCore ALONE (it writes [EXPERT-OFFLOAD-FINAL]), wait ENGINE_GRACE.
# 2. TERM the group and the tree, wait REAP_WAIT.  3. KILL group, tree, session,
# then anything still holding the port. Idempotent and NOT interruptible.
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

# Both upstream features announce themselves from inside the paging host
# callback / the routing hook, i.e. only once a DECODE forward has run — so
# neither can be checked at /health time the way the prefetch keys are. Called
# from report_stats, which the traps also reach.
report_features() {
  local _pn
  if [[ "${EXPERTS_PRUNING}" == "1" ]]; then
    grep -o '\[EXPERT-PRUNE\] entered NPU pruning path:.*$' "${SERVE_LOG}" 2>/dev/null \
      | sort -u | sed 's/^/[prune]  /'
    grep -q '\[EXPERT-PRUNE\] entered NPU pruning path' "${SERVE_LOG}" 2>/dev/null || \
      echo "[prune]  WARNING: no '[EXPERT-PRUNE] entered NPU pruning path' — pruning NEVER RAN. The
         key was swallowed, OFFLOAD was off, or no decode step reached the paging path."
    if [[ "${EXPERTS_PRUNING_DEBUG}" == "1" ]]; then
      _pn="$(grep -c '\[EXPERT-PRUNE-LAYER-JSON\]' "${SERVE_LOG}" 2>/dev/null)"
      echo "[prune]  [EXPERT-PRUNE-LAYER-JSON] records: ${_pn:-0}"
      # pruned_routes counts POSITIONS; saved_h2d_count counts experts that
      # actually left the demand set. prune_mask has no per-source atomicity, so
      # a pruned position whose expert survives in another MoE row still pays
      # its H2D — this ratio is how much of the pruning bought a transfer.
      python3 - "${SERVE_LOG}" 2>/dev/null <<'PY' | sed 's/^/[prune]  /'
import json, sys
tag = "[EXPERT-PRUNE-LAYER-JSON] "
pruned = saved = miss = rows = 0
for line in open(sys.argv[1], errors="ignore"):
    if tag not in line:
        continue
    try:
        d = json.loads(line.split(tag, 1)[1].strip())
    except Exception:
        continue
    pruned += d.get("pruned_routes", 0)
    saved += d.get("saved_h2d_count", 0)
    miss += d.get("miss_routes", 0)
    rows += 1
if rows:
    ratio = f"{saved / pruned:.3f}" if pruned else "n/a (nothing pruned)"
    print(f"layer-records={rows} miss_routes={miss} pruned_routes={pruned} "
          f"saved_h2d_count={saved} saved/pruned={ratio}")
    if pruned and saved / pruned < 0.8:
        print("saved/pruned < 0.8: those pruned positions whose expert is still "
              "paged in for another MoE row lost mixture mass and saved no transfer.")
PY
    fi
  fi
  if [[ "${ACT_ROUTE}" == "1" ]]; then
    grep -o '\[ACTIVATION-ROUTING\].*$' "${SERVE_LOG}" 2>/dev/null | sort -u | sed 's/^/[actrt]  /'
    if ! grep -q '\[ACTIVATION-ROUTING\].*eligible=True' "${SERVE_LOG}" 2>/dev/null; then
      echo "[actrt]  WARNING: no contract line with eligible=True — try_activation_route rejected every
         forward and routing stayed BASELINE. The contract line above names the fields it saw."
    elif [[ "${ACT_ROUTE_BACKEND}" == anchor_union_fused_native ]] \
         && ! grep -q '\[ACTIVATION-ROUTING\].*fused=True' "${SERVE_LOG}" 2>/dev/null; then
      echo "[actrt]  WARNING: eligible but fused=False everywhere — supports_fused_route() rejected the
         row count (not in fused_rows) or the dspark contract, so every layer took the slower
         reference backend. Routing is correct; the throughput number is not the fused one."
    fi
  fi
  return 0
}

# Called from the happy path AND the traps, so an interrupted run still surfaces
# its artefacts. Idempotent.
report_stats() {
  (( STATS_REPORTED )) && return 0
  STATS_REPORTED=1
  report_features          # both upstream features only log once decode has run
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
    grep -q 'decode steps=0 ' "${_f}" && echo "[!!]     decode steps=0 — armed but never reached the
         paging decode path. Not a result."
    sed 's/^/  /' "${_f}"
  else
    echo "[stats]  WARNING: no summary under ${STATS_DIR}. Traces above: config=0 keys swallowed,
         topology=0 never registered, armed=0 hook missing, first sample=0 decode never paged."
  fi
  grep -q '\[EXPERT-OFFLOAD-FINAL\]' "${SERVE_LOG}" 2>/dev/null && {
    echo "[stats]  [EXPERT-OFFLOAD-FINAL] from serve.log:"
    awk '/\[EXPERT-OFFLOAD-FINAL\]/{f=1} f{print} f&&/^=+$/{r++; if(r>=2) exit}' \
      "${SERVE_LOG}" | head -60 | sed 's/^/  /'; }
  grep -q 'host callback failed' "${SERVE_LOG}" 2>/dev/null && \
    echo "[stats]  WARNING: expert-offload host callback failures — paging did not complete"
  # The stall timer latches off rather than raising when the device cannot
  # timestamp inside a captured graph; the rows would otherwise just be absent.
  [[ "${EXPERT_PREFETCH_WAIT_TIMING}" == "1" ]] && \
    grep -q 'elapsed_time unavailable' "${SERVE_LOG}" 2>/dev/null && \
    echo "[stats]  WARNING: prefetch-stall timing was DISABLED at runtime — pf_wait rows are empty."
  return 0
}

on_signal() {
  trap '' PIPE           # a dead tee must not kill this shell before reap() runs
  echo >&2; echo "[signal] caught SIG${1} — stopping the server. Further ^C is IGNORED until" \
                 "the tree is down." >&2
  # Clamped, not zeroed: 20s still lets the EngineCore write its summary.
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
    # Stated explicitly, including thinking=False, so a run can never silently
    # sit in the wrong mode.
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
    CMD+=( --mode perf --summarizer default_perf ); n="${P_PERF}"   # --merge-ds is acc-only
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
  [[ "${got}" == "${want}" ]] || die "server is not honouring chat_template_kwargs. The template may
       use a different key (edit build_cmd: thinking -> enable_thinking). PROBE=0 to override."
  return 0
}

# ── build the serve argv ──────────────────────────────────────────────────────
# expert_offload_config is a CLOSED key set: an unknown key raises at startup.
# TOP-LEVEL keys use .get(), so a typo there is ignored SILENTLY — hence the
# post-health greps. Keys are emitted only when non-default, so an untouched knob
# keeps the engine's default and old runs stay comparable.

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
  # Pruning: the threshold is the whole feature — with the engine default of
  # [0,...] the enable flag alone prunes nothing (guarded above).
  if [[ "${EXPERTS_PRUNING}" == "1" ]]; then
    p+=",\"experts_pruning_enabled\":true"
    [[ -n "${EXPERTS_PRUNING_THRESHOLD}" ]] && \
      p+=",\"experts_pruning_threshold\":${EXPERTS_PRUNING_THRESHOLD}"
    [[ "${EXPERTS_PRUNING_DEBUG}" == "1" ]] && p+=",\"experts_pruning_debug\":true"
  fi
  [[ "${ENABLE_MULTI_CARD}" == "1" ]] && p+=",\"enable_multi_card\":true"
  p+=",\"moe_offload_debug\":false"     # per-layer host-callback trace: invalid for timing
  ADDL_PARTS+=( "\"expert_offload_config\":{${p}}" )
fi
[[ -n "${REMOE_GATE}" ]] && ADDL_PARTS+=( "\"moe_gate_override_path\":\"${REMOE_GATE}\"" )
# activation_routing is a SEPARATE top-level section with its own CLOSED key set
# (ActivationRoutingConfig raises on an unknown key). Only `enabled` + `backend`
# are emitted by default: every other field is left unwritten so the engine's
# dspark derivation fills it, which is what the PR means by "needs no hand-tuned
# values in the common case".
if [[ "${ACT_ROUTE}" == "1" ]]; then
  a="\"enabled\":true,\"backend\":\"${ACT_ROUTE_BACKEND}\""
  [[ -n "${ACT_ROUTE_VERIFY_BLOCK}" ]]      && a+=",\"verify_block_size\":${ACT_ROUTE_VERIFY_BLOCK}"
  [[ -n "${ACT_ROUTE_PROTECTED_ROWS}" ]]    && a+=",\"protected_rows\":${ACT_ROUTE_PROTECTED_ROWS}"
  [[ -n "${ACT_ROUTE_SUFFIX_POOL_TOP_K}" ]] && a+=",\"suffix_pool_top_k\":${ACT_ROUTE_SUFFIX_POOL_TOP_K}"
  [[ -n "${ACT_ROUTE_FUSED_ROWS}" ]]        && a+=",\"fused_rows\":${ACT_ROUTE_FUSED_ROWS}"
  ADDL_PARTS+=( "\"activation_routing\":{${a}}" )
fi
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
# validate_activation_routing raises on use_async_scheduling, so the flag is
# forced rather than left to the build's default. It is a SECOND changed
# variable in the A/B — see the note printed with the plan.
[[ "${ACT_ROUTE}" == "1" ]] && SERVE+=( --no-async-scheduling )
if (( GRAPH_ON )); then
  SERVE+=( --compilation-config \
    "{\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\",\"cudagraph_capture_sizes\":[${CAPTURE_SIZES}]}" )
else
  SERVE+=( --enforce-eager )
fi
[[ -n "${SPEC_CFG}" ]] && SERVE+=( --speculative-config "${SPEC_CFG}" )
SERVE+=( --additional-config "${ADDL}" )

ensure_dataset_cfg        # the mrcr preset needs a config the package does not ship

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
  # script's next write would take SIGPIPE and die mid-reap, log gone too.
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
echo "  prune   : ${EXPERTS_PRUNING} thr=${EXPERTS_PRUNING_THRESHOLD:-engine default [0,..] = prunes nothing} debug=${EXPERTS_PRUNING_DEBUG}"
if [[ "${ACT_ROUTE}" == "1" ]]; then
  echo "  actroute: 1 backend=${ACT_ROUTE_BACKEND} verify_block=${ACT_ROUTE_VERIFY_BLOCK:-derived(${VERIFY_BLOCK})} protected=${ACT_ROUTE_PROTECTED_ROWS:-default(3)} suffix_pool=${ACT_ROUTE_SUFFIX_POOL_TOP_K:-default(2)} fused_rows=${ACT_ROUTE_FUSED_ROWS:-derived([${VERIFY_BLOCK},512])}"
else
  echo "  actroute: 0 (baseline routing — every verify-block row keeps its own top-k)"
fi
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
  (( GRAPH_ON )) || echo "  warn    : in EAGER prefetch overlaps nothing, so pf_wait reads near zero
            for reasons unrelated to its cost. Only meaningful under graph capture."; }
[[ "${EXPERT_SUBSTITUTION}" == "1" ]] && \
  echo "  note    : substitution ON — it CHANGES ROUTING. Not comparable to a run without it."
[[ "${EXPERTS_PRUNING}" == "1" ]] && {
  echo "  note    : pruning ON — it CHANGES ROUTING and DESTROYS mixture weight (a pruned route's
            weight becomes 0 with no renormalisation). It also changes what the decode statistics
            MEAN, not just their value: |G| becomes post-prune, so hit_pre/hit_post rise because
            pruning deletes misses and only misses."
  [[ "${EXPERTS_PRUNING_DEBUG}" == "1" ]] && \
    echo "  warn    : EXPERTS_PRUNING_DEBUG=1 logs one JSON record per MoE layer per decode step from
            inside the paging host callback — serve.log grows fast and TPOT here is NOT a baseline."; }
[[ "${ACT_ROUTE}" == "1" ]] && {
  echo "  note    : activation routing ON — suffix rows of each verify block are masked to the block's
            expert pool, so it CHANGES ROUTING. It also moves the expert predictor's target: the head
            was trained against the unconstrained top-k, so pf_waste rises and pred_prec falls without
            the head changing."
  echo "  note    : --no-async-scheduling is FORCED by this feature, so it is a SECOND changed variable.
            If async scheduling is on by default in this build, take the ACT_ROUTE=0 baseline with
            --no-async-scheduling too, or the comparison carries both changes."
  [[ "${ACT_ROUTE_BACKEND}" == anchor_union_reference ]] && \
    echo "  warn    : anchor_union_reference is the DEBUG backend — byte-identical routing, much slower.
            Accuracy parity only, never a throughput number."; }
[[ "${EXPERTS_PRUNING}" == "1" && "${ACT_ROUTE}" == "1" ]] && \
  echo "  note    : both upstream features ON — their effects are SUB-ADDITIVE (activation routing
            shrinks the activated set, which leaves pruning fewer non-resident routes to drop). Take
            the three single-feature runs before reading the combination."

if [[ "${OFFLOAD}" == "1" ]]; then
  echo "  thresh  : offload_threshold=${THR} MoE rows (${NDE_MIN}/${TOPK}); floor = moe_rows*topk = ${NDE_FLOOR}"
  # Above the threshold decode leaves the LRC path entirely: the run completes
  # and measures nothing.
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
    # MC2 is admitted only when num_tokens*topk <= one rank's slots. Miss it and
    # every step goes ALLTOALL -> shard prefill pool -> nothing recorded.
    _slots=$(( NDE_MIN / EP_SIZE ))
    (( DECODE_TOKENS * TOPK > _slots )) && guard "multi-card decode never reaches MC2:
       decode_tokens*topk=$(( DECODE_TOKENS * TOPK )) > per-rank admission slots=${_slots}."
    echo "  mc      : admission slots=${_slots}/rank; multi-card records NO per-layer cache stats
            and EXPERT_PREFETCH_NUM does not reach it."
  fi
  # _prefill_load_layer runs in the outer Python of update_weights: once at
  # capture, never on replay — so a captured size above the bound replays against
  # stale pool weights.
  if (( GRAPH_ON )); then
    _unsafe=""; for _s in ${CAPTURE_SIZES//,/ }; do (( _s * DP > THR )) && _unsafe+="${_s} "; done
    [[ -n "${_unsafe}" ]] && echo "  warn    : capture size(s) ${_unsafe%% } exceed the safe bound
            (size x dp > ${THR}) — those graphs replay against stale prefill-pool weights."
  fi
fi

# supports_fused_route() takes the fused kernel ONLY when the forward's row count
# is literally in fused_rows; any other size silently takes the reference
# backend. The derived list is [verify_block, 512], and a steady-state uniform
# decode always pads to verify_block, so the smaller captures are the ones to
# know about.
if [[ "${ACT_ROUTE}" == "1" && "${ACT_ROUTE_BACKEND}" == anchor_union_fused_native ]]; then
  _fr="${ACT_ROUTE_FUSED_ROWS:-[${VERIFY_BLOCK},512]}"
  echo "  fused   : fused_rows=${_fr}; the fused kernel runs only at those exact row counts"
  if (( GRAPH_ON )); then
    _nofuse=""
    for _s in ${CAPTURE_SIZES//,/ }; do
      [[ ",$(tr -d '[] ' <<<"${_fr}")," == *",${_s},"* ]] || _nofuse+="${_s} "
    done
    [[ -n "${_nofuse}" ]] && echo "  warn    : capture size(s) ${_nofuse%% } are not in fused_rows — those
            replays take the slower reference backend. Harmless when decode always pads to
            ${VERIFY_BLOCK}; widen ACT_ROUTE_FUSED_ROWS if a smaller size is actually replayed."
  fi
  # the fused path's dspark contract, which route() falls back from silently.
  _nre="$(cfg n_routed_experts)"; _sf="$(cfg scoring_func)"
  [[ "${_nre}" == 256 && "${TOPK}" == 6 && "${_sf}" == sqrtsoftplus ]] \
    || echo "  warn    : cannot confirm the fused dspark contract (wants n_routed_experts=256 top_k=6
            scoring_func=sqrtsoftplus; config.json reads ${_nre:-?}/${TOPK}/${_sf:-?}).
            supports_fused_route() would return False and every layer takes the reference backend —
            the post-run [ACTIVATION-ROUTING] line settles it with fused=True/False."
fi

for t in ${TASK_LIST}; do
  preset "${t}"
  echo "     - ${t}  ${P_DS}  temp=${P_TEMP} max_out=${P_MAXOUT}${P_CLAMP} acc_prompts=${P_ACC} perf_prompts=$(( P_PERF * P_PARTS )) avg@n=${P_AVGN}  published(non-think): ${P_REF}"
done
echo "  out     : ${OUT_DIR}"
echo "  hf_home : ${HF_HOME}  (lcb builds its dataset cache here, not in ${DATASETS_CACHE})"
echo "  repro   : $(for v in CARD PORT DP MTP NUM_SPEC_TOKENS OFFLOAD NUM_DEVICE_EXPERTS NUM_DEVICE_LAYERS \
      ENABLE_MULTI_CARD PREFETCH EXPERT_PREFETCH_NUM EXPERT_PREFETCH_TOKENS EXPERT_PREFETCH_WAIT_TIMING \
      EXPERT_PREDICTOR EXPERT_PREDICTOR_CKPT EXPERT_SUBSTITUTION EXPERT_SUBSTITUTION_THRESHOLD REMOE_GATE \
      EXPERTS_PRUNING EXPERTS_PRUNING_THRESHOLD EXPERTS_PRUNING_DEBUG \
      ACT_ROUTE ACT_ROUTE_BACKEND ACT_ROUTE_VERIFY_BLOCK ACT_ROUTE_PROTECTED_ROWS \
      ACT_ROUTE_SUFFIX_POOL_TOP_K ACT_ROUTE_FUSED_ROWS \
      ENFORCE_EAGER MAX_MODEL_LEN MAX_OUTPUT_LEN DECODE_STATS CSV TASKS RUN THINK THINK_EFFORT NUM_PROMPTS PERF_PROMPTS AVG_N MODEL
   do printf '%s=%q ' "${v}" "${!v-}"; done)$0"
echo "────────────────────────────────────────────────────────────"
printf '  $ '; printf '%q ' "${SERVE[@]}"; echo
for step in "${PLAN[@]}"; do build_cmd "${step%:*}" "${step#*:}"; printf '  $ '; printf '%q ' "${CMD[@]}"; echo; done
[[ "${DRY_RUN}" == "1" ]] && { echo "[DRY_RUN] nothing launched."; exit 0; }

# ── preflight — everything that can fail in seconds, before a 4-minute load ───

# Without vllm-ascend, DeviceConfig raises while BUILDING THE ARG PARSER, before
# any flag here is read. Name the cause.
python3 -c "
import sys
from importlib.metadata import entry_points
if not list(entry_points(group='vllm.platform_plugins')): sys.exit(1)
for m in ('vllm_ascend','torch_npu'): __import__(m)
" 2>/dev/null || die "vllm-ascend / torch_npu will not import, or no vllm.platform_plugins entry
       point is registered. Did you 'source env_a5_0260.sh' in this shell?"
command -v ais_bench >/dev/null 2>&1 || die "ais_bench not on PATH"
[[ -n "${PKG_CFG}" && -d "${PKG_CFG}" ]] || die "cannot locate the ais_bench configs dir"

# Imports that only happen AFTER every request is sent, so a miss costs a whole
# pass: the perf summarizer's tokenizer, and math500's math_verify scorer.
[[ "${RUN}" == acc ]] || python3 -c "
from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('${TOKENIZER}', trust_remote_code=True)" \
  2>/dev/null || die "the perf summary would fail on TOKENIZER='${TOKENIZER}'"
[[ "${RUN}" == perf || " ${TASK_LIST} " != *" math500 "* ]] || python3 -c "
import math_verify, latex2sympy2_extended" 2>/dev/null \
  || die "math500 is scored by math_verify, which will not import: pip install math-verify==0.5.2"

# lcb loads through the repo's own loading script, which datasets 4.x removed —
# in every mode, so this is not exempt for perf.
[[ " ${TASK_LIST} " != *" lcb "* ]] || python3 -c "
import datasets, sys; sys.exit(int(datasets.__version__.split('.')[0]) >= 4)" 2>/dev/null \
  || die "lcb needs a datasets release that still runs a repo loading script: pip install 'datasets<=3.6.0'"

# mrcr's bin filter needs the o200k_base ranks file, which tiktoken downloads at
# DATASET-LOAD time — i.e. after the model is up, and through whatever proxy is
# set. Fetch it here instead: this both primes the cache while a network exists
# and fails in seconds when it does not.
[[ " ${TASK_LIST} " != *" mrcr "* ]] || python3 -c "
import tiktoken; tiktoken.get_encoding('o200k_base')" 2>/dev/null \
  || guard "tiktoken cannot load o200k_base, and mrcr's bin filter needs it. Seed the cache once,
       from any host that can reach the internet, then copy it to ${TIKTOKEN_CACHE_DIR}:
         TIKTOKEN_CACHE_DIR=<dir> python3 -c \"import tiktoken; tiktoken.get_encoding('o200k_base')\"
       or download the file and name it by the sha1 of its URL:
         wget -O <dir>/fb374d419588a4632f3f557e76b4b70aebbca790 \\
           https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
[[ " ${TASK_LIST} " != *" mrcr "* ]] || echo "[tik  ] o200k_base ready (cache ${TIKTOKEN_CACHE_DIR})"

# A full filesystem under the dataset build cache or the results dir fails LATE
# and vaguely (FINDINGS). lcb builds an Arrow copy and needs the most.
free_gib() { df -Pk "$1" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1048576}'; }
_need=1; [[ " ${TASK_LIST} " == *" lcb "* ]] && _need=5
for _d in "${HF_HOME}" "${OUT_DIR}"; do
  mkdir -p "${_d}" 2>/dev/null
  _free="$(free_gib "${_d}")"
  [[ "${_free}" =~ ^[0-9]+$ ]] || { echo "[space] cannot read free space for ${_d}" >&2; continue; }
  echo "[space] ${_free} GiB free for ${_d} (want >= ${_need})"
  (( _free < _need )) && guard "only ${_free} GiB free on the fs holding ${_d}: a dataset build there
       reports 'Not enough disk space. Needed: Unknown size'. Free space, or move HF_HOME / OUT_DIR."
done

# AIS_BENCH_DATASETS_CACHE is a ROOT that AISBench appends the config's own
# relative path to. That path may be quoted either way, may name a FILE
# (aime2025) or a directory plus a separate file_name= (math500), and may be an
# f-string with a {placeholder} per sub-dataset (mgsm) — expanded and COUNTED
# here, because a directory-level check once passed a clone with no .tsv in it.
_miss=0
for t in ${TASK_LIST}; do
  preset "${t}"
  _f="$(find "${PKG_CFG}/datasets" -name "${P_DS}.py" -print -quit 2>/dev/null)"
  [[ -n "${_f}" ]] || { echo "[!!] dataset config '${P_DS}' not installed" >&2; _miss=1; continue; }
  (( P_PARTS_OK )) || guard "cannot count the sub-datasets of ${P_DS} (Config.fromfile failed on
       ${_f}). --num-prompts applies to EACH of them, so the totals above may be multiplied."
  _rel="$(grep -oE "(^|[[:space:](,])path=f?['\"][^'\"]*['\"]" "${_f}" | head -1 | sed -E "s/^.*path=f?['\"]//;s/['\"]$//")"
  _fn="$(grep -oE "file_name=f?['\"][^'\"]*['\"]" "${_f}" | head -1 | sed -E "s/^file_name=f?['\"]//;s/['\"]$//")"
  [[ "${_rel}" == /* ]] && _abs="${_rel}" || _abs="${DATASETS_CACHE}/${_rel}"
  [[ -n "${_fn}" ]] && _abs="${_abs%/}/${_fn}"
  _zip="$(sed -E 's#^(.*/)?datasets/##;s#/.*$##' <<<"${_rel}")"   # opencompass zip = top data dir
  _have=0; _want=1; _what="${_abs}"
  if [[ "${_abs}" == *'{'* ]]; then
    _what="$(sed -E 's/\{[^}]*\}/*/g' <<<"${_abs}")"; _want="${P_PARTS}"
    for _g in ${_what}; do [[ -e "${_g}" ]] && (( _have++ )); done
  else
    # A directory that exists but holds no file is the mgsm failure again, so a
    # dir only counts when something is actually in it.
    [[ -n "${_rel}" && -e "${_abs}" ]] && _have=1
    [[ "${_have}" == 1 && -d "${_abs}" && -z "$(find "${_abs}" -type f -print -quit 2>/dev/null)" ]] && _have=0
  fi
  if (( _have >= _want )); then
    echo "[data ] ${t}: ${_what} (${_have}/${_want} present)"
  else
    echo "[!!] ${t}: ${_have}/${_want} data file(s) at ${_what:-?} (path= read from ${_f})" >&2
    echo "     mkdir -p '${DATASETS_CACHE}/ais_bench/datasets' && cd '${DATASETS_CACHE}/ais_bench/datasets' \\" >&2
    if [[ "${P_GET}" == hfcli:* ]]; then
      _repo="${P_GET#hfcli:}"; _sub="${_repo#*:}"; _repo="${_repo%%:*}"
      echo "       && hf download ${_repo} --repo-type dataset --include '8needle/*' \\
       --local-dir '${DATASETS_CACHE}/ais_bench/datasets/${_sub}'   # 'hf' replaces huggingface-cli" >&2
    elif [[ "${P_GET}" == hf:* ]]; then
      _repo="${P_GET#hf:}"; _rev=""
      [[ "${_repo}" == *@* ]] && { _rev="${_repo#*@}"; _repo="${_repo%@*}"; }
      echo "       && git lfs install && git clone https://huggingface.co/datasets/${_repo}${_rev:+ \\
       && git -C '${_repo##*/}' checkout ${_rev}   # main no longer carries the files the loader reads}" >&2
    else
      echo "       && wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/${_zip:-${t}}.zip && unzip ${_zip:-${t}}.zip" >&2
    fi
    _miss=1; continue
  fi
  # git-lfs pointers are a few hundred bytes of text where the data should be:
  # present, so every count passes, and empty of data.
  _dir="${_what}"; [[ -d "${_dir}" ]] || _dir="$(dirname "${_what}")"
  while IFS= read -r _p; do
    head -c 40 "${_p}" 2>/dev/null | grep -q 'git-lfs' && { guard "${t}: ${_p} is a git-lfs POINTER,
       not data — the clone ran without 'git lfs install'. Re-clone, or 'git -C \"${_dir}\" lfs pull'."
       break; }
  done < <(find "${_dir}" -maxdepth 1 -type f -size -1024c 2>/dev/null | head -20)
done
(( _miss )) && die "missing dataset config or data"

_pids="$(port_pids)"
if [[ "${SKIP_SERVE}" == "1" ]]; then
  echo "[port ] :${PORT} pids=${_pids:-none}"
elif (exec 4<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
  die "port ${PORT} already in use${_pids:+ by PID(s) ${_pids}}"
else
  # Prove the HTTP path works BEFORE a four-minute model load: the port is free,
  # so a real request must come back unreachable. Anything else means the health
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
  # necessarily got there when bash returns from `&`. One `ps` races it, blanks
  # SERVE_PGID and silently demotes reap to a tree walk. So poll.
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
      (( _since > HEALTH_GRACE )) && die "engine serving on :${PORT} but ${HEALTH_URL} silent for
       ${HEALTH_GRACE}s — connectivity, not load: check its 'Starting vLLM server on' line."
    fi
    (( i % 30 == 29 )) && echo "[serve] still waiting ($(( (i+1) * 2 ))s of ${WAIT}s) ..."
    sleep 2
  done
  (( ok )) || die "server not ready in ${WAIT}s (see ${SERVE_LOG})"
  echo "[serve] healthy"

  # The engine's own echoed lines turn a silently swallowed top-level key back
  # into a failure.
  [[ -n "${REMOE_GATE}" ]] && { grep -q '\[GATE-OVERRIDE\] applied' "${SERVE_LOG}" \
    || guard "no '[GATE-OVERRIDE] applied' in the log — REMOE_GATE was ignored and this run would
       measure the BASE router."; }
  if [[ "${EXPERT_PREDICTOR}" != fate ]]; then
    grep -o '\[PREFETCH-AI\].*$' "${SERVE_LOG}" | sort -u | sed 's/^/  predict : /'
    grep -q '\[PREFETCH-AI\] armed' "${SERVE_LOG}" || guard "no '[PREFETCH-AI] armed' in the log — the
       head did not load or no layer resolved a capture site, so every prefetch came from fate."
  fi
  [[ "${EXPERT_PREFETCH_WAIT_TIMING}" == "1" ]] && \
    { grep -q '\[PREFETCH-WAIT\] timing enabled' "${SERVE_LOG}" \
      || guard "no '[PREFETCH-WAIT] timing enabled' in the log — the key was not applied and this run
       measures no stall."; }
  # validate_activation_routing logs this from __init__ and raises on any
  # contract violation, so its ABSENCE on a healthy server means the section was
  # never resolved: missing, enabled=false, or a non-dspark drafter.
  if [[ "${ACT_ROUTE}" == "1" ]]; then
    grep -o 'Activation routing enabled:.*$' "${SERVE_LOG}" | sort -u | sed 's/^/  actroute: /'
    grep -q 'Activation routing enabled:' "${SERVE_LOG}" \
      || guard "no 'Activation routing enabled:' in the log — resolve_activation_routing_config
       returned None, so every routing hook stayed inert and this run is BASELINE routing."
  fi
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
  # AISBench's progress table needs a real TTY and is also the liveness signal,
  # so the heartbeat only runs when there is none.
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

# ── quiesce, shut down, then report — IN THAT ORDER (see FINDINGS) ───────────

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

# ── FINDINGS — each line is something that silently changed a number ────────
#
#  ^C MUST NOT ABANDON THE CLEANUP. setsid puts the server in its own session, so
#  the terminal's SIGINT never reaches it and this script's trap is all that stops
#  it. Four defects once conspired: no HUP trap; an interruptible reap(), so a 2nd
#  ^C killed the script between TERM and KILL; SERVE_PGID read by one `ps` racing
#  setsid(); and the tee in this script's group, so ^C killed it and the next write
#  took SIGPIPE. pgrep -P also cannot see a worker re-parented to init (hence
#  session_pids), and reap() must NEVER `pgrep -f EngineCore` host-wide — that once
#  killed a concurrent benchmark. Do not "simplify" reap(), its traps, or the tee.
#
#  ORDER IS QUIESCE -> ENGINE -> GROUP -> KILL -> REPORT. The collector's idle
#  snapshot needs an alive, idle server; [EXPERT-OFFLOAD-FINAL] needs the EngineCore
#  to get its own SIGTERM before the API server force-kills it at timeout=0s.
#
#  curl DOES NOT WORK HERE: env_a5_0260.sh puts ${CONDA_PREFIX}/lib first, so
#  /usr/bin/curl dies in the loader (exit 127) — indistinguishable from "not ready".
#
#  --config-dir REPLACES ais_bench's config root, it is not additive. Without the
#  symlinks ensure_dirs() makes, every pass dies at config resolution: a healthy
#  idle server and a benchmark that never starts.
#
#  MoE ROWS != DECODE TOKENS WHEN DP > 1: prepare() all-gathers, so the MoE layer
#  sees DP x decode_tokens rows, and that is what offload_threshold compares. Above
#  it every step takes the prefill pool — no paging, prefetch or statistics, and the
#  run looks normal. Pool floor = moe_rows*topk, so each spec token costs topk*dp
#  slots: dspark k=5 is 6 rows, floor 36 at topk=6 — exactly the default pool, with
#  nothing to spare. The same bound gates MC2 in multi-card, and a captured graph
#  above it replays stale pool weights (_prefill_load_layer runs once, at capture).
#
#  PREFETCH: _TOKENS is prediction width, _NUM the transfer budget (single-card
#  only), and the METHOD is per target layer — a head drives only its own layers,
#  fate keeps the rest, 40 layers contribute either way, so pf_loads_step/pf_loads
#  at 40.0000 proves coverage did not shift. A mismatched checkpoint fails silently
#  (in_dim = hc_mult*hidden_size; L = num_hidden_layers or minus num_hash_layers;
#  the basis half is undetectable). pf_wait is the ONLY cost measurement — device
#  events, so it works under replay, and a run with it on is not a TPOT baseline.
#  In EAGER the callback runs inline and synchronizes: PREFETCH=1 is SLOWER there.
#
#  READ avg@n ONLY: AISBench de-interleaves with the wrong stride, so pass@n and
#  cons@n come out as 1-(1-p)^n and p^n. Pass@1 IS avg@n. Keep the gpqa cot config:
#  the _str variant measured ~8 points lower because first_option_postprocess takes
#  the FIRST regex match, scoring a self-correction at its pre-correction letter.
#
#  THE SMALL SETS ARE FOR PAIRED A/B, NOT PUBLISHED PARITY. At 30-130 items one
#  question is 1-3 points: compare question by question (--dump-eval-details),
#  never two headline numbers. That is why they are greedy.
#
#  --num-prompts IS PER SUB-DATASET: test_range="[:N]" on every entry of the
#  config's *_datasets list (mmlu_pro 14 categories, mgsm 11 languages).
#  NUM_PROMPTS=20 once sent 280 prompts and PERF_PROMPTS=30 sent 420, so the script
#  treats both as totals and passes total // subsets (>= 1), counted from the config.
#
#  DATASET PATHS FAIL IN THREE SILENT WAYS. (1) "Not enough disk space. Needed:
#  Unknown size" = ZERO free bytes on the ROOT fs, where `datasets` builds lcb.
#  (2) A HF repo can stop carrying the files its loader reads — juletxara/mgsm was
#  converted to parquet, so a clone creates the dir, passes a directory check, and
#  dies at load. (3) git-lfs pointers are present, text, and empty of data. Paths
#  also vary: either quoting, an f-string {placeholder}, a FILE, or dir + file_name=.
#
#  A REMOE GATE ON THE WRONG BASIS DOES NOT CRASH: QuaRot's R1 is absorbed into
#  mlp.gate, so same shapes, same norms, every check passes — the model just stops
#  emitting EOS and every prompt runs to max_out_len.
#
#  EXPERT PRUNING IS ALL IN THE THRESHOLD. experts_pruning_enabled on its own does
#  nothing: the engine default experts_pruning_threshold is [0,...] and weak_mask is
#  `weight < row_sum * thr[rank]`, never true at 0 — so the flag alone gives a run
#  byte-identical to pruning off, with a healthy log and a plausible summary. The
#  list is RANK-INDEXED over top_k and its length must EQUAL top_k, or
#  dynamic_pruning_unsorted raises on the first decode forward, i.e. after the model
#  load and the health check. thr[0]=0 is what keeps the strongest expert unprunable.
#  Pruning also only prunes a route that is weak AND non-resident, and a prefetched
#  expert is already resident when it runs — so it removes only demand prefetch
#  MISSED, which is why hit_pre/hit_post/pred_acc all rise under it and are not
#  comparable with a non-pruning run. It is single-card only: multi-card never calls
#  it. And prune_mask has no per-source atomicity, so a position pruned in one MoE
#  row whose expert survives in another still pays its H2D — EXPERTS_PRUNING_DEBUG=1
#  prints pruned_routes and saved_h2d_count, and their ratio is how much of the
#  pruning actually bought a transfer.
#
#  ACTIVATION ROUTING HAS THREE SILENT FALLBACKS, ALL OF THEM "HEALTHY". (1) The
#  section is a CLOSED key set and needs BOTH enabled:true and a non-baseline
#  backend; miss either and resolve_activation_routing_config returns None and every
#  hook is inert. (2) try_activation_route's eligibility gate is long (roles present,
#  row count match, num_experts/top_k/scoring_func match, renormalize, no tid2eid,
#  no mix_placement, topk_group and num_expert_group in (None,1)) and failing ANY
#  field just returns None — baseline routing, no error. (3) supports_fused_route
#  needs the row count LITERALLY in fused_rows plus the full dspark contract (fp32
#  [N,256], top_k 6, sqrtsoftplus, scale 1.5, bias present); miss it and the slower
#  reference backend runs instead, with identical output. The one line that settles
#  all three is [ACTIVATION-ROUTING] ... eligible=<bool> fused=<bool>, which this
#  script prints after the run. Leave verify_block_size / fused_rows /
#  expected_router_layers EMPTY: writing a field lands it in user_keys and SUPPRESSES
#  the engine's derivation for it, even when the value equals the default. It also
#  forces --no-async-scheduling, so an ACT_ROUTE A/B changes TWO variables unless the
#  baseline carries that flag too.
#
#  PADDED ROWS ARE STILL FULLY ROUTED under activation routing: prepare() sizes the
#  roles buffer to num_tokens_padded and writes roles only inside the scheduled
#  verify segments, so padding keeps role 0 — which keeps it out of the candidate
#  pool but does NOT mask it, and the single-card paging path has no pad filter. As
#  activation routing bites, padding's share of |G| and of `loads` grows.
#
#  ALSO SILENT: multi-card records no per-layer cache statistics; draft MoE layers
#  are excluded from offload by a layer-NAME test; ais_bench exits 0 when the engine
#  is dead, because HTTP 500s are responses; and a chat_template kwarg the template
#  does not declare is dropped without a word — which is what probe_mode() is for.
# ─────────────────────────────────────────────────────────────────────────────