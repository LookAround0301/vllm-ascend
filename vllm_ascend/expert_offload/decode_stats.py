"""Decode-time statistics: MoE expert cache/prefetch accounting and
speculative-decoding acceptance, reduced to one end-of-run summary.

Design constraints this file is shaped by:

* The per-layer numbers are produced inside a ``_launch_host_func`` callback,
  which under ACL-graph replay re-executes with the argument tuple that was
  baked at capture time. Nothing per-step can be passed in as an argument, so
  the step boundary is *derived* here, from a layer being revisited.
* The callback may run on a report thread (one per subscribed stream), while
  the summary is printed from the main thread at shutdown. Every mutation is
  therefore under ``_lock``.
* Collection must add no device->host synchronisation. Everything recorded
  here is host data the paging path already holds.
"""

import atexit
import os
import threading
import time
from statistics import median

from vllm.logger import logger

# Metric table. Adding a row here propagates automatically to the printed
# summary AND to the CSV header (07_csv_export.md §3) — do not inline these
# names at use sites.
#   key      : series name
#   label    : printed label
#   source   : the per-layer value it reduces
#   reduce   : 'mean' across contributing layers, or 'sum' over them
#   in_csv   : whether it is a per-layer quantity worth a CSV column
_METRICS = (
    ("gsize",        "routed experts    |G| /layer",   "gsize",     "mean", True),
    ("psize",        "predicted experts |P| /layer",   "psize",     "mean", True),
    ("hit_post",     "cache hit rate (post-subst)",    "hit_post",  "mean", True),
    ("hit_pre",      "cache hit rate (pre-subst)",     "hit_pre",   "mean", True),
    ("loads",        "on-demand loads /layer/step",    "loads",     "mean", True),
    ("loads_step",   "on-demand loads /step (total)",  "loads",     "sum",  False),
    ("pf_loads",     "prefetch loads |N| /layer/step", "pf_loads",  "mean", True),
    ("pf_loads_step","prefetch loads /step (total)",   "pf_loads",  "sum",  False),
    ("pf_hit",       "loads avoided    |N&G| /layer",  "pf_hit",    "mean", True),
    ("pf_hit_step",  "loads avoided /step (total)",    "pf_hit",    "sum",  False),
    ("pf_waste",     "wasted transfers |N-G| /layer",  "pf_waste",  "mean", True),
    ("pf_waste_step","wasted transfers /step (total)", "pf_waste",  "sum",  False),
    ("subst",        "substituted experts /layer/step","subst",     "mean", True),
    ("pred_prec",    "prediction correct   |P&G|/|P|", "pred_prec", "mean", True),
    ("pred_acc",     "demand covered       |P&G|/|G|", "pred_acc",  "mean", True),
    ("pf_in_lrc",    "predicted resident   |A|/|P|",   "pf_in_lrc", "mean", True),
    ("pf_useful",    "prefetch useful    |N&G|/|N|",   "pf_useful", "mean", True),
    ("pf_wait",      "prefetch stall  ms /layer",      "pf_wait",   "mean", True),
    ("pf_wait_step", "prefetch stall  ms /step",       "pf_wait",   "sum",  False),
)

# The per-layer slots a single record_layer() call may fill.
_LAYER_KEYS = tuple(
    dict.fromkeys(source for _k, _l, source, _r, _c in _METRICS))

_CSV_COLUMNS = tuple(
    source for _k, _l, source, _r, in_csv in _METRICS if in_csv)

# watchdog cadence. Not a snapshot interval
_WATCHDOG_TICK_SECONDS = 1.0
_QUIET_SECONDS = 3.0
# liveness line
_HEARTBEAT_SECONDS = 60.0


def _summarize_series(series):
    """(mean, median, min, max, n) over a per-step series, or None if empty."""
    n = len(series)
    if n == 0:
        return None
    return (sum(series) / n, median(series), min(series), max(series), n)


class DecodeStatsCollector:
    """Aggregates per-(MoE layer, decode step) samples into a run summary.

    The hierarchy is deliberately two levels, not three: per-(layer, step)
    values are reduced across layers at step close into one scalar per metric
    per step, and the run summary is avg/median/min/max over that per-step
    series. min/max therefore read as "worst step / best step", which is the
    only reading that stays meaningful once layers are averaged out.
    """

    def __init__(self, csv_enabled=False, out_path=None, flush_every=200,
                 flush_seconds=30.0, rank=0):
        self._csv_enabled = csv_enabled
        self._out_path = out_path
        self._flush_every = flush_every
        # wall-clock trigger. See _maybe_due_locked for why both exist.
        self._flush_seconds = flush_seconds
        self._rank = rank
        self._lock = threading.Lock()
        self._final = False
        self._active = False

        self._num_layers = 0
        self._hash_layers = frozenset()
        self._multi_card = False

        self._step_vals = {key: {} for key in _LAYER_KEYS}
        self._seen = set()
        self._step_open = False

        self._series = {key: [] for key, _l, _s, _r, _c in _METRICS}
        self._layer_sum = {key: {} for key in _LAYER_KEYS}
        self._layer_cnt = {key: {} for key in _LAYER_KEYS}

        self._decode_steps = 0
        self._shortfalls = 0
        self._announced_first = False
        self._flushes = 0
        self._steps_at_last_flush = 0
        # monotonic, not wall time — immune to clock changes during a long run.
        self._time_at_last_flush = time.monotonic()
        # cached run header, so an interim artifact is self-describing too.
        self._header = None

        # watchdog state
        self._record_calls = 0
        self._last_record = time.monotonic()
        self._watchdog = None
        self._stop = threading.Event()
        # per-forward decode gate, set by the model runner before every forward.
        self._step_is_decode = True
        self._nondecode_steps = 0

        self._spec_len = []
        self._spec_rate = []
        self._spec_accepted = 0
        self._spec_proposed = 0
        self._spec_drafts = 0

        self._timestamp = time.strftime("%Y%m%d-%H%M%S")

    # -------------------------------------------------------------- #
    #  Lifecycle                                                       #
    # -------------------------------------------------------------- #

    @property
    def collecting(self) -> bool:
        """Cheap gate for hot-path callers, so they can skip the work entirely."""
        return self._active and not self._final

    def activate(self) -> None:
        """Arm collection. Called once, after every warmup and graph capture."""
        self._active = True
        # the clock starts when collection is armed, not at construction —
        # otherwise model loading and graph capture, which are minutes, would make
        # the very first decode step look overdue and flush an empty table.
        self._time_at_last_flush = time.monotonic()
        self._last_record = time.monotonic()

        if self._watchdog is None:
            self._watchdog = threading.Thread(
                target=self._watchdog_loop, name="decode-stats-watchdog",
                daemon=True)
            self._watchdog.start()
        logger.info("[DECODE-STATS] armed: collection is now live (snapshot every %d "
                    "steps or %.0fs, and %.0fs after decoding stops; output dir %s)",
                    self._flush_every, self._flush_seconds, _QUIET_SECONDS,
                    self._out_path or "<none>")

    def set_header(self, header) -> None:
        """Cache the run-configuration header for interim flushes.

        NEW. The header used to be supplied only by shutdown(), so every periodic
        artifact written during the run lacked the run/offload lines that make it
        readable on its own. Set once, right after activate().
        """
        with self._lock:
            self._header = header

    def set_topology(self, num_layers: int, hash_layers, multi_card=False) -> None:
        """Register the MoE-layer layout. hash_layers are excluded from every
        cross-layer reduction and from the CSV rows, per the requirement that
        the numbers describe gate-routed MoE layers only."""
        with self._lock:
            self._num_layers = num_layers
            self._hash_layers = frozenset(hash_layers)
            self._multi_card = multi_card

    # -------------------------------------------------------------- #
    #  Recording                                                     #
    # -------------------------------------------------------------- #
    
    def set_step_kind(self, is_decode: bool) -> None:
        """Declare whether the forward about to run is a pure decode step.

        Called once per forward by the model runner, from `execute_model`
        (with the real answer) and from `_dummy_run` (always False). Everything
        else — prefill, chunked prefill, a prefix-cache-hit remainder short
        enough to slip under `offload_threshold`, and the DP-idle dummy run —
        must not contribute samples, and none of them were excluded before.

        No lock: a bool store on the forward thread against a bool load on the
        report thread, and the two are ordered by the forward itself (eager runs
        the callback inline; replay is preceded by a stream synchronize).
        """
        self._step_is_decode = is_decode
        if not is_decode and self._active and not self._final:
            self._nondecode_steps += 1

    def note_shortfall(self) -> None:
        """One (layer, step) where a needed expert could not be given a slot.
        Reported separately because it breaks the loads<->hit-rate identity."""
        if not self.collecting:
            return
        with self._lock:
            self._shortfalls += 1

    def record_layer(self, layer_idx: int, *, gsize=None, psize=None,
                hit_post=None, hit_pre=None, loads=None, pf_loads=None,
                pf_hit=None, pf_waste=None, subst=None, pred_acc=None,
                pred_prec=None, pf_in_lrc=None, pf_useful=None,
                pf_wait=None) -> None:
        """Record one MoE layer's reactive pass within the current decode step.

        ``None`` means "no sample for this metric on this layer this step" and is
        dropped rather than counted as zero — which is what keeps the last MoE
        layer out of the prefetch metrics and a zero-transfer prefetch out of
        pf_useful.

        The step boundary is detected here, by revisiting a layer: the manager
        walks layers 0..L-1 once per decode forward, so the next appearance of an
        already-seen layer starts a new step. Hash layers participate in the
        boundary but contribute no values.
        """
        if not self.collecting:
            return
        # drop the whole forward if it is not a decode step for decode stats capture.
        if not self._step_is_decode:
            return
        values = {
            # gsize/psize/pf_loads/pred_prec added. Keys must match the
            # `source` column of _METRICS; nothing else keys off this dict.
            "gsize": gsize, "psize": psize,
            "hit_post": hit_post, "hit_pre": hit_pre,
            "loads": loads, "pf_loads": pf_loads,
            "pf_hit": pf_hit, "pf_waste": pf_waste,
            "subst": subst, "pred_acc": pred_acc, "pred_prec": pred_prec,
            "pf_in_lrc": pf_in_lrc, "pf_useful": pf_useful,
            # milliseconds the compute stream was blocked at this layer's prefetch join.
            "pf_wait": pf_wait,
        }
        # This method runs inside the paging host callback, which blocks its stream until it returns, 
        # so the summary formatting and two file writes were a decode stall that grew
        # with the run. _watchdog_loop owns snapshotting now.
        with self._lock:
            # re-check under the lock
            if self._final:
                return
            self._record_calls += 1
            self._last_record = time.monotonic()
            if layer_idx in self._seen:
                self._close_step_locked()
            self._seen.add(layer_idx)
            self._step_open = True
            if layer_idx in self._hash_layers:
                return
            # announce *after* the hash-layer return.
            if not self._announced_first:
                self._announced_first = True
                logger.info("[DECODE-STATS] first sample recorded (layer=%d) — "
                            "collection is working", layer_idx)
            for key, value in values.items():
                if value is None:
                    continue
                self._step_vals[key][layer_idx] = value
                self._layer_sum[key][layer_idx] = (
                    self._layer_sum[key].get(layer_idx, 0.0) + value)
                self._layer_cnt[key][layer_idx] = (
                    self._layer_cnt[key].get(layer_idx, 0) + 1)

    def _maybe_due_locked(self, quiet: bool = False) -> bool:
        """Is a snapshot due? Caller holds _lock.

        Marks the flush as taken here rather than in _flush, so two threads
        cannot both decide they are due for the same interval — record_layer runs
        on a host-callback report thread under graph replay.
        """
        if self._out_path is None:
            return False
        # require at least one new step
        new_steps = self._decode_steps - self._steps_at_last_flush
        if new_steps <= 0:
            return False
        now = time.monotonic()
        # `quiet` is the third trigger — decoding has stopped, which is
        # the end of the workload and the moment the artifact must be complete
        if (new_steps < self._flush_every
                and now - self._time_at_last_flush < self._flush_seconds
                and not quiet):
            return False
        self._steps_at_last_flush = self._decode_steps
        self._time_at_last_flush = now
        return True
    
    def _watchdog_loop(self) -> None:
        """Snapshot on a timer, off the paging callback's critical path.

        Closes three failure modes that all presented as "no summary and no
        CSV after the run":
        * both periodic triggers used to be evaluated inside ``record_layer``,
          which only runs while decoding — so once the workload finished nothing
          could snapshot again and the tail lived in memory until ``summarize()``,
          the call vLLM's process manager SIGKILLs;
        * ``record_layer`` runs inside ``_launch_host_func``, so its file writes
          stalled the compute stream;
        * a collector that never received a sample looked exactly like one that
          was never created. The heartbeat is emitted either way.
        """
        heartbeat_at = time.monotonic()
        while not self._stop.wait(_WATCHDOG_TICK_SECONDS):
            if self._final:
                return
            due = False
            beat = None
            with self._lock:
                if self._final:
                    return
                now = time.monotonic()
                quiet = (now - self._last_record) >= _QUIET_SECONDS
                if quiet and self._step_open:
                    self._close_step_locked()
                due = self._maybe_due_locked(quiet=quiet)
                if now - heartbeat_at >= _HEARTBEAT_SECONDS:
                    heartbeat_at = now
                    beat = (self._decode_steps, self._record_calls,
                            self._flushes, len(self._spec_len))
            # both outside the lock: logging and file I/O must not serialise
            # against record_layer on the report thread.
            if beat is not None:
                logger.debug("[DECODE-STATS] heartbeat: steps=%d records=%d "
                            "snapshots=%d spec_samples=%d out=%s",
                            beat[0], beat[1], beat[2], beat[3],
                            self._out_path or "<none>")
            if due and not self._final:
                self._flush(final=False)

    def record_spec_step_totals(self, accepted: int, proposed: int, n_drafts: int) -> None:
        """Record one decode step's speculative-decoding totals.
        Definitions follow vLLM's
        (https://docs.vllm.ai/en/stable/api/vllm/v1/spec_decode/metrics/):
        acceptance rate = accepted / proposed, acceptance length = 1 + accepted per
        draft, bonus token included by convention.
        """
        if not self.collecting or n_drafts <= 0 or proposed <= 0:
            return
        with self._lock:
            self._spec_len.append(1.0 + accepted / n_drafts)
            self._spec_rate.append(accepted / proposed)
            self._spec_accepted += accepted
            self._spec_proposed += proposed
            self._spec_drafts += n_drafts

    def _close_step_locked(self) -> None:
        """Reduce the open step across layers. Caller holds _lock."""
        if not self._step_open:
            return
        for key, _label, source, reduce_kind, _in_csv in _METRICS:
            layer_values = self._step_vals[source]
            if not layer_values:
                continue  # no contributing layer this step: no sample, not 0.0
            total = sum(layer_values.values())
            self._series[key].append(
                total if reduce_kind == "sum" else total / len(layer_values))
        self._decode_steps += 1
        for slot in self._step_vals.values():
            slot.clear()
        self._seen.clear()
        self._step_open = False

    # -------------------------------------------------------------- #
    #  Summary                                                         #
    # -------------------------------------------------------------- #

    def summarize(self, header=None) -> None:
        """Emit the final statistics. Idempotent and latched.

        Called from the model runner's shutdown() and from the atexit backstop.
        """
        self._flush(final=True, header=header)

    def _flush(self, final: bool, header=None) -> None:
        """Emit the current statistics to the log and to the output directory.

        The previous design emitted exactly once, from shutdown(), so anything
        that prevented that one call — vLLM's process manager SIGKILLing the
        EngineCore in the same second as SIGTERM — destroyed the entire
        measurement silently. Now the run's own artifacts are on disk from the
        first flush onwards and the final emit is a refinement, not the only
        chance.
        """
        with self._lock:
            if self._final:
                # was a warning. Demoted, because this can only ever
                # drop a RECOMPUTATION, never data: record_layer refuses to
                # record once _final is set (see its in-lock re-check), and
                # every flush rebuilds the whole summary from the running
                # accumulators. The only way to get here now is a watchdog tick
                # that decided it was due and then lost the lock to
                # summarize() — expected at shutdown, and already written.
                logger.debug("[DECODE-STATS] snapshot skipped: already "
                             "finalized (steps=%d)", self._decode_steps)
                return
            if final:
                self._close_step_locked()
                self._final = True
                # retire the watchdog with the collector, so it cannot
                # race the shutdown path or hold a reference after the summary.
                self._stop.set()
                # latch the trigger baseline. _close_step_locked() above
                # just incremented _decode_steps, and _steps_at_last_flush is
                # still pointing at the last PERIODIC snapshot — so without this
                # the next watchdog tick computes new_steps > 0, declares itself
                # due, and flushes into a finalized collector. That is the
                # "snapshot dropped: already finalized" line every clean
                # shutdown was producing.
                self._steps_at_last_flush = self._decode_steps
                self._time_at_last_flush = time.monotonic()
            # fall back to the cached header, so interim artifacts carry
            # the run configuration too and not just the metric table.
            if header is None:
                header = self._header
            else:
                self._header = header
            self._flushes += 1
            text = self._format_summary_locked(header)
            rows = self._csv_rows_locked()
            steps = self._decode_steps
            flush_no = self._flushes
        if final:
            logger.info("%s", text)
        else:
            # Periodic flushes log one line, not the whole table — the table is in
            # the file. Logging it every interval would bury serve.log.
            logger.debug("[DECODE-STATS] flush #%d at %d decode steps -> %s",
                        flush_no, steps, self._out_path)
        self._write_files(text, rows, final)

    def _write_files(self, text, rows, final: bool) -> None:
        """Write the summary .txt and, when enabled, the CSV. Overwrites in place.

        The .txt exists because the summary previously lived only in the logger,
        which put it at the mercy of log plumbing, of a mirror that had already been
        stopped, and of teardown ordering. A file is checkable with `ls` while the
        run is still going.
        """
        if self._out_path is None:
            return
        try:
            os.makedirs(self._out_path, exist_ok=True)
            path = os.path.join(
                self._out_path,
                f"decode_stats_summary_rank{self._rank}_{self._timestamp}.txt")
            with open(path, "w") as handle:
                handle.write(text)
                handle.write("\n")
            if final:
                logger.info("[DECODE-STATS] summary written to %s",
                            os.path.abspath(path))
        except Exception as exc:
            # Never take down a serving process over a statistics artifact.
            logger.warning("[DECODE-STATS] summary write failed: %s", exc)
        if not self._csv_enabled:
            return
        try:
            self._write_csv(rows, quiet=not final)
        except Exception as exc:
            logger.warning("[EXPERT-OFFLOAD-CSV] write failed: %s", exc)

    def _format_summary_locked(self, header) -> str:
        width = 84
        lines = ["", "=" * width, "[EXPERT-OFFLOAD-FINAL] decode statistics summary".center(width), "=" * width]
        for line in (header or {}).get("lines", []):
            lines.append(line)
        moe_layers = self._num_layers - len(self._hash_layers)
        lines.append(
            f"measured : decode steps={self._decode_steps}  "
            f"moe layers={moe_layers} (hash excluded={len(self._hash_layers)})  "
            f"slot shortfalls={self._shortfalls}")
        # collector state, always. An empty table used to give no way to
        # tell "never armed" from "armed but never fed" from "finalized early".
        lines.append(
            f"collector: active={self._active} final={self._final} "
            f"snapshots={self._flushes} record_calls={self._record_calls} "
            # forwards dropped as non-decode. decode_steps + non_decode
            # should equal the engine's total forward count
            f"non_decode={self._nondecode_steps} "
            f"out={self._out_path or '<none>'}")
        if self._multi_card:
            lines.append(
                "note     : multi-card offload path collects no per-layer "
                "statistics; cache/prefetch rows below are empty by design.")
        if self._decode_steps == 0:
            lines.append(
                "note     : no decode steps were measured (offload inactive, "
                "prefill-only run, or statistics never armed).")

        lines.append("-" * width)
        lines.append(f"{'metric':<32}{'mean':>10}{'median':>10}"
                     f"{'min':>10}{'max':>10}{'steps':>8}")
        lines.append("-" * width)
        for key, label, _source, _reduce, _in_csv in _METRICS:
            stats = _summarize_series(self._series[key])
            if stats is None:
                continue  # a section with no samples prints nothing
            mean, med, lo, hi, n = stats
            lines.append(f"{label:<32}{mean:>10.4f}{med:>10.4f}"
                         f"{lo:>10.4f}{hi:>10.4f}{n:>8d}")

        spec_len = _summarize_series(self._spec_len)
        spec_rate = _summarize_series(self._spec_rate)
        if spec_len is not None and spec_rate is not None:
            lines.append("-" * width)
            for label, stats in (("acceptance length (1+acc)", spec_len),
                                 ("acceptance rate  acc/prop", spec_rate)):
                mean, med, lo, hi, n = stats
                lines.append(f"{label:<32}{mean:>10.4f}{med:>10.4f}"
                             f"{lo:>10.4f}{hi:>10.4f}{n:>8d}")
            # Ratio-of-sums, i.e. the aggregate form vLLM itself reports, so
            # these numbers are directly comparable with published ones. It
            # differs from the mean-of-per-step-means above whenever the number
            # of drafting requests varies across steps.
            agg_rate = (self._spec_accepted / self._spec_proposed
                        if self._spec_proposed else 0.0)
            agg_len = (1.0 + self._spec_accepted / self._spec_drafts
                       if self._spec_drafts else 0.0)
            lines.append(
                f"{'  ratio-of-sums':<32}rate={agg_rate:.4f}  "
                f"length={agg_len:.4f}  drafts={self._spec_drafts}  "
                f"accepted={self._spec_accepted}/{self._spec_proposed}")

        lines.append("=" * width)
        return "\n".join(lines)

    def _csv_rows_locked(self):
        """Per-layer means, one row per non-hash MoE layer.

        Rows are built from the layer range rather than from observed samples,
        so the row set is stable across runs and a metric that never fired for
        a layer yields an empty cell (-> NaN in pandas/numpy) rather than a
        misleading 0.0 — 07_csv_export.md §5.
        """
        rows = []
        for layer_idx in range(self._num_layers):
            if layer_idx in self._hash_layers:
                continue
            row = {"layer": layer_idx}
            for column in _CSV_COLUMNS:
                count = self._layer_cnt[column].get(layer_idx, 0)
                row[column] = (
                    f"{self._layer_sum[column][layer_idx] / count:.6f}"
                    if count else "")
            rows.append(row)
        return rows

    def _write_csv(self, rows, quiet=False) -> None:
        import csv  # local: this is the only place it is needed

        # fixed filename per run (no _csv_written latch), rewritten on every
        # flush so the file always reflects the latest state. One file per run still,
        # because the timestamp is fixed at construction.
        directory = self._out_path or "."
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(
            directory, f"decode_stats_rank{self._rank}_{self._timestamp}.csv")
        columns = ["layer", *_CSV_COLUMNS]
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        if not quiet:
            logger.info(
                "[EXPERT-OFFLOAD-CSV] wrote %d layer rows over %d decode steps to %s",
                len(rows), self._decode_steps, os.path.abspath(path))


_COLLECTOR = None
_COLLECTOR_RESOLVED = False


def get_decode_stats():
    """Process-wide collector, or None when the feature is disabled.

    Resolved lazily on first use because AscendConfig is not built at import time.
    """
    global _COLLECTOR, _COLLECTOR_RESOLVED
    if _COLLECTOR_RESOLVED:
        return _COLLECTOR
    from vllm_ascend.ascend_config import get_ascend_config
    try:
        config = get_ascend_config()
    except Exception:
        return None  # config not built yet; retry on the next call
    _COLLECTOR_RESOLVED = True
    if not getattr(config, "decode_stats_enabled", False):
        # A None collector was indistinguishable from one that
        # existed and never recorded anything, which is half of every "no statistics" investigation.
        logger.info("[DECODE-STATS] disabled by config "
                    "(additional_config.decode_stats_enabled=false)")
        return None
    _COLLECTOR = DecodeStatsCollector(
        csv_enabled=config.decode_stats_csv,
        out_path=config.decode_stats_path,
        flush_every=config.decode_stats_flush_every,
        flush_seconds=getattr(config, "decode_stats_flush_seconds", 30.0),
        rank=int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))),
    )

    def _atexit_summary():
        try:
            _COLLECTOR.summarize(None)
        except Exception:
            pass  # logging may be partially torn down at interpreter exit

    atexit.register(_atexit_summary)
    return _COLLECTOR
