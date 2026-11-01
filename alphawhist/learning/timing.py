"""Optional wall-clock timing counters, constant memory per category.

Timing is entirely opt-in. Counters store count, sum, max and log2
histograms, so profiling millions of decisions does not retain raw samples.
Times measure wall-clock time, not CPU time. In multiworker jobs, sums of
per-worker durations must not be interpreted as one end-to-end wall time.
"""
from collections import defaultdict
from functools import wraps
import json
from pathlib import Path
from time import perf_counter_ns


class TimingCounters:
    def __init__(self):
        self.buckets = defaultdict(lambda: [0, 0, 0, {}])

    def add(self, label, nanoseconds):
        if nanoseconds < 0:
            raise ValueError("Negative duration")
        row = self.buckets[label]
        row[0] += 1
        row[1] += nanoseconds
        row[2] = max(row[2], nanoseconds)
        bucket = max(0, nanoseconds.bit_length() - 1)
        row[3][bucket] = row[3].get(bucket, 0) + 1

    def wrap(self, obj, name, category=None):
        """Time one instance method without modifying the original class."""
        original = getattr(obj, name)
        label = category or name

        @wraps(original)
        def timed(*args, **kwargs):
            start = perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                self.add(label, perf_counter_ns() - start)

        setattr(obj, name, timed)

    def merge(self, other):
        for label, (count, total, peak, histogram) in other.buckets.items():
            row = self.buckets[label]
            row[0] += count
            row[1] += total
            row[2] = max(row[2], peak)
            for bucket, n in histogram.items():
                row[3][bucket] = row[3].get(bucket, 0) + n

    @staticmethod
    def _percentile(histogram, count, percentile):
        """Upper edge of a power-of-two-nanosecond bucket (approximate)."""
        target = max(1, int(count * percentile + .999999))
        seen = 0
        for key, n in sorted(histogram.items()):
            seen += n
            if seen >= target:
                return (2 ** (key + 1)) / 1e6
        return 0.

    def summary(self):
        out = {}
        for label, (count, total, peak, histogram) in sorted(self.buckets.items()):
            out[label] = {
                "count": count,
                "total_seconds": total / 1e9,
                "mean_ms": total / count / 1e6 if count else 0.,
                "p50_ms_upper_bound": self._percentile(histogram, count, .5),
                "p95_ms_upper_bound": self._percentile(histogram, count, .95),
                "max_ms": peak / 1e6,
            }
        return out

    def save(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "metadata": metadata or {},
            "timing": self.summary(),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
