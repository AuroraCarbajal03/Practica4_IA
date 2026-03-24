import time
import tracemalloc


class MetricTracker:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.peak_kb = 0.0

    def start(self):
        tracemalloc.start()
        self.start_time = time.perf_counter()

    def stop(self):
        self.end_time = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        self.peak_kb = peak / 1024.0
        tracemalloc.stop()

    @property
    def elapsed_ms(self):
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000.0