from cbench.benchmarks.base import Benchmark
from cbench.config import ModelID


class TemperatureBenchmark(Benchmark):
    name = "temperature"
    description = "Temperature sweep (thinking disabled), 3 runs each"

    def get_variants(self) -> list[dict]:
        return [
            {"name": f"temp_{t}", "temperature": t, "model": ModelID.OPUS_4_6}
            for t in (0.0, 0.3, 0.5, 0.7, 1.0)
        ]

    def build_call_kwargs(self, variant: dict) -> dict:
        return {"temperature": variant["temperature"]}
