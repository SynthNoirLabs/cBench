"""Context compaction benchmark — tests Anthropic's native context compaction API.

Beta header: compact-2026-01-12
Uses context_management.edits with compact_20260112 type to trigger server-side
context summarization. Measures information retention after compaction.

This benchmark requires a custom multi-turn runner because compaction is triggered
during extended conversations, not single API calls.
"""

import time
from typing import Any

from cbench.benchmarks.base import Benchmark, BenchmarkResult
from cbench.client import BenchClient, CallResult
from cbench.config import BETA_COMPACTION, ModelID
from cbench.display import get_progress
from cbench.metrics import calculate_cost
from cbench.storage import save_results
from cbench.tasks.base import ScoringMethod, TaskDefinition

# Facts to embed in conversation turns — these will be tested after compaction
_CONVERSATION_FACTS = [
    ("The orbital period of planet Kepler-442b is 112.3 Earth days.", "112.3"),
    ("Dr. Elena Vasquez discovered the high-temperature superconductor compound LK-99B in 2024.", "LK-99B"),
    ("The ancient city of Tanis had a population of approximately 34,000 at its peak.", "34,000"),
    ("The Fibonacci prime at index 19 is 4181.", "4181"),
    ("Protocol ZETA-9 requires authentication tokens to be rotated every 73 minutes.", "73"),
    ("The Mariana Hadal Snailfish was recorded at a depth of 8,178 meters.", "8,178"),
    ("The Treaty of Westborough was signed on March 14, 1647, by seven nations.", "1647"),
    ("Element 119, Ununennium, has a predicted melting point of 0 degrees Celsius.", "Ununennium"),
]

# Filler messages to pad the conversation between facts
_FILLER_PROMPTS = [
    "Tell me about the history of computing in 500 words.",
    "Explain quantum entanglement in detail.",
    "Describe the process of photosynthesis step by step.",
    "What are the main differences between TCP and UDP? Be thorough.",
    "Explain how neural networks learn, covering backpropagation in detail.",
]


class CompactionBenchmark(Benchmark):
    name = "compaction"
    description = "Context compaction — information retention after server-side summarization"

    def get_variants(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "with_compaction",
                "use_compaction": True,
                "model": ModelID.OPUS_4_6,
            },
            {
                "name": "without_compaction",
                "use_compaction": False,
                "model": ModelID.OPUS_4_6,
            },
        ]

    def get_tasks(self) -> list[TaskDefinition]:
        """Each task tests recall of one fact after compaction."""
        tasks = []
        for i, (fact, answer) in enumerate(_CONVERSATION_FACTS):
            tasks.append(
                TaskDefinition(
                    name=f"recall_fact_{i}",
                    category="compaction",
                    prompt=f"Earlier in our conversation I told you a fact. What was it? Specifically: {fact.split()[0]} {fact.split()[1]}",
                    expected_answer=answer,
                    scoring_method=ScoringMethod.CONTAINS,
                    difficulty="hard",
                )
            )
        return tasks

    def build_call_kwargs(self, variant: dict[str, Any]) -> dict[str, Any]:
        return {}


class CompactionRunner:
    """Custom multi-turn runner for the compaction benchmark."""

    def __init__(self, output_dir: str = "results") -> None:
        self.output_dir = output_dir
        self.client = BenchClient()

    def run(self, benchmark: CompactionBenchmark, num_runs: int = 1) -> list[BenchmarkResult]:
        variants = benchmark.get_variants()
        facts = _CONVERSATION_FACTS
        total = len(variants) * len(facts) * num_runs
        results: list[BenchmarkResult] = []

        with get_progress() as progress:
            prog_task = progress.add_task("[cyan]compaction[/cyan]", total=total)

            for variant in variants:
                use_compaction = variant["use_compaction"]
                model = ModelID.OPUS_4_6

                for run_idx in range(num_runs):
                    # Phase 1: Build conversation with embedded facts
                    messages: list[dict[str, Any]] = []
                    total_input = 0
                    total_output = 0
                    _total_cost = 0.0
                    start_time = time.monotonic()

                    for i, (fact, _answer) in enumerate(facts):
                        # Inject fact
                        messages.append({
                            "role": "user",
                            "content": f"Remember this fact: {fact}",
                        })
                        messages.append({
                            "role": "assistant",
                            "content": "I've noted that fact. Got it.",
                        })

                        # Add filler to grow context
                        if i < len(_FILLER_PROMPTS):
                            filler_prompt = _FILLER_PROMPTS[i]
                            messages.append({"role": "user", "content": filler_prompt})

                            kwargs: dict[str, Any] = {
                                "model": model.value,
                                "messages": messages,
                                "max_tokens": 1024,
                            }

                            if use_compaction:
                                kwargs["extra_headers"] = {
                                    "anthropic-beta": BETA_COMPACTION,
                                }
                                kwargs["context_management"] = {
                                    "type": "compact_20260112",
                                    "trigger": {"type": "token_threshold", "token_count": 20000},
                                }

                            try:
                                response = self.client.client.messages.create(**kwargs)
                                resp_text = ""
                                for block in response.content:
                                    if block.type == "text":
                                        resp_text += block.text
                                messages.append({"role": "assistant", "content": resp_text})

                                total_input += response.usage.input_tokens
                                total_output += response.usage.output_tokens

                                # Apply compaction edits if present
                                edits = getattr(response, "context_management", None)
                                if edits and hasattr(edits, "edits"):
                                    for edit in edits.edits:
                                        if hasattr(edit, "type") and edit.type == "compact_20260112" and hasattr(edit, "messages"):
                                            # Replace messages with compacted version
                                            messages = list(edit.messages)
                            except Exception:
                                messages.append({
                                    "role": "assistant",
                                    "content": "I understand. Let me continue.",
                                })

                    # Phase 2: Query each fact
                    for i, (fact, answer) in enumerate(facts):
                        query = (
                            f"Earlier I asked you to remember several facts. "
                            f"One was about {fact.split()[3:6]}. "
                            f"What was the specific number or name mentioned?"
                        )
                        messages.append({"role": "user", "content": query})

                        kwargs = {
                            "model": model.value,
                            "messages": messages,
                            "max_tokens": 512,
                        }
                        if use_compaction:
                            kwargs["extra_headers"] = {
                                "anthropic-beta": BETA_COMPACTION,
                            }

                        try:
                            response = self.client.client.messages.create(**kwargs)
                            resp_text = ""
                            for block in response.content:
                                if block.type == "text":
                                    resp_text += block.text

                            total_input += response.usage.input_tokens
                            total_output += response.usage.output_tokens

                            messages.append({"role": "assistant", "content": resp_text})
                        except Exception:
                            resp_text = ""

                        elapsed = time.monotonic() - start_time
                        score = 1.0 if answer.lower() in resp_text.lower() else 0.0

                        usage_dict: dict[str, Any] = {
                            "input_tokens": total_input,
                            "output_tokens": total_output,
                        }
                        cost = calculate_cost(model, usage_dict)

                        call_result = CallResult(
                            input_tokens=total_input,
                            output_tokens=total_output,
                            wall_clock_seconds=elapsed,
                            response_text=resp_text,
                            response_length=len(resp_text),
                            cost_usd=cost,
                        )

                        task_def = TaskDefinition(
                            name=f"recall_fact_{i}",
                            category="compaction",
                            prompt=query,
                            expected_answer=answer,
                            scoring_method=ScoringMethod.CONTAINS,
                            difficulty="hard",
                        )

                        results.append(
                            BenchmarkResult(
                                benchmark_name="compaction",
                                variant_name=variant["name"],
                                task_name=task_def.name,
                                run_index=run_idx,
                                call_result=call_result,
                                score=score,
                                variant_config=variant,
                            )
                        )
                        progress.update(prog_task, advance=1)
                        time.sleep(0.3)

        save_results("compaction", results, self.output_dir)
        return results

    @staticmethod
    def estimate_cost(num_runs: int = 1) -> dict[str, Any]:
        num_facts = len(_CONVERSATION_FACTS)
        num_filler = len(_FILLER_PROMPTS)
        # Rough estimate: filler calls + query calls per variant per run
        calls_per_variant = (num_filler + num_facts) * num_runs
        total_calls = calls_per_variant * 2  # 2 variants

        # Average ~2000 input tokens per call (growing context), ~300 output
        avg_input = 2000
        avg_output = 300
        pricing = {
            "input": 5.00,  # Opus 4.6
            "output": 25.00,
        }
        cost = (
            (avg_input / 1_000_000) * pricing["input"] * total_calls
            + (avg_output / 1_000_000) * pricing["output"] * total_calls
        )

        return {
            "benchmark": "compaction",
            "variants": 2,
            "tasks": num_facts,
            "api_calls": total_calls,
            "estimated_cost": cost,
        }
