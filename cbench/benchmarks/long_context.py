"""Long context benchmark — needle-in-a-haystack across context sizes."""

from typing import Any

from cbench.benchmarks.base import Benchmark
from cbench.config import ModelID
from cbench.tasks.base import ScoringMethod, TaskDefinition

# Filler paragraph (~100 tokens each) repeated to build haystacks
_FILLER_PARAGRAPH = (
    "The development of artificial intelligence has progressed through several distinct phases. "
    "Early symbolic AI focused on logical reasoning and expert systems during the 1950s and 1960s. "
    "The connectionist approach brought neural networks, initially with limited layers and capacity. "
    "Deep learning emerged in the 2010s with improved architectures and GPU acceleration. "
    "Modern large language models represent a convergence of scale, data, and architectural "
    "innovations that have transformed natural language processing. These systems demonstrate "
    "capabilities that were previously thought to require explicit programming or human-level "
    "understanding. Transfer learning has enabled models trained on general corpora to adapt "
    "to highly specific downstream tasks with minimal fine-tuning. The interplay between "
    "scaling laws and emergent abilities continues to drive both research and commercial "
    "applications across industries worldwide. "
)

# The "needle" — a unique fact embedded in the haystack
_NEEDLE = "The secret activation code for Project Chimera is: AURORA-7749-DELTA."

# Approximate tokens per filler paragraph
_TOKENS_PER_PARAGRAPH = 110


def _build_haystack(target_tokens: int, needle_position: float = 0.5) -> str:
    """Build haystack text with a needle embedded at the given relative position."""
    num_paragraphs = target_tokens // _TOKENS_PER_PARAGRAPH
    needle_idx = int(num_paragraphs * needle_position)

    paragraphs = []
    for i in range(num_paragraphs):
        if i == needle_idx:
            paragraphs.append(_NEEDLE)
        paragraphs.append(f"[Section {i + 1}] {_FILLER_PARAGRAPH}")

    return "\n\n".join(paragraphs)


class LongContextBenchmark(Benchmark):
    name = "long_context"
    description = "Needle-in-a-haystack retrieval across context sizes (50K to 1M tokens)"

    def get_variants(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "50k_context",
                "target_tokens": 50_000,
                "model": ModelID.OPUS_4_6,
            },
            {
                "name": "200k_context",
                "target_tokens": 200_000,
                "model": ModelID.OPUS_4_6,
            },
            {
                "name": "500k_context",
                "target_tokens": 500_000,
                "model": ModelID.OPUS_4_6,
            },
            {
                "name": "1m_context",
                "target_tokens": 1_000_000,
                "model": ModelID.OPUS_4_6,
            },
        ]

    def get_tasks(self) -> list[TaskDefinition]:
        return [
            TaskDefinition(
                name="needle_in_haystack",
                category="long_context",
                prompt=(
                    "Based on all the information provided in the system message, "
                    "what is the secret activation code for Project Chimera? "
                    "Reply with ONLY the activation code, nothing else."
                ),
                expected_answer="AURORA-7749-DELTA",
                scoring_method=ScoringMethod.CONTAINS,
                difficulty="hard",
            ),
        ]

    def build_call_kwargs(self, variant: dict[str, Any]) -> dict[str, Any]:
        haystack = _build_haystack(variant["target_tokens"])
        kwargs: dict[str, Any] = {
            "system": haystack,
        }
        return kwargs
