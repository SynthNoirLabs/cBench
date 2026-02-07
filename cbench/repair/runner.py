"""Repair benchmark runner: orchestrates agent → pytest → score."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from cbench.client import CallResult
from cbench.config import PRICING, ModelID, is_claude_model
from cbench.display import get_progress
from cbench.repair.benchmark import FixtureInfo, RepairBenchmark
from cbench.storage import save_results


@dataclass
class RepairResult:
    benchmark_name: str = "code_repair"
    variant_name: str = ""
    fixture_name: str = ""
    difficulty: str = ""
    run_index: int = 0
    tests_total: int = 0
    tests_passed_before: int = 0
    tests_passed_after: int = 0
    score: float = 0.0  # tests_passed_after / tests_total
    agent_turns: int = 0
    wall_clock_seconds: float = 0.0
    call_result: CallResult = field(default_factory=CallResult)
    error: str | None = None


def _run_pytest(repo_dir: Path) -> tuple[int, int]:
    """Run pytest in a directory and return (passed, total)."""
    try:
        result = subprocess.run(
            ["python3", "-m", "pytest", "-v", "--tb=short", str(repo_dir)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(repo_dir),
        )
        output = result.stdout + result.stderr

        # Parse pytest output for pass/fail counts
        # Look for "X passed" and "X failed" in the summary line
        passed = 0
        failed = 0
        for line in output.splitlines():
            if "passed" in line or "failed" in line or "error" in line:
                import re

                p = re.search(r"(\d+) passed", line)
                f = re.search(r"(\d+) failed", line)
                e = re.search(r"(\d+) error", line)
                if p:
                    passed = int(p.group(1))
                if f:
                    failed = int(f.group(1))
                if e:
                    failed += int(e.group(1))

        total = passed + failed
        return passed, total
    except Exception:
        return 0, 0


def _run_agent_repair(
    fixture: FixtureInfo,
    work_dir: Path,
    model: ModelID,
    variant: dict,
    max_turns: int,
) -> tuple[int, CallResult]:
    """Run the repair agent on a fixture copy.

    The agent reads the source file, the test file, the test output,
    and iteratively fixes the source file until tests pass.

    Returns (turns_used, aggregated CallResult).
    """
    import anthropic

    client = anthropic.Anthropic(max_retries=3)

    source_file = work_dir / fixture.source_file
    test_file = work_dir / fixture.test_file

    source_code = source_file.read_text()
    test_code = test_file.read_text()

    # Run tests to get initial failure output
    test_result = subprocess.run(
        ["python3", "-m", "pytest", "-v", "--tb=short", str(test_file)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(work_dir),
    )
    test_output = test_result.stdout + test_result.stderr

    system_prompt = (
        "You are a code repair agent. Your job is to fix bugs in Python code to make "
        "failing tests pass. You will be given the source code, test file, and test output.\n\n"
        "When you identify the fix, output the COMPLETE corrected source file wrapped in "
        "```python\\n...\\n``` markers. Do not add new tests or modify the test file. "
        "Only fix the source code."
    )

    messages = [
        {
            "role": "user",
            "content": (
                f"## Source File: {fixture.source_file}\n```python\n{source_code}\n```\n\n"
                f"## Test File: {fixture.test_file}\n```python\n{test_code}\n```\n\n"
                f"## Test Output (failing)\n```\n{test_output}\n```\n\n"
                "Fix the bugs in the source file so all tests pass. "
                "Output the complete corrected source file."
            ),
        }
    ]

    total_input = 0
    total_output = 0
    total_cost = 0.0

    kwargs: dict = {
        "model": model.value,
        "system": system_prompt,
        "messages": messages,
        "max_tokens": 4096,
    }
    if "thinking" in variant:
        kwargs["thinking"] = variant["thinking"]

    turns = 0
    for turn in range(max_turns):
        turns += 1

        try:
            response = client.messages.create(**kwargs)
        except Exception:
            break

        # Extract response text
        response_text = ""
        for block in response.content:
            if block.type == "text":
                response_text += block.text

        usage = response.usage
        total_input += usage.input_tokens
        total_output += usage.output_tokens

        pricing = PRICING[model]
        total_cost += (usage.input_tokens / 1_000_000) * pricing["input_per_mtok"]
        total_cost += (usage.output_tokens / 1_000_000) * pricing["output_per_mtok"]

        # Extract code from response and write it
        import re

        code_match = re.search(r"```python\s*\n(.*?)```", response_text, re.DOTALL)
        if code_match:
            fixed_code = code_match.group(1).strip()
            source_file.write_text(fixed_code + "\n")

        # Run tests again
        test_result2 = subprocess.run(
            ["python3", "-m", "pytest", "-v", "--tb=short", str(test_file)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(work_dir),
        )

        passed, total = _run_pytest(work_dir)
        if total > 0 and passed == total:
            break  # All tests pass

        # If not all passing, give the agent another turn with updated test output
        new_test_output = test_result2.stdout + test_result2.stderr
        messages.append({"role": "assistant", "content": response_text})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Tests still failing. Updated output:\n```\n{new_test_output}\n```\n\n"
                    "Please fix the remaining issues. Output the complete corrected source file."
                ),
            }
        )
        kwargs["messages"] = messages

    call_result = CallResult(
        input_tokens=total_input,
        output_tokens=total_output,
        cost_usd=total_cost,
    )
    return turns, call_result


class RepairRunner:
    def __init__(self, output_dir: str = "results") -> None:
        self.output_dir = output_dir

    def run(
        self,
        benchmark: RepairBenchmark,
        num_runs: int = 1,
        max_turns: int = 15,
    ) -> list[RepairResult]:
        fixtures = benchmark.get_fixtures()
        variants = benchmark.get_variants()
        total = len(fixtures) * len(variants) * num_runs
        results: list[RepairResult] = []

        with get_progress() as progress:
            task_id = progress.add_task(
                "[cyan]code_repair[/cyan]",
                total=total,
            )

            for fixture in fixtures:
                for variant in variants:
                    model = variant.get("model", ModelID.OPUS_4_6)

                    for run_idx in range(num_runs):
                        # Create a temp copy of the fixture
                        with tempfile.TemporaryDirectory() as tmp:
                            work_dir = Path(tmp) / fixture.name
                            shutil.copytree(fixture.path, work_dir)

                            # Run tests before
                            passed_before, total_tests = _run_pytest(work_dir)

                            # Run agent repair
                            start = time.monotonic()
                            try:
                                turns, call_result = _run_agent_repair(
                                    fixture, work_dir, model, variant, max_turns
                                )
                            except Exception as e:
                                elapsed = time.monotonic() - start
                                results.append(
                                    RepairResult(
                                        variant_name=variant["name"],
                                        fixture_name=fixture.name,
                                        difficulty=fixture.difficulty,
                                        run_index=run_idx,
                                        tests_total=total_tests,
                                        tests_passed_before=passed_before,
                                        wall_clock_seconds=elapsed,
                                        error=str(e),
                                    )
                                )
                                progress.update(task_id, advance=1)
                                continue
                            elapsed = time.monotonic() - start

                            # Run tests after
                            passed_after, total_tests_after = _run_pytest(work_dir)
                            if total_tests_after > 0:
                                total_tests = total_tests_after

                            score = passed_after / total_tests if total_tests > 0 else 0.0

                            call_result.wall_clock_seconds = elapsed

                            results.append(
                                RepairResult(
                                    variant_name=variant["name"],
                                    fixture_name=fixture.name,
                                    difficulty=fixture.difficulty,
                                    run_index=run_idx,
                                    tests_total=total_tests,
                                    tests_passed_before=passed_before,
                                    tests_passed_after=passed_after,
                                    score=score,
                                    agent_turns=turns,
                                    wall_clock_seconds=elapsed,
                                    call_result=call_result,
                                )
                            )

                        progress.update(task_id, advance=1)
                        time.sleep(1.0)  # Rate limiting

        save_results("code_repair", results, self.output_dir)
        return results

    @staticmethod
    def estimate_cost(
        benchmark: RepairBenchmark,
        num_runs: int,
        max_turns: int,
    ) -> dict:
        fixtures = benchmark.get_fixtures()
        variants = benchmark.get_variants()

        # Estimate ~2000 input tokens and ~1000 output tokens per turn
        avg_input_per_turn = 2000
        avg_output_per_turn = 1000
        avg_turns = min(max_turns, 5)  # Assume ~5 turns on average

        total_cost = 0.0
        for variant in variants:
            model = variant.get("model", ModelID.OPUS_4_6)
            pricing = PRICING[model]

            calls = len(fixtures) * num_runs * avg_turns
            input_cost = (avg_input_per_turn / 1_000_000) * pricing["input_per_mtok"] * calls
            output_cost = (avg_output_per_turn / 1_000_000) * pricing["output_per_mtok"] * calls
            total_cost += input_cost + output_cost

        return {
            "benchmark": "code_repair",
            "fixtures": len(fixtures),
            "variants": len(variants),
            "num_runs": num_runs,
            "max_turns": max_turns,
            "estimated_cost": total_cost,
        }
