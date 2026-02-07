# cBench

Multi-provider AI model benchmark suite. Compares models across Anthropic, OpenAI, Google, and OpenRouter on tasks designed to differentiate current-generation models.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For all providers:

```bash
pip install -e ".[all-providers]"
```

Or individually: `pip install -e ".[openai]"` / `pip install -e ".[google]"`

## Configuration

Copy `.env.example` to `.env` and fill in your API keys:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=sk-or-...
```

Only the keys for providers you want to use are required.

## Usage

### List benchmarks and tasks

```bash
cbench list
```

### Run benchmarks

```bash
cbench run model_comparison
cbench run thinking_modes effort_levels
cbench run all
cbench run model_comparison --num-runs 3        # multiple runs for reliability stats
cbench run model_comparison --dry-run            # cost estimate only
cbench run model_comparison --no-confirm         # skip confirmation prompt
```

### Code review benchmark

Runs an LLM agent that explores a repository and produces a scored review.

```bash
cbench review /path/to/repo
cbench review /path/to/repo --variants opus_adaptive,haiku_no_thinking
cbench review /path/to/repo --num-runs 3 --max-turns 30
```

### Code repair benchmark

Agentic multi-turn bug fixing. The agent reads buggy code + failing tests, iteratively fixes the source until tests pass.

```bash
cbench repair
cbench repair calculator_bug api_handler_bug
cbench repair --variants opus_agent --max-turns 10
cbench repair --dry-run
```

### Cost estimation

```bash
cbench estimate
cbench estimate model_comparison temperature
```

### Analyze results

```bash
cbench analyze results/
cbench analyze results/ --format charts
cbench analyze results/ --format report
```

## Benchmarks

| Benchmark | What it tests |
|-----------|--------------|
| `thinking_modes` | Thinking enabled vs disabled vs adaptive |
| `effort_levels` | Low / medium / high effort (Claude only) |
| `budget_sweep` | Thinking token budget scaling |
| `model_comparison` | Cross-provider model comparison (12 models) |
| `temperature` | Temperature 0.0 / 0.5 / 1.0 |
| `streaming` | Streaming vs non-streaming |
| `caching` | Prompt caching on vs off |
| `review` | Agentic code review with LLM-as-Judge |
| `repair` | Agentic code repair (multi-turn bug fixing) |

## Tasks

8 tasks across 4 categories:

| Category | Task | Scoring | Difficulty |
|----------|------|---------|------------|
| math_reasoning | `combinatorics` | CONTAINS | hard |
| math_reasoning | `multi_step_word_problem` | CONTAINS | medium |
| code_generation | `longest_increasing_subsequence` | CODE_EXECUTION | hard |
| code_generation | `code_debug_fix` | CODE_EXECUTION | medium |
| code_generation | `code_refactor` | CODE_EXECUTION | hard |
| complex_analysis | `knights_and_knaves` | CONTAINS | hard |
| complex_analysis | `constraint_satisfaction` | CONTAINS | hard |
| api_design | `rest_api_design` | LLM_JUDGE | medium |

## Models

| Provider | Models |
|----------|--------|
| Anthropic | Claude Opus 4.6, Sonnet 4.5, Haiku 4.5 |
| OpenAI | GPT-5.2, GPT-5.1, GPT-5 Mini (all reasoning models) |
| Google | Gemini 3 Pro, Gemini 3 Flash, Gemini 2.5 Pro, Gemini 2.5 Flash |
| OpenRouter | Qwen3 Coder, DeepSeek V3 |

## Metrics

Standard output includes: score, cost, latency, input/output tokens, Score/$, Score/1K-tokens.

When running with `--num-runs > 1`, a summary table shows per-(variant, task) aggregates: mean score, standard deviation, pass^k (fraction of runs scoring >= 0.5), mean cost, and mean latency.

## Project structure

```
cbench/
  cli.py              CLI entry point
  config.py           Models, pricing, provider routing
  runner.py           Benchmark orchestrator
  scorer.py           Scoring (exact match, contains, code execution, LLM judge)
  metrics.py          Reliability metrics (pass^k, stddev)
  display.py          Rich table output
  storage.py          JSON results persistence
  client.py           Legacy Anthropic client
  providers/          Multi-provider clients (Anthropic, OpenAI, Google, OpenRouter)
  benchmarks/         Benchmark definitions (7 standard benchmarks)
  tasks/              Task definitions (8 tasks, 4 categories)
  review/             Code review benchmark (agentic, LLM-as-Judge)
  repair/             Code repair benchmark (agentic, multi-turn)
  analysis/           Charts and markdown reports
tests/                Unit tests
```

## License

MIT
