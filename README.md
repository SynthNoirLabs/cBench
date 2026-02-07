# cBench

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Multi-provider LLM benchmark suite. Compares 12 models across Anthropic, OpenAI, Google, and OpenRouter on reasoning, tool use, context capacity, and more.

## Models

| Provider | Models |
|----------|--------|
| Anthropic | Claude Opus 4.6, Sonnet 4.5, Haiku 4.5 |
| OpenAI | GPT-5.2, GPT-5.1, GPT-5 Mini (all reasoning models) |
| Google | Gemini 3 Pro, Gemini 3 Flash, Gemini 2.5 Pro, Gemini 2.5 Flash |
| OpenRouter | Qwen3 Coder, DeepSeek V3 |

## Benchmarks

15 benchmarks across standard, beta feature, and agentic categories:

| Benchmark | What it tests |
|-----------|--------------|
| `thinking_modes` | Thinking enabled vs disabled vs adaptive |
| `effort_levels` | Low / medium / high effort (Claude only) |
| `budget_sweep` | Thinking token budget scaling |
| `model_comparison` | Cross-provider model comparison (12 models) |
| `temperature` | Temperature 0.0 / 0.5 / 1.0 |
| `streaming` | Streaming vs non-streaming |
| `caching` | Prompt caching on vs off |
| `long_context` | Needle-in-a-haystack at 50K / 200K / 500K / 1M tokens |
| `large_output` | Output capacity at 16K / 64K / 128K max tokens |
| `tool_search` | Tool Search Tool beta — deferred loading vs full context |
| `programmatic_tools` | Programmatic tool calling beta — code execution for tool orchestration |
| `compaction` | Context compaction beta — info retention after summarization |
| `agent_teams` | Agent SDK multi-agent — single vs team performance |
| `review` | Agentic code review with LLM-as-Judge |
| `repair` | Agentic code repair (multi-turn bug fixing) |

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all-providers]"
```

Or install only the providers you need: `pip install -e .` (Anthropic only), `pip install -e ".[openai]"`, `pip install -e ".[google]"`.

## Configuration

Set your API keys in `.env` (copy from `.env.example`):

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=sk-or-...
```

Only the keys for providers you want to use are required.

## Usage

```bash
cbench list                                     # list all benchmarks
cbench run model_comparison                     # run a single benchmark
cbench run thinking_modes effort_levels         # run multiple
cbench run all                                  # run all standard benchmarks
cbench run model_comparison --num-runs 3        # multiple runs for reliability stats
cbench run model_comparison --dry-run           # cost estimate only
```

### Specialized benchmarks

These use dedicated runners and have their own CLI commands:

```bash
cbench compaction                               # context compaction (multi-turn, beta)
cbench agent-teams                              # agent SDK multi-agent coordination
cbench review /path/to/repo                     # agentic code review
cbench repair                                   # agentic code repair
```

All commands support `--dry-run`, `--no-confirm`, `--num-runs N`, and `--output-dir`.

### Analyze results

```bash
cbench analyze results/                         # charts + markdown report
cbench analyze results/ --format charts         # charts only
cbench analyze results/ --format report         # report only
```

## Metrics

Output includes: score, cost, latency, input/output tokens, Score/$, Score/1K-tokens.

With `--num-runs > 1`: mean score, standard deviation, pass^k, mean cost, and mean latency.

## Project structure

```
cbench/
  cli.py              CLI entry point
  config.py           Models, pricing, provider routing, beta headers
  runner.py           Benchmark orchestrator
  scorer.py           5 scoring methods (exact, contains, code exec, LLM judge, tool match)
  client.py           Anthropic client (tools, betas, thinking)
  providers/          Multi-provider clients (Anthropic, OpenAI, Google, OpenRouter)
  benchmarks/         11 standard benchmarks + 2 with custom runners
  tasks/              8 tasks across 4 categories
  review/             Agentic code review (LLM-as-Judge)
  repair/             Agentic code repair (multi-turn)
  analysis/           Charts and markdown reports
tests/                Unit tests
```

## License

MIT
