from cbench.tasks.base import BenchmarkTask, ScoringMethod, TaskDefinition


class MathReasoningTasks(BenchmarkTask):
    def get_tasks(self) -> list[TaskDefinition]:
        return [
            TaskDefinition(
                name="multi_step_word_problem",
                category="math_reasoning",
                prompt=(
                    "A store sells notebooks for $4 each and pens for $1.50 each. "
                    "Maria buys 3 notebooks and twice as many pens as notebooks. "
                    "She pays with a $50 bill. She then uses 40% of her change to "
                    "buy stickers at $0.75 each. How many stickers can she buy? "
                    "Show your work and give the final answer as just a number on "
                    "the last line."
                ),
                expected_answer="14",
                scoring_method=ScoringMethod.CONTAINS,
                difficulty="medium",
            ),
            TaskDefinition(
                name="combinatorics",
                category="math_reasoning",
                prompt="How many ways can you choose 5 items from a set of 20? Answer with just the number.",
                expected_answer="15504",
                scoring_method=ScoringMethod.CONTAINS,
                difficulty="medium",
            ),
        ]
