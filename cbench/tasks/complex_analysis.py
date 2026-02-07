from cbench.tasks.base import BenchmarkTask, ScoringMethod, TaskDefinition


class ComplexAnalysisTasks(BenchmarkTask):
    def get_tasks(self) -> list[TaskDefinition]:
        return [
            TaskDefinition(
                name="knights_and_knaves",
                category="complex_analysis",
                prompt=(
                    "On an island, knights always tell the truth and knaves always lie. "
                    "You meet two people, A and B. A says 'We are both knaves.' What are "
                    "A and B? Answer in the format 'A is a [knight/knave], B is a "
                    "[knight/knave]'."
                ),
                expected_answer="A is a knave, B is a knight",
                scoring_method=ScoringMethod.CONTAINS,
                difficulty="hard",
            ),
            TaskDefinition(
                name="constraint_satisfaction",
                category="complex_analysis",
                prompt=(
                    "A conference has 4 speakers (Alice, Bob, Carol, Dave) and 4 time slots "
                    "(9am, 10am, 11am, 1pm). Each speaker presents exactly once. The "
                    "constraints are:\n"
                    "1. Alice must present before Bob.\n"
                    "2. Carol cannot present at 9am or 1pm.\n"
                    "3. Dave must present immediately after Alice (in the next time slot).\n"
                    "4. Bob cannot present at 10am.\n\n"
                    "Find the unique valid schedule. Answer in the format:\n"
                    "9am: [name], 10am: [name], 11am: [name], 1pm: [name]"
                ),
                expected_answer="9am: Alice, 10am: Dave, 11am: Carol, 1pm: Bob",
                scoring_method=ScoringMethod.CONTAINS,
                difficulty="hard",
            ),
        ]
