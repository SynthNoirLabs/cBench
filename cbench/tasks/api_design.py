from cbench.tasks.base import BenchmarkTask, ScoringMethod, TaskDefinition


class ApiDesignTasks(BenchmarkTask):
    def get_tasks(self) -> list[TaskDefinition]:
        return [
            TaskDefinition(
                name="api_design",
                category="api_design",
                prompt=(
                    "Design a REST API for a library management system. The system must support:\n"
                    "1. Books (CRUD, search by title/author/ISBN)\n"
                    "2. Members (registration, profile management)\n"
                    "3. Borrowing (checkout, return, view history, handle overdue)\n"
                    "4. Reservations (reserve unavailable books, cancel, notify when available)\n\n"
                    "For each endpoint, specify:\n"
                    "- HTTP method and path\n"
                    "- Request body (if applicable)\n"
                    "- Response format with status codes\n"
                    "- Error cases\n\n"
                    "Use RESTful conventions, proper HTTP status codes, and include "
                    "pagination for list endpoints. Design for consistency and extensibility."
                ),
                expected_answer="",
                scoring_method=ScoringMethod.LLM_JUDGE,
                difficulty="hard",
                judge_rubric=(
                    "Score the API design on these dimensions (1-5 scale):\n\n"
                    "### Completeness (1-5)\n"
                    "1 = Covers <30% of requirements (missing entire domains)\n"
                    "2 = Covers ~50%, missing several operations\n"
                    "3 = Covers ~70%, addresses main operations\n"
                    "4 = Covers ~90%, only minor operations missed\n"
                    "5 = Full coverage of all requirements including edge cases\n\n"
                    "### REST Conventions (1-5)\n"
                    "1 = Ignores REST principles (verbs in URLs, wrong methods)\n"
                    "2 = Some REST, but inconsistent (mix of conventions)\n"
                    "3 = Mostly RESTful with minor deviations\n"
                    "4 = Proper REST: correct methods, nouns in paths, nesting\n"
                    "5 = Expert REST: HATEOAS-aware, proper resource modeling, idempotency\n\n"
                    "### Error Handling (1-5)\n"
                    "1 = No error cases mentioned\n"
                    "2 = Generic errors only (400, 500)\n"
                    "3 = Common errors covered (404, 409)\n"
                    "4 = Comprehensive errors with meaningful messages\n"
                    "5 = Full error taxonomy with problem+json or similar structure\n\n"
                    "### Consistency (1-5)\n"
                    "1 = Every endpoint uses different patterns\n"
                    "2 = Some patterns reused but many inconsistencies\n"
                    "3 = Mostly consistent naming and structure\n"
                    "4 = Highly consistent with clear conventions\n"
                    "5 = Perfectly consistent, predictable patterns throughout\n\n"
                    "### Extensibility (1-5)\n"
                    "1 = Rigid design, hard to extend\n"
                    "2 = Some room for growth but tightly coupled\n"
                    "3 = Reasonable structure, moderately extensible\n"
                    "4 = Well-structured for growth, versioning considered\n"
                    "5 = Expert design: versioned, paginated, filterable, future-proof\n\n"
                    "Respond with ONLY a JSON object:\n"
                    "{\n"
                    '  "completeness": <1-5>,\n'
                    '  "rest_conventions": <1-5>,\n'
                    '  "error_handling": <1-5>,\n'
                    '  "consistency": <1-5>,\n'
                    '  "extensibility": <1-5>,\n'
                    '  "reasoning": "<2-3 sentences explaining your scoring>"\n'
                    "}"
                ),
            ),
        ]
