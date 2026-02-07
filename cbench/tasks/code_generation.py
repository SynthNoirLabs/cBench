from cbench.tasks.base import BenchmarkTask, ScoringMethod, TaskDefinition


class CodeGenerationTasks(BenchmarkTask):
    def get_tasks(self) -> list[TaskDefinition]:
        return [
            TaskDefinition(
                name="longest_increasing_subsequence",
                category="code_generation",
                prompt=(
                    "Write a Python function `lis(arr: list[int]) -> int` that returns "
                    "the length of the longest increasing subsequence using O(n log n) "
                    "algorithm. Only output the function, no explanation."
                ),
                expected_answer="",
                scoring_method=ScoringMethod.CODE_EXECUTION,
                difficulty="hard",
                test_cases=[
                    {"input": [10, 9, 2, 5, 3, 7, 101, 18], "expected": 4},
                    {"input": [0, 1, 0, 3, 2, 3], "expected": 4},
                    {"input": [7, 7, 7, 7], "expected": 1},
                ],
            ),
            TaskDefinition(
                name="code_debug_fix",
                category="code_generation",
                prompt=(
                    "The following Python function is supposed to merge two sorted lists "
                    "into one sorted list, but it has bugs. Fix the function so all test "
                    "cases pass. Only output the corrected function, no explanation.\n\n"
                    "```python\n"
                    "def merge_sorted(a, b):\n"
                    "    result = []\n"
                    "    i = j = 0\n"
                    "    while i < len(a) and j < len(b):\n"
                    "        if a[i] <= b[j]:\n"
                    "            result.append(a[i])\n"
                    "            i += 1\n"
                    "        else:\n"
                    "            result.append(b[j])\n"
                    "            i += 1  # BUG: should increment j\n"
                    "    # BUG: missing remaining elements\n"
                    "    return result\n"
                    "```\n\n"
                    "Test cases:\n"
                    "- merge_sorted([1,3,5], [2,4,6]) should return [1,2,3,4,5,6]\n"
                    "- merge_sorted([], [1,2,3]) should return [1,2,3]\n"
                    "- merge_sorted([1], []) should return [1]\n"
                    "- merge_sorted([1,1,1], [1,1]) should return [1,1,1,1,1]"
                ),
                expected_answer="",
                scoring_method=ScoringMethod.CODE_EXECUTION,
                difficulty="medium",
                test_cases=[
                    {"input": {"a": [1, 3, 5], "b": [2, 4, 6]}, "expected": [1, 2, 3, 4, 5, 6]},
                    {"input": {"a": [], "b": [1, 2, 3]}, "expected": [1, 2, 3]},
                    {"input": {"a": [1], "b": []}, "expected": [1]},
                    {"input": {"a": [1, 1, 1], "b": [1, 1]}, "expected": [1, 1, 1, 1, 1]},
                ],
            ),
            TaskDefinition(
                name="code_refactor",
                category="code_generation",
                prompt=(
                    "Refactor the following Python code to be clean and efficient. The function "
                    "should compute the same result but use proper data structures and avoid "
                    "redundancy. Only output the refactored function, no explanation.\n\n"
                    "```python\n"
                    "def count_word_frequencies(text):\n"
                    "    words = text.lower().split()\n"
                    "    unique_words = []\n"
                    "    counts = []\n"
                    "    for word in words:\n"
                    "        word = word.strip('.,!?;:')\n"
                    "        found = False\n"
                    "        for i in range(len(unique_words)):\n"
                    "            if unique_words[i] == word:\n"
                    "                counts[i] = counts[i] + 1\n"
                    "                found = True\n"
                    "        if found == False:\n"
                    "            unique_words.append(word)\n"
                    "            counts.append(1)\n"
                    "    result = []\n"
                    "    for i in range(len(unique_words)):\n"
                    "        result.append((unique_words[i], counts[i]))\n"
                    "    result.sort(key=lambda x: x[1], reverse=True)\n"
                    "    return result\n"
                    "```\n\n"
                    "The refactored function must:\n"
                    "1. Be named `count_word_frequencies` with the same signature\n"
                    "2. Return the same output format: list of (word, count) tuples sorted by count descending\n"
                    "3. Handle punctuation stripping the same way"
                ),
                expected_answer="",
                scoring_method=ScoringMethod.CODE_EXECUTION,
                difficulty="hard",
                test_cases=[
                    {
                        "input": "the cat sat on the mat the cat",
                        "expected": [("the", 3), ("cat", 2), ("sat", 1), ("on", 1), ("mat", 1)],
                    },
                    {
                        "input": "hello, world! hello.",
                        "expected": [("hello", 2), ("world", 1)],
                    },
                    {
                        "input": "one",
                        "expected": [("one", 1)],
                    },
                ],
            ),
        ]
