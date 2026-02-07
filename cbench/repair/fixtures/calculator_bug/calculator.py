"""Simple calculator with an off-by-one bug in the factorial function."""


def add(a: float, b: float) -> float:
    return a + b


def subtract(a: float, b: float) -> float:
    return a - b


def multiply(a: float, b: float) -> float:
    return a * b


def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def factorial(n: int) -> int:
    """Compute n! — BUG: off-by-one, starts range at 2 but misses n itself."""
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n == 0:
        return 1
    result = 1
    for i in range(2, n):  # BUG: should be range(2, n + 1)
        result *= i
    return result


def power(base: float, exp: int) -> float:
    """Compute base^exp — works correctly."""
    return base ** exp
