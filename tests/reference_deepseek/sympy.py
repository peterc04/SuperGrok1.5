import math


def isprime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    return all(n % f for f in range(3, math.isqrt(n) + 1, 2))
