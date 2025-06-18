from math import factorial

def RuleOfProduct(n1, n2):
    return n1 * n2

def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

def Combinations(n, k):
    return factorial(n) / (factorial(k) * factorial(n - k))

def Permutations(n, k):
    return factorial(n) / factorial(n - k)

def CombinationWithRepetition(n, k):
    return Combinations(n + k - 1, k)

def PermutationWithRepetition(n, k):
    return n ** k

# p.24 Example 2.6
sampleSpace = Combinations(25, 5)
events = Combinations(10, 2) * Combinations(15, 3) + Combinations(10, 3) * \
         Combinations(15, 2) + Combinations(10, 4) * Combinations(15, 1) + \
         Combinations(10, 5) * Combinations(15, 0)

print(format(events / sampleSpace, ".3f"))