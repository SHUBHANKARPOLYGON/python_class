# 0/1 Knapsack Problem using Dynamic Programming

def knapsack(weights, values, capacity):
    n = len(values)
    # Create a DP table
    dp = [[0 for x in range(capacity + 1)] for x in range(n + 1)]

    # Build table dp[][] in bottom-up manner
    for i in range(n + 1):
        for w in range(capacity + 1):
            if i == 0 or w == 0:
                dp[i][w] = 0
            elif weights[i-1] <= w:
                dp[i][w] = max(values[i-1] + dp[i-1][w - weights[i-1]], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]

    return dp[n][capacity]

# Example values and weights
values = [60, 100, 120]
weights = [10, 20, 30]
capacity = 50

# Run knapsack
max_value = knapsack(weights, values, capacity)
print(f"The maximum value that can be put in a knapsack of capacity {capacity} is {max_value}")
