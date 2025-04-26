# Minimax Algorithm with Alpha-Beta Pruning

def minimax(depth, node_index, is_maximizing_player, values, alpha, beta):
    # Terminal condition (leaf node)
    if depth == 3:
        return values[node_index]

    if is_maximizing_player:
        best = float('-inf')

        # Recur for left and right children
        for i in range(2):
            val = minimax(depth + 1, node_index * 2 + i, False, values, alpha, beta)
            best = max(best, val)
            alpha = max(alpha, best)

            # Alpha Beta Pruning
            if beta <= alpha:
                break
        
        return best
    else:
        best = float('inf')

        # Recur for left and right children
        for i in range(2):
            val = minimax(depth + 1, node_index * 2 + i, True, values, alpha, beta)
            best = min(best, val)
            beta = min(beta, best)

            # Alpha Beta Pruning
            if beta <= alpha:
                break
        
        return best

# Example values at leaf nodes
values = [3, 5, 6, 9, 1, 2, 0, -1]

# Call minimax
print("The optimal value is:", minimax(0, 0, True, values, float('-inf'), float('inf')))
