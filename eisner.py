"""
Eisner's algorithm for projective dependency parsing.
O(n^3) dynamic programming — finds the optimal projective tree
that maximizes total edge score.
"""
import numpy as np

def eisner_parse(scores):
    """
    Eisner's algorithm for maximum spanning projective dependency tree.
    
    Args:
        scores: np.array of shape (n, n) where scores[h][d] = score of head h → dep d
                Row 0 is the ROOT node.
    
    Returns:
        heads: list of length n, where heads[d] = head of token d (0 = ROOT)
    """
    n = len(scores)
    
    # Complete spans: C[s][t][d][c]
    # s = start, t = end, d = direction (0=left, 1=right), c = complete (0=incomplete, 1=complete)
    C = np.full((n, n, 2, 2), -np.inf)
    bp = np.full((n, n, 2, 2), -1, dtype=int)
    
    # Base case: single words
    for i in range(n):
        C[i][i][0][1] = 0.0
        C[i][i][1][1] = 0.0
    
    # Fill in spans of increasing width
    for width in range(1, n):
        for s in range(n - width):
            t = s + width
            
            # Incomplete spans (creating a new edge)
            # Right incomplete: s→t (head s, dep t)
            for q in range(s, t):
                val = C[s][q][1][1] + C[q+1][t][0][1] + scores[s][t]
                if val > C[s][t][1][0]:
                    C[s][t][1][0] = val
                    bp[s][t][1][0] = q
            
            # Left incomplete: t→s (head t, dep s)
            for q in range(s, t):
                val = C[s][q][1][1] + C[q+1][t][0][1] + scores[t][s]
                if val > C[s][t][0][0]:
                    C[s][t][0][0] = val
                    bp[s][t][0][0] = q
            
            # Complete spans (closing an incomplete span)
            # Right complete
            for q in range(s+1, t+1):
                val = C[s][q][1][0] + C[q][t][1][1]
                if val > C[s][t][1][1]:
                    C[s][t][1][1] = val
                    bp[s][t][1][1] = q
            
            # Left complete
            for q in range(s, t):
                val = C[s][q][0][1] + C[q][t][0][0]
                if val > C[s][t][0][1]:
                    C[s][t][0][1] = val
                    bp[s][t][0][1] = q
    
    # Backtrack to find heads
    heads = [0] * n
    
    def backtrack(s, t, d, c):
        if s == t:
            return
        q = bp[s][t][d][c]
        if q < 0:
            return
        
        if c == 0:  # incomplete
            if d == 1:  # right: s→t
                heads[t] = s
                backtrack(s, q, 1, 1)
                backtrack(q+1, t, 0, 1)
            else:  # left: t→s
                heads[s] = t
                backtrack(s, q, 1, 1)
                backtrack(q+1, t, 0, 1)
        else:  # complete
            if d == 1:  # right complete
                backtrack(s, q, 1, 0)
                backtrack(q, t, 1, 1)
            else:  # left complete
                backtrack(s, q, 0, 1)
                backtrack(q, t, 0, 0)
    
    backtrack(0, n-1, 1, 1)
    return heads


if __name__ == "__main__":
    # Quick test
    scores = np.array([
        [0, 5, 1, 1],  # ROOT scores
        [0, 0, 3, 1],  # word 1 as head
        [0, 1, 0, 4],  # word 2 as head
        [0, 1, 1, 0],  # word 3 as head
    ], dtype=float)
    heads = eisner_parse(scores)
    print(f"Heads: {heads}")  # Expected: [0, 0, 1, 2] or similar optimal tree
