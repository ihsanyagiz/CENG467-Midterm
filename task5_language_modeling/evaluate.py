"""
Task 5: Language Modeling Evaluation
"""

import math

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_perplexity(avg_loss):
    """Compute perplexity from average cross-entropy loss."""
    return math.exp(min(avg_loss, 20))
