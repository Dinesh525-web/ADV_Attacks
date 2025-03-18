from rapidfuzz import fuzz

def evaluate_attack(original, attacked):
    """Compares original output vs. attacked output using fuzzy matching."""
    score = fuzz.ratio(original, attacked)
    return score
