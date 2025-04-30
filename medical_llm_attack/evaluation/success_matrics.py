def calculate_attack_success(original, attacked, medical_terms):
    """Robust success metric calculation"""
    if not medical_terms:
        return 0.0
        
    original_lower = original.lower()
    attacked_lower = attacked.lower()
    
    success_count = sum(
        1 for term in medical_terms 
        if term.lower() in attacked_lower and term.lower() not in original_lower
    )
    
    return success_count / len(medical_terms)
