from attacks.prompt_injection.injector import medical_prompt_injection
from attacks.embedding_perturb.fgsm_attack import fgsm_attack

def hybrid_attack(model, tokenizer, question, attack_params):
    """End-to-end hybrid attack pipeline"""
    try:
        # Stage 1: Prompt Injection
        injected_response = medical_prompt_injection(
            model, tokenizer, question, 
            attack_params.get("prefix", "Ignore previous instructions")
        )
        
        # Stage 2: Embedding Perturbation
        inputs = tokenizer(
            injected_response,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)
        
        adv_embeddings = fgsm_attack(
            model,
            inputs.input_ids,
            inputs.attention_mask,
            epsilon=attack_params.get("epsilon", 0.03)
        )
        
        # Stage 3: Adversarial Generation
        outputs = model.generate(
            inputs_embeds=adv_embeddings,
            attention_mask=inputs.attention_mask,
            max_new_tokens=200,
            temperature=0.7
        )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    except RuntimeError as e:
        print(f"Hybrid attack failed: {str(e)}")
        return ""
