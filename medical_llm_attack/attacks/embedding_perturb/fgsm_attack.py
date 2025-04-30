import torch
import torch.nn.functional as F

def fgsm_attack(model, input_ids, attention_mask, epsilon=0.01, max_perturb=0.1):
    """Memory-efficient FGSM attack with gradient clipping"""
    model.eval()
    
    # Convert to float32 for GPU compatibility
    input_ids = input_ids.to(model.device).long()
    attention_mask = attention_mask.to(model.device).float()
    
    # Create a fresh tensor that requires gradients
    with torch.no_grad():
        original_embeddings = model.get_input_embeddings()(input_ids)
    
    # Create a NEW leaf tensor with gradients enabled
    embeddings = original_embeddings.clone().detach().requires_grad_(True)
    
    # Forward pass with embeddings
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    
    # Calculate loss - adapt based on model type
    if hasattr(outputs, 'loss') and outputs.loss is not None:
        loss = outputs.loss
    elif hasattr(outputs, 'logits'):
        import torch.nn.functional as F
        logits = outputs.logits
        # Use logits to create a simple loss
        if len(logits.shape) == 2:  # Classification
            predicted = logits.argmax(dim=-1)
            loss = F.cross_entropy(logits, predicted)
        else:  # Language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), 
                                 shift_labels.reshape(-1))
    else:
        # Fallback - use any available tensor
        loss = outputs[0].mean() if isinstance(outputs, tuple) else outputs.mean()
    
    # Compute gradients
    loss.backward()
    
    # Now embeddings.grad should exist
    if embeddings.grad is None:
        raise ValueError("Gradient computation failed - check model outputs")
        
    with torch.no_grad():
        perturbation = epsilon * embeddings.grad.sign()
        perturbation = torch.clamp(perturbation, -max_perturb, max_perturb)
        adversarial_embeddings = embeddings + perturbation
    
    return adversarial_embeddings