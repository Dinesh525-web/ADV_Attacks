import torch
import torch.nn.functional as F

def pgd_attack(model, input_ids, attention_mask, epsilon=0.03, alpha=0.01, num_iter=10, max_perturb=0.1):
    """Projected Gradient Descent (PGD) attack implementation for embedding perturbation
    
    Args:
        model: The target model to attack
        input_ids: Token IDs of the input text
        attention_mask: Attention mask for the input
        epsilon: Maximum perturbation size
        alpha: Step size for each iteration
        num_iter: Number of attack iterations
        max_perturb: Maximum perturbation magnitude for any dimension
    
    Returns:
        Adversarially perturbed embeddings
    """
    model.eval()
    
    input_ids = input_ids.to(model.device).long()
    attention_mask = attention_mask.to(model.device).float()
    
    # Initialize adversarial embeddings as the original embeddings
    embeddings = model.get_input_embeddings()(input_ids)
    adv_embeddings = embeddings.clone().detach().requires_grad_(True)
    
    for _ in range(num_iter):
        outputs = model(inputs_embeds=adv_embeddings, attention_mask=attention_mask)
        
        # Calculate loss
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            loss = outputs.loss
        elif hasattr(outputs, 'logits'):
            logits = outputs.logits
            if len(logits.shape) == 2:  # classification
                predicted = logits.argmax(dim=-1)
                loss = F.cross_entropy(logits, predicted)
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_targets = logits[..., 1:, :].argmax(dim=-1).contiguous()
                loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1))
        else:
            first_tensor = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            loss = first_tensor.flatten().mean()
        
        model.zero_grad()
        if adv_embeddings.grad is not None:
            adv_embeddings.grad.data.zero_()
        loss.backward()
        
        with torch.no_grad():
            grad_sign = adv_embeddings.grad.sign()
            adv_embeddings += alpha * grad_sign
            # Project back to epsilon ball
            perturbation = torch.clamp(adv_embeddings - embeddings, min=-epsilon, max=epsilon)
            adv_embeddings = embeddings + perturbation
            adv_embeddings = torch.clamp(adv_embeddings, min=-max_perturb, max=max_perturb)
            adv_embeddings.requires_grad_(True)
    
    return adv_embeddings
