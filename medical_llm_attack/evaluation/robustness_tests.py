import torch
from transformers import pipeline

def stress_test(model, tokenizer, test_cases, max_length=100):
    results = []
    for case in test_cases:
        inputs = tokenizer(case, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=max_length,
                num_return_sequences=1
            )
        results.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return results

def check_response_consistency(model, tokenizer, question, num_samples=3):
    responses = []
    for _ in range(num_samples):
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        outputs = model.generate(
            inputs.input_ids,
            temperature=0.7,
            do_sample=True
        )
        responses.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return responses

def measure_performance(model, tokenizer, dataset, batch_size=4):
    from tqdm import tqdm
    losses = []
    
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        inputs = tokenizer(
            batch["text"],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        losses.append(outputs.loss.item())
    
    return sum(losses) / len(losses)
