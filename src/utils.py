import re
from datasets import load_dataset

def load_gsm8k_dataset(split="train", n=20):
    """
    Returns a list of dictionaries for cleaner iteration.
    """
    print(f"📚 Loading GSM8K ({n} examples)...")
    dataset = load_dataset("gsm8k", "main", split=split, streaming=True)
    
    batch_data = []
    count = 0
    for item in dataset:
        if count >= n: break
        truth = item['answer'].split("####")[-1].strip()
        # Clean commas from truth (e.g. 1,000 -> 1000)
        truth = truth.replace(",", "")
        
        batch_data.append({
            "question": item['question'],
            "truth": truth,
            "id": count
        })
        count += 1
    return batch_data

def parse_model_answer(text: str):
    if not text: return None
    # 1. Look for [[number]]
    match = re.search(r"\[\[(\d+)\]\]", text)
    if match: return match.group(1)
    
    # 2. Look for last number (robust fallback)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None