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


def load_bbh_object_count(n=20, split="test"):
    """
    Loads the 'object_counting' subset of Big Bench Hard.
    """
    print(f"📚 Loading BBH (Object Counting) | Split: {split} | n={n}")
    
    # We use 'lukaemon/bbh' which is a standard mirror of the dataset
    try:
        dataset = load_dataset("lukaemon/bbh", "object_counting", split=split, streaming=True)
    except Exception as e:
        print(f"⚠️ Failed to load BBH from HuggingFace: {e}")
        return []

    batch = []
    for i, item in enumerate(dataset):
        if i >= n: break
        
        # BBH items have 'input' (the question) and 'target' (the answer)
        # We strip the target to just get the number string
        truth = item['target'].strip()
        
        batch.append({
            "id": i,
            "question": item['input'],
            "truth": truth
        })
        
    print(f"✅ Loaded {len(batch)} examples.")
    return batch

def parse_count_answer(text: str):
    """
    Specific parser for counting tasks.
    Prioritizes explicit formats like "Answer: 3" or "[[3]]".
    """
    if not text: 
        return None
    
    # 1. Look for [[number]] (AdalFlow standard)
    bracket = re.search(r"\[\[(\d+)\]\]", text)
    if bracket:
        return bracket.group(1)

    # 2. Look for explicit format: "The answer is 3" or "Answer: 3"
    explicit = re.search(r"(?:answer is|answer:)\s*(\d+)", text, re.IGNORECASE)
    if explicit:
        return explicit.group(1)
        
    # 3. Fallback: Find all integers and take the LAST one
    # (Reasoning usually lists items first, then gives the count at the end)
    numbers = re.findall(r"\b\d+\b", text)
    if numbers:
        return numbers[-1]
        
    return None