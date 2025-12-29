import re
from datasets import load_dataset

def load_gsm8k_subset(split="train", n=20):
    """
    Loads real GSM8K data from Hugging Face.
    Returns a list of tuples: (Question, Ground_Truth_Number)
    """
    print(f"⏳ Loading GSM8K ({split} split, n={n})...")
    
    # Load streaming to avoid downloading the whole thing
    dataset = load_dataset("gsm8k", "main", split=split, streaming=True)
    
    data_pairs = []
    count = 0
    
    for item in dataset:
        if count >= n:
            break
            
        question = item['question']
        raw_answer = item['answer']
        
        # GSM8K format: "Reasoning... #### 1234"
        # We only want the "1234" part for validation
        ground_truth_num = raw_answer.split("####")[-1].strip()
        
        data_pairs.append((question, ground_truth_num))
        count += 1
        
    print(f"✅ Loaded {len(data_pairs)} examples.")
    return data_pairs

def parse_model_answer(text: str):
    """
    Robustly finds the last number in the text.
    Handles 'The answer is [[48]]' or just '48'.
    """
    # 1. Try to find double brackets [[number]] (Our target format)
    match = re.search(r"\[\[(\d+)\]\]", text)
    if match:
        return match.group(1)
    
    # 2. Fallback: Find the last number in the text (for loose grading)
    # This regex catches integers and decimals
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if numbers:
        return numbers[-1]
    
    return None