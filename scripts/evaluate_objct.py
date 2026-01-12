import sys
import os
import re

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.client import LocalLLMClient
from src.agentct import ObjectCountStudent
from src.utils import load_bbh_object_count

def parse_answer(text):
    """ Robust parser for evaluation """
    if not text: return None
    # 1. Try strict format
    match = re.search(r"Answer:\s*\[\[(\d+)\]\]", text)
    if match: return int(match.group(1))
    # 2. Try last number
    numbers = re.findall(r"\d+", text)
    if numbers: return int(numbers[-1])
    return None

def evaluate(name, student, dataset):
    print(f"\n🔍 Evaluating: {name}...")
    correct = 0
    for i, item in enumerate(dataset):
        q, truth_raw = item['question'], item['truth']
        
        # --- 1. SAFE CAST TRUTH TO INT ---
        try:
            truth = int(str(truth_raw).strip())
        except ValueError:
            # If truth is "5 apples", try to extract 5
            numbers = re.findall(r"\d+", str(truth_raw))
            truth = int(numbers[0]) if numbers else -999

        response = student(q)
        pred = parse_answer(response.data)
        
        # --- 2. COMPARE ---
        is_correct = (pred == truth)
        
        if is_correct:
            correct += 1
            # Optional: Print passes to confirm it's working
            # print(f"   ✅ {name} | Pred: {pred} | Truth: {truth}")
        else:
            print(f"   ❌ {name} | Pred: {pred} | Truth: {truth}")
            
    acc = correct / len(dataset)
    print(f"📈 {name} Accuracy: {acc:.2%}")
    return acc

def run_evaluation():
    # 1. Load Data (Use a fresh set if possible, or same for demo)
    test_data = load_bbh_object_count(n=5) 
    client = LocalLLMClient("Qwen/Qwen2.5-1.5B-Instruct")
    
    # 2. BASELINE RUN
    # Initialize fresh student (loads defaults from class definition)
    baseline_student = ObjectCountStudent(client, model_kwargs={})
    acc_base = evaluate("Baseline (Default)", baseline_student, test_data)
    
    # 3. OPTIMIZED RUN
    # Initialize student and inject loaded prompts
    if not os.path.exists("outputs/optimized_instructions.txt"):
        print("❌ Error: No optimized artifacts found. Run training first!")
        return

    opt_student = ObjectCountStudent(client, model_kwargs={})
    
    with open("outputs/optimized_instructions.txt", "r") as f:
        opt_student.system_prompt.data = f.read().strip()
        
    with open("outputs/optimized_reasoning.txt", "r") as f:
        opt_student.reasoning_strategy.data = f.read().strip()
        
    print(f"\n[INFO] Loaded Optimized Instructions:\n{opt_student.system_prompt.data[:100]}...")
    
    acc_opt = evaluate("Optimized (Trained)", opt_student, test_data)
    
    # 4. REPORT
    print("\n" + "="*30)
    print("       FINAL RESULTS       ")
    print("="*30)
    print(f"Baseline:   {acc_base:.2%}")
    print(f"Optimized:  {acc_opt:.2%}")
    print(f"Improvement: {acc_opt - acc_base:+.2%}")

if __name__ == "__main__":
    run_evaluation()