import sys
import os
import csv
from typing import Any, Callable, Dict, Tuple

# Path setup
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import adalflow as adal
from adalflow.eval.answer_match_acc import AnswerMatchAcc

# --- LOCAL IMPORTS ---
from src.client import LocalLLMClient
from src.agentct import ObjectCountStudent
from src.utils import load_bbh_object_count

# ==========================================
# 1. ADAL COMPONENT (Local Adapter)
# ==========================================
class ObjectCountAdalComponent(adal.AdalComponent):
    def __init__(self, client):
        # Initialize the Student
        task = ObjectCountStudent(client)
        
        # Define Evaluation Metric
        eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
        
        super().__init__(task=task, eval_fn=eval_fn)

    def prepare_task(self, sample: Dict) -> Tuple[Callable, Dict[str, Any]]:
        # Map dataset 'question' to agent's 'input_str'
        return self.task.call, {"input_str": sample["question"]}

    def prepare_eval(self, sample: Dict, y_pred: adal.GeneratorOutput) -> Tuple[float, Dict[str, Any]]:
        # Extract data for evaluation
        y_label = y_pred.data
        return self.eval_fn, {"y": y_label, "y_gt": sample["truth"]}

# ==========================================
# 2. DIAGNOSIS RUNNER
# ==========================================
def diagnose(component, dataset, split_name="test"):
    print(f"\n🔍 Starting DIAGNOSIS on '{split_name}' split ({len(dataset)} samples)...")
    
    correct = 0
    errors = 0
    results = []

    print(f"📝 Current Prompt:\n{component.task.system_prompt.data}\n")

    for i, sample in enumerate(dataset):
        # 1. Prepare & Run Task
        func, kwargs = component.prepare_task(sample)
        response = func(**kwargs)
        
        # 2. Evaluate
        _, eval_kwargs = component.prepare_eval(sample, response)
        acc = component.eval_fn(**eval_kwargs)
        
        pred_text = eval_kwargs['y']
        truth = eval_kwargs['y_gt']
        
        if acc == 1.0:
            print(f"✅ [{i+1}] PASS | Pred: {pred_text}")
            correct += 1
            status = "PASS"
        else:
            print(f"❌ [{i+1}] FAIL | Pred: {pred_text} | True: {truth}")
            errors += 1
            status = "FAIL"
            
        results.append({
            "id": i,
            "status": status,
            "question": sample["question"][:50] + "...",
            "prediction": pred_text,
            "truth": truth
        })

    # Summary
    accuracy = correct / len(dataset)
    print(f"\n📊 DIAGNOSIS REPORT ({split_name}):")
    print(f"   - Accuracy: {accuracy:.2%}")
    print(f"   - Errors:   {errors}")
    
    # Save Report
    filename = f"diagnosis_{split_name}.csv"
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "status", "question", "prediction", "truth"])
        writer.writeheader()
        writer.writerows(results)
    print(f"📂 Detailed report saved to {filename}")

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load Client
    print("⏳ Loading Student Model...")
    client = LocalLLMClient("Qwen/Qwen2.5-1.5B-Instruct")

    # 2. Setup Component
    component = ObjectCountAdalComponent(client)

    # 3. Load Data
    # You can change n=100 to diagnose the full set
    test_data = load_bbh_object_count(n=10, split="test")

    # 4. Run Diagnosis
    diagnose(component, test_data, split_name="test")