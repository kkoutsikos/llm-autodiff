import sys
import os
import uuid
import time
import csv
# Path setup to find 'src'
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.client import LocalLLMClient
from src.agentct import ObjectCountStudent
from src.utils import load_bbh_object_count, parse_count_answer
try:
    from adalflow.optim import Gradient
except ImportError:
    # Fallback for different versions
    from adalflow.core.types import Gradient
# ==========================================
# 1. CUSTOM DATA STRUCTURES
# ==========================================
class ResearchGradient:
    def __init__(self, data, from_response, to_pred, score=0.0):
        self.data = data
        self.from_response = from_response
        self.to_pred = to_pred
        self.score = score

class MockTrace:
    def __init__(self, data, name="mock"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.data = data
        self.component_trace = self


class CSVLogger:
    """Simple logger to save training stats to a CSV file."""
    def __init__(self, filename="training_log.csv"):
        self.filename = filename
        # Initialize file with headers
        with open(self.filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Accuracy", "Errors", "Format_Failures", "Prompt_Length"])
        print(f"📝 Logging metrics to {self.filename}")

    def log(self, epoch, acc, errors, format_fails, prompt_len):
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, acc, errors, format_fails, prompt_len])



# ==========================================
# 2. SMART RESEARCH OPTIMIZER (Handles Demos!)
# ==========================================
class ResearchOptimizer:
    def __init__(self, params, teacher_client):
        self.params = list(params)
        self.client = teacher_client 

    def step(self):
        for param in self.params:
            grads = getattr(param, "gradients", [])
            if not grads: continue
            
            print(f"\n[OPTIMIZER] 🧠 Analyzing {len(grads)} failures for: {param.role_desc}...")
            
            # 1. Build Error Log
            error_log = ""
            for i, grad in enumerate(grads):
                q = grad.from_response.data 
                truth = grad.to_pred.data
                critique = grad.data
                error_log += f"Ex {i+1}:\nQ: {q[:100]}...\nExpected: {truth}\nTeacher Note: {critique}\n\n"

            # 2. Select Meta-Prompt based on Parameter Type
            if "Example" in param.role_desc or "Demos" in param.role_desc:
                # === STRATEGY A: OPTIMIZE EXAMPLES ===
                print("[OPTIMIZER] 🔧 Tuning FEW-SHOT EXAMPLES...")
                meta_prompt = [
                    {"role": "system", "content": "You are a teacher creating exam prep materials."},
                    {"role": "user", "content": f"""
I have a student failing to count objects. 
Current Examples provided to student:
"{param.data}"

The student is still making these mistakes:
{error_log}

Task: Write NEW, BETTER examples (Q&A pairs) that specifically address these mistakes.
- If they miss negative constraints ("no flute"), include an example with "no X".
- If they miss formatting, show the correct format "Answer: [[N]]".
- Keep it concise.

Output ONLY the new text for the examples.
"""}
                ]
            else:
                # === STRATEGY B: OPTIMIZE INSTRUCTIONS ===
                print("[OPTIMIZER] 🔧 Tuning SYSTEM INSTRUCTIONS...")
                meta_prompt = [
                    {"role": "system", "content": "You are an expert Prompt Engineer."},
                    {"role": "user", "content": f"""
My agent is failing to count objects.
Current Instructions: "{param.data}"

Failures:
{error_log}

Task: Write BETTER instructions to fix this. 
- Be specific about listing items.
- Enforce the format "Answer: [[number]]".

Output ONLY the new instruction text.
"""}
                ]

            # 3. Generate & Update
            new_val = self.client.call(api_kwargs={"messages": meta_prompt})
            clean_val = new_val.strip().replace('"', '')
            
            print(f"\n[OPTIMIZER] 💡 New {param.role_desc}:/n{clean_val}\n")
            
            param.data = clean_val
            param.gradients = []

# ==========================================
# 3. MAIN LOOP
# ==========================================
def train():
    print("🚀 Starting Few-Shot BBH Experiment (CSV Logging)...")
    
    # Init Logger
    logger = CSVLogger("training_results.csv")

    print("\n[1/2] Loading TEACHER (Qwen 7B)...")
    teacher_client = LocalLLMClient("Qwen/Qwen2.5-7B-Instruct")
    print("\n[2/2] Loading STUDENT (Qwen 1.5B)...")
    student_client = LocalLLMClient("Qwen/Qwen2.5-1.5B-Instruct")

    student = ObjectCountStudent(student_client)
    optimizer = ResearchOptimizer(student.parameters(), teacher_client)
    
    train_data = load_bbh_object_count(n=5)
    
    for epoch in range(1, 6):
        print(f"\n================ EPOCH {epoch} ================")
        print(f"[INSTRUCTIONS] {student.system_prompt.data}")
        
        student.system_prompt.gradients = []
        student.few_shot_demos.gradients = []
        
        errors = 0
        correct = 0
        format_fails = 0
        
        for item in train_data:
            q, truth = item['question'], item['truth']
            
            response = student(q)
            pred_raw = response.data
            pred_parsed = parse_count_answer(pred_raw)
            
            if pred_parsed == truth:
                print(f"✅ PASS | {pred_parsed} == {truth}")
                correct += 1
            else:
                print(f"❌ FAIL | Pred: {pred_parsed} | True: {truth}")
                errors += 1
                if "Answer: [[" not in pred_raw: format_fails += 1
                
                critique = teacher_client.call(api_kwargs={"messages": [
                    {"role": "user", "content": f"Q: {q}\nStudent: {pred_raw}\nTarget: {truth}\nAnalyze error."}
                ]})
                
                grad = ResearchGradient(critique, MockTrace(q), MockTrace(truth))
                if not hasattr(student.system_prompt, "gradients"): student.system_prompt.gradients = []
                if not hasattr(student.few_shot_demos, "gradients"): student.few_shot_demos.gradients = []
                
                student.system_prompt.gradients.append(grad)
                student.few_shot_demos.gradients.append(grad)

        # LOGGING
        acc = correct / len(train_data)
        logger.log(epoch, acc, errors, format_fails, len(student.system_prompt.data))
        print(f"📊 Stats saved: Acc={acc:.2f}")

        if errors > 0:
            optimizer.step()
        else:
            print("✨ Converged!")
            break

if __name__ == "__main__":
    train()