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

try:
    from adalflow.optim import Gradient
except ImportError:
    # Fallback for different versions
    from adalflow.core.types import Gradient
    

from src.gradient import TextualBackwardEngine
from src.prompts import OPTIMIZE_INSTRUCTIONS_TEMPLATE, OPTIMIZE_DEMOS_TEMPLATE, OPTIMIZE_REASONING_TEMPLATE
from src.utils import load_bbh_object_count, parse_count_answer    
    
from src.utils import load_bbh_object_count, parse_count_answer

# ==========================================
# 1. HELPER: CSV LOGGER
# ==========================================
class CSVLogger:
    def __init__(self, filename="training_log.csv"):
        self.filename = filename
        with open(self.filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Accuracy", "Errors", "Format_Failures", "Total_Prompt_Len"])
        print(f"📝 Logging metrics to {self.filename}")

    def log(self, epoch, acc, errors, format_fails, prompt_len):
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, acc, errors, format_fails, prompt_len])

# ==========================================
# 2. SEQUENTIAL RESEARCH OPTIMIZER
# ==========================================
class ResearchOptimizer:
    def __init__(self, params, teacher_client):
        self.params = list(params)
        self.client = teacher_client 

    def step(self):
        """
        Executes a Sequential Optimization Step.
        Order: Instructions -> Reasoning -> Examples
        """
        
        # 1. Define Priority Order
        priority_map = {
            "Task Instructions": 0,
            "Reasoning Strategy": 1,
            "Few Shot Examples": 2
        }
        
        # Filter params with gradients and sort them
        active_params = [p for p in self.params if hasattr(p, "gradients") and p.gradients]
        sorted_params = sorted(active_params, key=lambda p: priority_map.get(p.role_desc, 99))
        
        if not sorted_params: return

        print(f"\n[OPTIMIZER] 🔄 Starting Sequential Update Cascade ({len(sorted_params)} steps)...")

        # 2. Track Context Changes (To prevent drift)
        context_updates = ""

        for param in sorted_params:
            grads = param.gradients
            print(f"\n   👉 Step {priority_map.get(param.role_desc)}: Optimizing {param.role_desc}...")
            
            # --- A. Build Error Log ---
            error_log = ""
            for i, grad in enumerate(grads):
                q = grad.from_response.data 
                truth = grad.to_pred.data
                critique = grad.data
                error_log += f"Ex {i+1}:\nQ: {q[:80]}...\nTarget: {truth}\nCritique: {critique}\n\n"

            # --- B. Select Template ---
            if "Example" in param.role_desc:
                template = OPTIMIZE_DEMOS_TEMPLATE
                system_role = "You are a teacher."
            elif "Reasoning" in param.role_desc:
                template = OPTIMIZE_REASONING_TEMPLATE
                system_role = "You are a Cognitive Scientist."
            else:
                template = OPTIMIZE_INSTRUCTIONS_TEMPLATE
                system_role = "You are an expert Prompt Engineer."

            # --- C. Format Prompt ---
            prompt_content = template.format(
                current_prompt=param.data,
                error_log=error_log
            )

            # --- D. INJECT CONTEXT ---
            if context_updates:
                prompt_content += (
                    f"\n\n🚨 CONTEXT UPDATE 🚨\n"
                    f"Other prompt parts have just changed:\n{context_updates}\n"
                    f"Ensure your update aligns with this."
                )

            # --- E. Call Teacher ---
            new_val = self.client.call(api_kwargs={
                "messages": [{"role": "system", "content": system_role},
                             {"role": "user", "content": prompt_content}]
            })
            
            clean_val = new_val.strip().replace('"', '')
            print(f"   💡 New {param.role_desc} Generated.")

            # --- F. Update ---
            param.data = clean_val
            param.gradients = [] 
            context_updates += f"\n[{param.role_desc}]: {clean_val}\n"

# ==========================================
# 3. MAIN LOOP
# ==========================================
def train():
    print("🚀 Starting Sequential 3-Parameter Experiment...")
    logger = CSVLogger("training_results.csv")

    teacher_client = LocalLLMClient("Qwen/Qwen2.5-7B-Instruct")
    student_client = LocalLLMClient("Qwen/Qwen2.5-1.5B-Instruct")

    student = ObjectCountStudent(student_client)
    optimizer = ResearchOptimizer(student.parameters(), teacher_client)
    backward_engine = TextualBackwardEngine(teacher_client)
    
    train_data = load_bbh_object_count(n=5)
    
    for epoch in range(1, 6):
        print(f"\n================ EPOCH {epoch} ================")
        print(f"[INSTRUCT]  {student.system_prompt.data[:100]}...")
        print(f"[REASONING] {student.reasoning_strategy.data[:100]}...")
        print(f"[DEMOS]     {student.few_shot_demos.data[:100]}...")
        
        # Clear Gradients
        student.system_prompt.gradients = []
        student.reasoning_strategy.gradients = []
        student.few_shot_demos.gradients = []
        
        errors = 0
        correct = 0
        format_fails = 0
        
        for item in train_data:
            q, truth = item['question'], item['truth']
            
            # Forward
            response = student(q)
            pred_raw = response.data
            pred_parsed = parse_count_answer(pred_raw)
            
            if pred_parsed == truth:
                print(f"✅ PASS")
                correct += 1
            else:
                print(f"❌ FAIL | Pred: {pred_parsed} | True: {truth}")
                errors += 1
                if "Answer: [[" not in pred_raw: format_fails += 1
                
                # Backward
                grad = backward_engine.compute_gradient(q, pred_raw, truth)
                
                # Distribute Gradient to ALL 3 parameters
                if not hasattr(student.system_prompt, "gradients"): student.system_prompt.gradients = []
                if not hasattr(student.reasoning_strategy, "gradients"): student.reasoning_strategy.gradients = []
                if not hasattr(student.few_shot_demos, "gradients"): student.few_shot_demos.gradients = []
                
                student.system_prompt.gradients.append(grad)
                student.reasoning_strategy.gradients.append(grad)
                student.few_shot_demos.gradients.append(grad)

        # Log
        acc = correct / len(train_data)
        total_len = len(student.system_prompt.data) + len(student.reasoning_strategy.data) + len(student.few_shot_demos.data)
        logger.log(epoch, acc, errors, format_fails, total_len)
        print(f"📊 Epoch Stats: Acc={acc:.2f} | Errors={errors}")

        if errors > 0:
            optimizer.step()
        else:
            print("✨ Converged!")
            break

if __name__ == "__main__":
    train()