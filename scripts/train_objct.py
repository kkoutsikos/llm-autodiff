import sys
import os
import uuid
import random
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
from src.prompts import (OPTIMIZE_INSTRUCTIONS_TEMPLATE, 
                         # OPTIMIZE_DEMOS_TEMPLATE, # 
                         OPTIMIZE_REASONING_TEMPLATE)
from src.utils import load_bbh_object_count, parse_count_answer    
    
from src.utils import load_bbh_object_count, parse_count_answer

# ==========================================
# 1. HELPER: CSV LOGGER
# ==========================================
class CSVLogger:
    def __init__(self, filename="training_results.csv"):
        self.filename = filename
        # Initialize file with headers
        with open(self.filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Accuracy", "Errors", "Avg_Prompt_Len"])

    def log(self, epoch, acc, errors, prompt_len):
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, acc, errors, prompt_len])
        print(f"📝 Logged Epoch {epoch} stats to {self.filename}")

# ==========================================
# 2. SEQUENTIAL RESEARCH OPTIMIZER
# ==========================================
class ResearchOptimizer:
    def __init__(self, params, teacher_client, verbose = True):
        self.params = list(params)
        self.client = teacher_client 
        self.verbose = verbose


    def step(self):
        """
        Executes a Sequential Optimization Step.
        Order: Instructions -> Reasoning -> Examples
        """
        
        # 1. Define Priority Order
        priority_map = {
            "Task Instructions": 0,
            "Reasoning Strategy": 1
            
        }

        # DEBUG: Print what the optimizer sees
        total_grads = sum(len(getattr(p, "gradients", [])) for p in self.params)
        print(f"\n[OPTIMIZER DEBUG] Total parameters: {len(self.params)}")
        print(f"[OPTIMIZER DEBUG] Total gradients found: {total_grads}")
        
        if total_grads == 0:
            print("⚠️ WARNING: Optimizer called but no gradients found! Check your attachment logic.")
            return

        # Filter params with gradients and sort them
        active_params = [
            p for p in self.params 
            if hasattr(p, "gradients") 
            and p.gradients 
            and p.role_desc in priority_map 
        ]
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

            if context_updates:
                prompt_content += (
                    f"\n\n🚨 CONTEXT UPDATE 🚨\n"
                    f"The other prompt parts have just changed:\n{context_updates}\n"
                    f"Use this context to align your update, but DO NOT copy this header."
                )

            if self.verbose:
                print(f"   [DEBUG] Sending meta-prompt to Teacher ({system_role})...")

            # C. Call Teacher
            new_val = self.client.call(api_kwargs={
                "messages": [{"role": "system", "content": system_role},
                             {"role": "user", "content": prompt_content}]
            })
            
            
            clean_val = new_val.strip().replace('"', '')
            
            
            if "🚨 CONTEXT UPDATE 🚨" in clean_val:
                # Take everything AFTER the header
                clean_val = clean_val.split("🚨 CONTEXT UPDATE 🚨")[-1].strip()
            
            # Fix 2: Remove Markdown headers often generated by 7B models
            clean_val = clean_val.replace("**Reasoning Strategy:**", "").strip()
            clean_val = clean_val.replace("Reasoning Strategy:", "").strip()
            
            print(f"   💡 New {param.role_desc}:\n   --------------------------------------------------\n{clean_val}\n   --------------------------------------------------")

            # E. Update
            param.data = clean_val
            param.gradients = [] 
            context_updates += f"\n[{param.role_desc}]: {clean_val}\n"

# ==========================================
# 3. MAIN LOOP
# ==========================================
def train():



    
    BATCH_SIZE = 8       
    DATASET_SIZE = 40    
    EPOCHS = 4

    print(f"🚀 Starting Mini-Batch Training (N={DATASET_SIZE}, Batch={BATCH_SIZE})...")
    logger = CSVLogger("training_results.csv")

    teacher_client = LocalLLMClient("Qwen/Qwen2.5-7B-Instruct")
    student_client = LocalLLMClient("Qwen/Qwen2.5-1.5B-Instruct")

    
    
    student = ObjectCountStudent(student_client, model_kwargs={})
    trainable_params = [
        student.system_prompt, 
        student.reasoning_strategy
        
    ]
    optimizer = ResearchOptimizer(trainable_params, teacher_client, verbose=True)
    backward_engine = TextualBackwardEngine(teacher_client)
    
    print("📚 Loading Data...")
    train_data = load_bbh_object_count(n=DATASET_SIZE)
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n\n================ EPOCH {epoch} ================")
        random.shuffle(train_data)
        
        epoch_correct = 0
        epoch_errors = 0
        
        batch_errors = 0
        current_batch_count = 0
        
        for i, item in enumerate(train_data):
            q, truth = item['question'], item['truth']
            
            response = student(q)
            pred_raw = response.data
            pred_parsed = parse_count_answer(pred_raw)
            
            if pred_parsed == truth:
                print(f"  ✅ Pass: {pred_parsed} == {truth}")
                epoch_correct += 1
            else:
                epoch_errors += 1
                batch_errors += 1
                print(f"  ❌ Fail: {pred_parsed} vs {truth}")
                
                grad = backward_engine.compute_gradient(q, pred_raw, truth)
                
                # Safe Gradient Attachment
                for param in [student.system_prompt, student.reasoning_strategy]:
                    if not hasattr(param, "gradients") or not isinstance(param.gradients, list):
                        param.gradients = []
                    param.gradients.append(grad)

            current_batch_count += 1
            
            # Mini-Batch Update Trigger
            if current_batch_count >= BATCH_SIZE or (i == len(train_data) - 1):
                if batch_errors > 0:
                    optimizer.step()
                else:
                    print(f"  ✨ Batch Perfect! No updates needed.")
                
                current_batch_count = 0
                batch_errors = 0
                student.system_prompt.gradients = []
                student.reasoning_strategy.gradients = []

        # --- LOGGING STEP ---
        epoch_acc = epoch_correct / DATASET_SIZE
        prompt_len = len(student.system_prompt.data) + len(student.reasoning_strategy.data)
        logger.log(epoch, epoch_acc, epoch_errors, prompt_len)
        print(f"📊 Epoch Stats: Acc={epoch_acc:.2%} | Errors={epoch_errors}")

    print("\n💾 Saving Optimized Prompts...")
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/optimized_instructions.txt", "w") as f: f.write(student.system_prompt.data)
    with open("outputs/optimized_reasoning.txt", "w") as f: f.write(student.reasoning_strategy.data)
    print("✅ Done.")

if __name__ == "__main__":
    train()