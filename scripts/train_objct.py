import sys
import os
import uuid
import random
import time
import csv
import re
# Path setup to find 'src'
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.prompts import render_gdlo_prompt
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
    def __init__(self, params, teacher_client):
        self.params = list(params)
        self.client = teacher_client 
        
        # Track history PER PARAMETER (using ID as key)
        self.best_history = {id(p): [] for p in self.params}
        self.failed_history = {id(p): [] for p in self.params}
        self.steps = 0

    def step(self):
        self.steps += 1
        print(f"\n⚡ [GDLO] Step {self.steps}: Generating Proposal...")
        
        for param in self.params:
            if not getattr(param, "gradients", None): 
                continue
            
            p_id = id(param)
            
            # 1. Render Prompt
            prompt_content = render_gdlo_prompt(
                param,
                param.gradients,
                steps=self.steps,
                past_history=self.best_history[p_id][-3:],     # Last 3 Best
                failed_proposals=self.failed_history[p_id][-3:] # Last 3 Failed
            )

            # 2. Call Teacher
            # Note: GDLO includes the system prompt inside the user message structure
            new_val = self.client.call(api_kwargs={
                "messages": [{"role": "user", "content": prompt_content}]
            })
            
            # 3. Clean & Update
            clean_val = new_val.strip().replace('"', '')
            if "<START>" in clean_val: clean_val = clean_val.split("<START>")[-1] # Basic cleaning
            
            # Store in history (Assuming success for now - in full AdalFlow we check score first)
            self.best_history[p_id].append(clean_val)
            
            print(f"   ✅ [GDLO] Updated {getattr(param, 'role_desc', 'Parameter')} (History Depth: {len(self.best_history[p_id])})")
            
            # Apply Update
            param.data = clean_val
            param.gradients = []

# ==========================================
# 3. MAIN LOOP
# ==========================================
def parse_count_answer(text):
    if not text: return None
    match = re.search(r"Answer:\s*\[\[(\d+)\]\]", text, re.IGNORECASE)
    if match: return int(match.group(1))
    numbers = re.findall(r"\d+", text)
    if numbers: return int(numbers[-1])
    return None

def train():
    # CONFIG
    BATCH_SIZE = 4       
    DATASET_SIZE = 20    
    EPOCHS = 4
    
    print(f"🚀 Starting GDLO Training (N={DATASET_SIZE})...")
    
    # Clients
    teacher_client = LocalLLMClient("Qwen/Qwen2.5-7B-Instruct")
    student_client = LocalLLMClient("Qwen/Qwen2.5-1.5B-Instruct")
    
    # Student (Single System Prompt)
    student = ObjectCountStudent(student_client, model_kwargs={})
    
    # Engine
    optimizer = ResearchOptimizer([student.system_prompt], teacher_client)
    backward_engine = TextualBackwardEngine(teacher_client)
    
    # Data
    train_data = load_bbh_object_count(n=DATASET_SIZE)
    
    # Logging
    with open("training_results_gdlo.csv", "w", newline='') as f:
        csv.writer(f).writerow(["Epoch", "Accuracy", "Errors", "Prompt_Len"])

    for epoch in range(1, EPOCHS + 1):
        print(f"\n\n================ EPOCH {epoch} ================")
        random.shuffle(train_data)
        
        epoch_correct = 0
        batch_errors = 0
        current_batch = 0
        
        for i, item in enumerate(train_data):
            q, truth_raw = item['question'], item['truth']
            try:
                truth = int(str(truth_raw).strip())
            except:
                truth = -999

            # Forward
            response = student(q)
            pred = parse_count_answer(response.data)
            
            # Check
            if pred == truth:
                print(f"  ✅ Pass: {pred} == {truth}")
                epoch_correct += 1
            else:
                print(f"  ❌ Fail: {pred} vs {truth}")
                batch_errors += 1
                
                # Backward
                grad = backward_engine.compute_gradient(q, student.system_prompt.data, response.data, truth)
                if not hasattr(student.system_prompt, "gradients"):
                    student.system_prompt.gradients = []
                student.system_prompt.gradients.append(grad)

            current_batch += 1
            
            # GDLO Update Step
            if current_batch >= BATCH_SIZE:
                if batch_errors > 0:
                    optimizer.step()
                else:
                    print("  ✨ Batch Perfect!")
                
                current_batch = 0
                batch_errors = 0
                student.system_prompt.gradients = []

        # Log Stats
        acc = epoch_correct / DATASET_SIZE
        print(f"📊 Epoch Stats: Acc={acc:.2%}")
        with open("training_results_gdlo.csv", "a", newline='') as f:
            csv.writer(f).writerow([epoch, acc, DATASET_SIZE-epoch_correct, len(student.system_prompt.data)])

    print("\n💾 Saving Optimized Artifacts...")
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/optimized_instructions.txt", "w") as f: 
        f.write(student.system_prompt.data)
    print("✅ Done.")

if __name__ == "__main__":
    train()