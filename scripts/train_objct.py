import sys
import os
import uuid
import time

# Path setup to find 'src'
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.client import LocalLLMClient
from src.agent import ObjectCountStudent
from src.utils import load_bbh_object_count, parse_count_answer
from adalflow.optim.types import Gradient

# ==========================================
# 1. THE RESEARCH OPTIMIZER
# ==========================================
class ResearchOptimizer:
    def __init__(self, params, teacher_client):
        self.params = list(params)
        self.client = teacher_client # Uses the Strong Model (7B)

    def step(self):
        for param in self.params:
            if not getattr(param, "gradients", None): continue
            
            print(f"\n[OPTIMIZER] 🧠 Teacher (7B) is analyzing {len(param.gradients)} failures...")
            
            # 1. Aggregate Critiques from the batch
            error_log = ""
            for i, grad in enumerate(param.gradients):
                q = grad.from_response.data
                truth = grad.to_pred.data
                critique = grad.data
                # Truncate question for prompt safety
                error_log += f"Example {i+1}:\n- Question: {q[:150]}...\n- Target: {truth}\n- Teacher Critique: {critique}\n\n"

            # 2. Meta-Prompt: The "Gradient Descent" Logic
            meta_prompt = [
                {"role": "system", "content": "You are an expert Prompt Engineer optimizing a small LLM."},
                {"role": "user", "content": f"""
I have a small student model (1.5B parameters) trying to count objects.
Current Prompt: "{param.data}"

It failed on these examples:
{error_log}

Task: Write a BETTER prompt that guides the small model to success.
Strategies to apply:
1. Chain of Thought: Ask it to LIST the valid objects first.
2. Formatting: Enforce 'Answer: [[number]]' at the end.
3. Simplicity: Keep instructions short and clear.

Output ONLY the new prompt text. Do not explain.
"""}
            ]

            # 3. Generate New Prompt (Update Weights)
            new_prompt = self.client.call(api_kwargs={"messages": meta_prompt})
            
            # Clean up potential artifacts
            clean_prompt = new_prompt.strip().replace('"', '')
            print(f"[OPTIMIZER] 💡 New Prompt Proposed: {clean_prompt}")
            
            # Apply Update
            param.data = clean_prompt
            param.reset_gradients()

# ==========================================
# 2. DATA WRAPPER
# ==========================================
class MockTrace:
    def __init__(self, data, name="mock"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.data = data

# ==========================================
# 3. MAIN TRAINING LOOP
# ==========================================
def train():
    print("🚀 Starting Dual-Model BBH Experiment...")

    # --- LOAD MODELS ---
    # This fits on Colab T4 because we use 4-bit quantization for both.
    
    print("\n[1/2] Loading TEACHER (Qwen 2.5 7B)...")
    teacher_client = LocalLLMClient("Qwen/Qwen2.5-7B-Instruct")
    
    print("\n[2/2] Loading STUDENT (Qwen 2.5 1.5B)...")
    student_client = LocalLLMClient("Qwen/Qwen2.5-1.5B-Instruct")

    # --- SETUP COMPONENTS ---
    student = ObjectCountStudent(student_client)
    optimizer = ResearchOptimizer(student.parameters(), teacher_client)
    
    # Load Data (5 hard examples to start)
    train_data = load_bbh_object_count(n=5)
    
    print(f"\n[INIT PROMPT] {student.system_prompt.data}")

    # --- EPOCH LOOP ---
    for epoch in range(1, 4):
        print(f"\n================ EPOCH {epoch} ================")
        student.system_prompt.reset_gradients()
        errors = 0
        
        for item in train_data:
            q, truth = item['question'], item['truth']
            
            # 1. FORWARD PASS (Student 1.5B)
            # The student is fast!
            response = student(q)
            pred_raw = response.data
            pred_parsed = parse_count_answer(pred_raw)
            
            # 2. EVALUATION
            if pred_parsed == truth:
                print(f"✅ PASS | Pred: {pred_parsed} == True: {truth}")
            else:
                print(f"❌ FAIL | Pred: {pred_parsed} (Raw len: {len(pred_raw)}) | True: {truth}")
                errors += 1
                
                # 3. BACKWARD PASS (Teacher 7B)
                # The teacher analyzes the mistake ("Calculates Gradient")
                critique_prompt = [
                    {"role": "system", "content": "You are a logic teacher."},
                    {"role": "user", "content": f"Question: {q}\nStudent Answer: {pred_raw}\nTarget: {truth}\n\nAnalyze the student's error. Did they miss an item? Did they count non-objects? Be specific."}
                ]
                critique = teacher_client.call(api_kwargs={"messages": critique_prompt})
                
                # Store the gradient
                grad = Gradient(
                    data=critique,
                    score=0.0,
                    from_response=MockTrace(q, "Input"),
                    to_pred=MockTrace(truth, "Truth"),
                    data_id=str(uuid.uuid4())
                )
                student.system_prompt.add_gradient(grad)

        # 4. OPTIMIZATION STEP
        if errors > 0:
            print(f"\n📉 Found {errors} errors. Running Optimization Step...")
            optimizer.step()
        else:
            print("✨ Converged! Perfect accuracy achieved.")
            break

if __name__ == "__main__":
    train()