import sys
import os
import uuid
import time

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
# 1. CUSTOM DATA STRUCTURES (The Fix)
# ==========================================
class ResearchGradient:
    """
    A simple container to hold the critique and context.
    We use this INSTEAD of AdalFlow's Gradient to avoid version conflicts.
    """
    def __init__(self, data, from_response, to_pred, score=0.0):
        self.data = data               # The Teacher's Critique (String)
        self.from_response = from_response # The Student's Output (MockTrace)
        self.to_pred = to_pred         # The Ground Truth (MockTrace)
        self.score = score

class MockTrace:
    def __init__(self, data, name="mock"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.data = data
        self.component_trace = self
# ==========================================
# 2. THE RESEARCH OPTIMIZER
# ==========================================
class ResearchOptimizer:
    def __init__(self, params, teacher_client):
        self.params = list(params)
        self.client = teacher_client 

    def step(self):
        for param in self.params:
            # We access the list directly to find our ResearchGradient objects
            grads = getattr(param, "gradients", [])
            if not grads: continue
            
            print(f"\n[OPTIMIZER] 🧠 Teacher (7B) is analyzing {len(grads)} failures...")
            
            # 1. Aggregate Critiques
            error_log = ""
            for i, grad in enumerate(grads):
                # NOW this works because 'grad' is our custom ResearchGradient
                q = grad.from_response.data 
                truth = grad.to_pred.data
                critique = grad.data
                
                error_log += f"Example {i+1}:\n- Question: {q[:150]}...\n- Target: {truth}\n- Teacher Critique: {critique}\n\n"

            # 2. Meta-Prompt
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

            # 3. Generate New Prompt
            new_prompt = self.client.call(api_kwargs={"messages": meta_prompt})
            clean_prompt = new_prompt.strip().replace('"', '')
            print(f"[OPTIMIZER] 💡 New Prompt Proposed: {clean_prompt}")
            
            # 4. Apply Update & Clear
            param.data = clean_prompt
            param.gradients = []

# ==========================================
# 3. MAIN TRAINING LOOP
# ==========================================
def train():
    print("🚀 Starting Dual-Model BBH Experiment (Robust Mode)...")

    # --- LOAD MODELS ---
    print("\n[1/2] Loading TEACHER (Qwen 2.5 7B)...")
    teacher_client = LocalLLMClient("Qwen/Qwen2.5-7B-Instruct")
    
    print("\n[2/2] Loading STUDENT (Qwen 2.5 1.5B)...")
    student_client = LocalLLMClient("Qwen/Qwen2.5-1.5B-Instruct")

    # --- SETUP COMPONENTS ---
    student = ObjectCountStudent(student_client)
    optimizer = ResearchOptimizer(student.parameters(), teacher_client)
    
    # Load Data
    train_data = load_bbh_object_count(n=15)
    
    print(f"\n[INIT PROMPT] {student.system_prompt.data}")

    # --- EPOCH LOOP ---
    for epoch in range(1, 11):
        print(f"\n================ EPOCH {epoch} ================")
        
        # Ensure gradients are cleared at start of epoch
        student.system_prompt.gradients = []
        
        errors = 0
        
        for item in train_data:
            q, truth = item['question'], item['truth']
            
            # 1. FORWARD (Student)
            response = student(q)
            pred_raw = response.data
            pred_parsed = parse_count_answer(pred_raw)
            
            # 2. EVAL
            if pred_parsed == truth:
                print(f"✅ PASS | Pred: {pred_parsed} == True: {truth}")
            else:
                print(f"❌ FAIL | Pred: {pred_parsed} (Raw len: {len(pred_raw)}) | True: {truth}")
                errors += 1
                
                # 3. CRITIQUE (Teacher)
                critique_prompt = [
                    {"role": "system", "content": "You are a logic teacher."},
                    {"role": "user", "content": f"Question: {q}\nStudent Answer: {pred_raw}\nTarget: {truth}\n\nAnalyze the student's error. Did they miss an item? Did they count non-objects? Be specific."}
                ]
                critique = teacher_client.call(api_kwargs={"messages": critique_prompt})
                
                # 4. STORE GRADIENT (FIX IS HERE)
                # We use ResearchGradient() instead of Gradient()
                grad = ResearchGradient(
                    data=critique,
                    score=0.0,
                    from_response=MockTrace(q, "Input"),
                    to_pred=MockTrace(truth, "Truth")
                )
                
                # We append directly to the list to avoid type checks
                if not hasattr(student.system_prompt, "gradients"):
                    student.system_prompt.gradients = []
                student.system_prompt.gradients.append(grad)

        # 4. OPTIMIZE
        if errors > 0:
            print(f"\n📉 Found {errors} errors. Running Optimization Step...")
            optimizer.step()
        else:
            print("✨ Converged! Perfect accuracy achieved.")
            break

if __name__ == "__main__":
    train()