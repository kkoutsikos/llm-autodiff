import sys
import os
import time

# Ensure we can import from 'src'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.client import get_client
from src.agent import MathStudent
from src.utils import load_gsm8k_subset, parse_model_answer # <--- NEW IMPORT

from adalflow.optim import TextualGradientOptimizer
from adalflow.optim.types import Gradient

def train():
    # 1. Setup Logging
    log_file = "experiment_log.txt"
    with open(log_file, "w") as f:
        f.write(f"Experiment Started: {time.ctime()}\n" + "="*50 + "\n")

    print("--- Initializing Phase 1 Replication ---")
    client = get_client()
    student = MathStudent(client)
    
    optimizer = TextualGradientOptimizer(
        params=student.parameters(),
        model_client=client,
        model_kwargs={}
    )

    # 2. LOAD REAL DATASET
    # We load 10 examples for the first test run. Increase 'n' later for full training.
    train_data = load_gsm8k_subset(split="train", n=10)
    
    print(f"\n[START] Initial Prompt: {student.system_prompt.data}")

    # 3. Training Loop
    for epoch in range(1, 4):
        print(f"\n--- Epoch {epoch} ---")
        student.system_prompt.reset_gradients()
        
        correct_count = 0
        gradients_collected = 0
        
        for x, y_true in train_data:
            # Forward Pass
            response_obj = student(x)
            y_pred_raw = response_obj.data
            y_parsed = parse_model_answer(y_pred_raw)
            
            # Evaluate (Compare extracted number string)
            # We strip commas (e.g., "1,000" -> "1000") for fair comparison
            is_correct = (y_parsed.replace(",", "") == y_true.replace(",", ""))
            
            if is_correct:
                print(f"✅ PASS | Pred: {y_parsed} | True: {y_true}")
                correct_count += 1
            else:
                print(f"❌ FAIL | Pred: {y_parsed} (Raw: {y_pred_raw[:15]}...) | True: {y_true}")
                
                # Teacher Critique
                diag_prompt = [
                    {"role": "system", "content": "You are a teacher grading a math test."},
                    {"role": "user", "content": f"""
                        Question: {x}
                        Student Answer: {y_pred_raw}
                        Target Answer: {y_true}
                        
                        Critique the student's error. 
                        Did they get the math wrong? Or just the format?
                        The target format is the number inside brackets: [[{y_true}]].
                    """}
                ]
                
                critique = client.call(api_kwargs={"messages": diag_prompt})
                
                grad = Gradient(
                    data=critique,
                    score=0.0,
                    from_response=response_obj,
                    to_pred=y_true
                )
                student.system_prompt.add_gradient(grad)
                gradients_collected += 1

        # Calculate Epoch Metrics
        accuracy = (correct_count / len(train_data)) * 100
        print(f"\n📊 Epoch {epoch} Accuracy: {accuracy:.1f}%")

        # Save to Log
        with open(log_file, "a") as f:
            f.write(f"\nEpoch {epoch} | Accuracy: {accuracy:.1f}%\n")
            f.write(f"Prompt: {student.system_prompt.data}\n")

        # Optimize
        if gradients_collected > 0:
            print(f"Applying {gradients_collected} gradients...")
            optimizer.step()
            print(f"[UPDATED] Prompt: {student.system_prompt.data[:100]}...")
        else:
            print("🚀 Converged! Stopping early.")
            break

if __name__ == "__main__":
    train()