import sys
import uuid
import os
import time
sys.path.append(os.getcwd())


# Ensure we can import from 'src'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.client import get_client
from src.agent import MathStudent
from src.utils import load_gsm8k_subset, parse_model_answer 

from adalflow.core.model_client import ModelClient
try:
    
    from adalflow.optim.text_grad.tgd_optimizer import TGDOptimizer as TextualGradientOptimizer
except ImportError:
    
    from adalflow.optim import TGDOptimizer as TextualGradientOptimizer


try:
    from adalflow.optim import Gradient
except ImportError:
    # Fallback for different versions
    from adalflow.core.types import Gradient


class MockTrace:
    def __init__(self, data, name="mock_node"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.data = data
        self.component_trace = self 

def train():
    print("🚀 Starting Offline Training...")
    client = get_client()
    student = MathStudent(client)
    
    optimizer = TextualGradientOptimizer(
        params=student.parameters(),
        model_client=client,
        model_kwargs={}
    )

    train_data = load_gsm8k_dataset(n=10)
    print(f"\n[INIT PROMPT] {student.system_prompt.data}")

    for epoch in range(1, 4):
        print(f"\n--- EPOCH {epoch} ---")
        student.system_prompt.reset_gradients()
        errors = 0
        
        for x, y_true in train_data:
            response_obj = student(x)
            y_pred_raw = response_obj.data
            y_parsed = parse_model_answer(y_pred_raw)
            
            if y_parsed and y_parsed.replace(",", "") == y_true.replace(",", ""):
                print(f"✅ PASS | {y_parsed} == {y_true}")
            else:
                print(f"❌ FAIL | Pred: {y_parsed} (Raw: {y_pred_raw[:30]}...) | True: {y_true}")
                errors += 1
                
                prompt = [
                    {"role": "system", "content": "You are a teacher grading a math test."},
                    {"role": "user", "content": f"Question: {x}\nStudent: {y_pred_raw}\nTarget: {y_true}\nCritique the error.Provide feedback to help them improve."}
                ]
                critique = client.call(api_kwargs={"messages": prompt})
                
                # WRAPPER FIX
                safe_response = MockTrace(y_pred_raw, name="StudentResponse")
                safe_truth = MockTrace(y_true, name="GroundTruth")
                
                grad = Gradient(
                    data=critique, 
                    score=0.0, 
                    from_response=safe_response, 
                    to_pred=safe_truth,
                    data_id=str(uuid.uuid4())
                )
                student.system_prompt.add_gradient(grad)

        if errors > 0:
            print(f"📉 Found {errors} errors. Optimizing...")
            optimizer.step()
            print(f"[NEW PROMPT] {student.system_prompt.data[:150]}...")
        else:
            print("✨ Converged!")
            break

if __name__ == "__main__":
    train()