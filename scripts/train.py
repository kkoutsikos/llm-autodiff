from src.client import get_client
from src.agent import MathStudent

# Import Optimizer logic
from adalflow.optim import TextualGradientOptimizer
from adalflow.optim.types import Gradient

def train():
    print("--- Initializing Offline Training ---")
    
    # 1. Setup
    client = get_client() 
    student = MathStudent(client)
    
    optimizer = TextualGradientOptimizer(
        params=student.parameters(),
        model_client=client,
        model_kwargs={}
    )

    # 2. Dataset (Constraint: Answer must be [[number]])
    train_data = [
        ("What is 10 + 10?", "[[20]]"),
        ("Calculate 5 * 5.", "[[25]]"),
        ("If x=2, what is x+3?", "[[5]]")
    ]
    
    print(f"\n[START] Prompt: {student.system_prompt.data}")

    # 3. Training Loop
    for epoch in range(1, 4):
        print(f"\n--- Epoch {epoch} ---")
        student.system_prompt.reset_gradients() # ZeroGrad
        
        epoch_errors = 0
        
        for x, y_true in train_data:
            # Forward
            response_obj = student(x)
            y_pred = response_obj.data.strip()
            
            # Evaluate
            if y_true not in y_pred:
                print(f"FAIL: Got '{y_pred}' | Exp '{y_true}'")
                epoch_errors += 1
                
                # Backward (Teacher Critique)
                diag_prompt = [
                    {"role": "system", "content": "You are a teacher."},
                    {"role": "user", "content": f"Critique this answer: '{y_pred}'. Expected format: '{y_true}'. Explain the format error."}
                ]
                critique = client.call(api_kwargs={"messages": diag_prompt})
                
                # Gradient Creation
                grad = Gradient(
                    data=critique,
                    score=0.0,
                    from_response=response_obj,
                    to_pred=y_true
                )
                student.system_prompt.add_gradient(grad)
            else:
                print(f"PASS: {x}")
        
        # Optimize
        if epoch_errors > 0:
            print("Optimizing...")
            optimizer.step()
            print(f"[UPDATED] Prompt: {student.system_prompt.data}")
        else:
            print("Converged!")
            break

if __name__ == "__main__":
    train()