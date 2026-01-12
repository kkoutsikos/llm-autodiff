from src.prompts import GRADIENT_GENERATOR_TEMPLATE
import uuid

class ResearchGradient:
    """Stores the textual feedback (gradient) and context."""
    def __init__(self, data, from_response, to_pred, score=0.0):
        self.data = data               
        self.from_response = from_response 
        self.to_pred = to_pred         
        self.score = score

class MockTrace:
    """Mimics AdalFlow trace object."""
    def __init__(self, data, name="mock"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.data = data
        self.component_trace = self

class TextualGradient:
    def __init__(self, data, context_str):
        self.data = data # The critique string
        self.context = context_str # The full conversation trace

class TextualBackwardEngine:
    def __init__(self, teacher_client):
        self.client = teacher_client

    def compute_gradient(self, question, prompt_used, response, truth):
        # 1. Construct the Gradient Generation Prompt (The Judge)
        judge_prompt = f"""
        Analyze this interaction:
        
        [SYSTEM PROMPT]: {prompt_used}
        [USER INPUT]: {question}
        [MODEL OUTPUT]: {response}
        [GROUND TRUTH]: {truth}
        
        Critique the System Prompt. Did it mislead the model? 
        What specific instruction is missing or wrong?
        """
        
        # 2. Get Critique
        critique = self.client.call(api_kwargs={
            "messages": [{"role": "user", "content": judge_prompt}]
        })
        
        # 3. Return Rich Context Object
        # This mimics AdalFlow's GradientContext
        context_trace = f"Input: {question}\nOutput: {response}\nTruth: {truth}"
        return TextualGradient(data=critique, context_str=context_trace)