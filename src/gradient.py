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

class TextualBackwardEngine:
    """Converts a Failure -> Textual Gradient using the Teacher."""
    def __init__(self, teacher_client):
        self.teacher = teacher_client

    def compute_gradient(self, question, student_response, truth):
        # 1. Prepare Prompt
        prompt_content = GRADIENT_GENERATOR_TEMPLATE.format(
            question=question,
            student_response=student_response,
            ground_truth=truth
        )

        # 2. Call Teacher (Backward Pass)
        critique = self.teacher.call(api_kwargs={
            "messages": [
                {"role": "system", "content": "You are an AI optimization assistant."},
                {"role": "user", "content": prompt_content}
            ]
        })

        # 3. Return Gradient Object
        return ResearchGradient(
            data=critique.strip(),
            from_response=MockTrace(question, "Input"),
            to_pred=MockTrace(truth, "Truth")
        )