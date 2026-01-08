from src.prompts import GRADIENT_GENERATOR_TEMPLATE
import uuid

class ResearchGradient:
    """
    A container for the 'Textual Gradient'.
    Stores the feedback (data) and the context (inputs/outputs).
    """
    def __init__(self, data, from_response, to_pred, score=0.0):
        self.data = data               # The text feedback (The "Gradient")
        self.from_response = from_response 
        self.to_pred = to_pred         
        self.score = score

class MockTrace:
    """Helper to store data in a way that mimics AdalFlow traces."""
    def __init__(self, data, name="mock"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.data = data

class TextualBackwardEngine:
    """
    The Engine that runs the 'Backward Pass'.
    It takes a failure and uses the Teacher Model to generate a Gradient.
    """
    def __init__(self, teacher_client):
        self.teacher = teacher_client

    def compute_gradient(self, question, student_response, truth):
        # 1. Prepare the Prompt using our Template
        prompt_content = GRADIENT_GENERATOR_TEMPLATE.format(
            question=question,
            student_response=student_response,
            ground_truth=truth
        )

        # 2. Run the Backward Pass (Call Teacher)
        critique = self.teacher.call(api_kwargs={
            "messages": [
                {"role": "system", "content": "You are an AI optimization assistant."},
                {"role": "user", "content": prompt_content}
            ]
        })

        # 3. Package into a Gradient Object
        return ResearchGradient(
            data=critique.strip(),
            from_response=MockTrace(question, "Input"),
            to_pred=MockTrace(truth, "Truth")
        )