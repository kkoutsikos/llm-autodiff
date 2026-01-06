import uuid
import adalflow.core.types as types
from adalflow.core.component import Component
from adalflow.core.parameter import Parameter

class ObjectCountStudent(Component):
    def __init__(self, client):
        super().__init__()
        self.client = client
        
        # Initial Prompt:
        # We start with a generic prompt to give the optimizer room to improve it.
        # Ideally, it should learn to add "List items first" or "Think step by step".
        self.system_prompt = Parameter(
            data="Count the objects in the text below. Question: {{input_str}}",
            role_desc="Object Counting Instructions",
            requires_opt=True,
            param_type=types.ParameterType.PROMPT
        )

    def call(self, input_str: str):
        # 1. Render the template
        rendered = self.system_prompt.data.replace("{{input_str}}", input_str)
        
        # 2. Call the Student Model (1.5B)
        # We pass 'input_str' as the USER message content via our Client adapter
        response_text = self.client.call(api_kwargs={"input_str": rendered})
        
        # 3. Return traceable output
        return types.GeneratorOutput(
            id=str(uuid.uuid4()), 
            data=response_text
        )