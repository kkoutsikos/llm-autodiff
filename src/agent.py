import adalflow as adal
# Import from specific submodules to avoid ImportError
from adalflow.core.component import Component
from adalflow.core.generator import Generator
from adalflow.optim.parameter import Parameter
from adalflow.core.types import GeneratorOutput 

class MathStudent(Component):
    def __init__(self, client):
        super().__init__()
        self.client = client  # <--- Use client directly
        
        self.system_prompt = Parameter(
            data="You are a math helper. Question: {{input_str}}",
            role_desc="Math Instructions",
            requires_opt=True,
            param_type=adal.ParameterType.PROMPT
        )

    def call(self, input_str: str):
        # 1. Manually Render the Prompt
        # This ensures the 'system_prompt' parameter is actually used
        rendered_prompt = self.system_prompt.data.replace("{{input_str}}", input_str)
        
        # 2. Call the Model Client Directly
        # This bypasses the 'Generator' class which was causing the TypeError
        response_text = self.client.call(api_kwargs={"prompt": rendered_prompt})
        
        # 3. Package the result for the Optimizer
        # The optimizer looks at 'id' and 'data'
        return GeneratorOutput(
            id=str(uuid.uuid4()), # Unique ID for this run
            data=response_text,   # The model's answer
        )