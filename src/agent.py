import adalflow as adal
from adalflow.core import Component, Generator, Parameter

class MathStudent(Component):
    def __init__(self, client):
        super().__init__()
        
        # The prompt we want to optimize
        self.system_prompt = Parameter(
            data="You are a math helper. Question: {{input_str}}",
            role_desc="Math Instructions",
            requires_opt=True,
            param_type=adal.ParameterType.PROMPT
        )
        
        # The Generator Node (Wraps the client)
        self.generator = Generator(
            model_client=client,
            model_kwargs={"temperature": 0.5} 
        )

    def call(self, input_str: str):
        # 1. Render prompt manually to ensure AdalFlow tracks it
        rendered_prompt = self.system_prompt.data.replace("{{input_str}}", input_str)
        
        # 2. Call Generator 
        # We pass 'prompt' (string) for the model, and 'prompt_kwargs' for the tracer
        return self.generator(prompt_kwargs={"input_str": input_str}, prompt=rendered_prompt)