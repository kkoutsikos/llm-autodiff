
"""
Meta-Prompts for the Research Optimizer and Backward Engine.
"""

# =====================================================
# META-PROMPT STRATEGY A: OPTIMIZING INSTRUCTIONS
# =====================================================
OPTIMIZE_INSTRUCTIONS_TEMPLATE = """
You are an expert Prompt Engineer optimizing a small Language Model (1.5B parameters).

CURRENT STATE:
The model is using these instructions:
"{current_prompt}"

THE PROBLEM:
The model is failing on specific examples. 
Here is a log of the failures and the Teacher's critiques:
{error_log}

YOUR TASK:
Write a NEW, IMPROVED version of the instructions.
- Analyze the Teacher's critiques to find the root cause.
- If the model forgets to count, emphasize "List items".
- If the model fails formatting, enforce "Answer: [[number]]".
- Keep the instructions concise but strict.

OUTPUT:
Return ONLY the text of the new instructions. Do not use quotes or markdown code blocks.
"""

# =====================================================
# META-PROMPT STRATEGY B: OPTIMIZING FEW-SHOT EXAMPLES
# =====================================================
OPTIMIZE_DEMOS_TEMPLATE = """
You are a Teacher preparing Few-Shot Examples for a student model.

CURRENT STATE:
The student is currently seeing these examples:
"{current_prompt}"

THE PROBLEM:
The student is confused and making errors.
Failures:
{error_log}

YOUR TASK:
Write a NEW set of Few-Shot Examples (Question & Answer pairs).
- The new examples must specifically address the errors shown above.
- If the student counts incorrectly, provide a clearer step-by-step example.
- Ensure the format matches:
  Q: ...
  A: ... Answer: [[N]]

OUTPUT:
Return ONLY the text of the new examples. Do not use quotes.
"""

# =====================================================
# BACKWARD ENGINE: TEXTUAL GRADIENT GENERATOR
# =====================================================
GRADIENT_GENERATOR_TEMPLATE = """
You are a Gradient Generator for an LLM Optimization System.

OBJECTIVE:
The Student Model failed to answer the question correctly.
Your goal is to generate a "Textual Gradient" (Feedback) that explains WHY it failed.
This feedback will be used to update the System Prompt.

CONTEXT:
1. Question: "{question}"
2. Student Answer: "{student_response}"
3. Correct Answer: "{ground_truth}"

ANALYSIS INSTRUCTIONS:
- Compare the Student Answer to the Correct Answer.
- Identify the specific point of failure (e.g., missed constraint, calculation error, formatting error).
- Do NOT just provide the right answer. Explain the *reasoning gap*.

GRADIENT (Feedback):
"""