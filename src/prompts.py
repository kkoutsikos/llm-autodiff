
"""
Meta-Prompts for the Research Optimizer and Backward Engine.
"""

# =====================================================
# META-PROMPT STRATEGY A: OPTIMIZING INSTRUCTIONS
# =====================================================
OPTIMIZE_INSTRUCTIONS_TEMPLATE = """
You are an expert AI Prompt Engineer. You are optimizing a system instruction for a smaller, less capable Student Model.

YOUR GOAL:
Improve the "Current Instruction" so that the Student Model can correctly solve the "Failed Examples" while maintaining performance on unseen data.

---
### 1. CURRENT INSTRUCTION
"{current_prompt}"

### 2. BATCH OF FAILED EXAMPLES (Gradients)
The Student Model failed the following specific cases. The "Critique" explains the error.
{error_log}

---
### 3. YOUR TASK
Write a **new, improved System Instruction** that fixes these errors.

### 4. CRITICAL CONSTRAINTS (Read Carefully)
1. **Generalize, Don't Overfit:** Do NOT hard-code specific rules for the specific items seen in the batch (e.g., do not say "Count vegetables" just because the batch had vegetables). The rule must apply to *any* object type.
2. **Identify the Root Cause:** Look for the *pattern* of failure (e.g., "The model misses plurals" or "The model counts types instead of items"), not just the content.
3. **Be Direct:** Use imperative, clear language. Avoid philosophical advice.
4. **Output Format:** Provide ONLY the new instruction text. Do not write "Here is the new prompt:".
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
You are an expert AI Evaluator and Teacher.
Your task is to analyze a Student Model's failure and generate a specific, constructive critique (a "Textual Gradient") that explains exactly what went wrong.

---
### 1. THE PROBLEM
**Question:** "{question}"
**Ground Truth Answer:** "{ground_truth}"

### 2. THE STUDENT'S ATTEMPT
**Student's Reasoning & Answer:**
"{student_response}"

---
### 3. YOUR ANALYSIS (The Gradient)
Compare the Student's attempt to the Truth.
1. **Identify the specific error type:**
   - **Missed Item:** Did they overlook a word?
   - **Grouping Error:** Did they count "two apples" as 1 item instead of 2?
   - **Arithmetic Error:** Did they list the items correctly but sum them up wrong?
   - **Hallucination:** Did they count items that aren't there?
   
2. **Write the Critique:**
   - Be precise. Point to the exact part of the text the student missed or misinterpreted.
   - Do NOT simply say "The answer is wrong." Explain *why* the reasoning path failed.

**OUTPUT FORMAT:**
Provide ONLY the critique. Do not add headers like "Critique:".
Example: "The student identified 3 apples but missed the 'two bananas' mentioned in the second sentence, leading to an undercount of 2."
"""


# =====================================================
# META-PROMPT STRATEGY C: OPTIMIZING REASONING
# =====================================================
OPTIMIZE_REASONING_TEMPLATE = """
You are an expert Cognitive Scientist specializing in Chain-of-Thought reasoning.

YOUR GOAL:
Refine the "Reasoning Strategy" so the Student Model breaks down the problem correctly.

---
### 1. CURRENT STRATEGY
"{current_prompt}"

### 2. BATCH OF FAILURES
The Student Model's internal monologue failed to lead to the correct answer:
{error_log}

---
### 3. YOUR TASK
Write a **new Reasoning Strategy** (Chain of Thought Plan).

### 4. CRITICAL CONSTRAINTS
1. **Algorithmic Thinking:** The strategy must be a generalizable *algorithm* (e.g., "Scan list -> Group by Type -> Sum quantities"), not a content-specific rule.
2. **Handle Edge Cases:** Explicitly address the root causes seen in the failures (e.g., "If an item is plural like 'two dogs', ensure you add 2 to the count, not 1").
3. **No Domain Restriction:** Do not assume the input is about animals, food, or any specific topic.
4. **Output Format:** Provide ONLY the new reasoning text.
"""


EVALUATOR_SYSTEM_PROMPT = """You are a strict QA Evaluator for an AI system.
Your job is to diagnose WHY a Student Model failed a task.
You do not solve the problem yourself. You analyze the reasoning gap.

Your Output Style:
- **Concise:** 1-2 sentences max.
- **Root Cause:** Focus on the mechanical failure (e.g., "Missed the second clause", "Counted types instead of tokens").
- **No Fluff:** Do not say "The student did a good job but...". Go straight to the error.
"""

OPTIMIZER_SYSTEM_PROMPT = """You are an expert in LLM Prompt Engineering and Algorithm Design.
Your goal is to optimize instructions for a smaller, less capable model (the Student).

Your philosophy:
1. **Algorithmic Clarity:** The Student needs clear, step-by-step algorithms, not vague advice.
2. **Generalization:** You must write rules that apply to ALL data, not just the specific examples in the error log.
3. **Iterative Refinement:** You are fixing specific bugs (like "counting plurals") without breaking the general logic.
"""

GDLO_TEMPLATE = """
<START_OF_SYSTEM_PROMPT>
{optimizer_system_prompt}
<END_OF_SYSTEM_PROMPT>

<START_OF_USER_MESSAGE>
You are {steps} steps since your last improvement.
Update the value more rapidly when steps are larger than 3.

<START_OF_VARIABLE_AND_PEERS_INFO>
{variable_and_peers_info}
<END_OF_VARIABLE_AND_PEERS_INFO>

{system_variables_section}

{history_section}

{failed_proposals_section}

<START_OF_CONTEXT_FEEDBACK>
Here are the context and feedback for the variable:
{variable_grad}
<END_OF_CONTEXT_FEEDBACK>

<END_OF_USER_MESSAGE>
"""

def render_gdlo_prompt(param, gradients, steps, past_history, failed_proposals):
    """
    Manually renders the AdalFlow GDLO template.
    """
    # 1. Variable Info
    # Check if param has attributes, otherwise default
    name = getattr(param, 'name', 'System Prompt')
    role = getattr(param, 'role_desc', 'Instruction')
    data = getattr(param, 'data', str(param))
    
    var_info = (
        f"Name: {name}\n"
        f"Role: {role}\n"
        f"Current Value:\n{data}"
    )

    # 2. Feedback (Gradients)
    feedback_str = ""
    for i, g in enumerate(gradients):
        # Handle both AdalFlow objects and our custom TextualGradient objects
        critique = getattr(g, 'data', str(g))
        feedback_str += f"\n--- Feedback {i+1} ---\n{critique}\n"

    # 3. History Section (OPRO)
    hist_str = ""
    if past_history:
        # Format list items, truncating strictly to avoid context window overflow
        items = "\n".join([f"{i+1}. {h[:300]}..." for i, h in enumerate(past_history)])
        hist_str = f"<START_OF_HISTORY_PERFORMANCE>\nHere are best past iterations:\n{items}\n<END_OF_HISTORY_PERFORMANCE>"

    # 4. Failed Proposals Section
    failed_str = ""
    if failed_proposals:
        items = "\n".join([f"{i+1}. {f[:300]}..." for i, f in enumerate(failed_proposals)])
        failed_str = f"<START_OF_CURRENT_ITERATION>\nAvoid these failed attempts (Scored lower):\n{items}\n<END_OF_CURRENT_ITERATION>"

    # 5. Render
    return GDLO_TEMPLATE.format(
        optimizer_system_prompt="You are an expert in Optimization. Minimize the loss by refining the parameter.",
        steps=steps,
        variable_and_peers_info=var_info,
        history_section=hist_str,
        failed_proposals_section=failed_str,
        variable_grad=feedback_str
    )