
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