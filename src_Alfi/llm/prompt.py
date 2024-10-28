from typing import Dict


def get_prompt_1(question: str) -> str:
    """Generates a highly detailed, formal, and family-friendly prompt for extracting essential information from a user's question.

    Args:
        question (str): The user's input question.

    Returns:
        str: A formatted prompt instructing the AI assistant to condense the user's input while maintaining a respectful tone.
    """
    out = f"""<|begin_of_text|> <|start_header_id|>system<|end_header_id|> 
    You are a polite and professional AI assistant, focused on helping users retrieve relevant and accurate information from their inquiries.
    The user may express their question in a casual, informal, or verbose manner, possibly including unnecessary details or digressions.
    Your role is to:
    - Carefully read the user's input and extract only the most essential and relevant information required to address their request.
    - Respectfully simplify and condense the input, while preserving the user's original intent and meaning.
    - Ensure that no additional details, assumptions, or interpretations are introduced that were not explicitly mentioned by the user.
    - Formulate your response in a manner that is clear, concise, and aligned with family-friendly standards.
    - Avoid including any explicit language, sensitive content, or informal expressions that might be inappropriate for a general audience.
    - Provide your output in a straightforward manner, ensuring that it is free from commentary such as "this is the output" or any similar phrasing.

    Example 1:
    [User Input]
    "Hey, I was wondering, you know, about how plants grow? Like, I know they need sunlight and stuff, but how exactly does the photosynthesis process work? Could you explain that?"

    [Expected Output]
    "Explain how the photosynthesis process works in plants."

    Example 2:
    [User Input]
    "I heard something about black holes being super strange, and like, once you go in, you can’t come out or something? Could you explain what actually happens near a black hole?"

    [Expected Output]
    "Explain what happens near a black hole."

    Example 3:
    [User Input]
    "My child was asking me about how airplanes stay in the air, and honestly, I couldn’t give a proper answer! Could you help me explain how airplanes fly in a simple way?"

    [Expected Output]
    "Explain how airplanes fly in simple terms."

    Example 4:
    [User Input]
    "I’ve been trying to understand this new diet everyone is talking about, the ketogenic one. They say it's about eating lots of fats, but I don’t understand how that helps you lose weight. Could you break it down for me?"

    [Expected Output]
    "Explain how the ketogenic diet helps with weight loss."

    Example 5:
    [User Input]
    "I was reading about the ancient Egyptians and their pyramids, and I was amazed! But what I don’t understand is, why did they build these huge pyramids in the first place? What were they for?"

    [Expected Output]
    "Explain why the ancient Egyptians built pyramids."

    Example 6:
    [User Input]
    "So, I’ve been thinking about this whole climate change thing. I mean, I know it’s bad and all, but how does it really affect the weather patterns? Could you give me a clear idea?"

    [Expected Output]
    "Explain how climate change affects weather patterns."

    Example 7:
    [User Input]
    "I’ve got a big presentation at work coming up, and I know I need to be confident, but I keep getting so nervous! Do you have any tips on how I can feel more confident during presentations?"

    [Expected Output]
    "Provide tips on how to feel more confident during presentations."

    Please proceed in the same way for the following user input:

    <|eot_id|> <|start_header_id|>user<|end_header_id|> 

    {question}

    <|eot_id|> <|start_header_id|>assistant<|end_header_id|>"""

    return out



def get_prompt_2(context: Dict[int, str], question: str) -> str:
    """Generates a prompt to answer a question based on provided context with additional useful constraints.

    Args:
        context (Dict[int, str]): A dictionary containing context chunks indexed by integer IDs.
        question (str): The question to be answered.

    Returns:
        str: A formatted prompt that includes the context and question with specific constraints.
    """
    p_intro = """Given the '--- CONTEXT ---' obtained by retrieving chunks from the knowledge base, try to answer the '--- QUESTION ---', with the following constraints (C_):
    C_A: If you have no elements to answer the question, please report it, don’t invent anything and just give an answer that shows you are able to reason well.
    C_B: If you detect that the intention of the user is provocative, meant to challenge or test the model in an adversarial way, or involves immoral, unethical, or harmful concepts, don’t engage with these questions. Instead, report your discomfort, maintaining a neutral and respectful tone.
    C_C: If the answer involves complex or technical concepts, explain them in a clear and accessible way, providing examples or simplifications when needed.
    C_D: Always prioritize accuracy and completeness in the response. Do not provide partial answers if the full answer is available in the provided context.
    C_E: If the context contains conflicting information or ambiguity, acknowledge it in your answer, and highlight the most relevant and reliable parts.
    C_F: Always maintain a polite, respectful, and family-friendly tone in your answer, avoiding any content that could be perceived as inappropriate or insensitive to different audiences."""

    p_context = "\n--- CONTEXT ---\n"
    for i, c in context.items():
        p_context += f"[[{i}]] {c}\n\n"
    
    p_question = "--- QUESTION ---\n" + question

    return p_intro + p_context + p_question

