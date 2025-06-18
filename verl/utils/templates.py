"""
Shared templates for different generation modes and tool usage patterns.
"""

TOOL_USE_TEMPLATE = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the assistant solves it. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user with the "
    "answer. During thinking, the assistant can invoke the Wikipedia "
    "search tool to search for fact information about specific topics "
    "if needed. The reasoning process and answer are enclosed within "
    "<think> and </think> tags respectively, and the search query and "
    "result are enclosed within <search> and </search> tags respectively. "
    "For example, <think>This is the reasoning process. <search> search "
    "query here </search> <search> search result here </search> This is the "
    "reasoning process.</think> The final answer is \\boxed{answer here}. "
    "The final exact answer is enclosed within \\boxed{} with latex format."
)

TOOL_USE_AND_INTERLEAVE_TEMPLATE = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the assistant solves it. The assistant first thinks about the "
    "reasoning process in the mind and then provides the user with the "
    "answer. During thinking, the assistant can invoke the Wikipedia "
    "search tool to search for fact information about specific topics "
    "if needed. The reasoning process and answer are enclosed within "
    "<think> and </think> tags respectively, and the search query and "
    "result are enclosed within <search> and </search> tags respectively. "
    "You conduct your reasoning within <think></think> and share partial "
    "answers within <answer></answer> as soon as you become confident about "
    "the intermediate results. You continue this pattern of "
    "<think></think><answer></answer><think></think><answer></answer> until "
    "you reach the final answer. For example, <think>This is the reasoning "
    "process. <search> search query here </search> <search> search result "
    "here </search> This is the reasoning process.</think> The final answer "
    "is \\boxed{answer here}. The final exact answer is enclosed within "
    "\\boxed{} with latex format."
)

CONFIDENCE_TEMPLATE = (
    "You are a helpful and highly intelligent assistant. After each sentence you write, "
    "please output your confidence in that statement as a floating point number between "
    "0 and 1 in square brackets, like [0.8]. A confidence of 1.0 means you are completely "
    "certain, while 0.0 means you are completely uncertain. Let's think step by step."
)

DEFAULT_TEMPLATE = (
    "You are a helpful and highly intelligent assistant. Let's think step by step."
)

# Template mappings
TEMPLATE_MAPPINGS = {
    "tool": TOOL_USE_TEMPLATE,
    "tool_interleaved": TOOL_USE_AND_INTERLEAVE_TEMPLATE,
    "confidence": CONFIDENCE_TEMPLATE,
    "default": DEFAULT_TEMPLATE,
}


def get_system_template(template_type: str = "default") -> str:
    """
    Get the system template based on the specified type.
    
    Args:
        template_type: Type of template to use. Options:
            - "tool": Basic tool use template
            - "tool_interleaved": Tool use with interleaved reasoning
            - "confidence": Confidence scoring template
            - "default": Simple assistant template
    
    Returns:
        str: The system template content
    
    Raises:
        ValueError: If template_type is not recognized
    """
    if template_type not in TEMPLATE_MAPPINGS:
        available_types = ", ".join(TEMPLATE_MAPPINGS.keys())
        raise ValueError(f"Unknown template type '{template_type}'. Available types: {available_types}")
    
    return TEMPLATE_MAPPINGS[template_type]


def format_system_message(template_type: str = "default", custom_content: str = None) -> dict:
    """
    Create a formatted system message for chat templates.
    
    Args:
        template_type: Type of template to use
        custom_content: If provided, use this instead of template lookup
    
    Returns:
        dict: Formatted system message with role and content
    """
    content = custom_content if custom_content is not None else get_system_template(template_type)
    return {
        "role": "system",
        "content": content
    } 