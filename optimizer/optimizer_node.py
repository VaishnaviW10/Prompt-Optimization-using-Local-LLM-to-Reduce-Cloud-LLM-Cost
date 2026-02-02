from optimizer.prompt_templates import OPTIMIZER_SYSTEM_PROMPT
from optimizer.token_utils import count_tokens


from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model="phi3:mini",
    temperature=0.0,
    max_tokens=50, 
    keep_alive="5m"
)

def optimize_prompt(state: dict) -> dict:
    """
    Stage 3 LangGraph node
    Reads 'optimized_query' from Stage 2 output
    Returns final LLM response
    """

    # USE THE SAME KEY AS STAGE 2
    optimized_query = state["optimized_query"]

    # Call LLM with optimized query
    final_response = llm.invoke(optimized_query)

    return {"final_response": final_response}


