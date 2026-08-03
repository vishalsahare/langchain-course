from dotenv import load_dotenv

load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""

    print(f"===> Executing get_product_price: {product}")
    prices = {"laptop": 1000.00, "phone": 500.00, "tablet": 300.00}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount (price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the discounted price."""

    print(f"===> Executing apply_discount(price={price}, discount_tier={discount_tier}")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# Difference 2: Without @tool, we must MANUALLY define the JSON schema for each function.
# This is exactly what LangChain's @tool decorator generates automatically
# from the function's type hints and docstring.
llm_with_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "required": ["product"],
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "required": ["price", "discount_tier"],
                "properties": {
                    "product": {
                    "type": "string",
                    "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    }
                }
            }
        }
    },
]

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_trace(messages):
    return ollama.chat(model=MODEL, messages=messages, tools=llm_with_tools)

@traceable(name="Langchain Agent Loop")
def run_agent(question: str):
    """Run the agent."""

    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            ),
        },
        {"role": "user", "content": question},
    ]

    for iteration in range(MAX_ITERATIONS + 1):
        print(f"--- Iteration {iteration} ---")

        response = ollama_chat_trace(messages=messages)
        ai_message = response.message

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"DEBUG type      : {type(ai_message)}")
            print(f"DEBUG content   : {repr(ai_message.content)}")
            print(f"DEBUG response  : {getattr(ai_message, 'response_metadata', {})}")
            content = ai_message.content
            if not content:
                content = ai_message.additional_kwargs.get("thinking", "")
            print(f"\nFinal Answer: {content}\n")
            return content

        # Execute each tool call and feed results back into the conversation
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments
        
        print(f"    [Tool Selected] {tool_name} with args {tool_args}")
        
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found")
        
        result = tool_to_use(**tool_args)
        print(f"    [Tool result] {result}")

        messages.append(ai_message)
        messages.append({"role": "tool", "content": str(result)})

    print(f"Error: Max iteration reach without a final answer.")
    return None

if __name__ == "__main__":
    print("Hello LangChain Agent(.bind_tools)!")
    print()
    response = run_agent(
        "What is the price of a laptop after applying a gold discount?"
    )
