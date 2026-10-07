from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openrouter import OpenRouterResponses


# ============================================================
# 1. DEFINE A TOOL
# ============================================================
# This is a normal Python function.
#
# The AI agent can decide to call this function when it needs
# to perform an arithmetic calculation.
#
# Important:
# The LLM does NOT execute this Python function itself.
# The Agent framework executes it on the LLM's behalf.
# ============================================================
load_dotenv()  # Load environment variables from .env file


def calculator(
    var1: float,
    var2: float,
    action: str,
) -> dict:
    """ 
    Perform arithmetic.

    Actions:
    - add: var1 + var2
    - subtract: var1 - var2
    - multiply: var1 * var2
    - divide: var1 / var2
    - percentage: var1 percent of var2
    - modulo: remainder after dividing var1 by var2
    - exponentiation: var1 raised to var2."""

    # 'match' checks which operation the AI requested.
    match action:

        # Example:
        # calculator(10, 5, "add")
        # → 15
        case "add":
            result = var1 + var2

        # Example:
        # calculator(10, 5, "subtract")
        # → 5
        case "subtract":
            result = var1 - var2

        # Example:
        # calculator(10, 5, "multiply")
        # → 50
        case "multiply":
            result = var1 * var2

        # Example:
        # calculator(10, 5, "divide")
        # → 2
        case "divide":

            # Prevent division by zero.
            if var2 == 0:
                return {
                    "error": "Division by zero is not allowed."
                }

            result = var1 / var2

        # Percentage calculation.
        #
        # Example:
        # 15% of 800
        #
        # var1 = 15
        # var2 = 800
        #
        # (15 / 100) * 800 = 120
        case "percentage":
            result = (var1 / 100) * var2

        # Exponentiation.
        #
        # Example:
        # 2 ** 3 = 8
        case "exponentiation":
            result = var1 ** var2
        # Modulo operation.
        #
        # Example:
        # 17 % 5 = 2
        case "modulo":
            result = var1 % var2
        # If the AI sends an operation that we don't support.
        case _:
            return {
                "error": f"Invalid action: {action}"
            }

    # Return the calculation result to the Agent.
    #
    # The Agent will send this result back to the LLM so that
    # the LLM can understand the result and explain it to the user.
    return {
        "var1": var1,
        "var2": var2,
        "action": action,
        "result": result,
    }


# ============================================================
# 2. CREATE THE AI AGENT
# ============================================================
#
# The Agent is the main component that connects:
#
#     User
#       ↓
#     Agent
#       ↓
#     LLM
#       ↓
#     Tool (calculator)
#       ↓
#     Tool Result
#       ↓
#     LLM
#       ↓
#     Final Answer
#
# Agno manages much of this interaction for us.
# ============================================================

agent = Agent(

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------
    # This tells Agno which LLM the agent should use.
    #
    # OpenAIResponses is Agno's model integration for
    # OpenAI's Responses API.
    # --------------------------------------------------------
    model=OpenRouterResponses(
        id="openai/gpt-oss-20b"
    ),

    # Give our agent a name.
    name="Personal Assistant",

    # --------------------------------------------------------
    # INSTRUCTIONS
    # --------------------------------------------------------
    # These instructions become the agent's behavior/rules.
    #
    # Think of them as the agent's "system instructions".
    # --------------------------------------------------------
    instructions=[
        "You are a helpful personal assistant.",

        # This tells the LLM that it has access to our
        # calculator tool and should use it when necessary.
        "Use the calculator tool when arithmetic calculations are required.",

        # After using the tool, the LLM should explain
        # the answer clearly to the user.
        "Explain the result clearly.",
    ],

    # --------------------------------------------------------
    # TOOLS
    # --------------------------------------------------------
    # We give the calculator function to the Agent.
    #
    # Agno inspects this Python function and makes it
    # available as a tool that the LLM can request.
    # --------------------------------------------------------
    tools=[calculator],

    # Tell Agno to format the final response using Markdown.
    markdown=True,
)


# ============================================================
# 3. START THE PROGRAM
# ============================================================
#
# This condition means:
#
# "Run the following code only when this file is executed
# directly."
#
# For example:
#
#     python main.py
#
# ============================================================

if __name__ == "__main__":
    print("MAIN.PY STARTED")
    print("Agent created successfully")
    print("Sending request to OpenRouter...")
    # Send the user's request to the Agent.
    agent.print_response(

        # User's question
        "What is the remainder when 17 is divided by 5?",

        # stream=True means the response is displayed
        # progressively as the LLM generates it.
        stream=True,
    )
