import json
import os

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 0. LOAD ENVIRONMENT VARIABLES
# ============================================================
# Your .env file should contain:
#
# OPENROUTER_API_KEY=sk-or-v1-xxxxxxxx
#
# NEVER hard-code your API key in this file.
# NEVER commit your .env file to GitHub.

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY is not set. "
        "Add it to your .env file."
    )


# ============================================================
# 1. CREATE OPENROUTER CLIENT
# ============================================================
# We use the official OpenAI Python SDK, but point it to
# OpenRouter instead of OpenAI.
#
# OpenRouter provides an OpenAI-compatible API.

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)


# ============================================================
# 2. DEFINE OUR PYTHON TOOLS
# ============================================================
# These are normal Python functions.
#
# The LLM does NOT execute these functions directly.
#
# The flow is:
#
# LLM → requests tool
# Python application → executes tool
# Python application → sends result back to LLM
#
# This is a very important concept in Agentic AI.


def calculate_bmi(weight_kg: float, height_m: float) -> dict:
    """
    Calculate BMI using weight in kilograms
    and height in metres.
    """

    if weight_kg <= 0 or height_m <= 0:
        return {
            "error": "Weight and height must be positive numbers."
        }
    if height_m > 3:
        return {
            "error": "Height seems too large. Please provide height in metres."
        }
    bmi = weight_kg / (height_m ** 2)

    return {
        "weight_kg": weight_kg,
        "height_m": height_m,
        "bmi": round(bmi, 2),
    }


def get_demo_weather(city: str) -> dict:
    """
    Return simulated weather data.

    IMPORTANT:
    This is NOT a real weather API.
    It is only being used to demonstrate
    how an agent can call a tool.
    """

    return {
        "city": city,
        "temperature_c": 29,
        "condition": "Sunny",
        "simulated": True,
    }


def calculatar(var1: float, var2: float, action: str) -> dict:
    """
    Calculator function that performs basic arithmetic operations like addition, substaction,
    multiplication, and division based on the provided action. It can also perform percentage calculations and exponentiation. The function takes two numerical inputs (var1 and var2) and a string action that specifies the operation to be performed. It returns a dictionary containing the input values and the result of the operation.
    """
    match action:
        case "add":
            result = var1 + var2
        case "subtract":
            result = var1 - var2
        case "multiply":
            result = var1 * var2
        case "divide":
            if var2 == 0:
                return {"error": "Division by zero is not allowed."}
            result = var1 / var2
        case "percentage":
            result = (var1 / 100) * var2
        case "exponentiation":
            result = var1 ** var2
        case _:
            return {"error": f"Invalid action: {action}"}

    return {
        "var1": var1,
        "var2": var2,
        "action": action,
        "result": result,
    }


# ============================================================
# 3. DESCRIBE TOOLS TO THE LLM
# ============================================================
# The LLM cannot automatically see our Python functions.
#
# We need to tell the model:
#
# - What tools are available
# - What each tool does
# - What parameters each tool expects
#
# IMPORTANT:
# This is the Chat Completions tool format.
# It is different from the Responses API format.

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate_bmi",
            "description": (
                "Calculate BMI using weight in kilograms "
                "and height in metres."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "weight_kg": {
                        "type": "number",
                        "description": "Weight in kilograms.",
                    },
                    "height_m": {
                        "type": "number",
                        "description": "Height in metres.",
                    },
                },
                "required": [
                    "weight_kg",
                    "height_m",
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_demo_weather",
            "description": (
                "Get simulated weather data for a city. "
                "This is demonstration data, not live weather."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Name of the city.",
                    },
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "funciton",
        "function": {
            "name": "calculatar",
            "description": "Perform basic arithmetic operations like addition, subtraction, multiplication, division, percentage calculations, and exponentiation.",
            "parameters": {

                "type": "object",
                "properties": {
                    "var1": {
                        "type": "number",
                        "description": "The first numerical input."
                    },
                    "var2": {
                        "type": "number",
                        "description": "The second numerical input."
                    },
                    "action": {
                        "type": "string",
                        "description": (
                            "The arithmetic operation to perform. "
                            "Valid actions are: 'add', 'subtract', 'multiply', "
                            "'divide', 'percentage', 'exponentiation'."
                        )
                    }
                },
                "required": ["var1", "var2", "action"],
                "additionalProperties": False
            }
        }
    }
]


# ============================================================
# 4. MAP TOOL NAMES TO PYTHON FUNCTIONS
# ============================================================
# When the LLM says:
#
# "Call calculate_bmi"
#
# our Python program needs to know which actual
# Python function to execute.
#
# That's what this dictionary does.

available_tools = {
    "calculate_bmi": calculate_bmi,
    "get_demo_weather": get_demo_weather,
    "calculatar": calculatar,
}


# ============================================================
# 5. AGENT LOOP
# ============================================================
# This is the heart of our simple AI agent.
#
# The agent repeatedly:
#
# 1. Sends the conversation to the LLM
# 2. Checks whether the LLM wants to call a tool
# 3. Executes the requested tool
# 4. Sends the tool result back to the LLM
# 5. Repeats until the LLM gives a final answer
#
# This is the basic "agent loop".
#
# Later, frameworks such as Agno, CrewAI and LangGraph
# can manage more complex versions of this loop.


def run_agent(user_task: str) -> str:
    """
    Run the agent until it produces a final answer.
    """

    # --------------------------------------------------------
    # Conversation history
    # --------------------------------------------------------
    #
    # The conversation contains everything the model needs
    # to understand what has happened so far.

    conversation = [
        {
            "role": "user",
            "content": user_task,
        }
    ]

    # Safety mechanism:
    # Prevent the agent from looping forever.

    max_iterations = 10

    for iteration in range(max_iterations):

        print(
            f"\n[Agent loop: iteration {iteration + 1}]"
        )

        # ----------------------------------------------------
        # CALL THE LLM
        # ----------------------------------------------------
        #
        # We use Chat Completions because our OpenRouter
        # integration is using the OpenAI-compatible API.

        response = client.chat.completions.create(
            model="openrouter/free",

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful personal assistant. "
                        "Use available tools when needed. "
                        "Never claim simulated weather is live weather. "
                        "Explain tool results clearly."
                    ),
                },
                *conversation,
            ],

            tools=tools,

            # Allows the model to decide whether a tool
            # should be called.
            tool_choice="auto",
        )

        # ----------------------------------------------------
        # GET THE ASSISTANT MESSAGE
        # ----------------------------------------------------

        message = response.choices[0].message

        # ----------------------------------------------------
        # NO TOOL CALL?
        # ----------------------------------------------------
        #
        # If the model did not request a tool, then we have
        # reached the final answer.

        if not message.tool_calls:

            return message.content or ""

        # ----------------------------------------------------
        # STORE THE ASSISTANT MESSAGE
        # ----------------------------------------------------
        #
        # We need to preserve the assistant's tool-call
        # request in the conversation.

        conversation.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in message.tool_calls
                ],
            }
        )

        # ----------------------------------------------------
        # EXECUTE EACH TOOL REQUESTED BY THE MODEL
        # ----------------------------------------------------

        for tool_call in message.tool_calls:

            # Name of the requested function
            tool_name = tool_call.function.name

            # Arguments arrive as a JSON string
            arguments = json.loads(
                tool_call.function.arguments
            )

            print(f"[Tool requested]: {tool_name}")
            print(f"[Arguments]: {arguments}")

            # ------------------------------------------------
            # FIND THE PYTHON FUNCTION
            # ------------------------------------------------

            try:

                function = available_tools[tool_name]

                # Execute the actual Python function
                result = function(**arguments)

            except KeyError:

                result = {
                    "error": f"Unknown tool: {tool_name}"
                }

            except Exception as error:

                result = {
                    "error": str(error)
                }

            print(f"[Tool result]: {result}")

            # ------------------------------------------------
            # SEND TOOL RESULT BACK TO THE LLM
            # ------------------------------------------------
            #
            # The LLM requested the tool.
            #
            # We executed it.
            #
            # Now we give the result back to the LLM.
            #
            # The LLM can then decide what to do next.

            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

    # --------------------------------------------------------
    # MAX ITERATIONS REACHED
    # --------------------------------------------------------

    raise RuntimeError(
        "Agent stopped because it reached "
        "the maximum number of iterations."
    )


# ============================================================
# 6. RUN THE AGENT
# ============================================================
#
# This creates a simple command-line interface.
#
# Example:
#
# You: Calculate BMI for 70 kg and 1.75 meters
#
# The LLM should decide:
#
#       ↓
# calculate_bmi(...)
#       ↓
# Python executes function
#       ↓
# BMI result
#       ↓
# LLM explains result
#
# ============================================================


if __name__ == "__main__":

    print("🤖 Personal AI Agent")
    print("Using OpenRouter")
    print("Type 'exit' to quit.\n")

    while True:

        user_input = input("You: ").strip()

        # Exit the program
        if user_input.lower() == "exit":

            print("Agent: Goodbye!")

            break

        # Ignore empty input
        if not user_input:
            continue

        # Run the agent
        try:

            answer = run_agent(user_input)

            print(f"\nAgent: {answer}\n")

        except Exception as error:

            print(f"\nError: {error}\n")
