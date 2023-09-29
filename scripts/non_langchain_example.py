from typing import Literal

import math

import inspect

from transformers import pipeline


##########################################################
# Step 1: Define the functions you want to articulate. ###
##########################################################


def calculator(
    input_a: float,
    input_b: float,
    operation: Literal["add", "subtract", "multiply", "divide"],
):
    """
    Computes a calculation.

    Args:
    input_a (float) : Required. The first input.
    input_b (float) : Required. The second input.
    operation (string): The operation. Choices include: add to add two numbers, subtract to subtract two numbers, multiply to multiply two numbers, and divide to divide them.
    """
    match operation:
        case "add":
            return input_a + input_b
        case "subtract":
            return input_a - input_b
        case "multiply":
            return input_a * input_b
        case "divide":
            return input_a / input_b


def cylinder_volume(radius, height):
    """
    Calculate the volume of a cylinder.

    Parameters:
    - radius (float): The radius of the base of the cylinder.
    - height (float): The height of the cylinder.

    Returns:
    - float: The volume of the cylinder.
    """
    if radius < 0 or height < 0:
        raise ValueError("Radius and height must be non-negative.")

    volume = math.pi * (radius**2) * height
    return volume


#############################################################
# Step 2: Let's define some utils for building the prompt ###
#############################################################


def format_functions_for_prompt(*functions):
    formatted_functions = []
    for func in functions:
        source_code = inspect.getsource(func)
        docstring = inspect.getdoc(func)
        formatted_functions.append(
            f"OPTION:\n<func_start>{source_code}<func_end>\n<docstring_start>\n{docstring}\n<docstring_end>"
        )
    return "\n".join(formatted_functions)


##############################
# Step 3: Construct Prompt ###
##############################


def construct_prompt(user_query: str):
    formatted_prompt = format_functions_for_prompt(calculator, cylinder_volume)
    formatted_prompt += f"\n\nUser Query: Question: {user_query}\n"

    prompt = (
        "<human>:\n"
        + formatted_prompt
        + "Please pick a function from the above options that best answers the user query and fill in the appropriate arguments.<human_end>"
    )
    return prompt


#######################################
# Step 4: Execute the function call ###
#######################################


def execute_function_call(model_output):
    # Ignore everything after "Reflection" since that is not essential.
    function_call = (
        model_output[0]["generated_text"]
        .strip()
        .split("\n")[1]
        .replace("Initial Answer:", "")
        .strip()
    )

    try:
        return eval(function_call)
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    # Build the model
    text_gen = pipeline(
        "text-generation",
        model="Nexusflow/NexusRaven-13B",
        device="cuda",
    )

    # Comp[ute a Simple equation
    prompt = construct_prompt("What is 1+10?")
    model_output = text_gen(
        prompt, do_sample=False, max_new_tokens=400, return_full_text=False
    )
    result = execute_function_call(model_output)

    print("Model Output:", model_output)
    print("Execution Result:", result)

    prompt = construct_prompt(
        "I have a cake that is about 3 centimenters high and 200 centimeters in diameter. How much cake do I have?"
    )
    model_output = text_gen(
        prompt,
        do_sample=False,
        max_new_tokens=400,
        return_full_text=False,
        stop=["\nReflection:"],
    )
    result = execute_function_call(model_output)

    print("Model Output:", model_output)
    print("Execution Result:", result)
