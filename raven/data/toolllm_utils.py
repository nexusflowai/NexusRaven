import random
import re
from datetime import datetime
from enum import Enum
from enum import Enum as ENUM
import copy
from functions.base import FunctionCallBenchmark
from utils.alpaca_utils import generate_and_test_functions_from_docstring
from datasets import load_dataset
from langchain.tools.base import StructuredTool
import pydantic
from utils.toolllm_utils import (
    fix_indentation,
    rename_ast_args,
    convert_datetimes_to_strings,
)

import inspect
from utils.utils import dict_to_func_call, compare_function_calls

correctness = []


class ToolLLM(FunctionCallBenchmark):
    NAME = "ToolAlpaca_Simulated"
    DATA_PATH = "Nexusflow/toolllm_eval"

    def build(self):
        self.dataset = load_dataset(self.DATA_PATH)["train"]
        for idx, item in enumerate(self.dataset):
            sample = {"prompt": None, "completion": None}
            random.shuffle(item["context"])
            yield item

    def build_tools(self):
        # Each sample has its own tools, so we'll build it dynamically
        tools = []
        return tools

    def take_step_and_compute_sample_accuracy(
        self, agent_information, prompt, ground_truth
    ):
        from registry import TASK_REGISTRY, LLM_REGISTRY, AGENT_REGISTRY

        global correctness

        codex = None
        initial = None
        instruction, docstring, context = prompt
        initial = copy.deepcopy(locals())
        for codex in context:
            exec(codex)
        final = locals()
        # This will contain all the fucntions we defined in the previous exercise.
        diff = {k: final[k] for k in final.keys() - initial.keys()}
        # Convert to functions
        tools = []
        for e in list(diff.keys()):
            tools.append(StructuredTool.from_function(eval(e)))

        new_agent = AGENT_REGISTRY(agent_information.llm, tools, agent_information.type)

        old_len = len(correctness)
        try:
            call = new_agent.build_agent().run(instruction)
        except pydantic.error_wrappers.ValidationError:
            pass
        except pydantic.error_wrappers.ValidationError:
            pass
        except StopIteration:
            pass
        # Some models will return the call instead of issuing the call
        try:
            if bool(re.match(r"\w+\(.*\)$", call)):
                exec(call)
        except:
            pass

        new_len = len(correctness)

        # Ensure that the function was correctly called.
        if old_len + 1 == new_len:
            name, args = [(e, k) for e, k in correctness[-1].items()][0]
            gpt_call = dict_to_func_call(name, args)
            exec(ground_truth)
            name, args = [(e, k) for e, k in correctness[-1].items()][0]
            gt_call = dict_to_func_call(name, args)
            accuracy = compare_function_calls(gpt_call, gt_call)
            # Drop the GT
            correctness = correctness[:-1]
            print(prompt[0], gpt_call, ground_truth, accuracy)
        else:
            gpt_call = "invalid call"
            accuracy = 0

        self.accuracy.append(accuracy)

    def item_to_prompt(self, item):
        return (item["Input"], item["Output"], item["context"])

    def item_to_completion(self, item):
        return item["Output"]


import ast
import re
import astunparse


def fix_indentation(code_str, indent_level=4):
    lines = code_str.split("\n")
    min_indent = float("inf")
    in_multiline_string = False
    multiline_string_indent = 0
    fixed_lines = []

    # Find the minimum indentation level
    for line in lines:
        stripped = line.lstrip()
        if '"""' in stripped or "'''" in stripped:
            if stripped.count('"""') % 2 == 1 or stripped.count("'''") % 2 == 1:
                in_multiline_string = not in_multiline_string
                multiline_string_indent = len(line) - len(stripped)

        if (
            not in_multiline_string and stripped
        ):  # Ignore empty lines and lines within multi-line strings
            min_indent = min(min_indent, len(line) - len(stripped))

    if min_indent == float("inf"):
        min_indent = 0  # In case all lines are empty or in a multiline string

    in_multiline_string = False  # Reset for the second pass

    # Re-indent the code
    for line in lines:
        stripped = line.lstrip()

        if '"""' in stripped or "'''" in stripped:
            if stripped.count('"""') % 2 == 1 or stripped.count("'''") % 2 == 1:
                in_multiline_string = not in_multiline_string
                multiline_string_indent = len(line) - len(stripped)

        if in_multiline_string:
            new_indentation = (len(line) - len(stripped)) - multiline_string_indent
            fixed_lines.append(" " * new_indentation + stripped)
        else:
            if stripped:  # Ignore empty lines
                indentation = len(line) - len(stripped)
                new_indentation = max(0, (indentation - min_indent))
                fixed_lines.append(" " * new_indentation + stripped)
            else:
                fixed_lines.append("")

    return "\n".join(fixed_lines)


def safe_validate_arguments(f):
    from pydantic import validate_arguments, Field, BaseModel
    from pydantic.fields import Undefined
    from functools import wraps
    import inspect

    names_to_fix = {n for n in BaseModel.__dict__ if not n.startswith("_")}

    @wraps(f)
    def wrapper(*args, **kwargs):
        kwargs = {
            n[:-1] if n.removesuffix("_") in names_to_fix else n: v
            for n, v in kwargs.items()
        }
        return f(*args, **kwargs)

    def _create_param(p: inspect.Parameter) -> inspect.Parameter:
        default = Undefined if p.default is inspect.Parameter.empty else p.default
        return p.replace(name=f"{p.name}_", default=Field(default, alias=p.name))

    sig = inspect.signature(f)
    sig = sig.replace(
        parameters=[
            _create_param(p) if n in names_to_fix else p
            for n, p in sig.parameters.items()
        ]
    )

    wrapper.__signature__ = sig
    wrapper.__annotations__ = {
        f"{n}_" if n in names_to_fix else n: v for n, v in f.__annotations__.items()
    }

    return wrapper


def rename_ast_args(func_str):
    # Parse the function string to an AST (Abstract Syntax Tree)
    tree = ast.parse(func_str)

    # Initialize a list to store the positions of arguments to be renamed
    rename_positions = []

    # Traverse the AST to find function definitions and their arguments
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for i, arg in enumerate(node.args.args):
                # If argument name starts with an underscore, store its position
                if arg.arg.startswith("_"):
                    rename_positions.append((node, i))

    # Rename the arguments
    for node, i in rename_positions:
        old_arg = node.args.args[i].arg
        new_arg = old_arg[1:] + "_"
        node.args.args[i].arg = new_arg

    # Convert the modified AST back to a string
    return astunparse.unparse(tree).strip()


def convert_datetimes_to_strings(func_call_str):
    # Define a regular expression pattern to match datetime objects
    datetime_pattern = r"\b(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\b"

    # Replace datetime objects with their string representations
    return re.sub(datetime_pattern, r'"\1"', func_call_str)
