from typing import Any, Callable, Dict, List, Tuple

import ast

import json

from datasets import Dataset


def build_functions(d: Dataset) -> Dict[str, Callable]:
    function_names = []
    for function_dict in d:
        name = function_dict["name"]
        description = function_dict["description"]
        args_dicts = function_dict["args_dicts"]

        args_dicts = sorted(args_dicts, key=lambda d: d["required"], reverse=True)

        args_strs = []
        for arg_dict in args_dicts:
            if arg_dict["required"]:
                default_str = ""
            else:
                default = arg_dict["default"]
                if isinstance(default, str) and default.startswith("JSON:"):
                    default = default.removeprefix("JSON:")
                    default = json.loads(default)
                default_str = f" = {default}"

            if arg_dict["type"] == "None":
                type_str = ""
            else:
                type_str = f": {arg_dict['type']}"

            args_str = f"{arg_dict['name']}{type_str}{default_str}"
            args_strs.append(args_str)

        function_str = f'def {name}({", ".join(args_strs)}):\n    """{description}"""\n    return ("{name}", locals())'
        exec(function_str)
        function_names.append(name)

    namespace = locals()
    functions = {f: namespace[f] for f in function_names}
    return functions


def parse_function_call_to_name_and_args(
    function: str,
) -> Tuple[str, List[str], Dict[str, Any]]:
    function = ast.parse(function).body[0]
    function_name = function.value.func.id
    keywords = function.value.keywords
    keywords = {k.arg: ast.literal_eval(k.value) for k in keywords}

    args = [ast.literal_eval(arg) for arg in function.value.args]

    """
    We use integers only for evaluation data since floats are tricky to evaluate
    e.g. the string representation of "3.33" may be "3.329999999" which yields an exact match of 0
    """
    keywords = {k: int(v) if isinstance(v, float) else v for k, v in keywords.items()}

    return function_name, args, keywords
