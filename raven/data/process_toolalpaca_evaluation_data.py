import os

import json

import re

import requests

from collections import defaultdict

from datasets import Dataset

from raven import ROOT_DIR


class ToolAlpacaDataPostprocessHelper:
    def run(self) -> None:
        response = requests.get(
            "https://raw.githubusercontent.com/tangqiaoyu/ToolAlpaca/main/data/eval_simulated.json"
        )
        content = response.content

        file = json.loads(content)
        eval_dataset = defaultdict(list)
        for item in file:
            (
                docstring,
                function_mapping,
            ) = self.generate_valid_python_functions_with_types(
                item["Function_Description"]
            )
            for instruction, golden_turns in zip(
                item["Instructions"], item["Golden_Answers"]
            ):
                # Transform the golden answers into a function call
                # We only want single turn for now.
                include = True
                # Write to the file
                parsed_turns = []
                for golden_answer in golden_turns:
                    golden_name = golden_answer["Action"]

                    if not golden_name in item["Function_Description"]:
                        print("ERROR", golden_name)
                        include = False
                        continue

                    golden_name = golden_name.replace("-", "_")
                    golden_call = golden_answer["Action_Input"]

                    golden_call = golden_call.replace(
                        """"symbols": "symbols": """, """"symbols": """
                    )
                    function_call = function_mapping[golden_name]
                    try:
                        function_call = self.construct_python_call(
                            function_call, golden_call
                        )
                    except json.decoder.JSONDecodeError:
                        # Skip this sample
                        continue
                    parsed_turns.append(function_call)

                # Serialize the list into a JSON string
                if len(parsed_turns) == 0:
                    continue

                serialized_parsed_turns = json.dumps(parsed_turns)
                serialized_backup = json.dumps((instruction, golden_turns))
                call = {
                    "docstring": docstring,
                    "instruction": instruction,
                    "golden_answer": serialized_parsed_turns,
                    "backup": serialized_backup,
                }
                key = f"turns_{len(golden_turns)}"
                if include:
                    eval_dataset[key].append(call)

        # Check the length of each list in the dict
        lengths = [len(v) for v in eval_dataset.values()]

        # If they aren't all the same, find the max length and pad the shorter ones
        if len(set(lengths)) > 1:
            max_length = max(lengths)
            for key, value in eval_dataset.items():
                if len(value) < max_length:
                    eval_dataset[key] = value + [{}] * (max_length - len(value))

        # Now the lengths should be the same and Dataset.from_dict() should work
        hf_dataset = Dataset.from_dict(eval_dataset)

        output_path = os.path.join(
            ROOT_DIR, "data", "resources", "toolalpaca_queries.json"
        )
        with open(output_path, "w") as f:
            json.dump(list(hf_dataset.to_iterable_dataset()), f, indent=4)

    def construct_python_call(self, function_call: str, golden_call: str) -> str:
        # Step 1: Parsing function_call
        function_match = re.match(r"(\w+)\((.*)\)", function_call)
        if not function_match:
            raise ValueError("Invalid function_call format")

        function_name = function_match.group(1)
        function_args_str = function_match.group(2).strip()

        # Handle no-argument case
        if not function_args_str:
            return f"{function_name}()"

        # Extract argument names
        arg_pairs = function_args_str.split(",")
        arg_names = [arg_pair.split(":")[0].strip() for arg_pair in arg_pairs]

        # Step 2: Parsing golden_call
        arg_values_dict = json.loads(golden_call)

        # Step 3: Generating actual Python call
        actual_call_args = []
        for arg_name in arg_names:
            # Default to None if the argument name is not found in arg_values_dict
            value = arg_values_dict.get(arg_name, None)

            if isinstance(value, str):
                actual_call_args.append(f'{arg_name}="{value}"')
            else:
                actual_call_args.append(f"{arg_name}={value}")

        actual_call = f'{function_name}({", ".join(actual_call_args)})'
        return actual_call

    def generate_valid_python_functions_with_types(self, functions):
        """
        Modify the function to use valid Python types for the parameters
        """
        python_code = ""
        type_mapping = {
            "Array[Object": "list[dict]",
            "Array": "list",
            "string": "str",
            "integer": "int",
            "array": "list",
            "object": "dict",
            "boolean": "bool",
            "Object.": "dict",
            "number": "float",
        }
        function_mapping = {}

        for function_name, function_desc in functions.items():
            if function_desc == "":
                continue

            function_name = function_name.replace("-", "_")

            summary = function_desc.split("\n")[0]

            operation_id = function_name
            try:
                parameters = json.loads(
                    function_desc.split("\n")[1].replace("Parameters: ", "")
                )
            except:
                # Try using normal
                try:
                    parameters = json.loads(
                        function_desc.split("\n")[0].replace("Parameters: ", "")
                    )
                except:
                    continue
            full_parameters = []

            function_name = operation_id
            function_parameters = []
            for param_name, param in parameters.items():
                param_type = None
                for key, value in type_mapping.items():
                    if key in param:
                        if param_type:
                            # Multiple types
                            raise ValueError
                        param_type = value
                        break
                if not param_type:
                    # No types
                    print(param)
                    raise ValueError

                param_default = None
                match = re.search(r"default is (.+)", param)
                if match:
                    extracted_value = match.group(1)
                    param_default = extracted_value
                param_required = (
                    not "Optional." in param
                )  # "required" in param or "Required" in param

                if param_required:
                    function_parameters.append(f"{param_name}: {param_type}")
                else:
                    function_parameters.append(
                        f"{param_name}: {param_type} = {param_default}"
                    )

                full_parameters.append(
                    {
                        "name": param_name,
                        "description": param,
                        "param_type": param_type,
                        "required": param_required,
                        "default": param_default,
                    }
                )

            function_parameters_str = ", ".join(function_parameters)

            docstring_lines = [
                '"""',
                f"{summary}",
                "",
            ]
            if len(parameters) > 0:
                docstring_lines.append("Parameters:")
            for param in full_parameters:
                param_name = param["name"]
                param_desc = param.get("description", "")
                param_type = param["param_type"]
                param_required = (
                    " required" if param.get("required", False) else " optional"
                )
                docstring_lines.append(
                    f"    {param_name} ({param_type}){param_required}: {param_desc}"
                )

            docstring_lines.append('"""')
            docstring_str = "\n".join(docstring_lines)

            if function_name == "":
                continue

            function_prototype = f"\nOPTION:\n<func_start>def {function_name}({function_parameters_str})<func_end>\n<docstring_start>\n{docstring_str}\n<docstring_end>\n"
            function_mapping[
                function_name
            ] = f"{function_name}({function_parameters_str})"

            python_code += function_prototype

        return python_code, function_mapping


if __name__ == "__main__":
    ToolAlpacaDataPostprocessHelper().run()
