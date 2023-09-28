from typing import Any, Dict, List

from dataclasses import dataclass

from argparse import ArgumentParser

import os

import json

import ast

import re

from itertools import chain

from datasets import Dataset, concatenate_datasets, load_dataset

from raven.utils import build_functions


@dataclass
class UploadStandardizedAPIListHelper:
    hf_path: str
    subset: str

    def run(self) -> None:
        d = concatenate_datasets(
            [
                self.cve_cpe(),
                self.emailrep(),
                self.virustotal(),
                self.toolalpaca(),
                self.toolllm(),
            ]
        )

        d: Dataset = d.map(
            function=self.standardize_arguments,
            input_columns="args_dicts",
            desc="Standardizing arguments",
        )

        build_functions(d)

        d.push_to_hub(
            repo_id=self.hf_path,
            config_name=self.subset,
        )

    def standardize_arguments(
        self, ds: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        for d in ds:
            d["name"] = d["name"].replace("-", "_")  # Argument names must be pythonic

            if d["type"] in ["str", "list", "bool", "int", "dict", "None"]:
                pass
            elif "list" in d["type"].lower():
                d["type"] = "list"
            elif "array" in d["type"].lower():
                d["type"] = "list"
            elif "bool" in d["type"].lower():
                d["type"] = "bool"
            elif "int" in d["type"].lower():
                d["type"] = "int"
            elif "str" in d["type"].lower():
                d["type"] = "str"
            elif "json" in d["type"].lower():
                d["type"] = "str"
            elif "enum" in d["type"].lower():
                d["type"] = "str"
            elif "datetime" in d["type"].lower():
                d["type"] = "str"
            else:
                raise ValueError(f"Unrecognized argument type `{d['type']}`")

        return {"args_dicts": ds}

    def cve_cpe(self) -> Dataset:
        p = os.path.split(__file__)[0]
        function_definitions_path = os.path.join(
            p, "resources", "cve_cpe_function_definitions.py"
        )
        with open(function_definitions_path) as f:
            content = f.read()

        content = ast.parse(content).body

        function_dicts: List[Dict[str, Any]] = []
        function: ast.FunctionDef
        for function in content:
            description = function.body[0].value.value

            args_str = description.split("Args:")[1].strip()
            args_strs = filter(None, args_str.split("\n"))
            args_dicts = []
            for arg_str in args_strs:
                name_type_str = arg_str[: arg_str.find(":")].strip()
                name = name_type_str.split()[0].strip()
                description = arg_str[arg_str.find(":") :].strip()
                arg_dict = {
                    "name": name,
                    "type": "None",
                    "required": False,  # All optional
                    "description": description,
                    "default": "None",
                }
                args_dicts.append(arg_dict)

            function_dict = {
                "dataset": "cve_cpe",
                "name": function.name,
                "description": function.body[0].value.value,
                "args_dicts": args_dicts,
            }
            function_dicts.append(function_dict)

        return Dataset.from_list(function_dicts)

    def emailrep(self) -> Dataset:
        """
        Function structure from https://github.com/sublime-security/emailrep.io-python/blob/9e187c9f8d1ba25fae0e32fc92e4ccc380b77933/emailrep/__init__.py#L6

        Argument descriptions
        - emailrep_post https://docs.sublimesecurity.com/reference/get_-email
        - emailrep_report https://docs.sublimesecurity.com/reference/post_report

        We additionally add a required `key` argument to the `emailrep_report` arguments since it's required to create a report.
        We remove the `description` and `tags` arguments since they are arbitrary.
        We remove the `timestamp` and `expires` args since they are never used.

        """

        emailrep_post = {
            "dataset": "emailrep",
            "name": "emailrep_post",
            "description": """
Query an email address

Parameters
email (string, required): Email address being queried
""",
            "args_dicts": [
                {
                    "name": "email",
                    "description": "Email address being queried",
                    "type": "string",
                    "default": "None",
                    "required": True,
                },
            ],
        }

        emailrep_report = {
            "dataset": "emailrep",
            "name": "emailrep_report",
            "description": """
Reports an email address. Date of malicious activity defaults to the current time unless otherwise specified.

Parameters:
email (string, required): Email address being reported.
key (string, required): The API key of the user.
""",
            "args_dicts": [
                {
                    "name": "email",
                    "description": "Email address being reported.",
                    "type": "string",
                    "default": "None",
                    "required": True,
                },
                {
                    "name": "key",
                    "description": "The API key of the user.",
                    "type": "string",
                    "default": "None",
                    "required": True,
                },
            ],
        }

        return Dataset.from_list([emailrep_post, emailrep_report])

    def virustotal(self) -> Dataset:
        """
        All function documentation is sourced from https://developers.virustotal.com/reference/overview
        since the existing VirusTotal Python client does not provide function definitions https://github.com/VirusTotal/vt-py/. vt-py just calls into the overall VirusTotal http endpoint

        """
        p = os.path.split(__file__)[0]
        function_definitions_path = os.path.join(
            p, "resources", "virustotal_function_definitions.py"
        )
        with open(function_definitions_path) as f:
            content = f.read()

        content = ast.parse(content).body

        pattern = r"-\s(\w.+):\s(\w+),\s(\w+),\s(.+)"
        pattern = re.compile(pattern)

        function_dicts: List[Dict[str, Any]] = []
        function: ast.FunctionDef
        for function in content:
            args_list = list(function.args.args)
            defaults = list(function.args.defaults)
            defaults = [None] * (len(args_list) - len(defaults)) + defaults

            args_dicts = []
            for arg, default in zip(args_list, defaults):
                arg_dict = {
                    "name": arg.arg,
                    "type": arg.annotation.id,
                    "required": default is None,
                    "description": "",
                    "default": "None" if default is None else default.value,
                }
                args_dicts.append(arg_dict)

            function_dict = {
                "dataset": "virustotal",
                "name": function.name,
                "description": function.body[0].value.value,
                "args_dicts": args_dicts,
            }
            function_dicts.append(function_dict)

        return Dataset.from_list(function_dicts)

    def toolalpaca(self) -> Dataset:
        d = load_dataset("Nexusflow/toolalpaca_simulated", split="train")
        all_docstrings: List[str] = [_d["docstring"] for _d in d["turns_1"]]

        func_pattern = re.compile("<func_start>(.*?)<func_end>", re.S)
        docstring_pattern = re.compile("<docstring_start>(.*?)<docstring_end>", re.S)

        function_strs = []
        for docstrings_to_parse in all_docstrings:
            func_signatures = re.findall(func_pattern, docstrings_to_parse)
            docstrings = re.findall(docstring_pattern, docstrings_to_parse)
            for func_signature, docstring in zip(func_signatures, docstrings):
                function_str = f"{func_signature}:\n    {docstring.strip()}"
                function_strs.append(function_str)

        function_strs = list(dict.fromkeys(function_strs))  # dedupe

        function_dicts = []
        for function_str in function_strs:
            function: ast.FunctionDef = ast.parse(function_str).body[0]
            description = function.body[0].value.value
            description = description.replace("{", "").replace("}", "")

            args_dicts = []
            for arg in function.args.args:
                if isinstance(arg.annotation, ast.Name):
                    type_str = arg.annotation.id
                else:
                    type_str = arg.annotation.value.id
                arg_dict = {
                    "name": arg.arg,
                    "description": "",
                    "type": type_str,
                    "default": "None",  # No defaults
                    "required": True,
                }
                args_dicts.append(arg_dict)

            function_dict = {
                "dataset": "toolalpaca",
                "name": function.name,
                "description": description.strip(),
                "args_dicts": args_dicts,
            }
            function_dicts.append(function_dict)
        return Dataset.from_list(function_dicts)

    def toolllm(self) -> None:
        d = load_dataset("Nexusflow/toolllm_eval", split="train")
        contexts: List[List[str]] = d["context"]
        contexts = chain.from_iterable(contexts)
        contexts = dict.fromkeys(contexts)
        contexts: List[str] = list(contexts)

        datetime_pattern = r"(\(\(\d{4} - \d{2}\) - \d{2}\))"
        datetime_pattern = re.compile(datetime_pattern)

        def map_single(context: str) -> Dict[str, Any]:
            context = datetime_pattern.sub(r'"\1"', context)

            function: ast.FunctionDef = ast.parse(context).body[0]

            args_list = list(function.args.args)
            defaults = list(function.args.defaults)
            defaults = [None] * (len(args_list) - len(defaults)) + defaults

            args_dicts = []
            for arg, default in zip(args_list, defaults):
                if default is None:
                    default = "None"
                    required = True
                else:
                    default = ast.literal_eval(default)
                    if isinstance(default, str):
                        default = default.replace('"', "")
                        default = f'"{default}"'
                    default = f"JSON:{json.dumps(default)}"
                    required = False

                arg_dict = {
                    "name": arg.arg,
                    "description": "",
                    "type": arg.annotation.id,
                    "default": default,
                    "required": required,
                }
                args_dicts.append(arg_dict)

            description = function.body[0].value.value
            description = description.strip()
            description = description.replace("{", "[")
            description = description.replace("}", "]")
            function_dict = {
                "dataset": "toolllm",
                "name": function.name,
                "description": description,
                "args_dicts": args_dicts,
            }
            return function_dict

        function_dicts = list(map(map_single, contexts))
        return Dataset.from_list(function_dicts)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf_path", type=str, required=True)
    parser.add_argument("--subset", type=str, required=True)
    args = parser.parse_args()

    UploadStandardizedAPIListHelper(**vars(args)).run()
