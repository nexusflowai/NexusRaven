from typing import Any, Dict, List, Tuple

from dataclasses import dataclass

import os

from argparse import ArgumentParser

from collections import defaultdict

import re

import json

from datasets import Dataset, concatenate_datasets, load_dataset

from raven import ROOT_DIR


@dataclass
class ToolLLMFormatQueriesHelper:
    hf_path: str
    toolllm_queries_subset: str
    toolllm_api_list_subset: str
    standardized_queries_subset: str
    standardized_api_list_subset: str

    def run(self) -> None:
        api_list = load_dataset(
            self.hf_path, name=self.standardized_api_list_subset, split="train"
        )
        queries = load_dataset(
            self.hf_path, name=self.standardized_queries_subset, split="train"
        )

        api_list, queries = self.concatenate_toolllm(api_list, queries)

        python_functions = list(
            map(self.convert_to_python_function, api_list.to_iterable_dataset())
        )
        python_files = self.group_python_functions_by_name(python_functions)
        python_files = Dataset.from_list(python_files)

        json_function_descriptions = list(
            map(
                self.convert_to_json_function_description,
                api_list.to_iterable_dataset(),
            )
        )
        json_function_descriptions = self.group_json_functions_by_name(
            json_function_descriptions
        )
        json_function_descriptions = Dataset.from_list(json_function_descriptions)

        custom_files = Dataset.from_list(self.get_custom_files())

        json_query_dicts = list(
            map(self.convert_to_json_query, queries.to_iterable_dataset())
        )
        self.add_query_id(json_query_dicts)

        toolllm_queries = Dataset.from_list(json_query_dicts)
        toolllm_api_list = concatenate_datasets(
            [python_files, json_function_descriptions, custom_files]
        )

        toolllm_queries.push_to_hub(
            repo_id=self.hf_path,
            config_name=self.toolllm_queries_subset,
        )
        toolllm_api_list.push_to_hub(
            repo_id=self.hf_path,
            config_name=self.toolllm_api_list_subset,
        )

    def concatenate_toolllm(
        self, api_list: Dataset, queries: Dataset
    ) -> Tuple[Dataset, Dataset]:
        """
        ToolLLM data is not yet fully integrated into this repository, so here we take a chunk from the
        Nexusflow/NexusRaven_API_evaluation dataset
        """
        toolllm_api_list = load_dataset(
            path="Nexusflow/NexusRaven_API_evaluation",
            name="standardized_api_list",
            split="train",
        )
        toolllm_api_list = toolllm_api_list.filter(
            lambda s: s == "toolllm", input_columns="dataset"
        )
        api_list = api_list.filter(lambda s: s != "toolllm", input_columns="dataset")
        api_list = concatenate_datasets([api_list, toolllm_api_list])

        toolllm_queries = load_dataset(
            path="Nexusflow/NexusRaven_API_evaluation",
            name="standardized_queries",
            split="train",
        )
        toolllm_queries = toolllm_queries.filter(
            lambda s: s == "toolllm", input_columns="dataset"
        )
        queries = queries.filter(lambda s: s != "toolllm", input_columns="dataset")
        queries = concatenate_datasets([queries, toolllm_queries])

        return api_list, queries

    def convert_to_python_function(self, api_dict: Dict[str, Any]) -> Dict[str, str]:
        name = api_dict["name"]
        description = api_dict["description"]
        args_dicts = api_dict["args_dicts"]

        # Convert to lower since ToolLLM only accepts lowercase function names
        lowered_name = self.to_lower_str(name)
        description = description.replace(name, lowered_name)
        name = lowered_name
        for arg_dict in args_dicts:
            lowered_name = self.to_lower_str(arg_dict["name"])
            description = description.replace(arg_dict["name"], lowered_name)
            arg_dict["name"] = lowered_name

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

        function_str = f'def {name}({", ".join(args_strs)}):\n    """{description}"""\n    return "Is successful."'

        return {
            "dataset": api_dict["dataset"],
            "function_str": function_str,
        }

    def group_python_functions_by_name(
        self, function_dicts: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        groups = defaultdict(list)
        for function_dict in function_dicts:
            groups[function_dict["dataset"]].append(function_dict["function_str"])

        python_files = []
        for group_name, group_function_strs in groups.items():
            group_function_str = "\n\n".join(group_function_strs)
            group_dict = {
                "file_path": f"data/toolenv/tools/Customized/{group_name}/api.py",
                "file_content": group_function_str,
            }
            python_files.append(group_dict)

        return python_files

    def convert_to_json_function_description(
        self, api_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        name = api_dict["name"]
        description = api_dict["description"]
        args_dicts = api_dict["args_dicts"]

        # Convert to lower since ToolLLM only accepts lowercase function names
        lowered_name = self.to_lower_str(name)
        description = description.replace(name, lowered_name)
        name = lowered_name
        for arg_dict in args_dicts:
            lowered_name = self.to_lower_str(arg_dict["name"])
            description = description.replace(arg_dict["name"], lowered_name)
            arg_dict["name"] = lowered_name

        required_parameters = []
        optional_parameters = []
        for arg_dict in args_dicts:
            tool_llm_arg_dict = arg_dict.copy()
            tool_llm_arg_dict.pop("required")

            required = arg_dict["required"]
            if required:
                list_to_append_to = required_parameters
            else:
                list_to_append_to = optional_parameters
            list_to_append_to.append(tool_llm_arg_dict)

        function_dict = {
            "name": name,
            "url": "",
            "description": description,
            "method": "",
            "required_parameters": required_parameters,
            "optional_parameters": optional_parameters,
        }

        return {
            "dataset": api_dict["dataset"],
            "function_dict": function_dict,
        }

    def group_json_functions_by_name(
        self, function_dicts: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        groups = defaultdict(list)
        for function_dict in function_dicts:
            groups[function_dict["dataset"]].append(function_dict["function_dict"])

        json_function_dicts = []
        for group_name, group_function_dicts in groups.items():
            tool_dict = {
                "tool_description": group_name,
                "tool_name": group_name,
                "title": group_name,
                "standardized_name": group_name,
                "api_list": group_function_dicts,
            }

            json_function_dict = {
                "file_path": f"data/toolenv/tools/Customized/{group_name}.json",
                "file_content": json.dumps(tool_dict, indent=4),
            }
            json_function_dicts.append(json_function_dict)

        return json_function_dicts

    def get_custom_files(self) -> List[Dict[str, str]]:
        resources_path = os.path.join(ROOT_DIR, "data", "resources")
        rapidapi = {
            "file_path": "toolbench/inference/Downstream_tasks/rapidapi.py",
            "file_content": open(os.path.join(resources_path, "rapidapi.py")).read(),
        }
        server = {
            "file_path": "toolbench/inference/server.py",
            "file_content": open(os.path.join(resources_path, "server.py")).read(),
        }

        return [rapidapi, server]

    def convert_to_json_query(self, query_dict: Dict[str, Any]) -> Dict[str, str]:
        api_list = []
        for context_function in query_dict["context_functions"]:
            context_function_dict = {
                "api_name": self.to_lower_str(context_function),
                "category_name": "Customized",
                "tool_name": query_dict["dataset"],
            }
            api_list.append(context_function_dict)

        return {
            "api_list": api_list,
            "query": query_dict["prompt"],
        }

    def add_query_id(self, json_query_dicts: List[Dict[str, str]]) -> None:
        query_id = 80000
        for d in json_query_dicts:
            d["query_id"] = query_id
            query_id += 1

    @staticmethod
    def to_lower_str(s: str) -> str:
        # `lowerCamelCase` becomes `lowerxxxcamelxxxcase`
        s = re.sub("([A-Z])", r"xxx\1", s)
        s = s.lower()
        return s


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf_path", type=str, required=True)
    parser.add_argument("--toolllm_api_list_subset", type=str, required=True)
    parser.add_argument("--toolllm_queries_subset", type=str, required=True)
    parser.add_argument("--standardized_api_list_subset", type=str, required=True)
    parser.add_argument("--standardized_queries_subset", type=str, required=True)
    args = parser.parse_args()

    ToolLLMFormatQueriesHelper(**vars(args)).run()
