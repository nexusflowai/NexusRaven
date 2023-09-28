from typing import Any, Dict, List

from dataclasses import dataclass

from collections import defaultdict

from argparse import ArgumentParser

import re

import json

from datasets import load_dataset

from raven.utils import build_functions, parse_function_call_to_name_and_args


@dataclass
class StandardizedQueriesHelper:
    hf_path: str
    standardized_queries_subset: str
    raw_queries_subset: str
    standardized_api_list_subset: str

    def run(self) -> None:
        api_list = load_dataset(
            self.hf_path, name=self.standardized_api_list_subset, split="train"
        )
        dataset_to_context_functions = defaultdict(list)
        for e in api_list:
            dataset_to_context_functions[e["dataset"]].append(e["name"])

        raw_queries = load_dataset(
            self.hf_path, name=self.raw_queries_subset, split="train"
        )
        standardized_queries = raw_queries.map(
            function=self._map_single,
            input_columns=["dataset", "query_dict"],
            remove_columns=raw_queries.column_names,
            fn_kwargs={"dataset_to_context_functions": dataset_to_context_functions},
        )

        functions = build_functions(api_list)
        for standardized_query in standardized_queries:
            reference_function_name = standardized_query["python_function_name"]
            reference_input_args_dict = json.loads(
                standardized_query["python_args_dict"]
            )

            try:
                functions[reference_function_name](**reference_input_args_dict)
            except Exception as e:
                print(reference_function_name, reference_input_args_dict)
                raise e

        standardized_queries.push_to_hub(
            repo_id=self.hf_path,
            config_name=self.standardized_queries_subset,
        )

    def _map_single(
        self,
        dataset: str,
        query_dict: str,
        dataset_to_context_functions: Dict[str, List[str]],
    ) -> Dict[str, str]:
        query_dict: Dict[str, Any] = json.loads(query_dict)
        match dataset:
            case "cve_cpe":
                d = self.cve_cpe(query_dict)
            case "emailrep" | "virustotal":
                context_functions = dataset_to_context_functions[dataset]
                d = self.parse_input_output(query_dict, context_functions)
            case "toolalpaca":
                d = self.toolalpaca(query_dict)
            case _:
                raise ValueError(f"Unrecognized dataset `{dataset}`")

        d["dataset"] = dataset
        return d

    def cve_cpe(self, query_dict: Dict[str, Any]) -> Dict[str, str]:
        function = query_dict["reference"]
        function = function.removeprefix("r = ")
        function = function.replace("nvdlib.", "")
        query_dict["reference"] = function

        context_functions = ["searchCVE", "searchCPE"]
        return self.parse_input_output(query_dict, context_functions)

    def parse_input_output(
        self, query_dict: Dict[str, Any], context_functions: List[str]
    ) -> Dict[str, str]:
        function_name, _, args_dict = parse_function_call_to_name_and_args(
            query_dict["reference"]
        )
        d = {
            "prompt": query_dict["query"],
            "python_function_name": function_name,
            "python_args_dict": json.dumps(args_dict),
            "context_functions": context_functions,
        }
        return d

    def toolalpaca(self, query_dict: Dict[str, Any]) -> Dict[str, str]:
        query_dict = query_dict["turns_1"]  # Just the first turn

        # These function names must match the API name later on
        reference = json.loads(query_dict["golden_answer"])[0]
        function_name, _, args_dict = parse_function_call_to_name_and_args(reference)

        func_pattern = re.compile("<func_start>def (.*?)<func_end>", re.S)
        matches = func_pattern.findall(query_dict["docstring"])
        context_functions = []
        for match in matches:
            paren_idx = match.find("(")
            match = match[:paren_idx]
            context_functions.append(match)

        return {
            "prompt": query_dict["instruction"],
            "python_function_name": function_name,
            "python_args_dict": json.dumps(args_dict),
            "context_functions": context_functions,
        }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf_path", type=str, required=True)
    parser.add_argument("--standardized_queries_subset", type=str, required=True)
    parser.add_argument("--raw_queries_subset", type=str, required=True)
    parser.add_argument("--standardized_api_list_subset", type=str, required=True)
    args = parser.parse_args()

    StandardizedQueriesHelper(**vars(args)).run()
