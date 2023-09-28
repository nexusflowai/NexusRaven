from typing import Callable, Dict, List

from dataclasses import dataclass

import json

# Necessary imports to successfully eval ToolLLM functions
from datetime import datetime
from enum import Enum as ENUM

from datasets import Dataset, load_dataset


@dataclass
class ToolLLMEvaluationDataHelper:
    hf_path: str
    standardized_queries_subset: str

    def __post_init__(self) -> None:
        raw_queries_subset = self.standardized_queries_subset.replace(
            "standardized", "raw"
        )
        self.dataset = load_dataset(
            path=self.hf_path,
            name=raw_queries_subset,
            split="train",
        )
        self.dataset = self.dataset.filter(
            lambda s: s == "toolllm", input_columns="dataset"
        )
        self.processed_dataset: Dataset | None = None

    def build_functions(self) -> Dict[str, Callable]:
        context_functions: List[List[str]] = []
        prompts: List[str] = []
        reference_function_calls: List[str] = []
        functions: Dict[str, Callable] = dict()
        for example in self.dataset:
            example = json.loads(example["query_dict"])
            function_strs = example["context"]

            function_str = None
            function_name = None
            function_names = []
            for function_str in function_strs:
                # Replace braces since they cause problems for langchain's formatting
                function_str = function_str.replace("{", "")
                function_str = function_str.replace("}", "")

                function_name = function_str[: function_str.find("(")].removeprefix(
                    "def "
                )

                # Remove the function body to use our own for evaluation
                function_str = function_str[: function_str.find("args_dict")]
                function_str = f"""
{function_str}return ("{function_name}", {{k: int(v) if isinstance(v, float) else v for k, v in locals().items()}})
"""

                exec(function_str)

                function_names.append(function_name)

            namespace = locals()
            new_functions = {n: namespace[n] for n in function_names}
            functions.update(new_functions)

            context_functions.append(function_names)

            prompts.append(example["Input"])
            reference_function_calls.append(example["Output"])

        self.processed_dataset = Dataset.from_dict(
            {
                "context_functions": context_functions,
                "prompt": prompts,
                "reference_function_call": reference_function_calls,
            }
        )

        return functions

    def get_eval_dataset(self) -> Dataset:
        if self.processed_dataset is None:
            self.build_functions()

        return self.processed_dataset
