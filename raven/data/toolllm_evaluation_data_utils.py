from dataclasses import dataclass

# Necessary imports to successfully eval ToolLLM functions
from datetime import datetime
from enum import Enum as ENUM

from datasets import load_dataset


@dataclass
class ToolLLM:
    hf_path: str
    standardized_queries_subset: str

    def __post_init__(self) -> None:
        raw_queries_subset = self.standardized_queries_subset.replace("standardized", "raw")
        self.dataset = load_dataset(
            path=self.hf_path,
            name=raw_queries_subset,
            split="train",
        )

    def build_functions(self):
        item = None
        instruction = item["Input"]
        reference = item["Output"]
        function_strs = item["context"]

        function_str = None
        initial_namespace = set(locals())
        for function_str in function_strs:
            exec(function_str)
        final_namespace = locals()

        new_function_names = set(final_namespace) - initial_namespace
        functions = {n: final_namespace[n] for n in new_function_names}

        return functions
