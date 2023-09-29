from typing import Dict, List

from dataclasses import dataclass

from argparse import ArgumentParser

import os

import re

import json

from datasets import Dataset, load_dataset

from raven import CACHE_DIR


@dataclass
class RunToolLLM:
    hf_path: str
    toolllm_api_list_subset: str
    toolllm_queries_subset: str
    toolllm_outputs_subset: str
    tool_name: str | None = None

    def run(self) -> None:
        toolllm_path = os.path.join(CACHE_DIR, "ToolBench")
        setup_command = f"""
rm -rf {toolllm_path}/output* {toolllm_path}/results.txt
git clone https://github.com/OpenBMB/ToolBench {toolllm_path}
cd {toolllm_path}
pip -q -q -q install -r requirements.txt --user
wget -nc 'https://docs.google.com/uc?export=download&id=1Vis-RxBstXLKC1W1agIQUJNuumPJrrw0&confirm=yes' -O data.zip
unzip -n data.zip
"""
        os.system(setup_command)

        files_to_move = load_dataset(
            path=self.hf_path,
            name=self.toolllm_api_list_subset,
            split="train",
        )
        for file_dict in files_to_move:
            file_path = os.path.join(toolllm_path, file_dict["file_path"])
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                f.write(file_dict["file_content"])

        query_dataset = load_dataset(
            path=self.hf_path,
            name=self.toolllm_queries_subset,
            split="train",
        )
        if self.tool_name is not None:
            query_dataset = query_dataset.filter(
                lambda l: l[0]["tool_name"] == self.tool_name,
                input_columns="api_list",
            )
        query_path = os.path.join(toolllm_path, "query.json")
        with open(query_path, "w") as f:
            f.write(json.dumps(list(query_dataset.to_iterable_dataset()), indent=4))

        os.system(
            f"""
cd {toolllm_path}
export PYTHONPATH=./
python toolbench/inference/qa_pipeline.py \
    --tool_root_dir data/toolenv/tools/ \
    --backbone_model toolllama \
    --model_path ToolBench/ToolLLaMA-7b \
    --max_observation_length 1024 \
    --observ_compress_method truncate \
    --method DFS_woFilter_w2 \
    --input_query_file query.json \
    --output_answer_file ./output_query.json \
    --api_customization
"""
        )

        results_path = os.path.join(toolllm_path, "results.txt")
        responses = self.parse_results_file_into_responses(results_path)
        responses = Dataset.from_dict({"response": responses})
        responses.push_to_hub(
            repo_id=self.hf_path,
            config_name=self.toolllm_outputs_subset,
        )

    def parse_results_file_into_responses(self, results_path: str) -> List[List[Dict]]:
        with open(results_path) as f:
            lines = map(str.strip, f.readlines())

        curr: List[Dict] = []
        res: List[List[Dict]] = []
        for line in lines:
            if line == "========":
                if len(curr) > 1:
                    res.append(curr)
                curr = []
            else:
                d = json.loads(line)
                if "function_call" in d:
                    d["function_call"] = self.to_upper_str(d["function_call"])

                curr.append(d)

        res.append(curr)

        return res

    def to_upper_str(self, s: str) -> str:
        # `lowerxxxcamelxxxcase` becomes `lowerCamelCase`
        s = re.sub("xxx([a-z])", lambda x: x.group(1).upper(), s)
        return s


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf_path", type=str, required=True)
    parser.add_argument("--toolllm_api_list_subset", type=str, required=True)
    parser.add_argument("--toolllm_queries_subset", type=str, required=True)
    parser.add_argument("--toolllm_outputs_subset", type=str, required=True)
    parser.add_argument("--tool_name", type=str)
    args = parser.parse_args()

    RunToolLLM(**vars(args)).run()
