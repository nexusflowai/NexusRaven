from dataclasses import dataclass

from argparse import ArgumentParser

import os

import json

from datasets import Dataset, concatenate_datasets, load_dataset

from raven import ROOT_DIR


@dataclass
class RawQueriesHelper:
    hf_path: str
    subset: str

    def run(self) -> None:
        d = concatenate_datasets(
            [
                self.upload_local_queries("cve_cpe"),
                self.upload_local_queries("emailrep"),
                self.upload_local_queries("virustotal"),
                self.toolalpaca(),
                self.toolllm(),
            ]
        )
        d.push_to_hub(
            repo_id=self.hf_path,
            config_name=self.subset,
        )

    def upload_local_queries(self, dataset: str) -> Dataset:
        with open(
            os.path.join(ROOT_DIR, "data", "resources", f"{dataset}_queries.json")
        ) as f:
            content = json.load(f)

        dicts = list(map(json.dumps, content))
        return Dataset.from_dict(
            {
                "dataset": [dataset] * len(dicts),
                "query_dict": dicts,
            }
        )

    def toolalpaca(self) -> Dataset:
        d = load_dataset("Nexusflow/toolalpaca_simulated", split="train")
        dicts = list(map(json.dumps, d.to_iterable_dataset()))
        return Dataset.from_dict(
            {
                "dataset": ["toolalpaca"] * len(dicts),
                "query_dict": dicts,
            }
        )

    def toolllm(self) -> Dataset:
        d = load_dataset("Nexusflow/toolllm_eval", split="train")
        dicts = list(map(json.dumps, d.to_iterable_dataset()))
        return Dataset.from_dict(
            {
                "dataset": ["toolllm"] * len(dicts),
                "query_dict": dicts,
            }
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf_path", type=str, required=True)
    parser.add_argument("--subset", type=str, required=True)
    args = parser.parse_args()

    RawQueriesHelper(**vars(args)).run()
