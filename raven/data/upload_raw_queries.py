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
                self.upload_local_queries("toolalpaca"),
                self.upload_local_queries("toolllm"),
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf_path", type=str, required=True)
    parser.add_argument("--subset", type=str, required=True)
    args = parser.parse_args()

    RawQueriesHelper(**vars(args)).run()
