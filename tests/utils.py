from typing import Dict

from datasets import Dataset


def setup_local_push_to_hub() -> Dict[str, Dataset]:
    """
    Monkey patch the `push_to_hub` function so that no datasets are actually pushed to the hub
    """
    datasets_pushed = dict()
    def push_to_hub_local(dataset: Dataset, repo_id: str, config_name: str):
        datasets_pushed[(repo_id, config_name)] = dataset

    Dataset.original_push_to_hub = Dataset.push_to_hub
    Dataset.push_to_hub = push_to_hub_local

    return datasets_pushed


def cleanup_local_push_to_hub() -> None:
    Dataset.push_to_hub = Dataset.original_push_to_hub
