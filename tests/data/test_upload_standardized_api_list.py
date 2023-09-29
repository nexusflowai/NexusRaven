from datasets import load_dataset

from raven.data.upload_standardized_api_list import UploadStandardizedAPIListHelper

from tests.utils import cleanup_local_push_to_hub, setup_local_push_to_hub


class TestUploadStandardizedAPIListHelper:
    def test_e2e(self) -> None:
        datasets_pushed = setup_local_push_to_hub()

        helper = UploadStandardizedAPIListHelper(
            hf_path="test_hf_path",
            subset="test_standardized_api_list_subset",
        )
        helper.run()

        expected_keys = [
            ("test_hf_path", "test_standardized_api_list_subset")
        ]
        actual_keys = list(datasets_pushed)
        assert expected_keys == actual_keys

        production_dataset = load_dataset(
            path="Nexusflow/NexusRaven_API_evaluation",
            name="standardized_api_list",
            split="train",
        )
        reproduced_dataset = datasets_pushed[("test_hf_path", "test_standardized_api_list_subset")]
        for prod_row, reprod_row in zip(production_dataset, reproduced_dataset):
            assert prod_row == reprod_row

        cleanup_local_push_to_hub()
