from datasets import load_dataset

from raven.data.upload_raw_queries import RawQueriesHelper

from tests.utils import cleanup_local_push_to_hub, setup_local_push_to_hub


class TestRawQueriesHelper:
    def test_e2e(self) -> None:
        datasets_pushed = setup_local_push_to_hub()

        helper = RawQueriesHelper(
            hf_path="test_hf_path",
            subset="test_raw_queries_subset",
        )
        helper.run()

        expected_keys = [
            ("test_hf_path", "test_raw_queries_subset")
        ]
        actual_keys = list(datasets_pushed)
        assert expected_keys == actual_keys

        production_dataset = load_dataset(
            path="Nexusflow/NexusRaven_API_evaluation",
            name="raw_queries",
            split="train",
        )
        reproduced_dataset = datasets_pushed[("test_hf_path", "test_raw_queries_subset")]
        for prod_row, reprod_row in zip(production_dataset, reproduced_dataset):
            assert prod_row == reprod_row

        cleanup_local_push_to_hub()
