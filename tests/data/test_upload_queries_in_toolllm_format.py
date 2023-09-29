from raven.data.upload_queries_in_toolllm_format import ToolLLMFormatQueriesHelper

from tests.utils import setup_local_push_to_hub, cleanup_local_push_to_hub


class TestToolLLMFormatQueriesHelper:
    def test_e2e(self) -> None:
        datasets_pushed = setup_local_push_to_hub()

        helper = ToolLLMFormatQueriesHelper(
            hf_path="Nexusflow/NexusRaven_API_evaluation",
            toolllm_queries_subset="test_toolllm_queries_subset",
            toolllm_api_list_subset="test_toolllm_api_list_subset",
            standardized_queries_subset="standardized_queries",
            standardized_api_list_subset="standardized_api_list",
        )
        helper.run()

        expected_keys = [
            ("Nexusflow/NexusRaven_API_evaluation", "test_toolllm_queries_subset"),
            ("Nexusflow/NexusRaven_API_evaluation", "test_toolllm_api_list_subset"),
        ]
        actual_keys = list(datasets_pushed)
        assert expected_keys == actual_keys

        cleanup_local_push_to_hub()
