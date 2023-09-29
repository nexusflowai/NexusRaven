from os.path import join

from raven import ROOT_DIR
from raven.data.process_toolalpaca_evaluation_data import ToolAlpacaDataPostprocessHelper


class TestToolAlpacaDataPostprocessHelper:
    def test_e2e(self) -> None:
        fpath = join(ROOT_DIR, "data", "resources", "toolalpaca_queries.json")
        with open(fpath) as f:
            expected_content = f.read()

        helper = ToolAlpacaDataPostprocessHelper()
        helper.run()

        with open(fpath) as f:
            actual_content = f.read()

        assert expected_content == actual_content
