import json
import pathlib


class FixtureFactory:
    def __init__(self, test_module_name: str | pathlib.Path):
        if test_module_name.endswith(".py"):
            test_module_name = test_module_name[:-3]
        if isinstance(test_module_name, str):
            test_module_name = pathlib.Path(test_module_name)
        test_module_name = test_module_name.name
        self.fixture_file_path = (
            pathlib.Path(__file__).parent / "fixtures" / f"{test_module_name}.json"
        )
        assert self.fixture_file_path.exists(), self.fixture_file_path

    def __call__(self, fixture_name):
        fixture_data = json.loads(self.fixture_file_path.read_text())
        return fixture_data.get(fixture_name)
