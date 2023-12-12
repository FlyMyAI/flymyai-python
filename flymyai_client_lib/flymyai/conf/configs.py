import dataclasses
import pathlib


@dataclasses.dataclass
class BaseProjectSettings:
    project_dir: pathlib.Path
    username: pathlib.Path
    apikey: pathlib.Path

    @classmethod
    def load_settings(cls):
        ...
