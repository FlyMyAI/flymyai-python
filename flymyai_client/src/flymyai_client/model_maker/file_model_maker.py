import pathlib

from flymyai_client.src.flymyai_client.model_maker.base_model_maker import BaseModelMaker
from flymyai_client.src.flymyai_client.schema_obtainers.file_schema_loader import FileSchemaLoader


class FileModelMaker(BaseModelMaker):
    schema_loader_class = FileSchemaLoader

    def __init__(self, input_file: pathlib.Path, output_file: pathlib.Path):
        self._input_file = input_file
        super().__init__(output_file)

    def _init_schema_loader(self):
        return self.schema_loader_class(self._input_file)
