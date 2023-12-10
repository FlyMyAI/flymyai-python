import pathlib

from flymyai_client.src.flymyai_client.schema_obtainers.base_schema_loader import BaseSchemaLoader, LoadSchemaException


class FileSchemaLoader(BaseSchemaLoader):

    def __init__(self, input_file: pathlib.Path | str):
        if isinstance(input_file, pathlib.Path):
            input_file = pathlib.Path(input_file)
        if not input_file.exists():
            raise LoadSchemaException(f"OpenAPI schema file does not exist: {input_file.absolute()}")
        self._input_file = input_file

    def _get_openapi_schema(self) -> str:
        with open(self._input_file, 'r') as f:
            return f.read()
