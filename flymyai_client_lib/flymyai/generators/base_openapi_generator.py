import pathlib
import subprocess
from typing import Any


class ModelGenerationError(Exception):
    ...


class BaseOpenapiGenerator:
    _output_file: pathlib.Path

    @property
    def output_file(self):
        return self._output_file

    @output_file.setter
    def output_file(self, out_file: Any):
        self._output_file = out_file

    def generate_models(self, schemas_data: str):
        try:
            subprocess.check_call(
                [
                    "datamodel-codegen",
                    f"--output_file {self.output_file}",
                    f"--input-file-type jsonschema",
                ],
                stdin=schemas_data,
            )
        except subprocess.CalledProcessError as e:
            raise ModelGenerationError(*e.args)
