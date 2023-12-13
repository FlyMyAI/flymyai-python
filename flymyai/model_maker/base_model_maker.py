import json
import pathlib
from typing import Type

from generators.base_openapi_generator import (
    BaseOpenapiGenerator,
)
from schema_obtainers.base_schema_loader import (
    BaseSchemaLoader,
)


class BaseModelMaker:
    schema_loader_class: Type[BaseSchemaLoader] = BaseSchemaLoader
    schema_generator_class: Type[BaseOpenapiGenerator] = BaseOpenapiGenerator

    def __init__(self, output_file: pathlib.Path):
        self.schema_generator = self._init_schema_generator()
        self.schema_loader = self._init_schema_loader()
        self.schema_generator.output_file = output_file

    def _init_schema_loader(self):
        return self.schema_loader_class()

    def _init_schema_generator(self):
        return self.schema_generator_class()

    def generate_model_py(self):
        on_generate_data = json.loads(self.schema_loader.dynamic_schemas)
        self.schema_generator.generate_models(on_generate_data)
