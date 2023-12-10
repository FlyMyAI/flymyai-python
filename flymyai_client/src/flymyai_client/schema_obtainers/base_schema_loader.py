import json


class LoadSchemaException(Exception):
    ...


class BaseSchemaLoader:
    _openapi_schema: dict

    def _get_openapi_schema(self) -> str:
        raise NotImplemented

    @property
    def openapi_schema(self):
        if not hasattr(self, '_openapi_schema'):
            self._openapi_schema = json.loads(self._get_openapi_schema())
        return self._openapi_schema

    @property
    def dynamic_input_model(self):
        return self._openapi_schema['components']['schemas']['DynamicInputModel']

    @property
    def dynamic_output_model(self):
        return self._openapi_schema['components']['schemas']['DynamicOutputModel']

    @property
    def dynamic_schemas(self):
        return {
            'schemas': {
                'DynamicInputModel': self.dynamic_input_model,
                'DynamicOutputModel': self.dynamic_input_model
            }
        }
