from flymyai.core.response_factory.base_response_factory import ResponseFactory


class MaybeNotExistent(Exception): ...


class AsyncTaskResultFactory(ResponseFactory):
    def construct(self):
        return self._base_construct_from_httpx_response()
