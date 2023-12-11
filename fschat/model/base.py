import abc

class BaseModel:

    tokenizer = None
    model = None

    @abc.abstractmethod
    def chat(self, query: str, params: dict):
        pass

    @abc.abstractmethod
    def generate(self, query_lists: list, params: dict):
        pass
