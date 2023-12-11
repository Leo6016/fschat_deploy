import abc
from sentence_transformers import SentenceTransformer
import torch

class Embedding: 
    @abc.abstractmethod
    def get_embedding(self, sentence):
        pass

class SentenceEmbedding(Embedding):
    
    def __init__(self, model_path, device) -> None:
        self.device = device
        self.model = SentenceTransformer(model_path, device=device)

    def generate(self, params):
        sentence=params.get("batch_seq")
        output = self.model.encode(sentence, device=self.device)
        self.delete_memory([])
        return output.tolist()

    def delete_memory(self, tensor_list):
        with torch.cuda.device(self.device):
            for item in tensor_list:
                item = None 
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()



