import abc
from sentence_transformers import SentenceTransformer
import torch

class Baai_similarity:
    
    def __init__(self, model_path, device) -> None:
        self.device = device
        self.model = SentenceTransformer(model_path, device=device)

    def torch_gc(self):
        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def generate(self, data):
        with torch.no_grad():
            try:
                text_a = data.get('text_a')
                text_b = data.get('text_b')

                text = text_a + text_b
                embeddings = self.model.encode(text, normalize_embeddings=True)
                sim = embeddings[0:len(text_a)] @ embeddings[len(text_a):len(text)].T
                sim = sim.tolist()

                self.torch_gc()
            except Exception as e:
                self.torch_gc()
                raise e
            answer = {"response": sim}
            return answer





