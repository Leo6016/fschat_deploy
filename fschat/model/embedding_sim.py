import transformers
import torch.nn.functional as F
import torch

class embedding_sim:
    def __init__(self,model_path,device) -> None:
        self.model_path=model_path
        self.device=device
        self.model=transformers.AutoModel.from_pretrained(model_path, trust_remote_code=True).cuda(device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()

    def torch_gc(self):
        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                # 释放缓存分配器当前持有的所有未占用的缓存内存，以便这些内存可以在其他 GPU 应用程序中使用并在 nvidia-smi 中可见
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def generate(self,data):
        with torch.no_grad():
            try:
                text_a = data.get('text_a')
                text_b = data.get('text_b')
                
                input_a = self.tokenizer(text_a, return_tensors='pt').to(self.device)
                input_b = self.tokenizer(text_b, return_tensors='pt').to(self.device)

                encoded_a = self.model(**input_a)
                encoded_b = self.model(**input_b)

                sim = F.cosine_similarity(encoded_a[1].squeeze(dim=0), encoded_b[1].squeeze(dim=0), dim=0)
                sim = sim.tolist()
                self.torch_gc()
            except Exception as e:
                self.torch_gc()
                raise e
            answer = {"response": sim}
            return answer
