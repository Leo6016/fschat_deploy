
from transformers import AutoModelForSequenceClassification,LlamaTokenizer
import torch

class ziya_reward:
    def __init__(self,model_path,device) -> None:
        self.device=device
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True).half().cuda(self.device)
        self.model=self.model.eval()
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path,add_eos_token=True)

    def torch_gc(self):
        if torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def generate(self,data):
        query = data.get('query')
        response = data.get('response')
        prefix_user = "Human:"
        prefix_bot = "\n\nAssistant:"

        text = prefix_user+query+prefix_bot+response
        with torch.no_grad():
            try:
                batch = self.tokenizer(text, return_tensors="pt",padding=True,truncation=True,max_length=1024).to(self.device)
                reward = self.model(batch['input_ids'], attention_mask = batch['attention_mask'])
                answer = {
                    "res": reward.item(),
                    "status": 200,
                }
                if batch is not None:
                    del batch
                if reward is not None:
                    del reward
                self.torch_gc()
                return answer
            except Exception as e:
                if batch is not None:
                    del batch
                if reward is not None:
                    del reward
                self.torch_gc()
                raise Exception(e)
    






