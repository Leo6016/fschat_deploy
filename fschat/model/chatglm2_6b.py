from model.base import BaseModel
import torch

class ChatGLM6BModel(BaseModel):
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.inference_mode()
    def chat(self, query: str, params: dict):
        generation_config = self.model.generation_config.__dict__
        merged_data = generation_config
        if params:
            merged_data={**generation_config, **params}
        response, _ = self.model.chat(self.tokenizer, query, **merged_data)
        return response
    
    @torch.inference_mode() 
    def generate(self, query_lists: list, params: dict):

        generation_config = self.model.generation_config.__dict__
        merged_data = generation_config
        if params:
            merged_data={**generation_config, **params}

        input_ids = self.tokenizer(query_lists, return_tensors='pt', padding=True).to(self.device)
        response = self.model.generate(inputs=input_ids['input_ids'], **merged_data)

        res = []
        n = int(len(response) / len(query_lists))
        for i, item in enumerate(response):
            output = self.tokenizer.decode(item, skip_special_tokens=True)
            res.append(output[len(query_lists[int(i / n)]):])

        return res