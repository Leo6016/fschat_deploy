from model.base import BaseModel
import torch
from transformers.generation import GenerationConfig

class Qwen14BModel(BaseModel):
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.tokenizer.pad_token_id = 151643
        self.tokenizer.eos_token_id = 151643
        self.tokenizer.padding_side = "left"

    @torch.inference_mode()
    def chat(self, query: str, params: dict):
        if not params:
            params = {}
        response, _ = self.model.chat(self.tokenizer, query, history=None, **params)
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