"""
A model worker that executes the model based on vLLM.

See documentations at docs/vllm_integration.md
"""

import argparse
import asyncio
import json
import uuid
from typing import List
import traceback
import os

from fastapi import FastAPI, Request, BackgroundTasks, status, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import torch
import uvicorn
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from base_model_worker import BaseModelWorker
from utils import get_context_length, build_logger
from model.base import BaseModel

from model.chatglm2_6b import ChatGLM6BModel
from model.qwen_14B import Qwen14BModel
from model.internlm_20b import Internlm20BModel
from model.baichuan2_13b import Baichuan213BModel
from model.selfmodel_7B import SelfModel7BModel
from transformers import AutoModel, AutoTokenizer, BloomForCausalLM, pipeline, AutoModelForCausalLM

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            # 释放缓存分配器当前持有的所有未占用的缓存内存，以便这些内存可以在其他 GPU 应用程序中使用并在 nvidia-smi 中可见
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

app = FastAPI()

class HFWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        llm_engine: BaseModel,
        conv_template: str,
        worker_info,
    ):
        global logger
        super().__init__(
            logger,
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template,
        )
        self.info = worker_info
        logger.info(
            f"Loading the model {self.model_names} on worker {worker_id}, worker type: HF worker..."
        )
        self.tokenizer = llm_engine.tokenizer
        self.context_len = get_context_length(llm_engine.model.config)

        if not no_register:
            self.init_heart_beat()

    async def generate_stream(self, params):
        pass

    async def generate(self, params):
        self.call_ct += 1

        prompt = params.pop('prompt')
        request_id = params.pop('request_id')
        if type(prompt) is str:
            prompt = [prompt]

        try:
            res = engine.generate(prompt, params)
            answer = {
                "text": res,
                "error_code": 200
            }       
            message = {"request_id": request_id, "prompt": prompt, "params": params, "result": res}
            logger.info(message)
            torch_gc()
            return answer
        except Exception as e:
            traceback.print_exc()
            torch_gc()
            return {"text": str(e), "error_code": 500}
    
    async def chat(self, params):
        self.call_ct += 1
        prompt = params.pop('prompt')
        request_id = params.pop('request_id')
        response = None
        try:
            assert type(prompt) == str, "prompt must be str in chat interface"
            response = engine.chat(prompt,params)
            res = [response]
            answer = {
                "text": res,
                "error_code": 200
            } 
            
            message = {"request_id": request_id, "prompt": prompt, "params": params, "result": res}
            logger.info(message)
            torch_gc()
            return answer
        except Exception as e:
            traceback.print_exc()
            torch_gc()
            return {"text": str(e), "error_code": 500}
    
    def get_status(self):
        dicts = super().get_status()
        dicts.update(self.info)
        return dicts


def release_worker_semaphore():
    worker.semaphore.release()


def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()


def create_background_tasks(request_id):
    async def abort_request() -> None:
        await engine.engine.abort(request_id)

    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    background_tasks.add_task(abort_request)
    return background_tasks


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = random_uuid()
    params["request_id"] = request_id
    generator = worker.generate_stream(params)
    background_tasks = create_background_tasks(request_id)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_chat")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = random_uuid()
    params["request_id"] = request_id
    output = await worker.chat(params)
    release_worker_semaphore()
    return JSONResponse(output)

@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = random_uuid()
    params["request_id"] = request_id
    output = await worker.generate(params)
    release_worker_semaphore()
    return JSONResponse(output)



@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    return worker.count_token(params)


@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    return worker.get_conv_template()


@app.post("/model_details")
async def api_model_details(request: Request):
    return {"context_length": worker.context_len}

@app.post("/shutdown")
def shutdown():
    try:
        os.kill(os.getpid(), 9)
        return status.HTTP_200_OK
    except Exception as e:
        return HTTPException(status_code=500,detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--model-path", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--limit-worker-concurrency", type=int, default=1024)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--gpus", type=str, required=True)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )

    args = parser.parse_args()
    CUDA_DEVICE = "cuda:{}".format(args.gpus)
        
    worker_id = str(uuid.uuid4())[:8]
    logger = build_logger("hf_worker", "hf_worker_{}.log".format(worker_id), args.log_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True).half().cuda(CUDA_DEVICE)
    model.eval()

    model_names=[i.lower() for i in args.model_names]

    engine = None
    if 'chatglm2_hf' in model_names or 'chatglm3_hf' in model_names:
        engine = ChatGLM6BModel(model, tokenizer, CUDA_DEVICE)
    if 'qwen_hf' in model_names:
        engine = Qwen14BModel(model, tokenizer, CUDA_DEVICE)
    if 'internlm_hf' in model_names:
        engine = Internlm20BModel(model, tokenizer, CUDA_DEVICE)
    if 'baichuan2_hf' in model_names:
        engine = Baichuan213BModel(model, tokenizer, CUDA_DEVICE)
    if 'selfmodel_hf' in model_names:
        engine = SelfModel7BModel(model, tokenizer, CUDA_DEVICE)

    worker_info = {"host": args.host, "port": args.port, "worker-address": args.worker_address, "model-path": args.model_path,
                   "template": args.conv_template, "gpus": args.gpus, "start_type": "hf"}
    
    worker = HFWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        args.no_register,
        engine,
        args.conv_template,
        worker_info,
    )
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
