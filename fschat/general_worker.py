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
from vllm.utils import random_uuid

from base_model_worker import BaseModelWorker
from utils import build_logger

from model.embedding import SentenceEmbedding
from model.baai_sim import Baai_similarity
from model.embedding_sim import embedding_sim
from model.ziya_reward import ziya_reward

from datetime import datetime

supported_models=["embedding","baai_sim","embedding_sim","ziya_reward"]

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            # 释放缓存分配器当前持有的所有未占用的缓存内存，以便这些内存可以在其他 GPU 应用程序中使用并在 nvidia-smi 中可见
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

app = FastAPI()

class Worker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        conv_template: str,
        worker_info,
        engine
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
            f"Loading the model {self.model_names} on worker {worker_id}, worker type: general worker..."
        )

        if not no_register:
            self.init_heart_beat()

    async def generate_stream(self, params):
        pass

    async def generate(self, params):
        return engine.generate(params)
    
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
    parser.add_argument("--worker-address", type=str, default="http://122.224.74.10:21002")
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument(
        "--controller-address", type=str, default="http://122.224.74.10:21001"
    )
    parser.add_argument("--model-path", type=str, required=True)
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
    time=datetime.now().strftime('%Y%m%d%H%M%S')
    logger = build_logger("general_worker", f"{args.model_names}_{time}.log", args.log_dir)

    model_names=[i.lower() for i in args.model_names]

    engine = None
    try:
        if 'embedding_general' in model_names:
            engine = SentenceEmbedding(args.model_path, CUDA_DEVICE)
        if "baai_sim_general" in model_names:
            engine = Baai_similarity(args.model_path, CUDA_DEVICE)
        if "embedding_sim_general" in model_names:
            engine = embedding_sim(args.model_path, CUDA_DEVICE)
        if "ziya_reward_general" in model_names:
            engine = ziya_reward(args.model_path,CUDA_DEVICE)
    except Exception as e:
        logger.error(e)
        raise Exception(e)
    
    if not engine:
        raise Exception(f"Unsupported model. Current supported models: {','.join(supported_models)}")
        

    worker_info = {"host": args.host, "port": args.port, "worker-address": args.worker_address, "model-path": args.model_path, "gpus": args.gpus, "start_type": "general"}
    
    worker = Worker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        args.no_register,
        args.conv_template,
        worker_info,
        engine
    )
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
