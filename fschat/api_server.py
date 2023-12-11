"""A server that provides OpenAI-compatible RESTful APIs. It supports:

- Chat Completions. (Reference: https://platform.openai.com/docs/api-reference/chat)
- Completions. (Reference: https://platform.openai.com/docs/api-reference/completions)
- Embeddings. (Reference: https://platform.openai.com/docs/api-reference/embeddings)

Usage:
python3 -m fastchat.serve.openai_api_server
"""
import asyncio
import argparse
import json
import logging
import os
import copy
from typing import Generator, Optional, Union, Dict, List, Any

import aiohttp
import fastapi
import requests
from fastapi import Depends, HTTPException, Request,Header
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseSettings
import tiktoken
import uvicorn
from utils import ModelRequest, ErrorCode, ErrorResponse, UsageInfo, IPRequest, DeployCommand, killCommand, daemonInfo ,build_logger


log_dir="/home/ubuntu/fschat_deploy/logs"
logger = build_logger("api_server","api_server.log",log_dir)

conv_template_map = {}

fetch_timeout = aiohttp.ClientTimeout(total=3 * 3600)

AUTH_TOKEN= "20548cb5a329260ead027437cb22590e945504abd419e2e44ba312feda2ff29e"

async def fetch_remote(url, pload=None, name=None):
    async with aiohttp.ClientSession(timeout=fetch_timeout) as session:
        async with session.post(url, json=pload) as response:
            chunks = []
            if response.status != 200:
                ret = {
                    "text": await response.text(),
                    "error_code": ErrorCode.INTERNAL_ERROR,
                }
                return ret
            if "deploy" in url:
                ret={
                    "text": await response.text(),
                    "code":200
                }
                return ret

            async for chunk, _ in response.content.iter_chunks():
                chunks.append(chunk)
        output = b"".join(chunks)
    
    if name is not None:
        res = json.loads(output)
        if name != "":
            res = res[name]
        return res

    return output


class AppSettings(BaseSettings):
    # The address of the model controller.
    controller_address: str = "http://localhost:21001"
    api_keys: Optional[List[str]] = None


app_settings = AppSettings()
app = fastapi.FastAPI()
headers = {"User-Agent": "FastChat API Server"}
get_bearer_token = HTTPBearer(auto_error=False)


async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
) -> str:
    if app_settings.api_keys:
        if auth is None or (token := auth.credentials) not in app_settings.api_keys:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )
        return token
    else:
        # api_keys not set; allow all
        return None

def create_error_response(code: int, message: str) -> JSONResponse:
    logger.error(message)
    return JSONResponse(
        ErrorResponse(message=message, code=code).dict(), status_code=500
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return create_error_response(ErrorCode.VALIDATION_TYPE_ERROR, str(exc))

async def check_model(request) -> Optional[JSONResponse]:
    controller_address = app_settings.controller_address
    ret = None

    models = await fetch_remote(controller_address + "/list_models", None, "models")
    if request.model not in models:
        ret = create_error_response(
            ErrorCode.INVALID_MODEL,
            f"Only {'&&'.join(models)} allowed now, your model {request.model}",
        )
    return ret

async def check_length(request, prompt, max_tokens, worker_addr):
    if (
        not isinstance(max_tokens, int) or max_tokens <= 0
    ):  # model worker not support max_tokens=None
        max_tokens = 1024 * 1024

    context_len = await fetch_remote(
        worker_addr + "/model_details", {"model": request.model}, "context_length"
    )
    token_num = await fetch_remote(
        worker_addr + "/count_token",
        {"model": request.model, "prompt": prompt},
        "count",
    )
    return min(max_tokens, context_len - token_num)

def process_input(model_name, inp):
    if isinstance(inp, str):
        inp = [inp]
    elif isinstance(inp, list):
        if isinstance(inp[0], int):
            decoding = tiktoken.model.encoding_for_model(model_name)
            inp = [decoding.decode(inp)]
        elif isinstance(inp[0], list):
            decoding = tiktoken.model.encoding_for_model(model_name)
            inp = [decoding.decode(text) for text in inp]

    return inp

def _add_to_set(s, new_stop):
    if not s:
        return
    if isinstance(s, str):
        new_stop.add(s)
    else:
        new_stop.update(s)

async def get_worker_address(model_name: str) -> str:
    """
    Get worker address based on the requested model

    :param model_name: The worker's model name
    :return: Worker address from the controller
    :raises: :class:`ValueError`: No available worker for requested model
    """
    controller_address = app_settings.controller_address
    worker_addr = await fetch_remote(
        controller_address + "/get_worker_address", {"model": model_name}, "address"
    )

    # No available worker
    if worker_addr == "":
        raise ValueError(f"No available worker for {model_name}")
    return worker_addr

async def get_conv(model_name: str, worker_addr: str):
    conv_template = conv_template_map.get((worker_addr, model_name))
    if conv_template is None:
        conv_template = await fetch_remote(
            worker_addr + "/worker_get_conv_template", {"model": model_name}, "conv"
        )
        conv_template_map[(worker_addr, model_name)] = conv_template
    return conv_template

@app.post("/register_daemon")
async def register_daemon(params: daemonInfo, token:str=Header(...)):
    if token!=AUTH_TOKEN:
        raise HTTPException(status_code=401,detail="Authorization failed")
    controller_address = app_settings.controller_address
    res = await fetch_remote(controller_address+"/register_daemon",dict(params),"")
    return res

@app.get("/list_daemons", dependencies=[Depends(check_api_key)])
async def show_machines(token:str=Header(...)):
    if token!=AUTH_TOKEN:
        raise HTTPException(status_code=401,detail="Authorization failed")
    controller_address = app_settings.controller_address
    response=await fetch_remote(controller_address+"/list_daemons")
    return JSONResponse(content=json.loads(response),status_code=200)

@app.post("/show_gpu_info", dependencies=[Depends(check_api_key)])
async def show_gpu_info(request: IPRequest,token:str=Header(...)):
    if token!=AUTH_TOKEN:
        raise HTTPException(status_code=401,detail="Authorization failed")
    gpu_info=requests.get(f"http://{request.host}:20000/gpu_info")
    return gpu_info.json()

@app.get("/list_models", dependencies=[Depends(check_api_key)])
async def show_available_models(token:str=Header(...)):
    if token!=AUTH_TOKEN:
        raise HTTPException(status_code=401,detail="Authorization failed")
    controller_address = app_settings.controller_address
    ret = await fetch_remote(controller_address + "/refresh_all_workers")
    models = await fetch_remote(controller_address + "/list_models", None, "models")

    return models

@app.post("/generate", dependencies=[Depends(check_api_key)])
async def create_completion(request: ModelRequest,token:str=Header(...)):
    if token!=AUTH_TOKEN:
        raise HTTPException(status_code=401,detail="Authorization failed")
    if not request.params or "request_id" not in request.params or not request.model or 'prompt' not in request.params:
        return JSONResponse(status_code=400, content="bad reqeust, request must have model, params.request_id, params.prompt")

    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret
    
    if request.params.get('prompt', ''):
        request.params['prompt'] = process_input(request.model, request.params.get('prompt', ''))

    worker_addr = await get_worker_address(request.model)
    prompts = request.params.pop('prompt')
    if '' in dict(request.params):
        request.params.pop('')
    params = request.params
    text_completions = []
    for text in prompts:
        gen_params = copy.deepcopy(params)
        gen_params['prompt'] = text
        content = asyncio.create_task(
                generate_completion(gen_params, worker_addr)
            )
        text_completions.append(content)

    try:
        all_tasks = await asyncio.gather(*text_completions)
    except Exception as e:
        return create_error_response(ErrorCode.INTERNAL_ERROR, str(e))

    choices = []
    usage = UsageInfo()
    for i, content in enumerate(all_tasks):
        if content["error_code"] != 0:
            logger.error({"code":content["error_code"], "details":content["text"]})
            return create_error_response(content["error_code"], content["text"])
        choices.append(
            content["text"]
        )
        task_usage = UsageInfo.parse_obj(content["usage"])
        for usage_key, usage_value in task_usage.dict().items():
            setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)
    res = {'text': choices, "usage": UsageInfo.parse_obj(usage)}
    logger.info({"input":dict(params),"output":res})
    return res

@app.post("/hf_generate", dependencies=[Depends(check_api_key)])
async def create_completion(request: ModelRequest,token:str=Header(...)):
    if token!=AUTH_TOKEN:
        raise HTTPException(status_code=401,detail="Authorization failed")

    if not request.params or "request_id" not in request.params or not request.model or 'prompt' not in request.params:
        return JSONResponse(status_code=400, content="bad reqeust, request must have model, params.request_id, params.prompt")

    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret
    
    if request.params.get('prompt', ''):
        request.params['prompt'] = process_input(request.model, request.params.get('prompt', ''))

    worker_addr = await get_worker_address(request.model)

    params = request.params
    text_completions = []
    content = asyncio.create_task(
            generate_completion(params, worker_addr)
        )
    text_completions.append(content)

    try:
        all_tasks = await asyncio.gather(*text_completions)
    except Exception as e:
        return create_error_response(ErrorCode.INTERNAL_ERROR, str(e))

    choices = []
    usage = UsageInfo()
    for i, content in enumerate(all_tasks):
        if content["error_code"] != 200 and content["error_code"]!=0:
            return create_error_response(content["error_code"], content["text"])
        choices = content["text"]
    res = {'choices': choices}
    return res

@app.post("/hf_chat", dependencies=[Depends(check_api_key)])
async def create_completion(request: ModelRequest, token:str=Header(...)):

    if token!=AUTH_TOKEN:
        raise HTTPException(status_code=401,detail="Authorization failed")

    if not request.params or "request_id" not in request.params or not request.model or 'prompt' not in request.params:
        return JSONResponse(status_code=400, content="bad reqeust, request must have model, params.request_id, params.prompt")

    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret
    

    worker_addr = await get_worker_address(request.model)

    params = request.params
    text_completions = []
    content = asyncio.create_task(
            generate_completion(params, worker_addr, "/worker_chat")
        )
    text_completions.append(content)

    try:
        all_tasks = await asyncio.gather(*text_completions)
    except Exception as e:
        return create_error_response(ErrorCode.INTERNAL_ERROR, str(e))

    choices = []
    usage = UsageInfo()
    for i, content in enumerate(all_tasks):
        if content["error_code"] != 0 and content["error_code"] != 200:
            return create_error_response(content["error_code"], content["text"])
        choices = content["text"]
    res = {'choices': choices}
    return res

@app.post("/general_generate", dependencies=[Depends(check_api_key)])
async def create_completion(request: ModelRequest, token:str=Header(...)):
    if token!=AUTH_TOKEN:
        raise HTTPException(status_code=401,detail="Authorization failed")

    if not request.params or "request_id" not in request.params or not request.model:
        return JSONResponse(status_code=400, content="bad reqeust, request must have model and params.request_id")

    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret
    
    worker_addr = await get_worker_address(request.model)
    params = request.params
    try:
        res = await generate_completion(params, worker_addr)
        if "error_code" in res:
           logger.warning(f"input: {params}")
           return create_error_response(ErrorCode.INTERNAL_ERROR, str(res)) 
    except Exception as e:
        return create_error_response(ErrorCode.INTERNAL_ERROR, str(e))
    logger.info(f"input: {params}, output: {res}")
    return res

@app.post("/deploy")
async def deploy_model(params: DeployCommand, token:str=Header(...)):
    if token!=AUTH_TOKEN:
            raise HTTPException(status_code=401,detail="Authorization failed")
    try:
        # try:
        #     r=requests.get(f"http://{params.host}:20000/heartbeat",timeout=30)
        # except Exception:
        #     logger.error(f"Failed to connect to deploy agent on {params.host}")
        #     return JSONResponse(status_code=500,content={"error":f"Failed to connect to deploy agent on {params.host}"})      
        url=f"http://{params.host}:20000/deploy"
        try:
            r=await fetch_remote(url,params.dict())
            if "error_code" in r:
                logger.error(r["text"])
                return JSONResponse(status_code=500,content={"error":r["text"]})
            logger.info(r.get("text"))
            return JSONResponse(status_code=200,content=r.get("text"))
        except Exception as e:
            return create_error_response(ErrorCode.INTERNAL_ERROR,str(e))
    #     r=requests.post(url,json=params.dict()).json()
    #     if r.get("msg"):
    #         return JSONResponse(status_code=200,content=r)
    #     else:
    #         return JSONResponse(status_code=500,content={"error":str(r.get("error"))})
    except Exception as e:
        raise HTTPException(500,str(e))

@app.post("/kill")
async def kill(params: killCommand, token:str=Header(...)):
    if token!=AUTH_TOKEN:
        raise HTTPException(status_code=401,detail="Authorization failed")
    worker_info= await show_available_models(token)
    worker_addr_list=[]
    for model in worker_info:
        for i in range(1,worker_info[model]["worker_num"]+1):
            worker_addr_list.append(worker_info[model][str(i)]["extend_info"]["worker-address"])
    worker_addr=f"http://{params.host}:{params.port}"
    if worker_addr not in worker_addr_list:
        raise HTTPException(status_code=500,detail={"msg": f"model does not exist on http://{params.host}:{params.port}"})
    try:
        await shutdown(worker_addr)
    except Exception:
        worker_info= await show_available_models(token)
        worker_addr_list=[]
        for model in worker_info:
            for i in range(1,worker_info[model]["worker_num"]+1):
                worker_addr_list.append(worker_info[model][str(i)]["extend_info"]["worker-address"])
        if worker_addr not in worker_addr_list:
            logger.info(f"model shutdown success on http://{params.host}:{params.port}")
            return JSONResponse(content={"msg":f"model shutdown success on http://{params.host}:{params.port}"})
        logger.error(f"model shutdown failed on http://{params.host}:{params.port}")
        raise HTTPException(status_code=500,detail={"msg": f"model shutdown failed on http://{params.host}:{params.port}"})
  
async def generate_completion(payload: Dict[str, Any], worker_addr: str, url="/worker_generate"):
    return await fetch_remote(worker_addr + url, payload, "")

async def shutdown(worker_addr: str, url="/shutdown"):
    return await fetch_remote(worker_addr + url)

### GENERAL API - NOT OPENAI COMPATIBLE ###


# @app.post("/api/v1/token_check")
# async def count_tokens(request: APITokenCheckRequest):
#     """
#     Checks the token count for each message in your list
#     This is not part of the OpenAI API spec.
#     """
#     checkedList = []
#     for item in request.prompts:
#         worker_addr = await get_worker_address(item.model)

#         context_len = await fetch_remote(
#             worker_addr + "/model_details",
#             {"prompt": item.prompt, "model": item.model},
#             "context_length",
#         )

#         token_num = await fetch_remote(
#             worker_addr + "/count_token",
#             {"prompt": item.prompt, "model": item.model},
#             "count",
#         )

#         can_fit = True
#         if token_num + item.max_tokens > context_len:
#             can_fit = False

#         checkedList.append(
#             APITokenCheckResponseItem(
#                 fits=can_fit, contextLength=context_len, tokenCount=token_num
#             )
#         )

#     return APITokenCheckResponse(prompts=checkedList)


### END GENERAL API - NOT OPENAI COMPATIBLE ###


def create_openai_api_server():
    parser = argparse.ArgumentParser(
        description="FastChat ChatGPT-Compatible RESTful API server."
    )
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument(
        "--allow-credentials", action="store_true", help="allow credentials"
    )
    parser.add_argument(
        "--allowed-origins", type=json.loads, default=["*"], help="allowed origins"
    )
    parser.add_argument(
        "--allowed-methods", type=json.loads, default=["*"], help="allowed methods"
    )
    parser.add_argument(
        "--allowed-headers", type=json.loads, default=["*"], help="allowed headers"
    )
    parser.add_argument(
        "--api-keys",
        type=lambda s: s.split(","),
        help="Optional list of comma separated API keys",
    )
    parser.add_argument(
        "--ssl",
        action="store_true",
        required=False,
        default=False,
        help="Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'.",
    )
    args = parser.parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )
    app_settings.controller_address = args.controller_address
    app_settings.api_keys = args.api_keys

    logger.info(f"args: {args}")
    return args


if __name__ == "__main__":
    args = create_openai_api_server()
    if args.ssl:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            ssl_keyfile=os.environ["SSL_KEYFILE"],
            ssl_certfile=os.environ["SSL_CERTFILE"],
        )
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
