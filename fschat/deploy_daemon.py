import subprocess
import os
import argparse
import time
import requests
import nvidia_smi

_GPU = False
_NUMBER_OF_GPU = 0

from fastapi import FastAPI, HTTPException,Header,status
from fastapi.responses import JSONResponse
import uvicorn

from utils import build_logger,DeployCommand

CONTROLLER_HEART_BEAT_EXPIRATION = 90

controller_address="http://10.31.29.02:6001"
python="/home/ubuntu/anaconda3/envs/deploy/bin/python"
log_path="/home/ubuntu/fschat_deploy/logs"

valid_methods=["hf","vllm","general"]

app = FastAPI()

deploy_timeout=1800

def bytes_to_megabytes(bytes):
    return round((bytes/1024)/1024,2)

def check_gpu():
    global _GPU
    global _NUMBER_OF_GPU
    nvidia_smi.nvmlInit()
    _NUMBER_OF_GPU = nvidia_smi.nvmlDeviceGetCount()
    if _NUMBER_OF_GPU > 0:
        _GPU = True

class deploy_daemon:
    def __init__(self,controller_address,daemon_address,daemon_name,daemon_port):
        self.controller_address=controller_address
        self.info={
            "daemon_name":daemon_name,
            "daemon_IP":daemon_address,
            "daemon_port":daemon_port
        }
        self.exists=False
        self.gpu_info={}
        try:
            response=requests.post(self.controller_address+"/register_daemon",json=self.info)
            if response.status_code==200:
                logger.info("Daemon init sucess")
            else:
                logger.info("Daemon already exist!!!")
                self.exists=True
        except Exception as e:
           logger.error(f"Daemon init failed: {e}")
        check_gpu()

    def get_gpu_usage(self):
        for i in range(_NUMBER_OF_GPU):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            self.gpu_info[i]={"current_usage":bytes_to_megabytes(info.used),
                              "total":bytes_to_megabytes(info.total)}
        return self.gpu_info

def check_complete(params):
    curr_list=requests.post(controller_address+"/list_models",timeout=5).json().get("models")
    if not curr_list:
        return False
    model_name=params["model_name"]
    model_addr=f"http://{params['host']}:{params['port']}"
    if model_name in curr_list:
        for item in curr_list[model_name]:
            if item!="worker_num":
                if curr_list[model_name][item]["extend_info"]["worker-address"]==model_addr:
                    return True
    return False

@app.post("/deploy")
async def deploy(params: DeployCommand):
    if params.deploy_method not in valid_methods:
        raise HTTPException(status_code=422, detail="Invalid deploy method. Must be 'vllm','hf','general'")
    model_names_methods=[]
    for i in range(len(params.model_names)):
        model_names_methods.append(params.model_names[i]+f"_{params.deploy_method}")
    model_names=",".join(model_names_methods) 
    model_check_params={
        "model_name":model_names_methods[0],
        "host":params.host,
        "port":params.port
    }
    if check_complete(model_check_params):
        return JSONResponse(status_code=500,content={"error":f"model {model_names} already exist on http://{params.host}:{params.port}"})
    gpu_list=",".join([str(i) for i in range(_NUMBER_OF_GPU)]) 
    other_params=""
    if params.additonal_params:
        for item in params.additonal_params:
            other_params=other_params+f"--{item.replace('_','-')} {params.additonal_params[item]}"
    if params.deploy_method=="vllm":
        gpus=",".join(params.gpu)
        command=f"CUDA_VISIBLE_DEVICES={gpus} {python} vllm_worker.py --host {params.host} --port {params.port} --model-path {params.model_path} --trust-remote-code --controller-address {controller_address} --worker-address http://{params.host}:{params.port} --model-names {model_names} --log-dir {log_path} --gpus {gpus} {other_params}"
    elif params.deploy_method=="hf":
        if len(params.gpu)>1:
            return JSONResponse(status_code=500,content={"error":"hf only supports single gpu"})
        command=f"CUDA_VISIBLE_DEVICES={gpu_list} {python} hf_worker.py --host {params.host} --port {params.port} --model-path {params.model_path} --controller-address {controller_address} --worker-address http://{params.host}:{params.port} --model-names {model_names} --log-dir {log_path} --gpus {params.gpu[0]} {other_params}"
    else:
        if len(params.gpu)>1:
            raise JSONResponse(status_code=500,content={"error":"general only supports single gpu"})
        command=f"CUDA_VISIBLE_DEVICES={gpu_list} {python} general_worker.py --host {params.host} --port {params.port} --model-path {params.model_path} --controller-address {controller_address} --worker-address http://{params.host}:{params.port} --model-names {model_names} --log-dir {log_path} --gpus {params.gpu[0]} {other_params}"
    try:
        with open("stdout.log", 'w') as stdout_file, open("stderr.log", 'w') as stderr_file:
            new_service_proc = subprocess.Popen(command, shell=True, stdout=stdout_file, stderr=stderr_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    start_time=time.time()

    logger.info(model_check_params)
    while time.time()-start_time<deploy_timeout:
        if new_service_proc.poll():
            with open("stderr.log", 'r') as stderr_file:
                error_message = stderr_file.read()
            logger.error(error_message)
            return JSONResponse(status_code=500, content={"error":str(error_message)})
        if check_complete(model_check_params):
            logger.info(f"{model_check_params} deploy success")
            return JSONResponse(status_code=200, content={"msg":f"{model_names} deploy success on http://{params.host}:{params.port}"})     
        time.sleep(5)    
    logger.error("Deploy timeout")
    return JSONResponse(status_code=500, content={"error:":f"{model_names} deploy failed, TIMEOUT: {deploy_timeout} seconds"})

@app.get("/gpu_info")
async def get_gpu_info():
    gpu_info=daemon.get_gpu_usage()
    return JSONResponse(content={"GPU_INFO":gpu_info},status_code=200)

@app.get("/heartbeat")
async def get_heartbeat():
    return status.HTTP_200_OK

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=20000)
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument(
        "--ssl",
        action="store_true",
        required=False,
        default=False,
        help="Enable SSL. Requires OS Environment variables 'SSL_KEYFILE' and 'SSL_CERTFILE'.",
    )
    parser.add_argument("--name",type=str,required=True)
    args = parser.parse_args()

    logger = build_logger("deploy_daemon", "deploy_daemon.log", args.log_dir)
    logger.info(f"args: {args}") 

    daemon=deploy_daemon(controller_address,daemon_address=args.host,daemon_name=args.name,daemon_port=args.port)
  
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
  

