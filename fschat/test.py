import requests
import time
from concurrent.futures import ThreadPoolExecutor

def deploy_model(port):
    url = "http://localhost:20000/deploy"
    payload = {
        "deploy_method": "general",
        "model_path": "/mnt/sdb/models/paraphrase-multilingual-mpnet-base-v2",
        "host": "localhost",
        "port": port,
        "model_names": ["embedding"],
        "gpu": [0]
    }
    try:
        response = requests.post(url, json=payload)
        return response.status_code, response.json()
    except Exception:
        return "err"

def heartbeat():
    url = "http://localhost:20000/heartbeat"
    try:
        response = requests.get(url,timeout=1)
    except Exception:
        return "err"
    return response.status_code

def main():
    port1=22001
    port2=22002
    st=time.time()
    # Use ThreadPoolExecutor to send two requests concurrently
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(deploy_model,port1),
            executor.submit(deploy_model,port2),
        ]

        # Wait for both requests to complete
        results = [future.result() for future in futures]

    for status_code, json_response in results:
        print(status_code, json_response)
    print(time.time()-st)
    
if __name__ == "__main__":
    main()

