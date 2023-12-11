
python3 -m controller --host 0.0.0.0 --port 21001 --log-dir /data/tzy/log/fschat

CUDA_VISIBLE_DEVICES=1 python vllm_worker.py --host 122.224.74.10 --port 21003 --model-path /mnt/sdb/models/Qwen-7B-Chat --trust-remote-code --controller-address http://122.224.74.10:21001 --worker-address http://122.224.74.10:21003 --model-names baichuan2-13b-chat /home/xgy/logs --gpus 0

CUDA_VISIBLE_DEVICES=0,1,2,3 python hf_worker.py --host 122.224.74.10 --port 21002 --model-path /mnt/sdb/models/Baichuan2-7B-Chat --controller-address http://122.224.74.10:21001 --worker-address http://122.224.74.10:21002 --model-names baichuan --log-dir /home/xgy/logs --gpus 0

python3 -m api_server --host 0.0.0.0 --port 8001 --controller-address http://10.31.29.2:21001

nohup python deploy_daemon.py --host 10.31.29.13 --port 20000 --log-dir /home/ubuntu/fschat_deploy/logs --name GPU13 &

pip install nvidia-ml-py3