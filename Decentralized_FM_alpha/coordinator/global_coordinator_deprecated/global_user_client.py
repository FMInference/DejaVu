import time
import socket
import argparse
import random
import json


class GlobalUserClient:
    def __init__(self, args):
        self.host_ip = args.coordinator_server_ip
        self.host_port = args.coordinator_server_port
        self.client_port = 9999 - random.randint(1, 5000)  # cannot exceed 10000

    def put_request_user_client(self, inference_details: dict):
        print("=========put_request_user_client=========")
        msg_dict = {
            'op': 'put_request_user_client',
            'hf_api_para': inference_details
        }
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            s.sendall(json.dumps(msg_dict).encode())
            msg_raw = s.recv(2048)
            msg = json.loads(msg_raw)
            print(f"=========Received=========")
            print(msg)
            print("---------------------------")

    def get_request_user_client(self, task_index: int):
        print("=========get_request_user_client=========")
        msg_dict = {
            'op': 'get_request_user_client',
            'task_index': task_index
        }
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            s.sendall(json.dumps(msg_dict).encode())
            msg_raw = s.recv(2048)
            msg = json.loads(msg_raw)
            print(f"=========Received=========")
            print(msg)
            print("---------------------------")


def main():
    parser = argparse.ArgumentParser(description='Test Job-Submit-Client')
    parser.add_argument('--coordinator-server-port', type=int, default=9102, metavar='N',
                        help='The port of coordinator-server.')
    parser.add_argument('--coordinator-server-ip', type=str, default='35.92.51.7', metavar='S',
                        help='The IP of coordinator-server.')
    parser.add_argument('--op', type=str, default='get', metavar='S',
                        help='The op: {get or put}.')
    parser.add_argument('--task-index', type=int, default=0, metavar='N',
                        help='The index of the submitted tasks.')
    parser.add_argument('--inputs', type=str, default='Hello world!', metavar='S',
                        help='The prompt sequence.')
    args = parser.parse_args()
    print(vars(args))
    client = GlobalUserClient(args)

    if args.op == 'get':
        client.get_request_user_client(args.task_index)
    elif args.op == 'put':
        inference_details = {
            'inputs': args.inputs,
            "parameters": {
                "max_new_tokens": 64,
                "return_full_text": False,
                "do_sample": True,
                "temperature": 0.8,
                "top_p": 0.95,
                "max_time": 10.0,
                "num_return_sequences": 3,
                "use_gpu": True
            }
        }
        client.put_request_user_client(inference_details)
    else:
        assert False


if __name__ == '__main__':
    main()
