import time
import socket
import argparse
import random
import json


class GlobalCoordinatorClient:
    def __init__(self, args):
        self.host_ip = args.global_coordinator_server_ip
        self.host_port = args.global_coordinator_server_port
        print("Global coordinator Host IP:", self.host_ip)
        print("Global coordinator Host Port:", self.host_port)
        self.client_port = 9999 - random.randint(1, 5000)  # cannot exceed 10000

    def put_request_cluster_coordinator(self, task_index: int, inference_result: str)-> dict:
        print("=========put_request_cluster_coordinator=========")
        msg_dict = {
            'op': 'put_request_cluster_coordinator',
            'task_index': task_index,
            'result': inference_result
        }
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            s.sendall(json.dumps(msg_dict).encode())
            msg_raw = s.recv(2048)
            return_msg = json.loads(msg_raw)
            print(f"=========Received=========")
            print(return_msg)
            print("---------------------------")
            return return_msg

    def get_request_cluster_coordinator(self) -> dict:
        print("=========get_request_cluster_coordinator=========")
        msg_dict = {
            'op': 'get_request_cluster_coordinator'
        }
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            s.sendall(json.dumps(msg_dict).encode())
            msg_raw = s.recv(2048)
            return_msg = json.loads(msg_raw)
            print(f"=========Received=========")
            print(return_msg)
            print("---------------------------")
            return return_msg


def main():
    parser = argparse.ArgumentParser(description='Test Job-Submit-Client')
    parser.add_argument('--global-coordinator-server-port', type=int, default=9102, metavar='N',
                        help='The port of coordinator-server.')
    parser.add_argument('--global-coordinator-server-ip', type=str, default='35.92.51.7', metavar='S',
                        help='The IP of coordinator-server.')
    parser.add_argument('--op', type=str, default='get', metavar='S',
                        help='The op: {get or put}.')
    parser.add_argument('--task-index', type=int, default=0, metavar='N',
                        help='The index of the submitted tasks.')
    args = parser.parse_args()
    print(vars(args))
    client = GlobalCoordinatorClient(args)

    if args.op == 'get':
        client.get_request_cluster_coordinator()
    elif args.op == 'put':
        inference_results = "Hello world reply."
        client.put_request_cluster_coordinator(args.task_index, inference_results)
    else:
        assert False


if __name__ == '__main__':
    main()