import socket
import argparse
import random
import json


class UserClient:
    def __init__(self, args):
        self.host_ip = args.coordinator_server_ip
        self.host_port = args.coordinator_server_port
        self.client_port = 9999 - random.randint(1, 5000)  # cannot exceed 10000

    def ask_coordinator_to_launch_vm(self, node_type: str):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            message_dict = {
                "op": "launch_vm_user",
                "node_type": node_type}
            s.sendall(json.dumps(message_dict).encode())
            msg = s.recv(1024)
            print(f"Received: {msg}")

    def ask_coordinator_node_info(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            message_dict = {
                "op": "check_node_status_user"}
            s.sendall(json.dumps(message_dict).encode())
            msg = s.recv(1024)
            print(f"Received: {msg}")

    def ask_coordinator_to_run_job(self, work_json):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            assert 'op' in work_json and work_json['op'] == 'run_job_user'
            s.sendall(json.dumps(work_json).encode())
            msg = s.recv(1024)
            print(f"Received: {msg}")


def main():
    parser = argparse.ArgumentParser(description='Test Job-Submit-Client')
    parser.add_argument('--coordinator-server-port', type=int, default=9002, metavar='N',
                        help='The port of coordinator-server.')
    parser.add_argument('--coordinator-server-ip', type=str, default='localhost', metavar='S',
                        help='The IP of coordinator-server.')
    parser.add_argument('--op', type=str, default='check_node_status_user', metavar='S',
                        help='')
    parser.add_argument('--node-type', type=str, default='a40.1x', metavar='S',
                        help='')
    args = parser.parse_args()
    # print(vars(args))
    client = UserClient(args)
    if args.op == 'check_node_status_user':
        client.ask_coordinator_node_info()
    elif args.op == 'launch_vm_user':
        client.ask_coordinator_to_launch_vm(args.node_type)
    elif args.op == 'run_job_user':
        work_json = {
            "op": "run_job_user",
            "job_type": "inference",
            "model_name": "gpt_175b",
            "data_set": "foo",
            "node_index": 0
        }
        client.ask_coordinator_to_run_job(work_json)


if __name__ == '__main__':
    main()
