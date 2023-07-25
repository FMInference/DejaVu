import time
import socket
import argparse
import random
import json


class JobSubmitClient:
    def __init__(self, args):
        self.host_ip = args.coordinator_server_ip
        self.host_port = args.coordinator_server_port
        if args.submit_job == 'heartbeats':
            self.client_port = 20000
        else:
            self.client_port = 9999 - random.randint(1, 5000) # cannot exceed 10000

    def submit_train_job(self, job_name: str):
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', self.client_port))
                s.connect((self.host_ip, self.host_port))
                s.sendall(b"train#submit#"+job_name.encode())
                msg = s.recv(1024)
                print(f"Received: {msg}")
            if msg.decode('utf-8').startswith('Fail'):
                print("try again in 10 seconds..")
                time.sleep(10)
            else:
                break
        
    def submit_inference_job(self, job_name: str, infer_data: str='foo'):
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', self.client_port))
                s.connect((self.host_ip, self.host_port))
                # s.sendall(b"inference#submit#"+job_name.encode())
                msg_dict = {
                    'task': 'inference',
                    'state': 'submit',
                    'job_name': job_name,
                    'infer_data': infer_data
                }
                s.sendall(json.dumps(msg_dict).encode())
                msg = s.recv(1024)
                print(f"Received: {msg}")
            if msg.decode('utf-8').startswith('Fail'):
                print("try again in 20 seconds..")
                time.sleep(20)
                self.client_port = 9999 - random.randint(1, 5000)
            else:
                break

    def client_heartbeats(self):
        # TODO, This is a dirty fix, change the client to multi-thread later;
        while True:
            time.sleep(100)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', self.client_port))
                s.connect((self.host_ip, self.host_port))
                msg_dict = {
                    'task': 'inference',
                    'state': 'client_heartbeats',
                }
                s.sendall(json.dumps(msg_dict).encode())
                msg = s.recv(1024)
                print(f"Received: {msg}")


def main():
    parser = argparse.ArgumentParser(description='Test Job-Submit-Client')
    parser.add_argument('--submit-job', type=str, default='inference', help='train or inference')
    parser.add_argument('--coordinator-server-port', type=int, default=9002, metavar='N',
                        help='The port of coordinator-server.')
    parser.add_argument('--coordinator-server-ip', type=str, default='localhost', metavar='S',
                        help='The IP of coordinator-server.')
    parser.add_argument('--job-name', type=str, default='lsf_gptJ_inf_4RTX2080Ti', metavar='S',
                        help='Support a fixed list of job first, this can be more flexible later.')
    parser.add_argument('--infer-data', type=str, default='foo', metavar='S', help='path of infer data')
    args = parser.parse_args()
    print(vars(args))
    client = JobSubmitClient(args)
    if args.submit_job == 'train':
        client.submit_train_job(args.job_name)
    elif args.submit_job == 'inference':
        client.submit_inference_job(args.job_name, infer_data=args.infer_data)
    elif args.submit_job == 'heartbeats':
        client.client_heartbeats()
    else:
        assert False


if __name__ == '__main__':
    main()
