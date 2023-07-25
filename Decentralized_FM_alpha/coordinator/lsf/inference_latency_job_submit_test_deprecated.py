import time
import socket
import argparse
import random
import json


class JobSubmitClient:
    def __init__(self, args):
        self.host_ip = args.coordinator_server_ip
        self.host_port = args.coordinator_server_port
        self.client_port = 9999 - random.randint(1, 5000)  # cannot exceed 10000

    def submit_inference_task(self, job_details: dict):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            s.sendall(b"inference#latency_job#" + json.dumps(job_details).encode())
            msg = s.recv(1024)
            print(f"Received Inference results: {msg}")


def main():
    parser = argparse.ArgumentParser(description='Test Latency Inference Job')
    parser.add_argument('--coordinator-server-port', type=int, default=9002, metavar='N',
                        help='The port of coordinator-server.')
    parser.add_argument('--coordinator-server-ip', type=str, default='localhost', metavar='S',
                        help='The IP of coordinator-server.')
    parser.add_argument('--input-seq', type=str, default='you are not very smart', metavar='S',
                        help='The inference input seq.')
    parser.add_argument('--max-new-tokens', type=int, default=20, metavar='S',
                        help='-')
    parser.add_argument('--return-full-text', type=lambda x: (str(x).lower() == 'true'),
                        default=True, metavar='S', help='-')
    parser.add_argument('--do-sample', type=lambda x: (str(x).lower() == 'true'),
                        default=True, metavar='S', help='-')
    parser.add_argument('--top-p', type=float, default=0.95, metavar='S',
                        help='-')
    parser.add_argument('--temperature', type=float, default=0.95, metavar='S',
                        help='-')
    parser.add_argument('--max-time', type=float, default=10.0, metavar='S',
                        help='-')
    parser.add_argument('--num-return-sequences', type=int, default=2, metavar='S',
                        help='-')


    args = parser.parse_args()
    # print(vars(args))
    client = JobSubmitClient(args)
    my_obj = {
        'inputs': args.input_seq,
        "parameters": {
            "max_new_tokens": args.max_new_tokens, "return_full_text": args.return_full_text,
            "do_sample": args.do_sample, "temperature": args.temperature, "top_p": args.top_p,
            "max_time": args.max_time, "num_return_sequences": args.num_return_sequences
        }
    }
    # print(json.dumps(my_obj).encode())
    client.submit_inference_task(my_obj)


if __name__ == '__main__':
    main()
