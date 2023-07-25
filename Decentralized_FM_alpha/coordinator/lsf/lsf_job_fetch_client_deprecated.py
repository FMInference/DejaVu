import time
import socket
import argparse
import json
import sys
sys.path.append("/nfs/iiscratch-zhang.inf.ethz.ch/export/zhang/export/fm/GPT-home-private/coordinator/global_coordinator")
from global_coordinator_client import GlobalCoordinatorClient


model_name_and_task_type_list = [
    ("image_generation", "stable_diffusion"),
    ("seq_generation", "gpt_j_6B")
]


class ReqeustFetchClient:
    def __init__(self, args):
        self.host_ip = args.coordinator_server_ip
        self.host_port = args.coordinator_server_port
        self.global_coord_client = GlobalCoordinatorClient(args)
        self.client_port = 20000

    def _fetcher_notify_server_heartbeats(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            msg_dict = {
                'task': 'inference',
                'state': 'fetcher_heartbeats',
            }
            s.sendall(json.dumps(msg_dict).encode())
            msg = s.recv(1024)
            print(f"Received: {msg}")

    def _fetcher_notify_server_enqueue_job(self, job_request):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            msg_dict = {
                'task': 'inference',
                'state': 'fetcher_heartbeats',
                'job_request': job_request
            }
            s.sendall(json.dumps(msg_dict).encode())
            msg = s.recv(1024)
            print(f"Received: {msg}")

    def execute_fetcher(self):
        # TODO, This is a dirty fix, change the client to multi-thread later;
        timer_count = 0
        while True:
            return_msg = None
            for task_model_tuple in model_name_and_task_type_list:
                return_msg = self.global_coord_client.get_request_cluster_coordinator(model_name=task_model_tuple[1],
                                                                                      task_type=task_model_tuple[0])
                if return_msg is not None:
                    break

            if return_msg:
                print(f"Enqueue request: <{return_msg['_id']}>")

            if return_msg is None:
                time.sleep(10)
                timer_count += 1
                if timer_count == 12:
                    self._fetcher_notify_server_heartbeats()
                    timer_count = 0
            else:
                self._fetcher_notify_server_enqueue_job(return_msg)


def main():
    parser = argparse.ArgumentParser(description='Test Job-Submit-Client')
    parser.add_argument('--coordinator-server-port', type=int, default=9002, metavar='N',
                        help='The port of coordinator-server.')
    parser.add_argument('--coordinator-server-ip', type=str, default='localhost', metavar='S',
                        help='The IP of coordinator-server.')
    parser.add_argument('--db-server-address', type=str,
                        default="http://xzyao:agway-fondly-ell-hammer-flattered-coconut@db.yao.sh:5984/", metavar='N',
                        help='Key value store address.')
    args = parser.parse_args()
    print(vars(args))
    client = ReqeustFetchClient(args)
    client.execute_fetcher()


if __name__ == '__main__':
    main()
