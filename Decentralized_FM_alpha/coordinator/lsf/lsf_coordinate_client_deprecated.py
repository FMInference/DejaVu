import json
import socket
import argparse
import os
from filelock import SoftFileLock
import time
from threading import Thread
import netifaces as ni
import requests


def client_message_parser(msg: bytes, context: str):
    msg_arg = msg.decode().split('#')
    if context == 'join_training':
        arg_dict = {'prime_ip': msg_arg[0],
                    'my_rank': int(msg_arg[1])}
    elif context == 'join_inference':
        arg_dict = {'prime_ip': msg_arg[0],
                    'my_rank': int(msg_arg[1]),
                    'port': int(msg_arg[2])}
    else:
        assert False
    return arg_dict


# The client port should be determined by the job-id which is unique. The ip + port will identify a worker.
class CoordinatorTrainClient:
    def __init__(self, args):
        self.host_ip = args.coordinator_server_ip
        self.host_port = args.coordinator_server_port
        self.client_port = int(args.lsf_job_no) % 10000 + 10000

    def notify_train_join(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            s.sendall(b"train#join")
            msg = s.recv(1024)
            print(f"Received: {msg}")
            msg_arg = client_message_parser(msg, 'join_training')
            return msg_arg['prime_ip'], msg_arg['my_rank']

    def notify_train_finish(self, message: str):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            s.sendall(b"train#finish#"+message.encode())
            msg = s.recv(1024)
            print(f"Received: {msg}")

class CoordinatorInferenceClient:
    def __init__(self, args):
        self.host_ip = args.coordinator_server_ip
        self.host_port = args.coordinator_server_port
        self.client_port = int(args.lsf_job_no) % 10000 + 10000
        self.__flag_keep_heart_beating = False
        self.__thread_keep_heart_beating = None
        self.__last_timestamp = time.time()
        self.__heartbeats_timelimit = args.heartbeats_timelimit

    def notify_inference_join(self):
        print("++++++++++++++++++notify_inference_join++++++++++++++++++")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            # s.sendall(b"inference#join#")
            msg_dict = {
                'task': 'inference',
                'state': 'join'
            }
            s.sendall(json.dumps(msg_dict).encode())
            msg = s.recv(1024)
            print(f"Received: {msg}")
            msg_arg = client_message_parser(msg, 'join_inference')
        return msg_arg['prime_ip'], msg_arg['my_rank'], msg_arg['port']

    def notify_inference_finish(self, rank: int, iter_time: float):
        print("++++++++++++++++++notify_inference_finish++++++++++++++++++")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            # s.sendall(b"inference#finish#"+message.encode())
            msg_dict = {
                'task': 'inference',
                'state': 'finish',
                'rank': rank,
                'iter_time': iter_time
            }
            s.sendall(json.dumps(msg_dict).encode())
            msg = s.recv(1024)
        print(f"Received: {msg}")
        return msg

    def notify_inference_heartbeat(self):
        print("++++++++++++++++++notify_inference_heartbeat++++++++++++++++++")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            # s.sendall(b"inference#finish#"+message.encode())
            msg_dict = {
                'task': 'inference',
                'state': 'worker_heartbeats',
            }
            s.sendall(json.dumps(msg_dict).encode())
            msg = s.recv(1024)
        print(f"Received: {msg}")
        return msg

    def notify_inference_dequeue_job(self, model_name):
        print("++++++++++++++++++notify_inference_dequeue_job++++++++++++++++++")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            # s.sendall(b"inference#finish#"+message.encode())
            msg_dict = {
                'task': 'inference',
                'state': 'worker_dequeue',
                'model_name': model_name
            }
            s.sendall(json.dumps(msg_dict).encode())
            msg = s.recv(8192)
            return_msg_dict = json.loads(msg)
        print(f"Received: {return_msg_dict}")
        return return_msg_dict

    def notify_inference_post_result(self, job_request):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            # s.sendall(b"inference#finish#"+message.encode())
            msg_dict = {
                'task': 'inference',
                'state': 'worker_post_result',
                'job_request': job_request
            }
            s.sendall(json.dumps(msg_dict).encode())
            msg = s.recv(8192)
        print(f"Received: {msg}")
        return msg
    
    def _keep_heart_beating(self):
        self.__flag_keep_heart_beating = True
        
        def _keep_heart_beating(self):
            while True:
                if not self.__flag_keep_heart_beating:
                    break
                self.notify_inference_heartbeat()
                time.sleep(self.__heartbeats_timelimit)
        
        self.__thread_keep_heart_beating = Thread(target=_keep_heart_beating, args=(self,))
        self.__thread_keep_heart_beating.start()
        
    def _stop_keep_heart_beating(self):
        self.__flag_keep_heart_beating = False
        if self.__thread_keep_heart_beating is not None:
            self.__thread_keep_heart_beating.join()

    def decorate_run_heart_beating_during(self, func):
        def decorated_func(*args, **kwargs):
            self._keep_heart_beating()
            ret = func(*args, **kwargs)
            self._stop_keep_heart_beating()
            return ret
        return decorated_func

    def decorate_run_heart_beating_before(self, func):
        def decorated_func(*args, **kwargs):
            last_timestamp = self.__last_timestamp
            current_timestamp = time.time()
            if current_timestamp - last_timestamp >= self.__heartbeats_timelimit:
                self.notify_inference_heartbeat()
                self.__last_timestamp = current_timestamp
            ret = func(*args, **kwargs)
            return ret
        return decorated_func
        

class CoordinatorInferenceFolderClient:
    def __init__(self, args, model_name:str):
        self.host_ip = args.coordinator_server_ip
        self.host_port = args.coordinator_server_port
        self.client_port = int(args.lsf_job_no) % 10000 + 10000
        self.working_directory = args.working_directory
        self.model_name = model_name
        self.dir_path = os.path.join(self.working_directory, self.model_name)
        lock_path = os.path.join(self.dir_path, self.model_name+'.lock')
        self.model_lock = SoftFileLock(lock_path, timeout=10)

    def notify_inference_join(self):
        print("++++++++++++++++++notify_inference_join++++++++++++++++++")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            # s.sendall(b"inference#join#")
            msg_dict = {
                'task': 'inference',
                'state': 'join'
            }
            s.sendall(json.dumps(msg_dict).encode())
            msg = s.recv(1024)
            print(f"Received: {msg}")
            msg_arg = client_message_parser(msg, 'join_inference')
            return msg_arg['prime_ip'], msg_arg['my_rank'], msg_arg['port']

    def notify_inference_finish(self, rank: int, iter_time: float):
        print("++++++++++++++++++notify_inference_finish++++++++++++++++++")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            # s.sendall(b"inference#finish#"+message.encode())
            msg_dict = {
                'task': 'inference',
                'state': 'finish',
                'rank': rank,
                'iter_time': iter_time
            }
            s.sendall(json.dumps(msg_dict).encode())
            msg = s.recv(1024)
        print(f"Received: {msg}")
        return msg

    def notify_inference_heartbeat(self):
        print("++++++++++++++++++notify_inference_heartbeat++++++++++++++++++")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            # s.sendall(b"inference#finish#"+message.encode())
            msg_dict = {
                'task': 'inference',
                'state': 'worker_heartbeats',
            }
            s.sendall(json.dumps(msg_dict).encode())
            msg = s.recv(1024)
        print(f"Received: {msg}")
        return msg

    def load_input_job_from_dfs(self):
        for filename in os.listdir(self.dir_path):
            # print(filename)
            if filename.startswith('input_'):
                doc_path = os.path.join(self.dir_path, filename)
                with self.model_lock:
                    with open(doc_path, 'r') as infile:
                        doc = json.load(infile)
                    # assert model_name == doc['task_api']['model_name']
                print(f"++++++++++++++load_input_job_from_dfs <{doc['_id']}>++++++++++++")
                return doc
        return None

    def save_output_job_to_dfs(self, result_doc):
        # print("++++++++++++++save_output_job_to_dfs++++++++++++")
        output_filename = 'output_' + result_doc['_id'] + '.json'
        output_path = os.path.join(self.dir_path, output_filename)
        with self.model_lock:
            with open(output_path, 'w') as outfile:
                json.dump(result_doc, outfile)
        input_filename = 'input_' + result_doc['_id'] + '.json'
        input_path = os.path.join(self.dir_path, input_filename)
        assert os.path.exists(input_path)
        os.remove(input_path)

    """
    def notify_inference_dequeue_job(self, model_name):
        print("++++++++++++++++++notify_inference_dequeue_job++++++++++++++++++")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            # s.sendall(b"inference#finish#"+message.encode())
            msg_dict = {
                'task': 'inference',
                'state': 'worker_dequeue',
                'model_name': model_name
            }
            s.sendall(json.dumps(msg_dict).encode())
            msg = s.recv(8192)
            return_msg_dict = json.loads(msg)
        print(f"Received: {return_msg_dict}")
        return return_msg_dict

    def notify_inference_post_result(self, job_request):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            # s.sendall(b"inference#finish#"+message.encode())
            msg_dict = {
                'task': 'inference',
                'state': 'worker_post_result',
                'job_request': job_request
            }
            s.sendall(json.dumps(msg_dict).encode())
            msg = s.recv(8192)
        print(f"Received: {msg}")
        return msg
    """


class CoordinatorHybridInferenceClient:
    def __init__(self, args):
        self.host_ip = args.coordinator_server_ip
        self.host_port = args.coordinator_server_port
        self.client_port = int(args.lsf_job_no) % 10000 + 10000
        self.node_type = args.node_type

    def notify_inference_join(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            s.sendall(b"inference#join#"+self.node_type.encode())
            msg = s.recv(1024)
            print(f"Received: {msg}")
            msg_arg = client_message_parser(msg, 'join_inference')
            return msg_arg['prime_ip'], msg_arg['my_rank'], msg_arg['port']

    def notify_inference_finish(self, message: str):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            s.sendall(b"inference#finish#"+message.encode())
            msg = s.recv(1024)
            print(f"Received: {msg}")


def main():
    parser = argparse.ArgumentParser(description='Test Coordinator-Client')
    parser.add_argument('--coordinator-type', type=str, default='train', help='train or inference')
    parser.add_argument('--coordinator-server-port', type=int, default=9002, metavar='N',
                        help='The port of coordinator-server.')
    parser.add_argument('--coordinator-server-ip', type=str, default='localhost', metavar='S',
                        help='The IP of coordinator-server.')
    parser.add_argument('--lsf-job-no', type=str, default='100', metavar='S',
                        help='Job-<ID> assigned by LSF.')
    args = parser.parse_args()
    print(vars(args))
    if args.coordinator_type == 'train':
        client = CoordinatorTrainClient(args)
        client.notify_train_join()
        client.notify_train_finish("0#6.88")
    elif args.coordinator_type == 'inference':
        client = CoordinatorInferenceClient(args)
        client.notify_inference_join()
        client.notify_inference_finish("0#6.88")
    else:
        assert False


if __name__ == '__main__':
    main()
