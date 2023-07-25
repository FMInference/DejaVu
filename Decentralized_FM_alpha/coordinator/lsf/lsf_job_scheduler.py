import os.path
from datetime import datetime
import time
import socket
import argparse
import json
import sys
import pycouchdb
from filelock import Timeout, SoftFileLock


model_name_and_task_type_list = [
    ("image_generation", "stable_diffusion"),
    ("seq_generation", "gpt_j_6B")
]


def alias_to_model_name(model_alias: str) -> str:
    # print(torch_type)

    mappings = {
        'stable_diffusion':'stable_diffusion',
        'Image: stable_diffusion': 'stable_diffusion',
        'gpt_j_6B': 'gpt_j_6B',
        'gpt-j-6B': 'gpt_j_6B',
        'EleutherAI/gpt-j-6B': 'gpt_j_6B',
        'gpt-neox-20b-new': 'gpt_neox',
        'T0pp-new': 't0_pp',
        't5-11b-new': 't5',
        'ul2-new': 'ul2',
        'opt_66B':'opt_66B',
        'multimodalart/latentdiffusion': None
    }
    return mappings[model_alias]


# assume both worker and client
class JobScheduler:
    def __init__(self, args):
        self.host_ip = args.coordinator_server_ip
        self.host_port = args.coordinator_server_port
        self.client_port = 20099
        self.working_directory = args.working_directory
        self.heartbeats_timelimit = 300

        self.db_server_address = args.db_server_address
        # server = pycouchdb.Server(args.db_server_address)
        # self.db = server.database("global_coordinator")
        # self.status_db = server.database("global_coordinator_status")
        self.model_locks = {}
        for task_model_tuple in model_name_and_task_type_list:
            model_name = task_model_tuple[1]
            path = os.path.join(self.working_directory, model_name)
            if not os.path.exists(path):
                os.mkdir(path)
            lock_path = os.path.join(path, model_name+'.lock')
            # if not os.path.exists(lock_path):
            #     with open(lock_path, mode='a'):
            #        pass
            model_lock = SoftFileLock(lock_path, timeout=1)
            self.model_locks[model_name] = model_lock

    def _get_db(self):
        server = pycouchdb.Server(self.db_server_address)
        db = server.database("global_coordinator")
        return db

    def _get_status_db(self):
        server = pycouchdb.Server(self.db_server_address)
        status_db = server.database("global_coordinator_status")
        return status_db


    def _job_scheduler_notify_server_heartbeats(self):
        print("=========_job_scheduler_notify_server_heartbeats=========")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            msg_dict = {
                'task': 'inference',
                'state': 'job_scheduler_heartbeats',
            }
            s.sendall(json.dumps(msg_dict).encode())
            msg = s.recv(1024)
            print(f"Received: {msg}")

    def _fetch_job_request_input_from_global_coordinator(self):
        # print("=========_fetch_job_request_input_from_global_coordinator=========")
        new_job_arr = []
        db = self._get_db()
        for doc in db.all():
            # print(doc)
            # print('job_type_info' in doc['doc'])
            doc = doc['doc']
            if 'job_type_info' in doc:
                model_alias = doc['task_api']['model_name']
                model_name = alias_to_model_name(model_alias)
                if model_name is None:
                    pass
                    # print(f"!!!!! Unknown model_name: {model_alias} !!!!!")
                else:
                    os.path.join(self.working_directory, model_name)
                    if doc['job_state'] == 'job_queued':
                        doc['job_state'] = 'job_running'
                        doc['time']['job_start_time'] = str(datetime.now())
                        doc = db.save(doc)
                        path = os.path.join(self.working_directory, model_name, 'input_' + doc['_id'] + '.json')
                        with self.model_locks[model_name]:
                            with open(path, 'w') as outfile:
                                json.dump(doc, outfile)
                        new_job_arr.append(doc['_id'])
        if len(new_job_arr) != 0:
            print("=========_fetch_job_request_input_from_global_coordinator=========")
            print("Get input job: ", new_job_arr)

    def _post_job_request_output_to_global_coordinator(self):
        return_job_arr = []
        db = self._get_db()
        for task_model_tuple in model_name_and_task_type_list:
            model_name = task_model_tuple[1]
            dir_path = os.path.join(self.working_directory, model_name)
            for filename in os.listdir(dir_path):
                # print(filename)
                if filename.startswith('output_'):
                    doc_path = os.path.join(dir_path, filename)
                    try:
                        with self.model_locks[model_name]:
                            with open(doc_path, 'r') as infile:
                                doc = json.load(infile)
                    except Timeout:
                        print("File lock timeout!")

                    assert 'task_api' in doc and doc['task_api']['outputs'] is not None
                    doc['job_state'] = 'job_finished'
                    doc['time']['job_end_time'] = str(datetime.now())
                    db.save(doc)
                    return_job_arr.append(doc['_id'])
                    new_doc_path = os.path.join(dir_path, 'posted_'+filename)
                    os.rename(doc_path, new_doc_path)
        if len(return_job_arr) != 0:
            print("=========_post_job_request_output_to_global_coordinator=========")
            print("Post output job:", return_job_arr)

    def execute_job_scheduler(self):
        last_time = time.time()
        print("Running: <_fetch_job_request_input_from_global_coordinator> and "
              "<_post_job_request_output_to_global_coordinator>")
        while True:
            self._fetch_job_request_input_from_global_coordinator()
            self._post_job_request_output_to_global_coordinator()
            current_time = time.time()
            if current_time - last_time > self.heartbeats_timelimit:
                last_time = current_time
                self._job_scheduler_notify_server_heartbeats()


def main():
    parser = argparse.ArgumentParser(description='LSF Job scheduler through DFS.')
    parser.add_argument('--coordinator-server-port', type=int, default=9002, metavar='N',
                        help='The port of coordinator-server.')
    parser.add_argument('--coordinator-server-ip', type=str, default='localhost', metavar='S',
                        help='The IP of coordinator-server.')
    parser.add_argument('--working-directory', type=str,
                        default='/cluster/scratch/biyuan/fetch_cache', metavar='S',
                        help='The IP of coordinator-server.')
    parser.add_argument('--db-server-address', type=str,
                        default="http://xzyao:agway-fondly-ell-hammer-flattered-coconut@db.yao.sh:5984/", metavar='N',
                        help='Key value store address.')
    args = parser.parse_args()
    print(vars(args))
    client = JobScheduler(args)
    client.execute_job_scheduler()


if __name__ == '__main__':
    main()
