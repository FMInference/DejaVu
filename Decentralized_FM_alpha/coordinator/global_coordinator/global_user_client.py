from datetime import datetime
import argparse
import pycouchdb


class GlobalUserClient:
    def __init__(self, args):
        self.db_server_address = args.db_server_address
        # server = pycouchdb.Server(args.db_server_address)
        # self.db = server.database("global_coordinator")
        # self.status_db = server.database("global_coordinator_status")
        self.task_keys = []
        self.active_models = {'gpt_j_6B', 'stable_diffusion'}

    def _get_db(self):
        server = pycouchdb.Server(self.db_server_address)
        db = server.database("global_coordinator")
        return db

    def _get_status_db(self):
        server = pycouchdb.Server(self.db_server_address)
        status_db = server.database("global_coordinator_status")
        return status_db

    def put_request_user_client(self, inference_details: dict):
        print("=========put_request_user_client=========")
        msg_dict = {
            'job_type_info': 'latency_inference',
            'job_state': 'job_queued',
            'time': {
                'job_queued_time': str(datetime.now()),
                'job_start_time': None,
                'job_end_time': None,
                'job_returned_time': None
            },
            'task_api': inference_details
        }
        db = self._get_db()
        doc = db.save(msg_dict)
        current_job_key = doc['_id']
        self.task_keys.append(current_job_key)
        print(f"=========[user client] put result in key value store=========")
        print("Current key:", current_job_key)
        print(doc)
        print("--------------------------------------------------")

    def get_request_user_client(self, request_key: str):
        print("=========get_request_user_client=========")
        db = self._get_db()
        doc = db.get(request_key)
        assert doc is not None
        print(f"=========[user client] get result in key value store=========")
        if doc['job_state'] == 'job_finished':
            doc['job_state'] = 'job_returned'
            doc['time']['job_returned_time'] = str(datetime.now())
            db.save(doc)
        print(doc)
        print("------------------------------------------------------")
        return doc

    def get_model_status_user_client(self):
        print("=========get_model_status_user_client=========")
        results = {}
        status_db = self._get_status_db()
        for status_doc in status_db.all():
            status_doc = status_doc['doc']
            # print(status_doc)
            if status_doc['model_name'] in self.active_models:
                current_key = status_doc['task_type'] + '/' + status_doc['model_name']
                if current_key not in results:
                    results[current_key] = status_doc
                else:
                    current_time = datetime.strptime(results[current_key]['last_heartbeat_time'], "%a %b %d %H:%M:%S %Y")
                    tmp_time = datetime.strptime(status_doc['last_heartbeat_time'], "%a %b %d %H:%M:%S %Y")
                    if tmp_time.timestamp() > current_time.timestamp():
                        results[current_key] = status_doc
        result_arr =[arr for arr in results.values()]
        for record in result_arr:
            print(record)
        print("------------------------------------------------------")
        return result_arr

    def get_model_time_estimate_user_client(self, task_type: str, model_name: str):
        print("=========get_model_status_user_client=========")
        last_time = None
        estimated_time = None
        db = self._get_db()
        for doc in db.all():
            doc = doc['doc']
            if ("job_type_info" in doc and doc['job_state'] == 'job_returned'
                    and doc['task_api']['model_name'] == model_name and doc['task_api']['task_type'] == task_type):
                start_time = datetime.strptime(doc['time']['job_queued_time'], '%Y-%m-%d %H:%M:%S.%f')
                if doc['time']['job_returned_time'] is not None:
                    end_time = datetime.strptime(doc['time']['job_returned_time'], '%Y-%m-%d %H:%M:%S.%f')
                else:
                    continue
                if last_time is None or last_time < start_time.timestamp():
                    last_time = start_time.timestamp()
                    estimated_time = end_time.timestamp() - start_time.timestamp()

        if estimated_time is None:
            estimated_record = 'N.A (No such record)'
        else:
            estimated_record = format(estimated_time, '.2f') + ' seconds'
        result = {
            "task_type": task_type,
            "model_name": model_name,
            "estimated_runtime": estimated_record
        }
        print(result)
        print("------------------------------------------------------")
        return result


def main():
    parser = argparse.ArgumentParser(description='Test Job-Submit-Client')
    parser.add_argument('--db-server-address', type=str,
                        default="http://xzyao:agway-fondly-ell-hammer-flattered-coconut@db.yao.sh:5984/", metavar='N',
                        help='Key value store address.')
    parser.add_argument('--op', type=str, default='get', metavar='S',
                        help='The op: {get or put}.')
    parser.add_argument('--request-key', type=str, default="6070cb8cfa50434192e060ed40c9a92e", metavar='N',
                        help='The index of the submitted tasks.')
    parser.add_argument('--inputs', type=str, default='Hello world!', metavar='S',
                        help='The prompt sequence.')
    parser.add_argument('--model-name', type=str, default='gpt-j-6B', metavar='S',
                        help='-')
    parser.add_argument('--task-type', type=str, default='seq_generation', metavar='S',
                        help='-')
    parser.add_argument('--max-new-tokens', type=int, default=128, metavar='N',
                        help='-')
    parser.add_argument('--num-return-sequences', type=int, default=3, metavar='N',
                        help='-')
    args = parser.parse_args()
    print(vars(args))
    client = GlobalUserClient(args)

    if args.op == 'get':
        client.get_request_user_client(args.request_key)
    elif args.op == 'put':
        inference_details = {
            'inputs': args.inputs,
            'model_name': args.model_name,
            'task_type': args.task_type,
            "parameters": {
                "max_new_tokens": args.max_new_tokens,
                "return_full_text": False,
                "do_sample": True,
                "temperature": 0.8,
                "top_p": 0.95,
                "max_time": 10.0,
                "num_return_sequences": args.num_return_sequences,
                "use_gpu": True
            },
            'outputs': None
        }
        client.put_request_user_client(inference_details)
    elif args.op == 'status':
        client.get_model_status_user_client()
    elif args.op == 'estimate':
        client.get_model_time_estimate_user_client(args.task_type, args.model_name)
    else:
        assert False


if __name__ == '__main__':
    main()
