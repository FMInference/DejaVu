import argparse
import pycouchdb
from datetime import datetime


class GlobalCoordinatorClient:
    def __init__(self, args):
        self.db_server_address = args.db_server_address
        # server = pycouchdb.Server(args.db_server_address)
        # self.db = server.database("global_coordinator")
        # self.status_db = server.database("global_coordinator_status")

    def _get_db(self):
        server = pycouchdb.Server(self.db_server_address)
        db = server.database("global_coordinator")
        return db

    def _get_status_db(self):
        server = pycouchdb.Server(self.db_server_address)
        status_db = server.database("global_coordinator_status")
        return status_db

    def put_request_cluster_coordinator(self, request_doc: dict, inference_result=None) -> dict:
        print("=========put_request_cluster_coordinator=========")
        # print(request_doc)

        request_doc['job_state'] = 'job_finished'
        request_doc['time']['job_end_time'] = str(datetime.now())
        if inference_result is not None:
            request_doc['task_api']['outputs'] = inference_result
        else:
            assert request_doc['task_api']['outputs'] is not None
        db = self._get_db()
        request_doc = db.save(request_doc)
        print(f"=========[cluster client] put result in key value store=========")
        # print(request_doc)
        print("-----------------------------------------------------------------")
        return request_doc

    def get_request_cluster_coordinator(self, job_type_info='latency_inference',
                                        model_name='gptj', task_type='seq_generation') -> dict:
        print("=========get_request_cluster_coordinator=========")
        # Note this is a preliminary version for latency based inference, we need to add more functionality here.
        db = self._get_db()
        for doc in db.all():
            # print(doc)
            # print('job_type_info' in doc['doc'])
            doc = doc['doc']
            if 'job_type_info' in doc and doc['job_type_info'] == job_type_info:
                if doc['task_api']['model_name'] == model_name and doc['task_api']['task_type'] == task_type:
                    if doc['job_state'] == 'job_queued':
                        doc['job_state'] = 'job_running'
                        doc['time']['job_start_time'] = str(datetime.now())
                        doc = db.save(doc)
                        print(f"=========[cluster client] get task in key value store=========")
                        # print(doc)
                        print("---------------------------------------------------------------")
                        return doc
        print(f"=========[cluster client] get task in key value store=========")
        print("None job in the queue")
        print("---------------------------------------------------------------")
        return None

    def post_model_heartbeats_cluster_coordinator(self, heartbeats_dict) -> dict:
        print("=========post_model_heartbeats_cluster_coordinator=========")
        assert 'task_type' in heartbeats_dict
        assert 'model_name' in heartbeats_dict
        assert 'cluster_location' in heartbeats_dict
        assert 'last_heartbeat_time' in heartbeats_dict
        status_db = self._get_status_db()
        status_doc = status_db.save(heartbeats_dict)
        print(f"=========[cluster client] post heartbeats in key value store=========")
        print(status_doc)
        print("---------------------------------------------------------------")
        return status_doc


def main():
    parser = argparse.ArgumentParser(description='Test Job-Submit-Client')
    parser.add_argument('--db-server-address', type=str,
                        default="http://xzyao:agway-fondly-ell-hammer-flattered-coconut@db.yao.sh:5984/", metavar='N',
                        help='Key value store address.')
    parser.add_argument('--op', type=str, default='put', metavar='S',
                        help='The op: {get or put}.')
    args = parser.parse_args()
    print(vars(args))
    client = GlobalCoordinatorClient(args)

    if args.op == 'get':
        client.get_request_cluster_coordinator()
    elif args.op == 'put':
        inference_results = ["Hello world reply."]
        req_doc={

        }
        client.put_request_cluster_coordinator(req_doc, inference_results)
    else:
        assert False


if __name__ == '__main__':
    main()