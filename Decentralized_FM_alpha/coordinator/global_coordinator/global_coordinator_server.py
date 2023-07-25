from datetime import datetime
import argparse
import json
import pycouchdb


class GlobalCoordinatorServer:
    def __init__(self, args):
        server = pycouchdb.Server(args.db_server_address)
        self.db = server.database("global_coordinator")
        self.status_db = server.database("global_coordinator_status")
        self.allocated_task_index = 0
        self.task_meta_key = None
        self.task_meta = None
        self._resume_server_from_pycouchdb()

    def _resume_server_from_pycouchdb(self):
        for entrance in self.db.all():
            if 'meta_start_time' in entrance['doc']:
                self.task_meta_key = entrance['key']
                self.task_meta = entrance['doc']
        if self.task_meta_key is None:
            self.task_meta_key = self.db.save({
                'meta_start_time': str(datetime.now()),
                'task_hash': {}
            })['_id']
            self.task_meta = self.db.get(self.task_meta_key)
            print(self.task_meta)

    def _allocate_task_index(self):
        current_index = self.allocated_task_index
        self.allocated_task_index += 1
        return current_index

    def check_job_key_value_info(self):
        record_count = 0
        for entrance in self.db.all():
            print("-----------------------------------------")
            doc = entrance['doc']
            print("key:", doc['_id'])
            if 'job_type_info' in doc:
                print("job_type_info:", doc['job_type_info'])
                print("job_state:", doc['job_state'])
                print("time:", doc['time'])
                print("task_api: <task_type>:", doc['task_api']['task_type'])
                print("task_api: <model_name>:", doc['task_api']['model_name'])
                print("task_api: <parameters>:", doc['task_api']['parameters'])
                print("task_api: <inputs>:", doc['task_api']['inputs'])
                if doc['task_api']['outputs']:
                    print("task_api: length of <outputs>:", len(doc['task_api']['outputs']))
                else:
                    print("task_api: <outputs>: not ready")
                record_count += 1
            else:
                print(doc)
        print("Total number of record:", record_count)

    def check_status_key_value_info(self):
        for entrance in self.status_db.all():
            print("-----------------------------------------")
            doc = entrance['doc']
            print(doc)

    def clear_key_value(self):
        keys = [doc['key'] for doc in self.db.all()]
        print(keys)
        for key in keys:
            self.db.delete(key)

    def clear_status_key_value(self):
        keys = [doc['key'] for doc in self.status_db.all()]
        print(keys)
        for key in keys:
            self.status_db.delete(key)


def main():
    parser = argparse.ArgumentParser(description='Test Coordinator-Server')
    parser.add_argument('--db-server-address', type=str,
                        default="http://xzyao:agway-fondly-ell-hammer-flattered-coconut@db.yao.sh:5984/", metavar='N',
                        help='Key value store address.')
    parser.add_argument('--clear-all', type=lambda x: (str(x).lower() == 'true'),
                        default=False, metavar='S',
                        help='Delete all cached results. Everything gets deleted.')
    parser.add_argument('--op', type=str,
                        default="check_job", metavar='N',
                        help='Key value store address.')
    args = parser.parse_args()
    print(vars(args))
    coordinator = GlobalCoordinatorServer(args)
    if args.clear_all:
        coordinator.clear_status_key_value()

    if args.op == 'check_job':
        coordinator.check_job_key_value_info()
    elif args.op == 'check_status':
        coordinator.check_status_key_value_info()
    else:
        assert False


if __name__ == '__main__':
    main()
