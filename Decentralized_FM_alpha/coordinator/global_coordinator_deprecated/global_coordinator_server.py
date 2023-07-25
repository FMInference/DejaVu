import socket
import argparse
import json
from pymemcache.client import base

MAX_TASK_INDXE=1000

class JsonSerde(object):
    def serialize(self, key, value):
        if isinstance(value, str):
            return value, 1
        return json.dumps(value), 2

    def deserialize(self, key, value, flags):
       if flags == 1:
           return value
       if flags == 2:
           return json.loads(value)
       raise Exception("Unknown serialization format")


class GlobalCoordinatorServer:
    def __init__(self, args):
        self.host = args.coordinator_server_ip
        self.port = args.coordinator_server_port
        self.allocated_task_index = 0
        self.key_value_client = base.Client(('127.0.0.1', 11111), serde=JsonSerde())
        self.to_do_tasks = []
        self._resume_server_from_memcache()

    def _allocate_task_index(self):
        current_index = self.allocated_task_index
        self.allocated_task_index += 1
        return current_index

    def _handle_put_request_user_client(self, ip, msg_dict):
        current_index = self._allocate_task_index()
        current_key = 'task_index_' + str(current_index)
        current_value = {
            'hf_api_para': msg_dict['hf_api_para'],
            'state': 'job_queued',
            'user_ip': ip
        }
        self.key_value_client.add(current_key, current_value)
        return_msg = {
            'task_index': current_index,
            'state': 'job_queued'
        }
        self.to_do_tasks.append(current_index)
        return json.dumps(return_msg)

    def _handle_get_request_user_client(self, ip, msg_dict):
        current_key = 'task_index_' + str(msg_dict['task_index'])
        current_value = self.key_value_client.get(current_key, default=None)
        # assert current_value is not None
        # assert ip == current_value['user_ip']
        if current_value is None or ip != current_value['user_ip']:
            return_msg = {
                'task_index': msg_dict['task_index'],
                'state': 'illegal request'
            }
        else:
            return_msg = {
                'task_index': msg_dict['task_index'],
                'state': current_value['state']
            }
            if current_value['state'] == 'job_finished':
                return_msg['result'] = current_value['result']
                self.key_value_client.delete(current_key)
        return json.dumps(return_msg)

    def _handle_get_request_cluster_coordinator(self, ip):
        if len(self.to_do_tasks) == 0:
            return_msg = {
                'task_index': -1,
            }
        else:
            current_index = self.to_do_tasks.pop(0)
            current_key = 'task_index_' + str(current_index)
            current_value = self.key_value_client.get(current_key, default=None)
            assert current_value is not None
            current_value['state'] = 'job_issued'
            current_value['server_ip'] = ip
            self.key_value_client.set(current_key, current_value)
            return_msg = {
                'task_index': current_index,
                'hf_api_para': current_value['hf_api_para'],
                'has_more_tasks': len(self.to_do_tasks) != 0
            }
        return json.dumps(return_msg)

    def _handle_put_request_cluster_coordinator(self, ip, msg_dict):
        current_key = 'task_index_' + str(msg_dict['task_index'])
        current_value = self.key_value_client.get(current_key, default=None)
        if current_value is None:
            return_msg = {
                'task_index': msg_dict['task_index'],
                'state': 'put_result_error'
            }
        else:
            assert current_value['server_ip'] == ip
            current_value['state'] = 'job_finished'
            current_value['result'] = msg_dict['result']
            self.key_value_client.set(current_key, current_value)
            return_msg = {
                'task_index': msg_dict['task_index'],
                'state': 'put_result_succeed'
            }
        return json.dumps(return_msg)

    def execute_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            while True:
                connection, address = s.accept()
                with connection:
                    ip, port = address
                    msg_raw = connection.recv(1024)
                    msg_dict = json.loads(msg_raw)
                    print(f"====<Ip:{ip}> op: {msg_dict['op']}====")
                    print(msg_dict)
                    print("----------------------------------------")
                    if msg_dict['op'] == 'put_request_user_client':
                        return_msg = self._handle_put_request_user_client(ip, msg_dict)
                    elif msg_dict['op'] == 'get_request_user_client':
                        return_msg = self._handle_get_request_user_client(ip, msg_dict)
                    elif msg_dict['op'] == 'get_request_cluster_coordinator':
                        return_msg = self._handle_get_request_cluster_coordinator(ip)
                    elif msg_dict['op'] == 'put_request_cluster_coordinator':
                        return_msg = self._handle_put_request_cluster_coordinator(ip, msg_dict)
                    else:
                        assert False, "Not recognized op"
                    connection.sendall(return_msg.encode())
                    connection.close()

    def check_status_from_memcache(self):
        for i in range(MAX_TASK_INDXE):
            current_key = 'task_index_' + str(i)
            current_value = self.key_value_client.get(current_key,default=None)
            if current_value:
                print(f"<key: {current_key}>")
                print(current_value)

    def _resume_server_from_memcache(self):
        for i in range(MAX_TASK_INDXE):
            current_key = 'task_index_' + str(i)
            current_value = self.key_value_client.get(current_key, default=None)
            if current_value:
                if current_value['state'] == 'job_issued':
                    self.to_do_tasks.append(i)
                self.allocated_task_index = i + 1


def main():
    parser = argparse.ArgumentParser(description='Test Coordinator-Server')
    parser.add_argument('--coordinator-server-port', type=int, default=9102, metavar='N',
                        help='The port of coordinator-server.')
    parser.add_argument('--coordinator-server-ip', type=str, default='0.0.0.0', metavar='S',
                        help='The IP of coordinator-server.')
    args = parser.parse_args()
    print(vars(args))
    coordinator = GlobalCoordinatorServer(args)
    coordinator.check_status_from_memcache()
    coordinator.execute_server()


if __name__ == '__main__':
    main()