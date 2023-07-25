import socket
import argparse


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
        self.client_port = int(args.unique_port) % 10000 + 10000

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
        self.client_port = int(args.unique_port) % 10000 + 10000

    def notify_inference_join(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            s.sendall(b"inference#join")
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


class CoordinatorHybridInferenceClient:
    def __init__(self, args):
        self.host_ip = args.coordinator_server_ip
        self.host_port = args.coordinator_server_port
        self.client_port = int(args.unique_port) % 10000 + 10000
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
    parser.add_argument('--unique-port', type=str, default='100', metavar='S',
                        help='Which port to use, each client should have different value of this.')
    args = parser.parse_args()
    print(vars(args))
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
