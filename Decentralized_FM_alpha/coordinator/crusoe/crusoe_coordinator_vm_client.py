import time
import socket
import argparse
import random
import json


class VMClient:
    def __init__(self, args):
        self.host_ip = args.coordinator_server_ip
        self.host_port = args.coordinator_server_port
        self.client_port = 9999

    def send_message_to_coordinate(self, message: str):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            message_dict = {
                "op": "send_message_vm",
                "message": message}
            s.sendall(json.dumps(message_dict).encode())
            msg = s.recv(1024)
            print(f"Received: {msg}")


def main():
    parser = argparse.ArgumentParser(description='Test Job-Submit-Client')
    parser.add_argument('--coordinator-server-port', type=int, default=9002, metavar='N',
                        help='The port of coordinator-server.')
    parser.add_argument('--coordinator-server-ip', type=str, default='localhost', metavar='S',
                        help='The IP of coordinator-server.')
    parser.add_argument('--message', type=str, default='Hello world from Crusoe!', metavar='S',
                        help='send a message to client.')
    args = parser.parse_args()
    # print(vars(args))
    client = VMClient(args)
    client.send_message_to_coordinate(args.message)


if __name__ == '__main__':
    main()
