import os
import json
import argparse
import shutil

# deprecated
node_ip_lists = []


# deprecated
def download_trace_logs(args, profix, postfix, ips=node_ip_lists):
    if os.path.isdir('./' + profix):
        # os.rmdir('./'+prefix)
        shutil.rmtree('./' + profix)
    os.mkdir('./' + profix)
    for i in range(args.world_size):
        os.system("scp -i ../binhang_ds3_aws_oregon.pem ubuntu@" + ips[i] +
                  ":~/GPT-home-private/trace_json/" + profix + '_' + str(i) + postfix + ' ./' + profix)


def merge_logs(args):
    result = []
    current_min_stamp = float('inf')
    for i in range(args.world_size):
        print(i)
        with open("../trace_json/" + args.profix + '/' + args.profix + '_' + str(i) + '_' + args.postfix + '.json') \
                as inputJson:
            current_trace = json.load(inputJson)
            inputJson.close()
            if i == 0:
                for log in current_trace:
                    current_min_stamp = min(log['ts'], current_min_stamp)
            for log in current_trace:
                log['pid'] = args.mode + ' node ' + str(i)
                log['ts'] = log['ts'] - current_min_stamp
            result.extend(current_trace)
    print(len(result))
    with open("../trace_json/" + args.profix + '_' + args.postfix + '.json', 'w') as outputJson:
        json.dump(result, outputJson)


def main():
    parser = argparse.ArgumentParser(description='Gpipe-GPT3')
    parser.add_argument('--world-size', type=int, default=12, metavar='N',
                        help='distributed cluster size (default: 3)')
    parser.add_argument('--mode', type=str, default='gpipe', metavar='S',
                        help='use which mode: gpipe or 1f1b.')
    parser.add_argument('--profix', type=str, default='gpt3_gpipe_b64_1_l2048_m2048_w12_p12_d1', metavar='S',
                        help='postfix of the tracing file name.')
    parser.add_argument('--postfix', type=str, default='tidy_profiling_real', metavar='S',
                        help='postfix of the tracing file name.')

    args = parser.parse_args()
    merge_logs(args)


if __name__ == '__main__':
    main()
