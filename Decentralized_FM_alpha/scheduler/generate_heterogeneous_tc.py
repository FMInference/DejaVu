import argparse
from generate_sim_com_matrices import *

# No space for IPs!!!!
private_ip = [
    "172.31.42.98",
    "172.31.35.226",
    "172.31.36.102",
    "172.31.47.227",
    "172.31.33.98",
    "172.31.34.225",
    "172.31.43.243",
    "172.31.35.113",
    "172.31.39.119",
    "172.31.37.117",
    "172.31.43.239",
    "172.31.38.234",
    "172.31.36.113",
    "172.31.38.112",
    "172.31.32.161",
    "172.31.32.95",
    "172.31.32.37",
    "172.31.33.162",
    "172.31.47.90",
    "172.31.43.87",
    "172.31.38.95",
    "172.31.44.222",
    "172.31.36.49",
    "172.31.41.174",
    "172.31.44.51",
    "172.31.38.179",
    "172.31.46.166",
    "172.31.46.38",
    "172.31.45.171",
    "172.31.39.40",
    "172.31.40.195",
    "172.31.42.66",
    "172.31.33.200",
    "172.31.45.68",
    "172.31.34.125",
    "172.31.32.250",
    "172.31.34.66",
    "172.31.33.127",
    "172.31.35.212",
    "172.31.45.206",
    "172.31.42.86",
    "172.31.34.85",
    "172.31.45.203",
    "172.31.42.73",
    "172.31.44.205",
    "172.31.44.203",
    "172.31.32.158",
    "172.31.41.156",
    "172.31.46.8",
    "172.31.35.132",
    "172.31.35.139",
    "172.31.34.138",
    "172.31.34.60",
    "172.31.46.59",
    "172.31.45.0",
    "172.31.35.128",
    "172.31.38.22",
    "172.31.39.21",
    "172.31.42.152",
    "172.31.35.24",
    "172.31.44.15",
    "172.31.41.140",
    "172.31.38.20",
    "172.31.37.16"
]


def get_delay_bandwidth(args):
    if args.case == '1':
        return simulate_1_datacenter(args.nodes)
    elif args.case == '2':
        return simulate_2_datacenter_spot_gpu(args.nodes)
    elif args.case == '3':
        return simulate_3_multi_universities(args.nodes)
    elif args.case == '4':
        return simulate_4_regional_geo_distributed(args.nodes)
    elif args.case == '5':
        return simulate_5_worldwide_geo_distributed(args.nodes)
    elif args.case == '5_2':
        return simulate_5_2_worldwide_geo_distributed(args.nodes)
    elif args.case == '6_1':
        return simulate_6_1_debug_homogeneous_tc(args.nodes)
    elif args.case == '6_2':
        return simulate_6_2_debug_pipeline(args.nodes)
    else:
        assert False


def generate_tc_scripts(args):
    assert args.nodes == len(private_ip)
    delay, bandwidth, _ = get_delay_bandwidth(args)
    with open("../scripts/tc_scripts/heterogeneous_setup_case"+str(args.case)+".sh", 'w') as script:
        tc_setting_dict = {}
        handle_i = 1
        for i in range(len(private_ip)):
            if i != args.rank:
                current_key = (delay[args.rank][i], bandwidth[args.rank][i])
                if current_key not in tc_setting_dict:
                    tc_setting_dict[current_key] = handle_i
                    handle_i += 1
        assert len(tc_setting_dict) <= 16
        # setup delay and bandwidth subclass qdisc
        script.write("sudo tc qdisc add dev ens3 root handle 1: prio bands {}\n"
                     .format(max(3, len(tc_setting_dict))))
        for key in tc_setting_dict.keys():
            current_delay, current_bandwidth = key
            handle_index = tc_setting_dict[key]
            limit_pkts = current_delay * 22500 * current_bandwidth
            script.write("sudo tc qdisc add dev ens3 parent 1:{} handle {}: netem delay {}ms rate {}Gbit limit {}\n"
                         .format(handle_index, handle_index*10, current_delay, current_bandwidth, limit_pkts))
        # setup filter
        for i in range(len(private_ip)):
            if i != args.rank:
                current_key = (delay[args.rank][i], bandwidth[args.rank][i])
                script.write("sudo tc filter add dev ens3 parent 1:0 protocol ip prio 1 u32 match ip dst {}/32 flowid 1:{}\n"
                             .format(private_ip[i], tc_setting_dict[current_key]))


def main():
    parser = argparse.ArgumentParser(description='Test PyTorch Distributed')
    parser.add_argument('--case', type=str, default='5', metavar='R',
                        help='which case to generate.')
    parser.add_argument('--rank', type=int, default=0, metavar='R',
                        help='rank for this IP')
    parser.add_argument('--nodes', type=int, default=64, metavar='R',
                        help='Total number of nodes')
    args = parser.parse_args()
    generate_tc_scripts(args)


if __name__ == '__main__':
    main()
