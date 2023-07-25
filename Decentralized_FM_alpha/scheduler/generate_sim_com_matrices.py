import numpy as np
import scipy.linalg
import sys
np.set_printoptions(threshold=sys.maxsize)


delay_bandwidth_dict = {
    "Oregon-Virginia": (67, 0.79),
    "Oregon-Ohio": (49, 1.10),
    "Oregon-Tokyo": (96, 0.523),
    "Oregon-Seoul": (124, 0.46),
    "Oregon-Singapore": (163, 0.341),
    "Oregon-Sydney": (139, 0.36),
    "Oregon-London": (136, 0.42),
    "Oregon-Frankfurt": (143, 0.404),
    "Oregon-Ireland": (124, 0.482),
    "Virginia-Ohio": (11, 1.12),
    "Virginia-Tokyo": (143, 0.524),
    "Virginia-Seoul": (172, 0.500),
    "Virginia-Singapore": (230, 0.364),
    "Virginia-Sydney": (197, 0.383),
    "Virginia-London": (76, 1.16),
    "Virginia-Frankfurt": (90, 1.02),
    "Virginia-Ireland": (67, 1.05),
    "Ohio-Tokyo": (130, 0.694),
    "Ohio-Seoul": (159, 0.529),
    "Ohio-Singapore": (197, 0.452),
    "Ohio-Sydney": (185, 0.484),
    "Ohio-London": (86, 1.05),
    "Ohio-Frankfurt": (99, 0.799),
    "Ohio-Ireland": (77, 1.14),
    "Tokyo-Seoul": (34, 1.10),
    "Tokyo-Singapore": (73, 1.01),
    "Tokyo-Sydney": (100, 0.761),
    "Tokyo-London": (210, 0.366),
    "Tokyo-Frankfurt": (223, 0.36),
    "Tokyo-Ireland": (199, 0.465),
    "Seoul-Singapore": (74, 1.14),
    "Seoul-Sydney": (148, 0.58),
    "Seoul-London": (238, 0.342),
    "Seoul-Frankfurt": (235, 0.358),
    "Seoul-Ireland": (228, 0.335),
    "Singapore-Sydney": (92, 0.816),
    "Singapore-London": (169, 0.500),
    "Singapore-Frankfurt": (155, 0.535),
    "Singapore-Ireland": (179, 0.492),
    "Sydney-London": (262, 0.326),
    "Sydney-Frankfurt": (265, 0.328),
    "Sydney-Ireland": (254, 0.344),
    "London-Frankfurt": (14, 1.14),
    "London-Ireland": (12, 1.09),
    "Frankfurt-Ireland": (24, 1.08)
}


def simulate_1_datacenter(nodes=64):
    print("Simulate case 0: on-demand datacenter.")
    delay = np.zeros((nodes, nodes))
    bandwidth = np.ones((nodes, nodes)) * 3.125
    gpu_per_instances = min(nodes//2, 8)
    instances = nodes // gpu_per_instances
    bandwidth_blocks = [
        np.ones((gpu_per_instances, gpu_per_instances))*96.875 for _ in range(instances)]
    bandwidth = bandwidth + scipy.linalg.block_diag(*bandwidth_blocks)
    regions = []
    for i in range(nodes):
        regions.append("instance_" + str(i // 8))
    print('delay(ms):', delay)
    print('bandwidth(Gbps):', bandwidth)
    return delay, bandwidth, regions


def simulate_2_datacenter_spot_gpu(nodes=64, group=(8, 4)):
    print("Simulate case 1: spot datacenter.")
    delay = np.zeros((nodes, nodes))
    bandwidth = np.ones((nodes, nodes)) * 1.25
    instance_num = group[0]
    gpu_per_instances = group[1]
    bandwidth_blocks = [np.ones((gpu_per_instances, gpu_per_instances)) * (98.75 if i < instance_num else 0)
                        for i in range(nodes//gpu_per_instances)]
    bandwidth = bandwidth + scipy.linalg.block_diag(*bandwidth_blocks)

    print('delay(ms):', delay)
    print('bandwidth(Gbps):', bandwidth)
    return delay, bandwidth, None


def simulate_3_multi_universities(nodes=64):
    print("Simulate case 2: multi universities. 0~31 in Ohio, 32~63 in Virginia.")
    delay = np.ones((nodes, nodes))
    bandwidth = np.ones((nodes, nodes)) * 5
    split = nodes//2
    regions = []
    for i in range(nodes):
        if i < split:
            regions.append("Ohio")
        else:
            regions.append("Virginia")
    for i in range(nodes):
        for j in range(nodes):
            if not ((i < split and j < split) or (i >= split and j >= split)):
                delay[i][j] = 11
                bandwidth[i][j] = 1.12
    print('delay:', delay)
    print('bandwidth:', bandwidth)
    return delay, bandwidth, regions


# Assume within region is 2 GB, 5 ms.
def simulate_4_regional_geo_distributed(nodes=64):
    print("Simulate case 3: regional geo distributed: 0~15 in Virgina; 17~33 in Oregon, 34~63 in Ohio")

    def in_virgina(index: int):
        return index < nodes//4

    def in_oregon(index: int):
        return nodes//4 <= index < nodes//2

    def in_california(index: int):
        return nodes//2 <= index < nodes*3//4

    def in_ohio(index: int):
        return index >= nodes*3//4

    regions = []
    for i in range(nodes):
        if in_virgina(i):
            regions.append("Virginia")
        elif in_oregon(i):
            regions.append("Oregon")
        elif in_california(i):
            regions.append("California")
        elif in_ohio(i):
            regions.append("Ohio")

    delay = np.ones((nodes, nodes)) * 10
    bandwidth = np.ones((nodes, nodes)) * 2
    for i in range(nodes):
        for j in range(i, nodes):
            if in_virgina(i) and in_oregon(j):
                delay[i][j] = 67
                delay[j][i] = 67
                bandwidth[i][j] = 1.15
                bandwidth[j][i] = 1.15
            elif in_virgina(i) and in_california(j):
                delay[i][j] = 59
                delay[j][i] = 59
                bandwidth[i][j] = 1.05
                bandwidth[j][i] = 1.05
            elif in_virgina(i) and in_ohio(j):
                delay[i][j] = 11
                delay[j][i] = 11
                bandwidth[i][j] = 1.12
                bandwidth[j][i] = 1.12
            elif in_oregon(i) and in_california(j):
                delay[i][j] = 12
                delay[j][i] = 12
                bandwidth[i][j] = 1.25
                bandwidth[j][i] = 1.25
            elif in_oregon(i) and in_ohio(j):
                delay[i][j] = 49
                delay[j][i] = 49
                bandwidth[i][j] = 1.10
                bandwidth[j][i] = 1.10
            elif in_california(i) and in_ohio(j):
                delay[i][j] = 52
                delay[j][i] = 52
                bandwidth[i][j] = 1.02
                bandwidth[j][i] = 1.02
    print('delay(ms):', delay)
    print('bandwidth(Gbps):', bandwidth)
    return delay, bandwidth, regions


# Assume within region is 2 GB, 5 ms.
def simulate_5_worldwide_geo_distributed(nodes=64):
    print("Simulate case 4: worldwide geo distributed (balanced)")
    cities = ["Oregon", "Virginia", "Ohio", "Tokyo", "Seoul", "London", "Frankfurt", "Ireland"]
    regions = []
    # for i in np.random.randint(low=0, high=len(cities), size=nodes):
    #    regions.append(cities[i])
    instances_per_region = nodes//len(cities)
    for i in range(len(cities)):
        for _ in range(instances_per_region):
            regions.append(cities[i])
    assert len(regions) == nodes

    def get_delay_bandwidth(region1: str, region2: str):
        if region1 == region2:
            return 10, 2
        else:
            if region1+'-'+region2 in delay_bandwidth_dict:
                return delay_bandwidth_dict[region1+'-'+region2]
            elif region2+'-'+region1 in delay_bandwidth_dict:
                return delay_bandwidth_dict[region2+'-'+region1]
            else:
                print(region1, region2)
                assert False

    delay = np.ones((nodes, nodes)) * 10
    bandwidth = np.ones((nodes, nodes)) * 2

    for i in range(nodes):
        for j in range(i, nodes):
            d_val, b_val = get_delay_bandwidth(regions[i], regions[j])
            delay[i][j] = d_val
            delay[j][i] = d_val
            bandwidth[i][j] = b_val
            bandwidth[j][i] = b_val
    print('delay(ms):', delay)
    print('bandwidth(Gbps):', bandwidth)
    return delay, bandwidth, regions


def simulate_5_2_worldwide_geo_distributed(nodes=64):
    print("Simulate case 5-2: worldwide geo distributed (not balanced)")
    # cities = ["Oregon", "Virginia", "Ohio",  "Tokyo", "Seoul", "Singapore", "Sydney", "London", "Frankfurt", "Ireland"]
    cities = ["Oregon", "Virginia", "Ohio", "Tokyo", "Seoul", "London", "Frankfurt", "Ireland"]
    regions = []
    print(np.__version__)
    np.random.seed(2022)
    for i in np.random.randint(low=0, high=len(cities), size=nodes):
        regions.append(cities[i])
    instances_per_region = nodes//len(cities)
    # for i in range(len(cities)):
    #    for _ in range(instances_per_region):
    #        regions.append(cities[i])
    assert len(regions) == nodes

    def get_delay_bandwidth(region1: str, region2: str):
        if region1 == region2:
            return 10, 2
        else:
            if region1+'-'+region2 in delay_bandwidth_dict:
                return delay_bandwidth_dict[region1+'-'+region2]
            elif region2+'-'+region1 in delay_bandwidth_dict:
                return delay_bandwidth_dict[region2+'-'+region1]
            else:
                print(region1, region2)
                assert False

    delay = np.ones((nodes, nodes)) * 10
    bandwidth = np.ones((nodes, nodes)) * 2

    for i in range(nodes):
        for j in range(i, nodes):
            d_val, b_val = get_delay_bandwidth(regions[i], regions[j])
            delay[i][j] = d_val
            delay[j][i] = d_val
            bandwidth[i][j] = b_val
            bandwidth[j][i] = b_val
    print("cities", cities)
    print("regions:")
    for i in range(len(regions)):
        print('{0:>10}'.format(regions[i]), end='| ')
        if i % 8 == 7:
            print()
    print('delay(ms):', delay)
    print('bandwidth(Gbps):', bandwidth)
    return delay, bandwidth, regions


def simulate_6_1_debug_homogeneous_tc(nodes=64, delay=50, bandwidth=1):
    print("Simulate case 6-1: homogeneous traffic control")
    delay = np.ones((nodes, nodes)) * delay
    bandwidth = np.ones((nodes, nodes)) * bandwidth
    print('delay(ms):', delay)
    print('bandwidth(Gbps):', bandwidth)
    return delay, bandwidth, None


def simulate_6_2_debug_pipeline(nodes=8):
    print("Simulate case 6-2: debug PP, nodes 8")
    delay = np.ones((nodes, nodes))
    bandwidth = np.ones((nodes, nodes)) * 5
    for i in range(nodes-1):
        delay[i][i+1] = 5*(i+1)
        delay[i+1][i] = 5*(i+1)
        bandwidth[i][i+1] = (10-i)//2
        bandwidth[i+1][i] = (10 - i) // 2
    print('delay(ms):', delay)
    print('bandwidth(Gbps):', bandwidth)
    return delay, bandwidth, None


def main():
    simulate_5_worldwide_geo_distributed()


if __name__ == '__main__':
    main()


