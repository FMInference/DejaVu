import random
import numpy as np
import config
import scheduler


def compute_random_pipeline_parallel_cost(candidate_partition=None):
    random_pipeline_parallel_cost = 0
    for i in range(scheduler.way - 1):
        cross_partition_cost = float('-inf')
        for j in range(scheduler.partition_size):
            cur_cost = scheduler.peer_delay[candidate_partition[i][j], candidate_partition[i+1][j]]/1e3 + \
                scheduler.send_activation_size * 8 / \
                scheduler.peer_bandwidth[candidate_partition[i]
                                         [j], candidate_partition[i+1][j]]
            if cross_partition_cost < cur_cost:
                cross_partition_cost = cur_cost
        random_pipeline_parallel_cost += cross_partition_cost
    return random_pipeline_parallel_cost


def random_candidates(nodes=None, population_size=None):
    candidate_partitions = []
    for i in range(population_size):
        cur_nodes = nodes.copy()
        random.seed = i
        random.shuffle(cur_nodes)
        candidate_partitions.append(cur_nodes)

    candidate_data_parallel_cost = []
    candidate_pipeline_parallel_cost = []
    candidate_total_cost = []
    candidate_min_total_cost = []
    for candidate_partition_idx, candidate_partition in enumerate(candidate_partitions):
        candidate_partition = [candidate_partition[i: i + scheduler.partition_size]
                               for i in range(0, scheduler.num_devices, scheduler.partition_size)]
        candidate_partitions[candidate_partition_idx] = candidate_partition
        data_parallel_cost = scheduler.compute_data_parallel_cost(
            candidate_partition=candidate_partition)
        pipeline_parallel_cost = compute_random_pipeline_parallel_cost(
            candidate_partition)
        candidate_data_parallel_cost.append(data_parallel_cost)
        candidate_pipeline_parallel_cost.append(2 * pipeline_parallel_cost)
        candidate_total_cost.append(data_parallel_cost +
                                    2 * pipeline_parallel_cost)
        candidate_min_total_cost.append(np.min(candidate_total_cost))
    return candidate_partitions, candidate_total_cost, candidate_min_total_cost, candidate_data_parallel_cost, candidate_pipeline_parallel_cost


if __name__ == "__main__":
    simulate_cases = [
        config.simulate_0_datacenter,
        config.simulate_1_datacenter_spot_gpu,
        config.simulate_2_multi_universities,
        config.simulate_3_regional_geo_distributed,
        config.simulate_4_worldwide_geo_distributed,
    ]

    import time
    for repetition in range(3):
        np.random.seed = repetition
        for case_idx, simulate_case in enumerate(simulate_cases):
            scheduler.peer_delay, scheduler.peer_bandwidth, scheduler.regions = simulate_case()
            start = time.perf_counter()

            candidate_partitions, candidate_total_cost, candidate_min_total_cost, candidate_data_parallel_cost, candidate_pipeline_parallel_cost = random_candidates(
                nodes=list(range(scheduler.num_devices)), population_size=5000)
            candidate_partition_idx = np.argmin(candidate_total_cost)
            candidate_partition = candidate_partitions[candidate_partition_idx]
            pipeline_parallel_path = list(range(scheduler.way))
            data_parallel_cost = candidate_data_parallel_cost[candidate_partition_idx]
            pipeline_parallel_cost = candidate_pipeline_parallel_cost[candidate_partition_idx]
            min_total_cost = candidate_total_cost[candidate_partition_idx]
            average_total_cost = np.average(candidate_total_cost)

            with open('data/random_scheduler_' + str(case_idx) + '_' + str(repetition) + '.npy', 'wb') as f:
                np.save(f, np.array(candidate_min_total_cost))
            end = time.perf_counter()
            print("run time(" + str(len(candidate_partitions)) +
                  " candidates): " + str(end - start) + " seconds")
            print("candidate partition: " + str(candidate_partition))
            print("pipeline parallel path: " + str(pipeline_parallel_path))
            print("min total cost: " + str(min_total_cost))
            print("average total cost: " + str(average_total_cost))
            print("data parallel cost: " + str(data_parallel_cost))
            print("pipeline parallel cost: " + str(pipeline_parallel_cost))
