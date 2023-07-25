import scheduler
import numpy as np
import config


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
            min_total_cost = float('inf')
            candidate_partition = None
            data_parallel_cost = None
            pipeline_parallel_cost = None
            pipeline_parallel_path = None
            pipeline_parallel_match = None

            candidate_partitions, all_cost_records, min_cost_records = scheduler.GCMA(
                nodes=list(range(scheduler.num_devices)), population_size=100, trails=4900, mode="baseline")
            candidate_partition_idx = np.argmin(all_cost_records)
            candidate_partition = [candidate_partitions[candidate_partition_idx][i: i + scheduler.partition_size]
                                   for i in range(0, scheduler.num_devices, scheduler.partition_size)]
            data_parallel_cost = scheduler.compute_data_parallel_cost(
                candidate_partition=candidate_partition)
            pipeline_parallel_cost, pipeline_parallel_path, pipeline_parallel_match = scheduler.compute_pipeline_parallel_cost(
                candidate_partition)
            min_total_cost = data_parallel_cost + 2 * pipeline_parallel_cost

            end = time.perf_counter()
            print("run time(" + str(len(all_cost_records)) +
                  " candidates): " + str(end - start) + " seconds")
            print("candidate partition: " + str(candidate_partition))
            print("pipeline parallel path: " + str(pipeline_parallel_path))
            print("total cost: " + str(min_total_cost))
            print("data parallel cost: " + str(data_parallel_cost))
            print("pipeline parallel cost: " + str(2 * pipeline_parallel_cost))

            candidate_pipeline = scheduler.get_pipelines(
                candidate_partition, pipeline_parallel_path, pipeline_parallel_match)

            ip_rank_map = [0] * scheduler.num_devices
            for pipeline_idx in range(scheduler.partition_size):
                for stage_idx in range(scheduler.way):
                    ip_rank_map[candidate_pipeline[stage_idx,
                                                   pipeline_idx]] = pipeline_idx * scheduler.way + stage_idx
            assert(np.sum(ip_rank_map) == np.sum(range(scheduler.num_devices)))

            if scheduler.regions != None:
                for pipeline_idx in range(scheduler.partition_size):
                    print("pipeline " + str(pipeline_idx) + ": ", end="")
                    for stage_idx in range(scheduler.way):
                        ip = ip_rank_map.index(
                            pipeline_idx * scheduler.way + stage_idx)
                        print(scheduler.regions[ip] +
                              (" " * (10 - len(scheduler.regions[ip]))), end=", ")
                    print()
            print(ip_rank_map)

            with open('data/hybrid_scheduler_' + str(case_idx) + '_' + str(repetition) + '.npy', 'wb') as f:
                np.save(f, np.array(min_cost_records))
