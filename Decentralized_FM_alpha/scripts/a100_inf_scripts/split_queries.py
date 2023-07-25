import json
import os
import argparse
from collections import defaultdict


def main(args):
    queries = []
    with open(args.input_file, 'r') as f:
        for line in f:
            queries.append(json.loads(line))
            
    clusters_by_engine = defaultdict(lambda : defaultdict(list))
    for i, q in enumerate(queries):
        # if q['engine'] != 'yalm':
        #     continue
        k = (q['echo'], q['logprobs'], q['max_tokens'], q['n'], q['temperature'], q['best_of'], q['top_p'],
             (tuple(sorted(q['stop'])) if q['stop'] is not None else None)
            )
        clusters_by_engine[q['engine']][k].append(q)
        
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
        
    token_chunk_size = args.generate_token_chunk_size #1000 * 100
    
    for engine, clusters in clusters_by_engine.items():
        
        try:
            os.system(f"mkdir -p {os.path.join(args.output_dir, engine)}")
        except:
            pass
        
        global_request = 0
        i_count = 0
        for k, v in clusters.items():
            n_gen = max(k[2], args.min_generate_token) # this regards generate_tokens < 100 as 100
            chunk_size = token_chunk_size // n_gen
            v = sorted(v, key=lambda q: -len(q['prompt'].split()))
            for i in range(len(v) // chunk_size + 1):
                qs = v[i*chunk_size: (i+1)*chunk_size]
                if len(qs) == 0:
                    continue
                with open(
                    os.path.join(args.output_dir, engine, f'request_{global_request}.jsonl'), 'w'
                ) as f:
                    for q in qs:
                        f.write(json.dumps(q) + '\n')
                        i_count += 1
                global_request += 1
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input-file', type=str, default=None, required=True,
                        help='input data file')
    parser.add_argument('--output-dir', type=str, default=None, required=True,
                        help='output data dir')
    parser.add_argument('--min-generate-token', type=str, default=100,
                        help='regard generate_tokens < 100 as 100')
    parser.add_argument('--generate-token-chunk-size', type=int, default=100*1000,
                        help='generate token chunk size. E.g. if we want each file contain 1000 queries, each generates 100 tokens, then this argument should be 100*1000.')

    args = parser.parse_args()
    
    main(args)
