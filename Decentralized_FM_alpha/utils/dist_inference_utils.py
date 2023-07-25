import json
from comm.comm_utils import *
from comm.hybrid_comm_utils import *

from task_datasets.inference_data import get_tokenizer
from flask import Flask, request
import threading
from coordinator.http_coordinate_client import CoordinatorInferenceHTTPClient


def distributed_inference_foo_iter(args, pipeline, device, request_processor,
                                   client: CoordinatorInferenceHTTPClient = None):
    total_time = 0
    if get_pipeline_parallel_rank() == 0:
        output_requests = []
        infer_data_loader = request_processor.get_dataloader(args.batch_size)
        for i, inputs in enumerate(infer_data_loader):
            input_ids = inputs['text'].to(device)
            output_ids_list = []
            current_iter_time = pipeline.inference_batch(input_ids, output_ids_list)
            request_processor.add_result(inputs, output_ids_list)
            if client is not None:
                client.update_status("running", returned_payload=
                {'progress': {'finished':i+1, 'total':len(infer_data_loader)}})
            
            if i > 0:
                total_time += current_iter_time
            if i >= args.num_iters-1:
                break
        averaged_time = total_time / (args.num_iters - 1 + 1e-9)
        print("Finished running ", args.num_iters,
              " iterations, averaged (exclude the first iter) run time:", averaged_time)
        # request_processor.write_scenario_state()
    else:
        i = 0
        while True:
            current_iter_time = pipeline.inference_batch(None)
            if i > 0:
                total_time += current_iter_time
            if i >= args.num_iters - 1:
                break
            i += 1
        averaged_time = total_time / (args.num_iters - 1 + 1e-9)
    return averaged_time


def distributed_inference_mask_iter(args, pipeline, device, request_processor,
                                    client: CoordinatorInferenceHTTPClient = None):
    
    total_time = 0
    if get_pipeline_parallel_rank() == 0:
        output_requests = []
        infer_data_loader = request_processor.get_dataloader(args.batch_size)
        for i, inputs in enumerate(infer_data_loader):
            input_ids = inputs['text'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            output_ids_list = []
            current_iter_time = pipeline.inference_batch(input_ids, output_ids_list, attention_mask=attention_mask)
            if i > 0:
                total_time += current_iter_time
            if i >= args.num_iters-1:
                break
        averaged_time = total_time / (args.num_iters - 1 + 1e-9)
        print("Finished running ", args.num_iters,
              " iterations, averaged (exclude the first iter) run time:", averaged_time)
            
    elif get_pipeline_parallel_rank() == pipeline.pipeline_group_size - 1:
        infer_data_loader = request_processor.get_dataloader(args.batch_size)
        for i, inputs in enumerate(infer_data_loader):
            input_ids = inputs['text'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            output_ids_list = []
            current_iter_time = pipeline.inference_batch(input_ids, output_ids_list, attention_mask=attention_mask)
            request_processor.add_result(inputs, output_ids_list, batch_time=current_iter_time)
            
            if client is not None:
                client.update_status("running", returned_payload=
                {'progress': {'finished': i + 1, 'total': len(infer_data_loader)}})
                
            if i > 0:
                total_time += current_iter_time
            if i >= args.num_iters-1:
                break
        averaged_time = total_time / (args.num_iters - 1 + 1e-9)
        
        # last node write
        request_processor.write_scenario_state()
            
    else:
        infer_data_loader = request_processor.get_dataloader(args.batch_size)
        for i, inputs in enumerate(infer_data_loader):
            input_ids = inputs['text'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            current_iter_time = pipeline.inference_batch(input_ids, attention_mask=attention_mask)
            if i > 0:
                total_time += current_iter_time
            if i >= args.num_iters-1:
                break
        averaged_time = total_time / (args.num_iters - 1 + 1e-9)
        
    return averaged_time


def distributed_hybrid_inference_foo_iter(args, pipeline, device, request_processor):
    total_time = 0
    if args.node_type == 'GPU' and get_gpu_pipeline_rank() == 0:
        output_requests = []
        batch_size = args.prompt_micro_batch_size * args.producer_buffer_size
        infer_data_loader = request_processor.get_dataloader(batch_size)
        for i, inputs in enumerate(infer_data_loader):
            input_ids = inputs['text'].to(device)
            output_ids_list = []
            current_iter_time = pipeline.inference_batch(input_ids, output_ids_list)
            request_processor.add_result(inputs, output_ids_list)

            if i > 0:
                total_time += current_iter_time
            if i >= args.num_iters - 1:
                break
        averaged_time = total_time / (args.num_iters - 1 + 1e-9)
        print("Finished running ", args.num_iters,
              " iterations, averaged (exclude the first iter) run time:", averaged_time)

        # request_processor.write_scenario_state()

    else:
        i = 0
        while True:
            current_iter_time = pipeline.inference_batch(None)
            if i > 0:
                total_time += current_iter_time
            if i >= args.num_iters - 1:
                break
            i += 1
        averaged_time = total_time / (args.num_iters - 1 + 1e-9)
    return averaged_time


def distributed_inference_mask_server(args, pipeline, device):
    
    tokenizer = get_tokenizer(args)
    tokenizer.model_max_length = args.input_seq_length
    
    input_ids = torch.ones([args.batch_size, args.input_seq_length]).long().cuda()
    attention_mask = torch.ones([args.batch_size, args.input_seq_length]).long().cuda()
    
    total_time = 0
    if get_pipeline_parallel_rank() == 0:
        
        app = Flask(__name__)
        sem = threading.Semaphore()
        
        @app.route('/', methods=['GET'])
        def hello_world():
            return '<p>The inference system is up.</p>'
        
        @app.route('/', methods=['POST'])
        def process_input():
            sem.acquire() # lock
            
            query = request.json
            parameters = query.get('parameters', {})
            
            # parameter TODO: current ignore
            generate_token_length = parameters.get('max_new_tokens', 10)
            return_full_text = parameters.get('return_full_text', False)
            do_sample = parameters.get('do_sample', True)
            temperature = parameters.get('temperature', 1)
            top_p = parameters.get('top_p', 1.0)
            num_return_sequences = parameters.get('num_return_sequences', 1)
            
            _tmp = torch.zeros(1, dtype=torch.int64, device=device)
            _tmp[:] = num_return_sequences
            pipeline.comm.broadcast(_tmp, src=0)
            pipeline.num_completions = _tmp.item()
            
            _tmp = torch.zeros(1, dtype=torch.int64, device=device)
            _tmp[:] = generate_token_length
            pipeline.comm.broadcast(_tmp, src=0)
            pipeline.generate_seq_length = _tmp.item()
            
            _tmp = torch.zeros(1, dtype=torch.float32, device=device)
            _tmp[:] = temperature
            pipeline.comm.broadcast(_tmp, src=0)
            args.temperature = _tmp.item()
            
            _tmp = torch.zeros(1, dtype=torch.float32, device=device)
            _tmp[:] = top_p
            pipeline.comm.broadcast(_tmp, src=0)
            args.top_p = _tmp.item()
            
            _tmp = torch.zeros(1, dtype=torch.uint8, device=device)
            _tmp[:] = do_sample
            pipeline.comm.broadcast(_tmp, src=0)
            if _tmp.item() == 0:
                args.temperature = 0
                
            pipeline.update_processors(args)
            #####
            
            inputs = tokenizer(query['inputs'], return_tensors='pt', 
                               padding='max_length', truncation=True,)
            
            input_ids = inputs['input_ids'].long().to(device)
            attention_mask = inputs['attention_mask'].long().to(device)
            
            pipeline.comm.broadcast(input_ids, src=0)
            pipeline.comm.broadcast(attention_mask, src=0)
            
            output_ids_list = []
            current_iter_time = pipeline.inference_batch(input_ids, output_ids_list, attention_mask=attention_mask)
            
            # TODO: force all communications in nccl, optimize later
            results = []
            for i in range(pipeline.num_completions):
                token_len = torch.zeros([1], dtype=torch.int64).cuda()
                pipeline.comm.recv(token_len, src=pipeline.pipeline_group_size - 1)
                result = torch.empty((1, token_len.item()), dtype=torch.long).cuda()
                pipeline.comm.recv(result, src=pipeline.pipeline_group_size - 1)
                if return_full_text:
                    results.append(query['inputs'] + tokenizer.decode(result[0]))
                else:
                    results.append(tokenizer.decode(result[0]))
            
            sem.release() # unlock
            
            return json.dumps(results)
            
        app.run(host='0.0.0.0', port=5001)
            
    elif get_pipeline_parallel_rank() == pipeline.pipeline_group_size - 1:

        while True:

            _tmp = torch.zeros(1, dtype=torch.int64, device=device)
            pipeline.comm.broadcast(_tmp, src=0)
            pipeline.num_completions = _tmp.item()

            _tmp = torch.zeros(1, dtype=torch.int64, device=device)
            pipeline.comm.broadcast(_tmp, src=0)
            pipeline.generate_seq_length = _tmp.item()

            _tmp = torch.zeros(1, dtype=torch.float32, device=device)
            pipeline.comm.broadcast(_tmp, src=0)
            args.temperature = _tmp.item()

            _tmp = torch.zeros(1, dtype=torch.float32, device=device)
            pipeline.comm.broadcast(_tmp, src=0)
            args.top_p = _tmp.item()

            _tmp = torch.zeros(1, dtype=torch.uint8, device=device)
            pipeline.comm.broadcast(_tmp, src=0)
            if _tmp.item() == 0:
                args.temperature = 0

            pipeline.update_processors(args)
            
            pipeline.comm.broadcast(input_ids, src=0)
            pipeline.comm.broadcast(attention_mask, src=0)
            
            output_ids_list = []
            current_iter_time = pipeline.inference_batch(input_ids, output_ids_list, attention_mask=attention_mask)
            
            for i in range(pipeline.num_completions):
                result = output_ids_list[i]
                token_len = torch.tensor(result['token_ids'].size(1), dtype=torch.long).cuda()
                pipeline.comm.send(token_len, dst=0)
                pipeline.comm.send(result['token_ids'].cuda(), dst=0)
            # s.send((json.dumps(output_ids_list)).encode())
            
    else:
        while True:

            _tmp = torch.zeros(1, dtype=torch.int64, device=device)
            pipeline.comm.broadcast(_tmp, src=0)
            pipeline.num_completions = _tmp.item()

            _tmp = torch.zeros(1, dtype=torch.int64, device=device)
            pipeline.comm.broadcast(_tmp, src=0)
            pipeline.generate_seq_length = _tmp.item()

            _tmp = torch.zeros(1, dtype=torch.float32, device=device)
            pipeline.comm.broadcast(_tmp, src=0)
            args.temperature = _tmp.item()

            _tmp = torch.zeros(1, dtype=torch.float32, device=device)
            pipeline.comm.broadcast(_tmp, src=0)
            args.top_p = _tmp.item()

            _tmp = torch.zeros(1, dtype=torch.uint8, device=device)
            pipeline.comm.broadcast(_tmp, src=0)
            if _tmp.item() == 0:
                args.temperature = 0

            pipeline.update_processors(args)
            
            pipeline.comm.broadcast(input_ids, src=0)
            pipeline.comm.broadcast(attention_mask, src=0)
            current_iter_time = pipeline.inference_batch(input_ids, attention_mask=attention_mask)
        
    return 0
