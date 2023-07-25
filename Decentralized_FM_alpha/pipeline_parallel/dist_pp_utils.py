from .dist_gpipe_pipeline_async import GpipeAsync
from .dist_gpipe_pipeline_finetune_async import GpipeAsync as GpipeFinetuneAsync
from .dist_pipeline_inference_greedy import DistGreedyInferenceAsync
from .dist_pipeline_inference_greedy_sync import DistGreedyInferenceSync
from .dist_pipeline_inference_mask_greedy import DistGreedyInferenceMaskAsync
from .dist_pipeline_inference_mask_sample import DistSampleInferenceMaskAsync
from .dist_pipeline_enc_dec_inference_mask_sample import DistSampleEncDecInferenceMaskAsync
from .dist_pipeline_enc_dec_inference_mask_sample_token_pipe import DistSampleEncDecInferenceMaskSync
from .dist_pipeline_inference_greedy_token_pipe_sync import DistGreedyInferenceTokePipeSync
from .dist_pipeline_inference_mask_greedy_token_pipe_sync import DistGreedyInferenceMaskTokenPipeSync
from .dist_pipeline_inference_mask_sample_token_pipe_sync import DistSampleInferenceMaskTokenPipeSync
from .dist_pipeline_inference_mask_sample_token_sparse_pipe_sync import DistSampleInferenceMaskTokenSparsePipeSync
from .dist_hybrid_inference_greedy_token_deprecated import DistHybridGreedyInference
from .dist_hybrid_inference_greedy_token_async_deprecated import DistHybridGreedyAsyncInference
from .dist_pipeline_inference_batch_homo_mask_sample_token_pipe import DistInferenceMaskTokenPipeHomoBatch
from .dist_pipeline_inference_batch_auto_mask_sample_token_pipe import DistInferenceMaskTokenPipeAutoBatch

def get_pp_module(args, vocab_size, num_classes, device, use_dp, rank=None):
    if args.pp_mode == 'gpipe':
        return GpipeAsync(args, vocab_size, num_classes, device, use_dp, rank=rank)
    # elif args.pp_mode =='gpipe_ao':
    #    return GpipeAsyncOffload(args, vocab_size, num_classes, device, use_dp, rank=rank)
    # elif args.pp_mode == '1f1b':
    #    return Pipe1F1BAsync(args, vocab_size, num_classes, device, use_dp)
    else:
        print("Not recognize this pipeline parallel mode.")
        assert False
        
        
def get_pp_finetune_module(args, config, device, use_dp, rank=None):
    
    if args.model_type == 'gpt2':
        from modules.dist_hf_gpt2_pp_train_module import GPTStageFirst, GPTStageLast, GPTStageMiddle
    elif args.model_type == 'gptneo':
        from modules.dist_hf_gptneo_pp_train_module import GPTStageFirst, GPTStageLast, GPTStageMiddle
    else:
        print(f"Not recognize this model type {args.model_type}")
        assert False
    
    if args.pp_mode == 'gpipe':
        return GpipeFinetuneAsync(args, config, device, use_dp,
                                  _StageFirst=GPTStageFirst, _StageLast=GPTStageLast, _StageMiddle=GPTStageMiddle)
    else:
        print("Not recognize this pipeline parallel mode.")
        assert False


def get_pp_inference_module(args, device, rank=None, be_coordinated=False):
    if args.pp_mode == 'pipe_async_greedy':
        return DistGreedyInferenceAsync(args, device, rank=rank)
    elif args.pp_mode == 'pipe_sync_greedy':
        return DistGreedyInferenceSync(args, device, rank=rank)
    elif args.pp_mode == 'pipe_sync_greedy_token_pipe':
        return DistGreedyInferenceTokePipeSync(args, device, rank=rank)
    elif args.pp_mode == 'pipe_sync_greedy_mask_token_pipe':
        return DistGreedyInferenceMaskTokenPipeSync(args, device, rank=rank)
    elif args.pp_mode == 'pipe_sync_sample_mask_token_pipe':
        return DistSampleInferenceMaskTokenPipeSync(args, device, rank=rank, be_coordinated=be_coordinated)
    elif args.pp_mode == 'pipe_sync_sample_mask_token_sparse_pipe':
        return DistSampleInferenceMaskTokenSparsePipeSync(args, device, rank=rank, be_coordinated=be_coordinated)
    elif args.pp_mode == 'pipe_sync_sample_mask_token_pipe_refactored':
        return DistInferenceMaskTokenPipeHomoBatch(args, device)
    elif args.pp_mode == 'pipe_sync_sample_mask_token_pipe_auto_batch':
        return DistInferenceMaskTokenPipeAutoBatch(args, device)
    elif args.pp_mode == 'pipe_async_greedy_mask':
        return DistGreedyInferenceMaskAsync(args, device, rank=rank)
    elif args.pp_mode == 'pipe_async_sample_mask':
        return DistSampleInferenceMaskAsync(args, device, rank=rank)
    elif args.pp_mode == 'pipe_async_sample_enc_dec_mask':
        return DistSampleEncDecInferenceMaskAsync(args, device, rank=rank)
    elif args.pp_mode == 'pipe_sync_sample_enc_dec_mask':
        return DistSampleEncDecInferenceMaskSync(args, device, rank=rank)
    elif args.pp_mode == 'pipe_hybrid_greedy_sync':
        return DistHybridGreedyInference(args, device, rank=rank)
    elif args.pp_mode == 'pipe_hybrid_greedy_async':
        return DistHybridGreedyAsyncInference(args, device, rank=rank)
    else:
        print("Not recognize this pipeline parallel mode.")
        assert False
