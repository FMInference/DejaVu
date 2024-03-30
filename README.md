# Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time

Large language models (LLMs) with hundreds of billions of parameters have sparked a new wave of exciting AI applications. However, they are computationally expensive at inference time. Sparsity is a natural approach to reduce this cost, but existing methods either require costly retraining, have to forgo LLM’s in-context learning ability, or do not yield wall-clock time speedup on modern hardware. We hypothesize that contextual sparsity, which are small, input-dependent sets of attention heads and MLP parameters that yield approximately the same output as the dense model for a given input, can address these issues. We show that contextual sparsity exists, that it can be accurately predicted, and that we can exploit it to speed up LLM inference in wall-clock time without compromising LLM’s quality or in-context learning ability. Based on these insights, we propose DejaVu, a system that uses a low-cost algorithm to predict contextual sparsity on the fly given inputs to each layer, along with an asynchronous and hardware-aware implementation that speeds up LLM inference. We validate that DejaVu can reduce the inference latency of OPT-175B by over 2×
 compared to the state-of-the-art FasterTransformer, and over 6×
 compared to the widely used Hugging Face implementation, without compromising model quality. The code is available at https://github.com/FMInference/DejaVu.

Paper Link: https://proceedings.mlr.press/v202/liu23am.html


This repo is consisting of three parts: (1) Training sparsity predictor (2) End-to-End Accuracy Benchmark (3) Generation Latency Benchmark.

## Training sparsity predictor
We collect training data by running model inference using Decentralized_FM_alpha. 

**Requirements**


```
    pip3 install --pre torch==1.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
    pip3 install cupy-cuda11x==11.0.0
    python3 -m cupyx.tools.install_library --cuda 11.x --library nccl
    pip3 install transformers
```

**Collect the training data**

To get started, you need to first collect the training data by runing model inference over c4 

```
DejaVu/Decentralized_FM_alpha/run_infer_opt_175b_collect_sp_data.sh
```
You need to specify the model checkpoint and data path. To get data, we provide the a script in DejaVu/Decentralized_FM_alpha/c4_train/get_data.py. By default, we sumsample 500 samples in the script. And to convert the model checkpoint from huggingface, we provide a script in DejaVu/Decentralized_FM_alpha/convert_opt_checkpoint.py

Also, you can specify where to store the training data inside DejaVu/Decentralized_FM_alpha/modules/hf_opt_module_save.py 

**Training the sparsity classifier**

All code related to training sparsity predictor is located in DejaVu/sparse_predictor.

We provide two script, one for training attention sparsity predictor DejaVu/sparse_predictor/run_c4_att.sh, one for training MLP sparsity predictor DejaVu/sparse_predictor/trainer_mlp.py. 

For detailed instruction, see DejaVu/sparse_predictor/README.md


## Accuracy Benchmark
We based our accuracy benchmark based on Decentralized_FM_alpha(https://github.com/DS3Lab/Decentralized_FM_alpha)

**Requirements**

```
    pip3 install --pre torch==1.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
    pip3 install cupy-cuda11x==11.0.0
    python3 -m cupyx.tools.install_library --cuda 11.x --library nccl
    pip3 install transformers
```

**Perplexity on c4**

To run evaluation using dense model, run 
```DejaVu/Decentralized_FM_alpha/run_infer_opt_175b_c4.sh```

To run evaluation using DejaVu model, run
```DejaVu/Decentralized_FM_alpha/run_infer_opt_175b_c4_sparse.sh```

Similar to collecting the data, you will need to specify 
(1) the model checkpoint path
(2) the sparsity predictor checkpoint path
(3) c4 data path

**Accuracy on downstream task**

We adopt lm-evaluation-harness for downstream task evaluation. 

1. Generate task data
```
cd lm-eval-harness-adapter
python generate_task_data.py --output-file wsc.jsonl --task-name wsc --num-fewshot 0
```

2. Run evaluation
```DejaVu/Decentralized_FM_alpha/run_infer_opt_175b_task_sparse.sh```

3. Evaluate model output
```
cd lm-eval-harness-adapter
python evaluate_task_result.py --result-file output_wsc.jsonl --task-name wsc --num-fewshot 0 --model-type opt
```

## Generation Latency
We provide pytorch based implementation that exploits cuda graph. 

**Requirements**

For best performance, please use docker. We provide a dockerfile with all requirement at DejaVu/Dejavu/Dockerfile

**Dense Model Latency Benchmark**
To benchmark latency with dense model, run

```torchrun --nproc_per_node=$NUM_GPUs benchmark_generation_opt.py --model-name $MODEL_NAME ```

Please specify the model checkpoint in DejaVu/Dejavu/benchmarks/benchmark_generation_opt.py with correspondence to $MODEL_NAME


**Sparse Model Latency Benchmark**

Sparse MLP Block
To benchmark latency with sparse MLP block model, run

```torchrun --nproc_per_node=$NUM_GPUs benchmark_generation_opt_mlp_sparse.py --model-name $MODEL_NAME --mlp-K $NUM_ACTIVE_NEURONS```

Please specify the model checkpoint in DejaVu/Dejavu/benchmarks/benchmark_generation_opt.py with correspondence to $MODEL_NAME
$NUM_ACTIVE_NEURONS indicate how many neurons to activate in the first fully connected layer in each MLP block. 

For example, for OPT-175B, mlp-k is set to 49152 by default, which will perform dense computation. Set mlp-K 4096 will perform sparse computation. We recommend setting mlp-K to be multiplied by 128.

Sparse Attention Blcok
Coming soon



## Citation

```
@InProceedings{pmlr-v202-liu23am,
  title = 	 {Deja Vu: Contextual Sparsity for Efficient {LLM}s at Inference Time},
  author =       {Liu, Zichang and Wang, Jue and Dao, Tri and Zhou, Tianyi and Yuan, Binhang and Song, Zhao and Shrivastava, Anshumali and Zhang, Ce and Tian, Yuandong and Re, Christopher and Chen, Beidi},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {22137--22176},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/liu23am/liu23am.pdf},
  url = 	 {https://proceedings.mlr.press/v202/liu23am.html},
}
```
