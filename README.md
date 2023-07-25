# Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time

This repo is consisting of two parts: (1) Training sparsity predictor (2) End-to-End Accuracy Benchmark (3) Generation Latency Benchmark.

## Training sparsity predictor
We collect training data by running model inference using Decentralized_FM_alpha. 

**Requirements**
    pip3 install --pre torch==1.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
    pip3 install cupy-cuda11x==11.0.0
    python3 -m cupyx.tools.install_library --cuda 11.x --library nccl
    pip3 install transformers

**Collect the training data**

To get started, you need to first collect the training data by runing model inference over c4 

```
DejaVu/Decentralized_FM_alpha/run_infer_opt_175b_collect_sp_data.sh
```
You need to specify the model checkpoint and data path. To get data, we provide the a script in DejaVu/Decentralized_FM_alpha/c4_train. And to convert the model checkpoint from huggingface, we provide a script in DejaVu/Decentralized_FM_alpha/convert_opt_checkpoint.py

Also, you can specify where to store the training data inside DejaVu/Decentralized_FM_alpha/modules/hf_opt_module_save.py 

**Training the sparsity classifier**

All code related to training sparsity predictor is located in DejaVu/sparse_predictor.

We provide two script, one for training attention sparsity predictor DejaVu/sparse_predictor/run_c4_att.sh, one for training MLP sparsity predictor DejaVu/sparse_predictor/trainer_mlp.py. 

For detailed instruction, see DejaVu/sparse_predictor/README.md


## Accuracy Benchmark
We based our accuracy benchmark based on Decentralized_FM_alpha(https://github.com/DS3Lab/Decentralized_FM_alpha)

**Requirements**
    pip3 install --pre torch==1.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
    pip3 install cupy-cuda11x==11.0.0
    python3 -m cupyx.tools.install_library --cuda 11.x --library nccl
    pip3 install transformers

**Collect Training Data to train sparsity classifier.**

### Usage and Examples

### Perplexity on c4

To run evaluation using dense model, run 
```DejaVu/Decentralized_FM_alpha/run_infer_opt_175b_c4.sh```

To run evaluation using DejaVu model, run
```DejaVu/Decentralized_FM_alpha/run_infer_opt_175b_c4_sparse.sh```

Similar to collecting the data, you will need to specify 
(1) the model checkpoint path
(2) the sparsity predictor checkpoint path
(3) c4 data path
