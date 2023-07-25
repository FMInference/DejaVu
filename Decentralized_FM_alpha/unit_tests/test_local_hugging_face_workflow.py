import argparse
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
import torch


def get_huggingface_tokenizer_model(args, device):
    if args.model_name == 't5-11b':
        tokenizer = AutoTokenizer.from_pretrained('t5-11b')
        model = T5ForConditionalGeneration.from_pretrained('t5-11b')
    elif args.model_name == 't0pp':
        tokenizer = AutoTokenizer.from_pretrained('bigscience/T0pp')
        model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp")
    elif args.model_name == 'ul2':
        tokenizer = AutoTokenizer.from_pretrained('google/ul2')
        model = T5ForConditionalGeneration.from_pretrained("google/ul2")
    elif args.model_name == 'gpt-j-6b':
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    elif args.model_name == 'gpt-neo-1.3B':
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    else:
        assert False, "Model not supported yet."
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    if args.fp16:
        model = model.half()
    model = model.to(device)
    return tokenizer, model


def main():

    # TODO: we assume len(payload) is 1, right?



    parser = argparse.ArgumentParser(description='Local Inference Runner with coordinator.')
    parser.add_argument('--job_id', type=str, default='-', metavar='S',
                        help='DB ID')
    parser.add_argument('--working-directory', type=str,
                        default='/cluster/scratch/biyuan/fetch_cache', metavar='S',
                        help='The IP of coordinator-server.')
    parser.add_argument('--model-name', type=str, default='gpt-neo-1.3B', metavar='S',
                        help='trained model path')
    parser.add_argument('--cuda-id', type=int, default=0, metavar='S',
                        help='cuda-id (default:0)')
    parser.add_argument('--temperature', type=float, default=0.9, metavar='N',
                        help='-')
    parser.add_argument('--top-p', type=float, default=0.0, metavar='N',
                        help='-')
    parser.add_argument('--max-tokens', type=int, default=32, metavar='N',
                        help='-')
    parser.add_argument('--fp16', action='store_true',
                        help='Run model in fp16 mode.')

    args = parser.parse_args()
    print(args)
    assert (torch.cuda.is_available())
    device = torch.device('cuda', args.cuda_id)

    tokenizer, model = get_huggingface_tokenizer_model(args, device)

    prompt = ["raw text", "where is Houston"]
    inputs = tokenizer(
        prompt,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    inputs.to(device)

    outputs = model.generate(
        **inputs, do_sample=True, top_p=None,
        temperature=1.0, top_k=1,
        max_new_tokens=args.max_tokens,
        return_dict_in_generate=True,
        output_scores=True,  # return logit score
        output_hidden_states=True,  # return embeddings
    )

    texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    print(texts)


if __name__ == '__main__':
    main()
