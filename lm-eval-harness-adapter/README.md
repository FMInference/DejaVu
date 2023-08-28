# LM-eval

This is an adapter of `lm-evaluation-harness`.

# QuickStart

1.Generate Input Data

```bash
python generate_task_data.py --output-file wsc.jsonl --task-name wsc --num-fewshot 0
```

2.Eval Outputs

```bash
python evaluate_task_result.py --result-file wsc_out.jsonl --task-name wsc --num-fewshot 0 --model-type opt
```
