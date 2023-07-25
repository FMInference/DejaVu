from transformers import AutoTokenizer
from llm_serving.model.wrapper import get_model

# Load the tokenizer. All OPT models with different sizes share the same tokenizer
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")
tokenizer.add_bos_token = False

# Load the model. Alpa automatically downloads the weights to the specificed path
model = get_model(model_name="alpa/bloom", path="/root/fm/models/opt_weights/")

# Generate
prompt = "Paris is the capital city of"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
output = model.generate(input_ids=input_ids, max_length=256, do_sample=True)
generated_string = tokenizer.batch_decode(output, skip_special_tokens=True)

print(generated_string)