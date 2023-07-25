from transformers import AutoTokenizer
from llm_serving.model.wrapper import get_model
from alpa.device_mesh import set_seed



set_seed(2006)

# Load the tokenizer. All OPT models with different sizes share the same tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
tokenizer.add_bos_token = False

# Generate
prompt = ["Paris is the capital city of", "Zurich is the capital city of"]
input_ids = tokenizer(prompt,padding=True,truncation=True, return_tensors="pt").input_ids

# Load the model. Alpa automatically downloads the weights to the specificed path
model = get_model(batch_size=2, model_name="alpa/opt-2.7b", path="/root/fm/models/opt_weights/")

output = model.generate(input_ids=input_ids, max_length=256, do_sample=True)
generated_string = tokenizer.batch_decode(output, skip_special_tokens=True)

print(generated_string)