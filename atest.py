from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch

# Use 'mps' for Apple Silicon GPU, or 'cpu' if MPS is not available
device = "mps" if torch.backends.mps.is_available() else "cpu"

model_path = "ibm-granite/granite-3.3-2b-instruct"

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if device == "mps" else torch.float32,
)
model.to(device)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Prepare conversation
conv = [{
    "role": "user",
    "content": "Give me a code to use langchain with ibm granite model using tranformers and how can I use it?"
}]

# Tokenize input
input_tensor = tokenizer.apply_chat_template(
    conv,
    return_tensors="pt",
    add_generation_prompt=True
).to(device)

# Set seed and generate output
set_seed(42)
output = model.generate(
    input_ids=input_tensor,
    max_new_tokens=512,
)

# Decode and print result
prediction = tokenizer.decode(output[0], skip_special_tokens=True)
print(prediction)

