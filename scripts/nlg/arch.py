"""Print the architecture of the Llama-3.1-8B model."""

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    # "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-8B",
    device_map="meta",
)
print(model)
