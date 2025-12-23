from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "microsoft/Orca-2-7b"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=False
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)


prompt = "Explain transformers like I'm five."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
