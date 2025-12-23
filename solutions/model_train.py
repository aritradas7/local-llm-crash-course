import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer

# ======================
# CONFIG
# ======================
#MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_ID = "D:/hf_models/LLaMA3"
DATASET_PATH = "sql_server_dba_instruction_dataset_30000.jsonl"
OUTPUT_DIR = "D:/hf_models/LLaMA3-SQLDBA"

# ======================
# QUANTIZATION (QLoRA)
# ======================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

# ======================
# LOAD DATASET (JSONL)
# ======================
dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train"
)

# ======================
# PROMPT FORMAT (ORCA STYLE)
# ======================
def format_prompt(example):
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
Explanation:
{example['response']['explanation']}

SQL Solution:
{example['response']['sql_code']}
"""

# ======================
# TRAINING CONFIG
# ======================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    report_to="none"
)

# ======================
# LoRA CONFIG
# ======================
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# ======================
# TRA
