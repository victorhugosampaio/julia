import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- CONFIGURAÇÕES ---
MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"
DATASET_PATH = "database_v1.json"
OUTPUT_DIR = "phi3-julia-lora-4gb-gpu"

# 1. CONFIGURAÇÃO DE QUANTIZAÇÃO (Essencial para 4GB)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 2. CARREGAR TOKENIZER E MODELO
print("Carregando o modelo...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager" # Compatível com GTX série 16
)

# Preparar para treinamento de baixa precisão
model = prepare_model_for_kbit_training(model)

# 3. CONFIGURAÇÃO DO LORA
lora_config = LoraConfig(
    r=8, 
    lora_alpha=16,
    target_modules=["qkv_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 4. PROCESSAMENTO DO DATASET
print("Processando o dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

def format_phi3(example):
    # Formata seguindo o template do Phi-3 usando suas colunas input e output
    text = f"<|user|>\n{example['input']}<|end|>\n<|assistant|>\n{example['output']}<|end|>"
    return {"text": text}

# Aplica a formatação
dataset = dataset.map(format_phi3)

def tokenize_function(examples):
    # max_length reduzido para 256 para economizar memória na GTX 1650
    return tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=256, 
        padding="max_length"
    )

# Tokeniza e remove todas as colunas antigas (input, output, command)
tokenized_dataset = dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=dataset.column_names
)

# 5. ARGUMENTOS DE TREINAMENTO (Otimizados para VRAM baixa)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    max_steps=100, # Ajuste o número de passos conforme necessário
    save_strategy="steps",
    save_steps=50,
    optim="paged_adamw_32bit",
    gradient_checkpointing=True,
    report_to="none",
    remove_unused_columns=False
)

# 6. INICIALIZAR O TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 7. INICIAR TREINO
print("Iniciando o treinamento na GPU...")
model.config.use_cache = False 
trainer.train()

# 8. SALVAR O RESULTADO
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Treinamento concluído! Adaptador salvo em: {OUTPUT_DIR}")