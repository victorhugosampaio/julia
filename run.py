# run_cpu_offload.py
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# ============================
# 1️⃣ CONFIGURAÇÕES
# ============================
BASE_MODEL = "Phi-3-mini-128k-instruct"
LORA_PATH = "phi3-julia-lora-4gb-gpu"
DEVICE = "cpu"
TORCH_DTYPE = torch.float32
OFFLOAD_DIR = "offload"  # pasta para offload do PEFT

os.makedirs(OFFLOAD_DIR, exist_ok=True)

# Depuração
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

print(f"🚀 Iniciando agente Julia em {DEVICE}")
print(f"   Modelo base: {BASE_MODEL}")
print(f"   Pesos LoRA: {LORA_PATH}\n")

# ============================
# 2️⃣ CARREGAR TOKENIZER
# ============================
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("✅ Tokenizador carregado")

# ============================
# 3️⃣ CARREGAR MODELO BASE + LORA COM OFFLOAD
# ============================
print("🔄 Carregando modelo base...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=TORCH_DTYPE,
    device_map={"": DEVICE},  # força CPU
    trust_remote_code=True,
    use_cache=True
)

print("🔄 Aplicando pesos LoRA com offload...")
model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
    device_map={"": DEVICE},
    offload_folder=OFFLOAD_DIR
)
model.eval()
model = model.to(DEVICE)
print("✅ Modelo carregado com sucesso!\n")

# ============================
# 4️⃣ FUNÇÃO DE GERAÇÃO DE RESPOSTA
# ============================
def gerar_resposta(pergunta, max_new_tokens=128):
    prompt = f"""
[INSTRUÇÃO]
Você é Julia, assistente doméstico inteligente. Responda de forma clara. Se houver ação, indique:

Comando: <json>
Assistente: <fala>

Usuário: {pergunta}
Assistente:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,       # sampling para respostas mais naturais
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    resposta_completa = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Assistente:" in resposta_completa:
        resposta = resposta_completa.split("Assistente:")[1].strip()
    else:
        resposta = resposta_completa
    return resposta

# ============================
# 5️⃣ FUNÇÃO PARA EXTRAIR COMANDO JSON
# ============================
def extrair_comando(resposta):
    if "Comando:" in resposta:
        parte_comando = resposta.split("Comando:")[1].strip()
        try:
            return json.loads(parte_comando)
        except:
            return None
    return None

# ============================
# 6️⃣ LOOP INTERATIVO NO CMD
# ============================
print("="*60)
print("🎤 Assistente Julia - CMD Interativo (CPU + Offload)")
print("Digite 'sair' para encerrar")
print("="*60 + "\n")

while True:
    pergunta = input("Você: ").strip()
    if pergunta.lower() in ["sair", "exit", "quit"]:
        print("👋 Até mais!")
        break
    if not pergunta:
        continue

    try:
        resposta = gerar_resposta(pergunta)
        print(f"Julia: {resposta}\n")

        comando = extrair_comando(resposta)
        if comando:
            print(f"[Detectada Ação Tuya]: {json.dumps(comando, ensure_ascii=False)}\n")

    except Exception as e:
        print(f"❌ Erro: {e}")
        torch.cuda.empty_cache()