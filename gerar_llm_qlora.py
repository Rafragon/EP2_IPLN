import pandas as pd
import numpy as np
import torch
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from scipy.special import softmax

# ==============================================================================
# 1. Preparação de Dados
# ==============================================================================
print("A carregar e limpar dados...")


def limpar_dados(df):
    df['resp_text'] = df['resp_text'].astype(
        str).str.replace(r'[^\w\s.,;:!?()-]', ' ', regex=True)
    df['resp_text'] = df['resp_text'].str.replace(
        r'\s+', ' ', regex=True).str.strip()
    return df


df_train = limpar_dados(pd.read_excel('train.xlsx').dropna(
    subset=['resp_text', 'clarity']).reset_index(drop=True))
df_test = limpar_dados(pd.read_excel(
    'test1-rotulado-para-teste-estatistico.xlsx'))

X_train_raw = df_train['resp_text'].tolist()
X_test_raw = df_test['resp_text'].tolist()

encoder = LabelEncoder()
y_train = encoder.fit_transform(df_train['clarity'].astype(str))

# ==============================================================================
# 2. Configurações do LLM e QLoRA
# ==============================================================================
NOME_MODELO = "Qwen/Qwen2.5-1.5B"

tokenizer = AutoTokenizer.from_pretrained(NOME_MODELO)
# LLMs decoders precisam de um pad_token explícito
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Quantização para 4-bits (Poupa muita VRAM na RTX 3060)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Configuração do LoRA (Treina apenas ~1% dos pesos do modelo)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS"
)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


def obter_modelo_compilado():
    modelo = AutoModelForSequenceClassification.from_pretrained(
        NOME_MODELO,
        num_labels=3,
        quantization_config=bnb_config,
        device_map="auto"
    )
    modelo.config.pad_token_id = tokenizer.pad_token_id
    modelo = prepare_model_for_kbit_training(modelo)
    modelo = get_peft_model(modelo, lora_config)
    return modelo


# ==============================================================================
# 3. Validação Cruzada (5-Fold OOF)
# ==============================================================================
print("\nIniciando Validação Cruzada 5-Fold com QLoRA...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
probabilidades_oof = np.zeros((len(X_train_raw), 3))

args_treino = TrainingArguments(
    output_dir="./tmp_llm",
    eval_strategy="no",
    save_strategy="no",
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Simula Batch Size de 16
    per_device_eval_batch_size=4,
    num_train_epochs=2,  # Apenas 2 épocas para não decorar o ruído
    weight_decay=0.01,
    fp16=True,
    report_to="none",
    logging_steps=50
)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_raw, y_train)):
    print(f"\n--- Treinando Fold {fold+1}/5 ---")

    X_f_train = [X_train_raw[i] for i in train_idx]
    y_f_train = y_train[train_idx]
    X_f_val = [X_train_raw[i] for i in val_idx]

    train_encodings = tokenizer(
        X_f_train, truncation=True, padding="max_length", max_length=256)
    val_encodings = tokenizer(X_f_val, truncation=True,
                              padding="max_length", max_length=256)

    dataset_train = TextDataset(train_encodings, y_f_train)
    dataset_val = TextDataset(val_encodings)

    modelo_fold = obter_modelo_compilado()

    trainer = Trainer(
        model=modelo_fold,
        args=args_treino,
        train_dataset=dataset_train
    )

    trainer.train()

    print("A gerar predições do fold...")
    pred_raw = trainer.predict(dataset_val).predictions
    probabilidades_oof[val_idx] = softmax(pred_raw, axis=1)

    del modelo_fold
    del trainer
    torch.cuda.empty_cache()

df_oof = pd.DataFrame(probabilidades_oof, columns=[
                      f'prob_llm_{c}' for c in encoder.classes_])
df_oof.to_csv('oof_llm.csv', index=False)

# ==============================================================================
# 4. Treino Final (100%) e Previsão do Teste 1
# ==============================================================================
print("\nTreinando LLM Final com 100% dos dados...")
train_encodings_full = tokenizer(
    X_train_raw, truncation=True, padding="max_length", max_length=256)
dataset_train_full = TextDataset(train_encodings_full, y_train)

modelo_final = obter_modelo_compilado()
trainer_final = Trainer(
    model=modelo_final,
    args=args_treino,
    train_dataset=dataset_train_full
)
trainer_final.train()

print("A prever o Teste 1...")
test_encodings = tokenizer(X_test_raw, truncation=True,
                           padding="max_length", max_length=256)
dataset_test = TextDataset(test_encodings)

probs_teste_raw = trainer_final.predict(dataset_test).predictions
probs_teste = softmax(probs_teste_raw, axis=1)

df_teste = pd.DataFrame(probs_teste, columns=[
                        f'prob_llm_{c}' for c in encoder.classes_])
df_teste.to_csv('prob_teste1_llm.csv', index=False)

print("\nConcluído! 'oof_llm.csv' e 'prob_teste1_llm.csv' guardados.")
