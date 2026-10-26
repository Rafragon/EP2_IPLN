import os
import sys
import pandas as pd
import numpy as np
import torch
import gc
import glob
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

if os.path.exists('oof_semantico.csv'):
    print("\n[SUCESSO] O arquivo 'oof_semantico.csv' já existe!")
    print("O script foi interrompido automaticamente para evitar re-treino.")
    sys.exit(0)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {'accuracy': accuracy_score(labels, np.argmax(predictions, axis=1))}


print("Carregando dados para o BERTimbau Large...")
df_train = pd.read_excel('train.xlsx')
df_train['resp_text'] = df_train['resp_text'].fillna('').astype(str)

X = np.array(df_train['resp_text'].tolist())
encoder = LabelEncoder()
y = encoder.fit_transform(df_train['clarity'])

# Alteração para a versão Large
nome_modelo = 'neuralmind/bert-large-portuguese-cased'
tokenizer = AutoTokenizer.from_pretrained(nome_modelo)

probabilidades_oof = np.zeros((len(X), len(encoder.classes_)))
dispositivo_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*50}\nA INICIAR FOLD {fold + 1} / 5 (BERT LARGE)\n{'='*50}")

    pasta_saida = f'./resultados_bert_large_oof_fold_{fold+1}'
    checkpoints_existentes = sorted(
        glob.glob(f"{pasta_saida}/checkpoint-*"), key=lambda x: int(x.split('-')[-1]))

    X_train, X_val = X[train_idx].tolist(), X[val_idx].tolist()
    y_train, y_val = y[train_idx], y[val_idx]

    train_encodings = tokenizer(
        X_train, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(X_val, truncation=True,
                              padding=True, max_length=512)

    train_dataset = TextDataset(train_encodings, y_train)
    val_dataset = TextDataset(val_encodings, y_val)

    if len(checkpoints_existentes) >= 2:
        print(
            f"[!] Checkpoints detectados para o Fold {fold + 1}. A saltar o treino...")
        ckpt_ep2 = checkpoints_existentes[-2]
        ckpt_ep3 = checkpoints_existentes[-1]

        modelo_base = AutoModelForSequenceClassification.from_pretrained(
            nome_modelo, num_labels=3, use_safetensors=True
        )
        # Reduzido batch de avaliação de 8 para 2 devido ao tamanho do modelo Large
        args_vazios = TrainingArguments(
            output_dir=pasta_saida, per_device_eval_batch_size=2, report_to="none")
        trainer = Trainer(model=modelo_base, args=args_vazios)

    else:
        print(f"Carregando o modelo Large do zero para o Fold {fold + 1}...")
        modelo = AutoModelForSequenceClassification.from_pretrained(
            nome_modelo, num_labels=3, use_safetensors=True
        )

        # Otimizações severas de VRAM para a RTX 3060 aguentar o BERT Large
        training_args = TrainingArguments(
            output_dir=pasta_saida,
            num_train_epochs=3,
            per_device_train_batch_size=2,       # Reduzido de 8 para 2
            # Aumentado para manter o batch efetivo em 32 (2x16)
            gradient_accumulation_steps=16,
            per_device_eval_batch_size=2,        # Reduzido de 8 para 2
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            learning_rate=1e-5,                  # LR menor é mais estável para modelos Large
            weight_decay=0.01,
            warmup_steps=314,
            label_smoothing_factor=0.1,          # Penaliza excesso de confiança no ruído
            # Ativa precisão mista (essencial para poupar VRAM e acelerar na RTX)
            fp16=True
        )

        trainer = Trainer(
            model=modelo,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

        trainer.train()

        checkpoints_existentes = sorted(
            glob.glob(f"{pasta_saida}/checkpoint-*"), key=lambda x: int(x.split('-')[-1]))
        if len(checkpoints_existentes) >= 2:
            ckpt_ep2 = checkpoints_existentes[-2]
            ckpt_ep3 = checkpoints_existentes[-1]
        else:
            ckpt_ep2 = ckpt_ep3 = checkpoints_existentes[-1]

    print(
        f"\nExtraindo predições das Épocas 2 e 3 para a validação do Fold {fold + 1}...")

    trainer.model = AutoModelForSequenceClassification.from_pretrained(
        ckpt_ep2, num_labels=3).to(dispositivo_gpu)
    probs_ep2 = softmax(trainer.predict(val_dataset).predictions, axis=1)

    trainer.model = AutoModelForSequenceClassification.from_pretrained(
        ckpt_ep3, num_labels=3).to(dispositivo_gpu)
    probs_ep3 = softmax(trainer.predict(val_dataset).predictions, axis=1)

    probs_fold = (probs_ep2 + probs_ep3) / 2.0
    probabilidades_oof[val_idx] = probs_fold

    del trainer
    if 'modelo' in locals():
        del modelo
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\nProcessamento Concluído! A guardar a matriz Out-of-Fold...")
df_oof = pd.DataFrame(probabilidades_oof, columns=[
                      f'prob_semantico_{c}' for c in encoder.classes_])
df_oof.to_csv('oof_semantico.csv', index=False)

predicoes_oof = np.argmax(probabilidades_oof, axis=1)
print(
    f"\nAcurácia do Especialista Semântico (BERT Large OOF): {accuracy_score(y, predicoes_oof):.4f}")
