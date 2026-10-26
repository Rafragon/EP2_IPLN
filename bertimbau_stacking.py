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

# 1. Verificação Global: O script já foi concluído antes?
if os.path.exists('oof_semantico.csv'):
    print("\n[SUCESSO] O arquivo 'oof_semantico.csv' já existe!")
    print("O script foi interrompido automaticamente para evitar o re-treino desnecessário.")
    sys.exit(0)

# 2. Classe de Dataset para o PyTorch


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


# 3. Carregamento dos Dados
print("A carregar os dados de treino...")
df_train = pd.read_excel('train.xlsx')
df_train['resp_text'] = df_train['resp_text'].fillna('').astype(str)

X = np.array(df_train['resp_text'].tolist())
encoder = LabelEncoder()
y = encoder.fit_transform(df_train['clarity'])

nome_modelo = 'neuralmind/bert-base-portuguese-cased'
tokenizer = AutoTokenizer.from_pretrained(nome_modelo)

probabilidades_oof = np.zeros((len(X), len(encoder.classes_)))

# Identificação manual e segura da GPU para evitar o erro de tensores dispersos
dispositivo_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 4. Validação Cruzada (5-Folds) com Retoma Automática
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n{'='*50}\nA INICIAR FOLD {fold + 1} / 5\n{'='*50}")

    pasta_saida = f'./resultados_bert_oof_fold_{fold+1}'
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

    # 4.1. Regra de Negócio: Treinar ou Saltar?
    if len(checkpoints_existentes) >= 2:
        print(
            f"[!] Checkpoints detectados para o Fold {fold + 1}. A saltar o treino e a carregar os pesos salvos...")
        ckpt_ep2 = checkpoints_existentes[-2]
        ckpt_ep3 = checkpoints_existentes[-1]

        # O Trainer exige um modelo na inicialização. Carregamos o base temporariamente.
        modelo_base = AutoModelForSequenceClassification.from_pretrained(
            nome_modelo, num_labels=3, use_safetensors=True
        )
        args_vazios = TrainingArguments(
            output_dir=pasta_saida, per_device_eval_batch_size=8, report_to="none")
        trainer = Trainer(model=modelo_base, args=args_vazios)

    else:
        print(f"A treinar o modelo do zero para o Fold {fold + 1}...")
        modelo = AutoModelForSequenceClassification.from_pretrained(
            nome_modelo, num_labels=3, use_safetensors=True
        )

        training_args = TrainingArguments(
            output_dir=pasta_saida,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=8,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            learning_rate=3e-5,
            weight_decay=0.01,
            warmup_steps=314
        )

        trainer = Trainer(
            model=modelo,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

        trainer.train()

        # Lê novamente os checkpoints recém-gerados
        checkpoints_existentes = sorted(
            glob.glob(f"{pasta_saida}/checkpoint-*"), key=lambda x: int(x.split('-')[-1]))
        if len(checkpoints_existentes) >= 2:
            ckpt_ep2 = checkpoints_existentes[-2]
            ckpt_ep3 = checkpoints_existentes[-1]
        else:
            ckpt_ep2 = ckpt_ep3 = checkpoints_existentes[-1]

    # 5. Snapshot Ensembling no Out-of-Fold (Época 2 e 3)
    print(
        f"\nA extrair predições das Épocas 2 e 3 para a validação do Fold {fold + 1}...")

    # Prever com Época 2 (Forçando a transferência para a GPU primeiro)
    trainer.model = AutoModelForSequenceClassification.from_pretrained(
        ckpt_ep2, num_labels=3).to(dispositivo_gpu)
    probs_ep2 = softmax(trainer.predict(val_dataset).predictions, axis=1)

    # Prever com Época 3 (Forçando a transferência para a GPU primeiro)
    trainer.model = AutoModelForSequenceClassification.from_pretrained(
        ckpt_ep3, num_labels=3).to(dispositivo_gpu)
    probs_ep3 = softmax(trainer.predict(val_dataset).predictions, axis=1)

    # Média matemática (Snapshot Ensembling)
    probs_fold = (probs_ep2 + probs_ep3) / 2.0

    # Inserir na matriz OOF principal
    probabilidades_oof[val_idx] = probs_fold

    # 6. Limpeza Extrema de Memória
    del trainer
    if 'modelo' in locals():
        del modelo
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 7. Exportação Final
print("\nProcessamento Concluído! A guardar a matriz Out-of-Fold...")
df_oof = pd.DataFrame(probabilidades_oof, columns=[
                      f'prob_semantico_{c}' for c in encoder.classes_])
df_oof.to_csv('oof_semantico.csv', index=False)

predicoes_oof = np.argmax(probabilidades_oof, axis=1)
print(
    f"\nAcurácia do Especialista Semântico (BERT Snapshot OOF): {accuracy_score(y, predicoes_oof):.4f}")
