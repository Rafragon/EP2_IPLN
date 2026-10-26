import pandas as pd
import numpy as np
import torch
import glob
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from scipy.special import softmax
from sklearn.preprocessing import LabelEncoder

# 1. Recuperar as classes exatas do treino
print("Carregando configurações...")
df_train = pd.read_excel('train.xlsx')
encoder = LabelEncoder()
encoder.fit(df_train['clarity'])

# 2. Carregar os textos de Teste
df_test = pd.read_excel('test1-rotulado-para-teste-estatistico.xlsx')
textos_teste = df_test['resp_text'].fillna('').astype(str).tolist()


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings): self.encodings = encodings
    def __getitem__(self, idx): return {k: torch.tensor(
        v[idx]) for k, v in self.encodings.items()}

    def __len__(self): return len(self.encodings['input_ids'])


print("Inicializando o Tokenizer do BERT Large...")
tokenizer = AutoTokenizer.from_pretrained(
    'neuralmind/bert-large-portuguese-cased')
test_encodings = tokenizer(
    textos_teste, truncation=True, padding=True, max_length=512)
test_dataset = TextDataset(test_encodings)

dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
probs_totais = np.zeros((len(textos_teste), 3))
modelos_contados = 0

# 3. Extrair predições dos 5 Folds do BERT Large
print("\nInicializando inferência massiva (Isto demorará alguns minutos)...")
for fold in range(1, 6):
    pasta_fold = f'./resultados_bert_large_oof_fold_{fold}'
    checkpoints = sorted(
        glob.glob(f"{pasta_fold}/checkpoint-*"), key=lambda x: int(x.split('-')[-1]))

    if not checkpoints:
        print(f"[AVISO] Pasta {pasta_fold} não encontrada ou vazia.")
        continue

    for ckpt in checkpoints[-2:]:
        print(f"Previsão com o modelo: {ckpt}...")
        modelo = AutoModelForSequenceClassification.from_pretrained(
            ckpt, num_labels=3).to(dispositivo)

        args_avaliacao = TrainingArguments(
            output_dir='./tmp', per_device_eval_batch_size=2, report_to="none")
        trainer = Trainer(model=modelo, args=args_avaliacao)
        probs = softmax(trainer.predict(test_dataset).predictions, axis=1)

        probs_totais += probs
        modelos_contados += 1

        del modelo
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# 4. Guardar Matriz de Teste
if modelos_contados > 0:
    probs_finais = probs_totais / modelos_contados
    df_probs = pd.DataFrame(probs_finais, columns=[
                            f'prob_semantico_{c}' for c in encoder.classes_])
    df_probs.to_csv('prob_teste1_semantico.csv', index=False)
    print(
        f"\n[SUCESSO] Ficheiro 'prob_teste1_semantico.csv' gerado usando a média de {modelos_contados} modelos Large!")
else:
    print("\n[ERRO] Nenhum checkpoint do BERT Large foi processado.")
