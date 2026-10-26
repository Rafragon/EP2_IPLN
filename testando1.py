import pandas as pd
import numpy as np
import torch
import re
import joblib
import glob
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from scipy.special import softmax
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# 1. Configurações e Caminhos
arquivo_teste = 'test1-rotulado-para-teste-estatistico.xlsx'
arquivo_treino = 'train.xlsx'
arquivo_saida = 'predicoes_stacking_teste1.csv'


def normalizacao_extrema(texto):
    texto_str = str(texto).lower() if pd.notnull(texto) else ""
    # As duas strings possuem exatamente 15 caracteres (correção aplicada)
    mapa_acentos = str.maketrans('áàâãéèêíïóôõöúç', 'aaaaeeeiioooouc')
    return re.sub(r'[^a-z0-9]', '', texto_str.translate(mapa_acentos))


# 2. Dados e Heurística
print("Carregando dados e dicionário de respostas prontas...")
df_train = pd.read_excel(arquivo_treino)
df_train['chave'] = df_train['resp_text'].fillna(
    '').astype(str).apply(normalizacao_extrema)
encoder = LabelEncoder()
encoder.fit(df_train['clarity'])
dicionario_respostas = df_train.groupby('chave')['clarity'].agg(
    lambda x: x.value_counts().index[0]).to_dict()

df_test = pd.read_excel(arquivo_teste)
df_test['resp_text'] = df_test['resp_text'].fillna('').astype(str)
textos_teste = df_test['resp_text'].tolist()
y_true = df_test['clarity'].tolist()

# 3. Inferência do Especialista Semântico (BERT K-Fold Snapshot)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings): self.encodings = encodings
    def __getitem__(self, idx): return {k: torch.tensor(
        v[idx]) for k, v in self.encodings.items()}

    def __len__(self): return len(self.encodings['input_ids'])


tokenizer = AutoTokenizer.from_pretrained(
    'neuralmind/bert-base-portuguese-cased')
test_dataset = TextDataset(
    tokenizer(textos_teste, truncation=True, padding=True, max_length=512))
dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

probs_bert_total = np.zeros((len(textos_teste), 3))
total_modelos = 0

print("\nInicializando inferência massiva do BERTimbau (5 Folds x 2 Épocas)...")
for fold in range(1, 6):
    # pasta_fold = f'./resultados_bert_oof_fold_{fold}'
    pasta_fold = f'./resultados_bert_large_oof_fold_{fold}'
    checkpoints = sorted(
        glob.glob(f"{pasta_fold}/checkpoint-*"), key=lambda x: int(x.split('-')[-1]))

    # Pega apenas nos dois últimos checkpoints (Época 2 e 3)
    for ckpt in checkpoints[-2:]:
        print(f"Processando {ckpt}...")
        modelo = AutoModelForSequenceClassification.from_pretrained(
            ckpt, num_labels=3).to(dispositivo)
        probs = softmax(Trainer(model=modelo).predict(
            test_dataset).predictions, axis=1)

        probs_bert_total += probs
        total_modelos += 1

        del modelo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if total_modelos == 0:
    print("[ERRO] Nenhum checkpoint encontrado nas pastas ./resultados_bert_oof_fold_X/")
    exit()

# Média final das probabilidades
probs_bert_media = probs_bert_total / total_modelos
df_bert_teste = pd.DataFrame(probs_bert_media, columns=[
                             f'prob_semantico_{c}' for c in encoder.classes_])

# 4. Fusão e Stacking
print("\nCarregando as matrizes dos especialistas clássicos e o Meta-Modelo...")
try:
    X_meta_teste = pd.concat([
        pd.read_csv('prob_teste1_sintatico.csv'),
        pd.read_csv('prob_teste1_morfologico.csv'),
        pd.read_csv('prob_teste1_lexical.csv'),
        # df_bert_teste,
        pd.read_csv('prob_teste1_semantico.csv'),
        pd.read_csv('prob_teste1_cnn.csv'),
        pd.read_csv('metadados_teste1.csv')
        # , pd.read_csv('prob_teste1_llm.csv')
    ], axis=1)
except FileNotFoundError as e:
    print(
        f"[ERRO] {e}\nGaranta que gerou os CSVs de teste no passo anterior.")
    exit()

meta_modelo = joblib.load('meta_modelo_stacking.pkl')
# meta_modelo = joblib.load('meta_modelo_xgboost.pkl')
predicoes_meta_num = meta_modelo.predict(X_meta_teste)
predicoes_meta_rotulos = encoder.inverse_transform(predicoes_meta_num)

# 5. Interceção Heurística e Avaliação
rotulos_finais = []
interceptados = 0
for idx, texto in enumerate(textos_teste):
    chave = normalizacao_extrema(texto)
    if chave in dicionario_respostas:
        rotulos_finais.append(dicionario_respostas[chave])
        interceptados += 1
    else:
        rotulos_finais.append(predicoes_meta_rotulos[idx])

print("\n" + "="*50)
print("🏆 RESULTADOS FINAIS NO CONJUNTO DE TESTE 1")
print("="*50)
print(
    f"Respostas corrigidas pela heurística de textos idênticos: {interceptados}")
print(f"Acurácia Absoluta: {accuracy_score(y_true, rotulos_finais):.4f}")
print("\nRelatório de Classificação:")
print(classification_report(y_true, rotulos_finais))

df_saida = pd.DataFrame({'resp_text': textos_teste,
                        'rotulo_real': y_true, 'predicao_ep2': rotulos_finais})
df_saida.to_csv(arquivo_saida, index=False, encoding='utf-8')
print(f"\nFicheiro de predições guardado como '{arquivo_saida}'.")
