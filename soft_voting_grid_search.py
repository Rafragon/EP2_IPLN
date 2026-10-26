import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ==============================================================================
# 1. Configurações e Heurística
# ==============================================================================
arquivo_treino = 'train.xlsx'
arquivo_teste = 'test1-rotulado-para-teste-estatistico.xlsx'


def normalizacao_extrema(texto):
    texto_str = str(texto).lower() if pd.notnull(texto) else ""
    mapa_acentos = str.maketrans('áàâãéèêíïóôõöúç', 'aaaaeeeiioooouc')
    return re.sub(r'[^a-z0-9]', '', texto_str.translate(mapa_acentos))


print("A carregar dicionário de textos idênticos do treino...")
df_train = pd.read_excel(arquivo_treino)
df_train = df_train.dropna(subset=['resp_text', 'clarity'])
df_train['chave'] = df_train['resp_text'].astype(
    str).apply(normalizacao_extrema)

encoder = LabelEncoder()
y_train_true = encoder.fit_transform(df_train['clarity'].astype(str))
dicionario_respostas = df_train.groupby('chave')['clarity'].agg(
    lambda x: x.value_counts().index[0]).to_dict()

# ==============================================================================
# 2. Carregar Matrizes OOF (Treino) para o Grid Search
# ==============================================================================
print("Carregando matrizes Out-of-Fold (OOF)...")
try:
    oof_sem = pd.read_csv('oof_semantico.csv').values
    oof_cnn = pd.read_csv('oof_cnn.csv').values
    oof_lex = pd.read_csv('oof_lexical.csv').values
    oof_mor = pd.read_csv('oof_morfologico.csv').values
    oof_sin = pd.read_csv('oof_sintatico.csv').values
except FileNotFoundError as e:
    print(f"[ERRO] Faltam ficheiros OOF: {e}")
    exit()

# ==============================================================================
# 3. Grid Search de Pesos Otimizados (Incrementos de 5%)
# ==============================================================================
print("Carregando Grid Search para pesos do Soft Voting (Isto levará alguns segundos)...")

melhor_acuracia_oof = 0
melhores_pesos = None

# Gera todas as combinações que somam 100 (em passos de 5)
for w_sem in range(0, 101, 5):
    for w_cnn in range(0, 101 - w_sem, 5):
        for w_lex in range(0, 101 - w_sem - w_cnn, 5):
            for w_mor in range(0, 101 - w_sem - w_cnn - w_lex, 5):
                w_sin = 100 - w_sem - w_cnn - w_lex - w_mor

                # Normaliza para 0.0 a 1.0
                pesos = np.array([w_sem, w_cnn, w_lex, w_mor, w_sin]) / 100.0

                # Média Ponderada Simples
                probs_hibridas = (
                    pesos[0] * oof_sem +
                    pesos[1] * oof_cnn +
                    pesos[2] * oof_lex +
                    pesos[3] * oof_mor +
                    pesos[4] * oof_sin
                )

                preds = np.argmax(probs_hibridas, axis=1)
                acc = accuracy_score(y_train_true, preds)

                if acc > melhor_acuracia_oof:
                    melhor_acuracia_oof = acc
                    melhores_pesos = pesos

print("\n--- MELHORES PESOS ENCONTRADOS ---")
print(f"BERT Large (Semântico) : {melhores_pesos[0]*100:.0f}%")
print(f"CNN (Local/Palavras)   : {melhores_pesos[1]*100:.0f}%")
print(f"TF-IDF SVC (Lexical)   : {melhores_pesos[2]*100:.0f}%")
print(f"FastText (Morfológico) : {melhores_pesos[3]*100:.0f}%")
print(f"POS Random Forest (Sint): {melhores_pesos[4]*100:.0f}%")
print(f"Acurácia OOF do Ensemble: {melhor_acuracia_oof:.4f}")

# ==============================================================================
# 4. Avaliação no Teste 1
# ==============================================================================
print("\nAplicando os pesos otimizados ao Teste 1...")
try:
    teste_sem = pd.read_csv('prob_teste1_semantico.csv').values
    teste_cnn = pd.read_csv('prob_teste1_cnn.csv').values
    teste_lex = pd.read_csv('prob_teste1_lexical.csv').values
    teste_mor = pd.read_csv('prob_teste1_morfologico.csv').values
    teste_sin = pd.read_csv('prob_teste1_sintatico.csv').values
except FileNotFoundError as e:
    print(f"[ERRO] Faltam ficheiros do Teste 1: {e}")
    exit()

df_test = pd.read_excel(arquivo_teste)
textos_teste = df_test['resp_text'].fillna('').astype(str).tolist()
y_test_true = df_test['clarity'].tolist()

# Calcula as probabilidades finais no teste
probs_finais_teste = (
    melhores_pesos[0] * teste_sem +
    melhores_pesos[1] * teste_cnn +
    melhores_pesos[2] * teste_lex +
    melhores_pesos[3] * teste_mor +
    melhores_pesos[4] * teste_sin
)

predicoes_iniciais = np.argmax(probs_finais_teste, axis=1)
predicoes_iniciais_rotulos = encoder.inverse_transform(predicoes_iniciais)

# ==============================================================================
# 5. Interceção Heurística e Relatório Final
# ==============================================================================
rotulos_finais = []
interceptados = 0

for idx, texto in enumerate(textos_teste):
    chave = normalizacao_extrema(texto)
    if chave in dicionario_respostas:
        rotulos_finais.append(dicionario_respostas[chave])
        interceptados += 1
    else:
        rotulos_finais.append(predicoes_iniciais_rotulos[idx])

print("\n" + "="*50)
print("🏆 RESULTADOS FINAIS DO SOFT VOTING NO TESTE 1")
print("="*50)
print(f"Respostas corrigidas pela heurística: {interceptados}")
print(f"Acurácia Absoluta: {accuracy_score(y_test_true, rotulos_finais):.4f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test_true, rotulos_finais))

df_saida = pd.DataFrame({
    'resp_text': textos_teste,
    'rotulo_real': y_test_true,
    'predicao_soft_voting': rotulos_finais
})
df_saida.to_csv('predicoes_softvoting_teste1.csv',
                index=False, encoding='utf-8')
