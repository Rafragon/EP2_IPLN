import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

print("A carregar probabilidades limpas geradas pelo passo anterior...")
probs_oof = np.load('probs_xgb_oof_limpo.npy')
y_oof = np.load('y_oof_limpo.npy')

df_train = pd.read_excel('train.xlsx').dropna(subset=['resp_text', 'clarity'])
encoder = LabelEncoder()
encoder.fit(df_train['clarity'].astype(str))
# Assume-se ordem alfabética: 0 = c1, 1 = c234, 2 = c5


def aplicar_limiares(probs, t_c1, t_c5):
    """Se c1 > t_c1, prevê c1. Senão, se c5 > t_c5, prevê c5. Senão, prevê c234."""
    preds = []
    for p in probs:
        if p[0] >= t_c1:
            preds.append(0)
        elif p[2] >= t_c5:
            preds.append(2)
        else:
            preds.append(1)  # Zona de dúvida vai para c234
    return np.array(preds)


print("A executar Grid Search de Limiares Otimizados...")
melhor_acc = 0
melhor_t_c1 = 0
melhor_t_c5 = 0

# Pesquisa limiares entre 30% e 60%
for t1 in np.arange(0.30, 0.61, 0.01):
    for t5 in np.arange(0.30, 0.61, 0.01):
        preds = aplicar_limiares(probs_oof, t1, t5)
        acc = accuracy_score(y_oof, preds)
        if acc > melhor_acc:
            melhor_acc = acc
            melhor_t_c1 = t1
            melhor_t_c5 = t5

print(f"\nLimiares Ideais Encontrados:")
print(f"Para ser c1, probabilidade mínima: {melhor_t_c1*100:.1f}%")
print(f"Para ser c5, probabilidade mínima: {melhor_t_c5*100:.1f}%")
print(f"Acurácia no OOF com limiares: {melhor_acc:.4f}")

# ==============================================================================
# Aplicação no Teste 1 e Heurística
# ==============================================================================
print("\nA aplicar limiares ao Teste 1...")
df_test = pd.read_excel('test1-rotulado-para-teste-estatistico.xlsx')
textos_teste = df_test['resp_text'].fillna('').astype(str).tolist()
y_test_true = df_test['clarity'].tolist()


def normalizacao_extrema(texto):
    texto_str = str(texto).lower() if pd.notnull(texto) else ""
    mapa_acentos = str.maketrans('áàâãéèêíïóôõöúç', 'aaaaeeeiioooouc')
    return re.sub(r'[^a-z0-9]', '', texto_str.translate(mapa_acentos))


df_train['chave'] = df_train['resp_text'].astype(
    str).apply(normalizacao_extrema)
dicionario_respostas = df_train.groupby('chave')['clarity'].agg(
    lambda x: x.value_counts().index[0]).to_dict()

probs_teste = np.load('probs_xgb_teste.npy')
preds_teste_num = aplicar_limiares(probs_teste, melhor_t_c1, melhor_t_c5)
preds_teste_rotulos = encoder.inverse_transform(preds_teste_num)

rotulos_finais = []
interceptados = 0
for idx, texto in enumerate(textos_teste):
    chave = normalizacao_extrema(texto)
    if chave in dicionario_respostas:
        rotulos_finais.append(dicionario_respostas[chave])
        interceptados += 1
    else:
        rotulos_finais.append(preds_teste_rotulos[idx])

print("\n" + "="*50)
print("RESULTADOS FINAIS (CLEANLAB + THRESHOLD MOVING)")
print("="*50)
print(f"Respostas corrigidas pela heurística: {interceptados}")
print(f"Acurácia Absoluta: {accuracy_score(y_test_true, rotulos_finais):.4f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test_true, rotulos_finais))

df_saida = pd.DataFrame({'resp_text': textos_teste,
                        'rotulo_real': y_test_true, 'predicao_final': rotulos_finais})
df_saida.to_csv('predicoes_cleanlab_threshold_teste1.csv',
                index=False, encoding='utf-8')
