import pandas as pd
import numpy as np
import re
from gensim.models import FastText
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# 1. Pré-processamento e Tokenização Simples


def tokenizar_texto(texto):
    """
    Remove pontuações, converte para minúsculas e divide o texto em palavras.
    """
    texto_str = str(texto).lower() if pd.notnull(texto) else ""
    # Expressão regular para extrair apenas blocos de caracteres alfanuméricos
    tokens = re.findall(r'\b\w+\b', texto_str)
    return tokens


print("Carregando os dados e tokenizando os textos...")
df_train = pd.read_excel('train.xlsx')
textos_tokenizados = [tokenizar_texto(txt) for txt in df_train['resp_text']]

# 2. Treino do Modelo FastText (Morfológico)
print("Treinando o modelo FastText do zero (Isto pode demorar alguns minutos)...")
# min_n e max_n definem o tamanho dos n-gramas de caracteres analisados.
modelo_ft = FastText(
    sentences=textos_tokenizados,
    vector_size=100,      # Dimensão dos embeddings
    window=5,             # Janela de contexto ao redor de cada palavra
    min_count=2,          # Ignora palavras que aparecem apenas 1 vez
    min_n=3,              # N-grama mínimo de caracteres (ex: 'pra')
    max_n=6,              # N-grama máximo de caracteres
    epochs=15,            # Passagens pelo dataset
    seed=42,
    workers=4
)

# 3. Geração de Embeddings por Documento (Média Vetorial)


def vetorizar_documento(modelo, tokens):
    """
    Calcula a média dos vetores de todas as palavras de um documento.
    O FastText gera vetores mesmo para palavras fora do vocabulário.
    """
    if len(tokens) == 0:
        return np.zeros(modelo.vector_size)

    # Extrai o vetor de cada palavra (gerado dinamicamente para palavras com erros)
    vetores = [modelo.wv[word] for word in tokens]
    return np.mean(vetores, axis=0)


print("Convertendo os textos em vetores matemáticos (Embeddings)...")
X = np.array([vetorizar_documento(modelo_ft, tokens)
             for tokens in textos_tokenizados])

encoder = LabelEncoder()
y = encoder.fit_transform(df_train['clarity'])

# 4. Configuração do Classificador (MLP) e Validação Cruzada
classificador_mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=0.001,
    learning_rate_init=0.001,
    max_iter=800,
    early_stopping=True,
    random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Treinando a Rede Neuronal (MLP) e a gerar probabilidades Out-of-Fold...")
probabilidades_oof = cross_val_predict(
    classificador_mlp, X, y, cv=cv, method='predict_proba', n_jobs=-1)
predicoes_oof = np.argmax(probabilidades_oof, axis=1)

print(
    f"\nAcurácia isolada do Especialista Morfológico: {accuracy_score(y, predicoes_oof):.4f}")
print(classification_report(y, predicoes_oof, target_names=encoder.classes_))

# 5. Guardar as probabilidades para a Camada 2 (Stacking)
df_oof = pd.DataFrame(probabilidades_oof, columns=[
                      f'prob_morfologico_{c}' for c in encoder.classes_])
df_oof.to_csv('oof_morfologico.csv', index=False)
print("Matriz de probabilidades guardada em 'oof_morfologico.csv'.")

# 6. Treino com 100% dos dados e Predição no Teste 1
print("\nTreinando MLP com 100% dos dados para inferência no teste...")
classificador_mlp.fit(X, y)

print("Carregando e processando o conjunto de teste...")
df_test = pd.read_excel('test1-rotulado-para-teste-estatistico.xlsx')
textos_teste_tokenizados = [tokenizar_texto(
    txt) for txt in df_test['resp_text']]
X_test = np.array([vetorizar_documento(modelo_ft, tokens)
                  for tokens in textos_teste_tokenizados])

probs_teste1 = classificador_mlp.predict_proba(X_test)
df_probs_teste1 = pd.DataFrame(
    probs_teste1, columns=[f'prob_morfologico_{c}' for c in encoder.classes_])
df_probs_teste1.to_csv('prob_teste1_morfologico.csv', index=False)
print("Probabilidades do teste salvas em 'prob_teste1_morfologico.csv'.")
