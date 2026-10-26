import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# 1. Pré-processamento Rápido


def limpar_texto(texto):
    """
    Remove URLs, caracteres especiais e números, mantendo apenas letras e espaços.
    """
    texto_str = str(texto).lower() if pd.notnull(texto) else ""
    texto_str = re.sub(r'http\S+|www\.\S+', '', texto_str)  # Remove URLs
    texto_str = re.sub(r'[^a-záàâãéèêíïóôõöúç\s]',
                       ' ', texto_str)  # Apenas letras
    texto_str = re.sub(r'\s+', ' ', texto_str).strip()  # Remove espaços extra
    return texto_str


print("Limpando os dados de treino...")
df_train = pd.read_excel('train.xlsx')
X_limpo = [limpar_texto(txt) for txt in df_train['resp_text']]

encoder = LabelEncoder()
y = encoder.fit_transform(df_train['clarity'])

# 2. Engenharia de Features Híbrida (FeatureUnion)
print("Construindo as matrizes TF-IDF de Palavras (1-3) e Caracteres (2-5)...")

# TF-IDF para Palavras Inteiras
tfidf_palavras = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 3),  # Unigramas, bigramas e trigramas de palavras
    sublinear_tf=True,  # Suaviza a contagem de palavras muito repetidas
    max_features=20000,
    min_df=2
)

# TF-IDF para Caracteres (A sua ideia dos 5 n-gramas)
tfidf_caracteres = TfidfVectorizer(
    # 'char_wb' respeita o limite das palavras, não junta o fim de uma com o início de outra
    analyzer='char_wb',
    # Extrai desde blocos de 2 até 5 letras (ex: 'pr', 'pra', 'praz', 'prazo')
    ngram_range=(2, 5),
    sublinear_tf=True,
    max_features=20000,
    min_df=2
)

# Une as duas estratégias numa matriz gigante de 40.000 características
union = FeatureUnion([
    ('palavras', tfidf_palavras),
    ('caracteres', tfidf_caracteres)
])

# 3. O Classificador LinearSVC Calibrado
# O C=0.1 aplica uma forte regularização para impedir que as 40.000 colunas causem overfitting
svc_base = LinearSVC(C=0.1, class_weight='balanced',
                     max_iter=2000, random_state=42)

# Envolvemos o SVC no CalibratedClassifierCV para forçá-lo a emitir probabilidades (0 a 1)
classificador_calibrado = CalibratedClassifierCV(
    estimator=svc_base, cv=3, method='sigmoid')

# Criamos o Pipeline final
pipeline_lexical = Pipeline([
    ('extrator_tfidf', union),
    ('classificador', classificador_calibrado)
])

# 4. Validação Cruzada e Geração de Probabilidades OOF
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Treinando o LinearSVC e a gerar probabilidades Out-of-Fold para o Stacking...")
probabilidades_oof = cross_val_predict(
    pipeline_lexical, X_limpo, y, cv=cv, method='predict_proba', n_jobs=-1)
predicoes_oof = np.argmax(probabilidades_oof, axis=1)

print(
    f"\nAcurácia isolada do Especialista Lexical: {accuracy_score(y, predicoes_oof):.4f}")
print(classification_report(y, predicoes_oof, target_names=encoder.classes_))

# 5. Guardar as probabilidades para a Camada 2 (Stacking)
df_oof = pd.DataFrame(probabilidades_oof, columns=[
                      f'prob_lexical_{c}' for c in encoder.classes_])
df_oof.to_csv('oof_lexical.csv', index=False)
print("Matriz de probabilidades guardada em 'oof_lexical.csv'.")
