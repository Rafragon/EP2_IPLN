import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

print("A carregar os dados de treino...")
df_train = pd.read_excel('train.xlsx').dropna(
    subset=['resp_text', 'clarity']).reset_index(drop=True)


def limpar_texto(texto):
    texto = str(texto).lower()
    return re.sub(r'[^\w\s]', '', texto)


X_train = df_train['resp_text'].apply(limpar_texto)
y_train = LabelEncoder().fit_transform(df_train['clarity'].astype(str))

# Construção do Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    # max_iter elevado para evitar avisos de convergência
    ('svc', LinearSVC(max_iter=5000, random_state=42, dual=True))
])

# Dividimos a busca em dois blocos: um focado em semântica (palavras) e outro em morfologia/ruído (caracteres)
parametros = [
    {
        # Bloco 1: Análise Baseada em Palavras (N-gramas longos)
        'tfidf__analyzer': ['word'],
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4)],
        'tfidf__min_df': [1, 2, 3, 5],
        'tfidf__max_df': [0.7, 0.85, 1.0],
        'tfidf__sublinear_tf': [True, False],
        'tfidf__use_idf': [True, False],
        'svc__C': [0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
        'svc__class_weight': [None, 'balanced']
    },
    {
        # Bloco 2: Análise Baseada em Fronteiras de Caracteres (Resistência a erros ortográficos)
        'tfidf__analyzer': ['char_wb'],
        'tfidf__ngram_range': [(2, 4), (3, 5), (4, 6)],
        'tfidf__min_df': [1, 2, 5],
        'tfidf__max_df': [0.7, 0.9, 1.0],
        'tfidf__sublinear_tf': [True, False],
        'tfidf__use_idf': [True, False],
        'svc__C': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        'svc__class_weight': [None, 'balanced']
    }
]

# Total estimado de combinações geradas: ~6.000 configurações.
# Com Validação Cruzada de 5-Folds, o XGBoost fará ~30.000 treinos independentes.
print(f"A iniciar o Grid Search Massivo (5-Fold CV)...")
print("Isto demorará várias horas. Deixe o computador trabalhar livremente.")
grid = GridSearchCV(pipeline, parametros, cv=5,
                    scoring='accuracy', n_jobs=-1, verbose=3)
grid.fit(X_train, y_train)

# Apresentação e gravação dos resultados
resultados = (
    "="*50 + "\n" +
    "MELHORES PARÂMETROS ENCONTRADOS PARA O LINEARSVC:\n" +
    "="*50 + "\n"
)

for param, valor in grid.best_params_.items():
    resultados += f"{param}: {valor}\n"

resultados += f"\nAcurácia OOF alcançada: {grid.best_score_:.4f}\n"

print("\n" + resultados)

# Guardar em ficheiro de texto para segurança
with open("melhores_parametros_svc.txt", "w", encoding="utf-8") as f:
    f.write(resultados)

print("Resultados guardados com sucesso no ficheiro 'melhores_parametros_svc.txt'.")
