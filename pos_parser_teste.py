import pandas as pd
import numpy as np
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# 1. Carregar o modelo avançado do spaCy para português
print("Carregando o modelo spaCy (pt_core_news_lg)...")
nlp = spacy.load("pt_core_news_lg")


def extrair_features_sintaticas(texto):
    """
    Transforma um texto num vetor de estatísticas gramaticais e sintáticas.
    """
    texto_str = str(texto) if pd.notnull(texto) else ""
    doc = nlp(texto_str)

    tamanho = len(doc)
    if tamanho == 0:
        return {'frac_noun': 0, 'frac_verb': 0, 'frac_adj': 0, 'frac_punct': 0,
                'densidade_passiva': 0, 'distancia_dependencia_media': 0,
                'profundidade_arvore_max': 0, 'tamanho_doc': 0}

    # Contadores
    pos_counts = {'NOUN': 0, 'VERB': 0, 'ADJ': 0, 'PUNCT': 0}
    vozes_passivas = 0
    distancias_dep = []
    profundidades = []

    for token in doc:
        # Part-of-Speech
        if token.pos_ in pos_counts:
            pos_counts[token.pos_] += 1

        # Dependency Parsing: Voz Passiva
        if token.dep_ in ['nsubj:pass', 'aux:pass']:
            vozes_passivas += 1

        # Dependency Parsing: Distância média entre a palavra e o seu "chefe" sintático
        distancias_dep.append(abs(token.i - token.head.i))

        # Dependency Parsing: Profundidade da árvore (quantos nós acima até à raiz)
        profundidades.append(len(list(token.ancestors)))

    return {
        'frac_noun': pos_counts['NOUN'] / tamanho,
        'frac_verb': pos_counts['VERB'] / tamanho,
        'frac_adj': pos_counts['ADJ'] / tamanho,
        'frac_punct': pos_counts['PUNCT'] / tamanho,
        'densidade_passiva': vozes_passivas / tamanho,
        'distancia_dependencia_media': np.mean(distancias_dep),
        'profundidade_arvore_max': np.max(profundidades),
        'tamanho_doc': tamanho
    }


# 2. Carregamento dos Dados
print("Carregando os dados e extrair matrizes sintáticas (Isto demorará alguns minutos)...")
df_train = pd.read_excel('train.xlsx')
X_textos = df_train['resp_text'].tolist()

# Extração paralela (se possível) ou sequencial das features
features_list = [extrair_features_sintaticas(txt) for txt in X_textos]
X = pd.DataFrame(features_list)

encoder = LabelEncoder()
y = encoder.fit_transform(df_train['clarity'])

# 3. Configuração do Modelo e Validação Cruzada (OOF para Stacking)
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Carregando Random Forest e gerando probabilidades Out-of-Fold para o Stacking...")
# cross_val_predict gera as probabilidades sem viciar o modelo, para a Camada 2
probabilidades_oof = cross_val_predict(
    rf_model, X, y, cv=cv, method='predict_proba')
predicoes_oof = np.argmax(probabilidades_oof, axis=1)

print(
    f"\nAcurácia isolada do Especialista Sintático: {accuracy_score(y, predicoes_oof):.4f}")
print(classification_report(y, predicoes_oof, target_names=encoder.classes_))

# Guardar as probabilidades para uso posterior no meta-modelo (Regressão Logística)
df_oof = pd.DataFrame(probabilidades_oof, columns=[
                      f'prob_sintatico_{c}' for c in encoder.classes_])
df_oof.to_csv('oof_sintatico.csv', index=False)
print("Matriz de probabilidades guardada em 'oof_sintatico.csv'.")

# 4. Extração da Análise de Características (Para garantir os 5% da nota do EP2)
rf_model.fit(X, y)
importancias = rf_model.feature_importances_
df_importancias = pd.DataFrame(
    {'Feature': X.columns, 'Importancia': importancias})
df_importancias = df_importancias.sort_values(
    by='Importancia', ascending=False)

print("\n--- IMPORTÂNCIA DAS CARACTERÍSTICAS SINTÁTICAS (Para o Relatório) ---")
print(df_importancias)

# 5. Treino com 100% dos dados e Predição no Teste 1
print("\nTreinando Random Forest com 100% dos dados para inferência no teste...")
rf_model.fit(X, y)

print("Carregando e extraindo features sintáticas do conjunto de teste...")
df_test = pd.read_excel('test1-rotulado-para-teste-estatistico.xlsx')
X_test_textos = df_test['resp_text'].tolist()

features_list_test = [extrair_features_sintaticas(
    txt) for txt in X_test_textos]
X_test = pd.DataFrame(features_list_test)

probs_teste1 = rf_model.predict_proba(X_test)
df_probs_teste1 = pd.DataFrame(
    probs_teste1, columns=[f'prob_sintatico_{c}' for c in encoder.classes_])
df_probs_teste1.to_csv('prob_teste1_sintatico.csv', index=False)
print("Probabilidades do teste salvas em 'prob_teste1_sintatico.csv'.")
