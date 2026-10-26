import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ==============================================================================
# 1. Carregar os Dados de Treino e Rótulos Reais
# ==============================================================================
print("A carregar os rótulos reais do conjunto de treino...")
df_train = pd.read_excel('train.xlsx')

encoder = LabelEncoder()
y_true = encoder.fit_transform(df_train['clarity'])

# ==============================================================================
# 2. Carregar as Matrizes Out-of-Fold (As 'Opiniões' dos Especialistas)
# ==============================================================================
print("A fundir as probabilidades dos 4 especialistas...")
try:
    X_meta = pd.concat([
        pd.read_csv('oof_sintatico.csv'),
        pd.read_csv('oof_morfologico.csv'),
        pd.read_csv('oof_lexical.csv'),
        pd.read_csv('oof_semantico.csv'),
        pd.read_csv('oof_cnn.csv'),
        # pd.read_csv('oof_llm.csv'),
        pd.read_csv('metadados_treino.csv')
    ], axis=1)
except FileNotFoundError as e:
    print(f"\n[ERRO] Ficheiro não encontrado: {e}")
    exit()

print(
    f"Matriz de Meta-Features construída: {X_meta.shape[0]} textos, {X_meta.shape[1]} features.")

# ==============================================================================
# 3. Otimização de Hiperparâmetros (Grid Search)
# ==============================================================================
print("\nA iniciar Grid Search no XGBoost (Isto demorará algum tempo)...")

xgb_base = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)

# Grelha de parâmetros para forçar o modelo a procurar a melhor arquitetura
param_grid = {
    'max_depth': [3, 5, 7],               # Profundidade de cada árvore
    'learning_rate': [0.01, 0.05, 0.1],   # Quão rápido o modelo aprende
    'n_estimators': [100, 300, 500],      # Número total de árvores
    # Percentagem de dados usada por árvore (evita overfitting)
    'subsample': [0.8, 1.0],
    # Percentagem de especialistas avaliados por árvore
    'colsample_bytree': [0.8, 1.0]
}

cv_grid = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv_grid,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_meta, y_true)
melhor_xgb = grid_search.best_estimator_

print(f"\nMelhores parâmetros encontrados pelo Grid Search:")
for param, valor in grid_search.best_params_.items():
    print(f" - {param}: {valor}")

# ==============================================================================
# 4. Avaliação Robusta do Melhor Modelo
# ==============================================================================
print("\nA executar Validação Cruzada Final com a melhor arquitetura XGBoost...")
probabilidades_stacking = cross_val_predict(
    melhor_xgb, X_meta, y_true, cv=cv_grid, method='predict_proba', n_jobs=-1)
predicoes_stacking = np.argmax(probabilidades_stacking, axis=1)

print("\n" + "="*50)
print("🏆 RESULTADOS FINAIS DO STACKING (XGBOOST OTIMIZADO)")
print("="*50)
print(
    f"Acurácia Geral do Sistema: {accuracy_score(y_true, predicoes_stacking):.4f}")
print("\nRelatório de Classificação Detalhado:")
print(classification_report(
    y_true, predicoes_stacking, target_names=encoder.classes_))

# ==============================================================================
# 5. Análise de Importância e Exportação
# ==============================================================================
print("\nA treinar o modelo definitivo com 100% dos dados OOF...")
melhor_xgb.fit(X_meta, y_true)

importancias = pd.DataFrame({
    'Feature (Probabilidade)': X_meta.columns,
    'Importância (XGBoost)': melhor_xgb.feature_importances_
}).sort_values(by='Importância (XGBoost)', ascending=False)

print("\n--- IMPORTÂNCIA DOS ESPECIALISTAS NA DECISÃO FINAL (Gain) ---")
print(importancias.to_string(index=False))

joblib.dump(melhor_xgb, 'meta_modelo_xgboost.pkl')
print("\nMeta-modelo guardado como 'meta_modelo_xgboost.pkl'.")
