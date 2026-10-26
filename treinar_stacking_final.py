import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

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
print("Fundindo as probabilidades dos especialistas...")

try:
    df_sintatico = pd.read_csv('oof_sintatico.csv')
    df_morfologico = pd.read_csv('oof_morfologico.csv')
    df_lexical = pd.read_csv('oof_lexical.csv')
    # O ficheiro gerado pelo BERTimbau
    # df_semantico = pd.read_csv('oof_semantico.csv')
    df_cnn = pd.read_csv('oof_cnn.csv')
    df_llm = pd.read_csv('oof_llm.csv')
    df_metadados = pd.read_csv('metadados_treino.csv')
except FileNotFoundError as e:
    print(f"\n[ERRO] Ficheiro não encontrado: {e}")
    print("Certifique-se de que os 6 ficheiros oof_*.csv estão na mesma pasta do script.")
    exit()

# Concatenar lado a lado. O resultado será um DataFrame com 12 colunas (4 modelos x 3 classes)
X_meta = pd.concat([df_sintatico, df_morfologico,
                   df_lexical,
                   # df_semantico,
                    df_cnn, df_metadados,
                    df_llm
                    ], axis=1)

print(
    f"Matriz de Meta-Features construída: {X_meta.shape[0]} textos, {X_meta.shape[1]} features (probabilidades).")

# ==============================================================================
# 3. Avaliação Robusta do Meta-Modelo
# ==============================================================================
# Utilizamos uma Regressão Logística regularizada (C=0.1) para evitar que
# o meta-modelo memorize e dependa de um único especialista.
meta_modelo = LogisticRegression(
    C=0.1, class_weight='balanced', max_iter=2000, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nA executar Validação Cruzada no Meta-Modelo (Camada 2)...")
# Gera as predições do meta-modelo sem o viciar, para sabermos a acurácia real da arquitetura
probabilidades_stacking = cross_val_predict(
    meta_modelo, X_meta, y_true, cv=cv, method='predict_proba', n_jobs=-1)
predicoes_stacking = np.argmax(probabilidades_stacking, axis=1)

print("\n" + "="*50)
print("🏆 RESULTADOS FINAIS DO ENSEMBLE HETEROGÉNEO (STACKING)")
print("="*50)
print(
    f"Acurácia Geral do Sistema: {accuracy_score(y_true, predicoes_stacking):.4f}")
print("\nRelatório de Classificação Detalhado:")
print(classification_report(
    y_true, predicoes_stacking, target_names=encoder.classes_))

# ==============================================================================
# 4. Treino Definitivo e Exportação
# ==============================================================================
print("\nA treinar o meta-modelo definitivo com 100% dos dados Out-of-Fold...")
meta_modelo.fit(X_meta, y_true)

# Exibir os pesos matemáticos (coeficientes) que o modelo atribuiu a cada especialista
# Soma os pesos absolutos para as 3 classes
pesos = np.abs(meta_modelo.coef_).sum(axis=0)
importancias = pd.DataFrame({
    'Feature (Probabilidade)': X_meta.columns,
    'Peso_Importancia': pesos
}).sort_values(by='Peso_Importancia', ascending=False)

print("\n--- IMPORTÂNCIA DOS ESPECIALISTAS NA DECISÃO FINAL ---")
print(importancias.to_string(index=False))
print("\nDica para o Relatório: Use a tabela acima para comprovar que o Stacking não dependeu apenas do BERT!")

# Guardar o modelo em disco para não precisar treiná-lo de novo na hora de testar
joblib.dump(meta_modelo, 'meta_modelo_stacking.pkl')
print("\nMeta-modelo guardado como 'meta_modelo_stacking.pkl'. Arquitetura pronta para a inferência!")
