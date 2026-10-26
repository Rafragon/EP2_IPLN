import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
import cleanlab

if __name__ == '__main__':
    print("A carregar os dados e as matrizes OOF...")
    df_train = pd.read_excel('train.xlsx').dropna(
        subset=['resp_text', 'clarity']).reset_index(drop=True)
    encoder = LabelEncoder()
    y_true = encoder.fit_transform(df_train['clarity'].astype(str))

    X_meta = pd.concat([
        pd.read_csv('oof_sintatico.csv'),
        pd.read_csv('oof_morfologico.csv'),
        pd.read_csv('oof_lexical.csv'),
        pd.read_csv('oof_cnn.csv'),
        pd.read_csv('oof_semantico.csv'),
        pd.read_csv('metadados_treino.csv')  # Injeção de metadados
    ], axis=1)

    print("A gerar probabilidades base do XGBoost para análise de ruído...")
    xgb_base = XGBClassifier(max_depth=3, learning_rate=0.05, n_estimators=300,
                             objective='multi:softprob', random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs_oof_xgb = cross_val_predict(
        xgb_base, X_meta, y_true, cv=cv, method='predict_proba', n_jobs=-1)

    print("A executar Cleanlab para detetar erros humanos...")
    mascara_erros = cleanlab.filter.find_label_issues(
        labels=y_true,
        pred_probs=probs_oof_xgb
    )

    mascara_limpa = ~mascara_erros
    X_meta_limpo = X_meta[mascara_limpa]
    y_limpo = y_true[mascara_limpa]

    print(
        f"Foram detetados e removidos {mascara_erros.sum()} textos com rótulos ruidosos/contraditórios!")
    print(
        f"Treinando o XGBoost Purificado com os {len(y_limpo)} textos restantes...")

    xgb_limpo = XGBClassifier(max_depth=3, learning_rate=0.05, n_estimators=300,
                              objective='multi:softprob', random_state=42, n_jobs=-1)
    xgb_limpo.fit(X_meta_limpo, y_limpo)

    joblib.dump(xgb_limpo, 'meta_modelo_xgb_limpo.pkl')

    print("A prever o Teste 1 com o modelo limpo...")
    X_meta_teste = pd.concat([
        pd.read_csv('prob_teste1_sintatico.csv'),
        pd.read_csv('prob_teste1_morfologico.csv'),
        pd.read_csv('prob_teste1_lexical.csv'),
        pd.read_csv('prob_teste1_cnn.csv'),
        pd.read_csv('prob_teste1_semantico.csv'),
        pd.read_csv('metadados_teste1.csv')  # Injeção de metadados
    ], axis=1)

    probs_teste_xgb = xgb_limpo.predict_proba(X_meta_teste)
    np.save('probs_xgb_teste.npy', probs_teste_xgb)
    np.save('probs_xgb_oof_limpo.npy', cross_val_predict(
        xgb_limpo, X_meta_limpo, y_limpo, cv=cv, method='predict_proba', n_jobs=-1))
    np.save('y_oof_limpo.npy', y_limpo)

    print("Concluído. Guardado 'probs_xgb_teste.npy' para o Ajuste de Limiares.")
