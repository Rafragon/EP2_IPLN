import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def limpar_dados(df):
    df['resp_text'] = df['resp_text'].astype(
        str).str.replace(r'[^\w\s.,;:!?()-]', ' ', regex=True)
    df['resp_text'] = df['resp_text'].str.replace(
        r'\s+', ' ', regex=True).str.strip()
    return df


print("Carregando e limpando dados...")
df_train = limpar_dados(pd.read_excel(
    'train.xlsx').dropna(subset=['resp_text', 'clarity']))
X_train_raw = df_train['resp_text'].to_numpy(dtype=object)

encoder = LabelEncoder()
y_train = encoder.fit_transform(df_train['clarity'].astype(str).to_numpy())

# Vetorização
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=25000, output_sequence_length=1000)
vectorizer.adapt(tf.convert_to_tensor(X_train_raw, dtype=tf.string))


def criar_modelo():
    model = tf.keras.Sequential([
        vectorizer,
        tf.keras.layers.Embedding(25000, 128),
        tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# 1. Gerar OOF (5-Fold)
print("\nGerando Out-of-Fold (OOF) em 5 Folds...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
probabilidades_oof = np.zeros((len(X_train_raw), 3))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_raw, y_train)):
    print(f"Treinando Fold {fold+1}...")
    X_f_train = tf.convert_to_tensor(X_train_raw[train_idx], dtype=tf.string)
    y_f_train = y_train[train_idx]
    X_f_val = tf.convert_to_tensor(X_train_raw[val_idx], dtype=tf.string)

    modelo = criar_modelo()
    modelo.fit(X_f_train, y_f_train, epochs=8, batch_size=32,
               verbose=0)  # 8 épocas diretas é seguro para CNN
    probabilidades_oof[val_idx] = modelo.predict(X_f_val, verbose=0)

pd.DataFrame(probabilidades_oof, columns=[
             f'prob_cnn_{c}' for c in encoder.classes_]).to_csv('oof_cnn.csv', index=False)

# 2. Gerar Probabilidades do Teste
print("\nTreinando CNN final com 100% dos dados para o Teste 1...")
modelo_final = criar_modelo()
modelo_final.fit(tf.convert_to_tensor(X_train_raw, dtype=tf.string),
                 y_train, epochs=10, batch_size=32, verbose=0)

df_test = limpar_dados(pd.read_excel(
    'test1-rotulado-para-teste-estatistico.xlsx'))
X_test_tensor = tf.convert_to_tensor(
    df_test['resp_text'].tolist(), dtype=tf.string)

probs_teste = modelo_final.predict(X_test_tensor, verbose=0)
pd.DataFrame(probs_teste, columns=[f'prob_cnn_{c}' for c in encoder.classes_]).to_csv(
    'prob_teste1_cnn.csv', index=False)
print("Ficheiros 'oof_cnn.csv' e 'prob_teste1_cnn.csv' gerados com sucesso.")
