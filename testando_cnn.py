import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# 1. Recuperar o codificador de rótulos do treino
df_train = pd.read_excel('train.xlsx')
df_train = df_train.dropna(subset=['resp_text', 'clarity'])
encoder = LabelEncoder()
encoder.fit(df_train['clarity'].astype(str))

# 2. Carregar e limpar os dados de Teste 1
print("A carregar e a normalizar o conjunto de teste...")
df_test = pd.read_excel('test1-rotulado-para-teste-estatistico.xlsx')

# Aplica a mesma limpeza profunda usada no treino
df_test['resp_text'] = df_test['resp_text'].astype(
    str).str.replace(r'[^\w\s.,;:!?()-]', ' ', regex=True)
df_test['resp_text'] = df_test['resp_text'].str.replace(
    r'\s+', ' ', regex=True).str.strip()

X_test = tf.convert_to_tensor(df_test['resp_text'].tolist(), dtype=tf.string)
y_true = df_test['clarity'].tolist()

# 3. Carregar o modelo CNN treinado
print("A carregar o modelo CNN...")
modelo_cnn = tf.keras.models.load_model('melhor_modelo_cnn.keras')

# 4. Fazer Predições
print("A gerar predições...")
predicoes_prob = modelo_cnn.predict(X_test)
predicoes_num = np.argmax(predicoes_prob, axis=1)

# Converter matriz numérica de volta para (c1, c234, c5)
predicoes_rotulos = encoder.inverse_transform(predicoes_num)

# 5. Avaliação e Exportação
print("\n" + "="*50)
print("RESULTADOS FINAIS DA CNN NO CONJUNTO DE TESTE 1")
print("="*50)
print(f"Acurácia: {accuracy_score(y_true, predicoes_rotulos):.4f}")
print("\nRelatório de Classificação:")
print(classification_report(y_true, predicoes_rotulos))

# Salvar dados para o teste estatístico
df_saida = pd.DataFrame({
    'resp_text': df_test['resp_text'],
    'rotulo_real': y_true,
    'predicao_cnn': predicoes_rotulos
})
df_saida.to_csv('predicoes_cnn_teste1.csv', index=False, encoding='utf-8')
print("\nFicheiro guardado como 'predicoes_cnn_teste1.csv'.")
