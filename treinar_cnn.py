from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Configuração para evitar alocação total da memória da GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 1. Carregamento dos Dados
print("Carregando dados...")
df = pd.read_excel('train.xlsx')
df = df.dropna(subset=['resp_text', 'clarity'])

# Limpeza definitiva (Regex): Remove \x95, \x96 e qualquer caractere não-padrão.
# Mantém apenas letras, números, espaços e pontuação básica (.,;:!?()-).
df['resp_text'] = df['resp_text'].astype(
    str).str.replace(r'[^\w\s.,;:!?()-]', ' ', regex=True)
# Remove espaços duplos residuais deixados pela limpeza
df['resp_text'] = df['resp_text'].str.replace(
    r'\s+', ' ', regex=True).str.strip()

X = df['resp_text'].to_numpy(dtype=object)

# Codificação dos rótulos (c1, c234, c5 para 0, 1, 2)
encoder = LabelEncoder()
y = encoder.fit_transform(df['clarity'].astype(str).to_numpy())

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Converte os arrays nativos para tensores de texto do TensorFlow
X_train = tf.convert_to_tensor(X_train, dtype=tf.string)
X_val = tf.convert_to_tensor(X_val, dtype=tf.string)


# 2. Configurações da CNN
VOCAB_SIZE = 25000
MAX_LENGTH = 1000  # Ajustado para cobrir a mediana/média de caracteres dos textos
EMBEDDING_DIM = 128
NUM_CLASSES = len(np.unique(y))
BATCH_SIZE = 32
EPOCHS = 50

# 3. Vetorização de Texto
print("Vetorizando texto...")
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_sequence_length=MAX_LENGTH
)
vectorizer.adapt(X_train)

# 4. Arquitetura do Modelo
model = tf.keras.Sequential([
    vectorizer,
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM),
    tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Callbacks para execução noturna


class PrintSeguroCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Época {epoch+1}/{self.params['epochs']} concluída | "
              f"Acurácia: {logs.get('accuracy', 0):.4f} - Loss: {logs.get('loss', 0):.4f} | "
              f"Val_Acurácia: {logs.get('val_accuracy', 0):.4f} - Val_Loss: {logs.get('val_loss', 0):.4f}")


callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'melhor_modelo_cnn.keras',
        monitor='val_loss',
        save_best_only=True
    ),
    PrintSeguroCallback()  # Adiciona a nossa impressão à prova de falhas
]

# 6. Treinamento
print("Iniciando treinamento seguro (sem barra de progresso)...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=0  # SILENCIA COMPLETAMENTE A BARRA PROBLEMÁTICA DO KERAS
)

# 7. Avaliação Final
print("Avaliando melhor modelo na base de validação...")
y_pred_prob = model.predict(X_val)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\nRelatório de Classificação:")
print(classification_report(y_val, y_pred, target_names=encoder.classes_))
