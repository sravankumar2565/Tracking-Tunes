import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LeakyReLU, Input, Dropout, BatchNormalization
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np
import ast
import joblib
import matplotlib.pyplot as plt

# Load and preprocess data
songs = pd.read_csv('training_data.csv').drop('track_id', axis=1)
songs = songs[songs['genre'] != 'International']
songs = songs.drop('dom_key', axis=1)
songs['mfcc'] = songs['mfcc'].apply(lambda x: np.array([float(i) for i in x.strip('[]').split(',')]))
songs['chroma'] = songs['chroma'].apply(lambda x: np.array(ast.literal_eval(x)))
mfcc_df = pd.DataFrame(songs['mfcc'].tolist(), index=songs.index)
chroma_df = pd.DataFrame(songs['chroma'].tolist(), index=songs.index)
songs = songs.drop(['mfcc', 'chroma'], axis=1)
songs = pd.concat([songs, mfcc_df, chroma_df], axis=1)
songs.columns = songs.columns.astype(str)

# Prepare features and labels
x, y = songs.drop('genre', axis=1), songs['genre']
y = y.astype('category').cat.codes
genre_labels = pd.get_dummies(songs['genre']).columns
joblib.dump(genre_labels, 'boi.pkl')

# Scale features
scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
x = x.values.reshape(x.shape[0], x.shape[1], 1)

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
joblib.dump(scaler, 'scaler.pkl')

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=1)
x_train, y_train = smote.fit_resample(x_train.reshape(x_train.shape[0], -1), y_train)
x_train = x_train.reshape(x_train.shape[0], x.shape[1], 1)

# One-hot encode labels
y_train = pd.get_dummies(y_train).values
y_test = pd.get_dummies(y_test).values

# Build the model
model = Sequential()
model.add(Input(shape=(x_train.shape[1], 1)))
# First Convolutional Block
model.add(Conv1D(64, kernel_size=3))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.2))
model.add(MaxPooling1D(pool_size=2))
# Second Convolutional Block
model.add(Conv1D(128, kernel_size=3))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.2))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
# Flatten and Dense layers
model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.3))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.3))
# Output layer (7 genres)
model.add(Dense(7, activation='softmax'))

# Compile the model with Top-3 Accuracy metric
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', lambda y_true, y_pred: tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)]
)

# Define callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6)

# Train the model
history = model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_test, y_test), callbacks=[early_stopping, learning_rate_reduction])

# Evaluate the model
test_loss, test_accuracy, test_top_3_accuracy = model.evaluate(x_test, y_test)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Top-3 Accuracy: {test_top_3_accuracy}")

# Option to save the model
k = int(input("Enter 1 or 0: "))
if k == 1:
    model.save('save2.keras')

# Print model summary
model.summary()

# Plot training & validation accuracy
plt.figure(figsize=(12, 5))

# Training vs Validation Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs. Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Training vs Validation Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs. Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Plot training vs. validation Top-3 Accuracy
plt.figure(figsize=(6, 5))
plt.plot(history.history['lambda'], label='Training Top-3 Accuracy')
plt.plot(history.history['val_lambda'], label='Validation Top-3 Accuracy')
plt.title('Training vs. Validation Top-3 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Top-3 Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
