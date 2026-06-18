import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from preprocess import X, test_X, label, test_label, tokens_dict

X = np.asarray(X, dtype=np.int32)
test_X = np.asarray(test_X, dtype=np.int32)
label = np.asarray(label, dtype=np.int32)
test_label = np.asarray(test_label, dtype=np.int32)
vocab_size = len(tokens_dict)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(4),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(X, label, epochs=10, validation_data=(test_X, test_label))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_X,  test_label, verbose=2)
