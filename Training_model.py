import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from converter_json_to_pythonlist import sentences, labels  # Assuming sentences and labels are imported correctly

# Constants
vocab_size = 10000
embedding_dim = 16
max_length = 32
padding_type = 'post'
trunc_type = 'post'
oov_tok = "<OOV>"
model_save_path = "model_nlp.h5"

# Split data into training and testing sets
training_size = int(len(sentences) * 0.8)
training_sentences = sentences[:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[:training_size]
testing_labels = labels[training_size:]

# Tokenize the sentences
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Convert labels to numpy arrays of floats
training_labels = np.array(training_labels).astype('float32')
testing_labels = np.array(testing_labels).astype('float32')

# Convert padded sequences to numpy arrays
training_padded = np.array(training_padded)
testing_padded = np.array(testing_padded)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
num_epochs = 30
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)
#saveing the model
model.save(model_save_path)
print("Training complete")
