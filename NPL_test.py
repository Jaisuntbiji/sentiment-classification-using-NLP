from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

loaded_model = load_model("model_nlp.h5")

sentences = [
    'granny starting to far spider in the garden might be real'
]

tokenizer = Tokenizer(num_words=100,oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences,maxlen=32,padding='post',truncating='post')

pred = loaded_model.predict(padded)
print(pred)