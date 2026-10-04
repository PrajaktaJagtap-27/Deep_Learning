
from tensorflow.keras.preprocessing.text import Tokenizer

sentences =[
    "food was good",
    "food was bad",
    "food was not good"
]

tokenizer = Tokenizer()

tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)

for sentence, sequence in zip(sentences, sequences):
   print("sentence:", sentence)
   print("sequence:", sequence)
   print("---------------------------------------")
