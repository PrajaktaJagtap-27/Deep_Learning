
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences =[
    "food was good",
    "food was bad",
    "food was not good"
]

tokenizer = Tokenizer()

tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)

print("original sequences:")
for sequence in  sequences:
    print(sequence,"Length :",len(sequence))

print("All sequences are of different length")

max_length = 4
padded_sequences = pad_sequences(
    sequences,
    maxlen=max_length,
    padding="pre"
    )

for sentences, sequence, padded in zip(sentences, sequences, padded_sequences):
   print("sentence:", sentences)
   print("Original sequence:", sequence)
   print("Padded sequence:", padded)
   print("---------------------------------------")
