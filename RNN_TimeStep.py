sentence ="foof was not good"

words = sentence.split()

print("Actual sentence is :",sentence)

for index, word in enumerate(words):
    print("Timesteps :",index+1," : ",word)
