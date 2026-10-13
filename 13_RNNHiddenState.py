words = ["food","was","not","good"]

hidden_state = "Empty memory" 

print("input tokens :",words)

print("initial hidden state:",hidden_state)

for index,word in enumerate(words):
    print("timestep :",index+1)
    print("current memory :",word)
    print("privious memory :",hidden_state)

    hidden_state = "memory after reading " + " ".join(words[:index + 1]) + ""

    print("updated memory :",hidden_state)

    print("-"*30)