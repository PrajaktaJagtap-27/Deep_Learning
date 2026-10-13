#ht = tanh(Wx * Xt  + Wh * ht-1 +b)
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def MarvellousRNNPredictions():
    print("Calculation of RNN")

    #food was not good 
    inputs = [1,2,5,3]

    Hidden_state = 0

    #RNN parameter
    Wx = 0.5
    Wh = 0.8
    bias = 0.1

    for  time_step, x in enumerate(inputs):
        previous_hidden_state = Hidden_state

        weighted_input = Wx * x
        weighted_memory = Wh * previous_hidden_state

        total = weighted_input  + weighted_memory + bias

        Hidden_state = np.tanh(total)

        print("Time step :",time_step+1)
        print("Input :",x)
        print("Hidden State :",Hidden_state)
        print("-"*30)
    #step 2
    print("Final hidden state",Hidden_state)

    #step 3 = output layer
    Wy = 1.0
    output_bias = 0.0

    output = (Wy * Hidden_state) + output_bias

    print("raw Output",output)
def main():
    MarvellousRNNPredictions()

if __name__ == "__main__":
    main()