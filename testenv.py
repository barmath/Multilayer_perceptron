import random

hidden_layer = []
hidden_layer_length = 3
input_length = 63

for _ in range(hidden_layer_length):
    temp = []
    for _ in range(input_length):
        temp.append(random.uniform(-0.5, 0.5))
    hidden_layer.append(temp)

for i in range(hidden_layer_length):
    temp = []
    print(hidden_layer[i])