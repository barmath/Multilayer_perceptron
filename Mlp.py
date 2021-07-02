#from Matrix_operations import Matrix_operations

import random
import numpy as np


class Mlp(object):
    def __init__(self, alpha, activation_function, hidden_layer_length, input_length, output_length):
        self.alpha = alpha
        self.hidden_layer_length = hidden_layer_length
        self.activation_function = activation_function
        self.input_length = input_length
        self.output_length = output_length
        self.setup_hidden_layer()
        self.setup_output_layer()
        self.set_biases()

    def show_structure(self):  
        print("hidden_layer_weight")
        print(self.hidden_layer_weight)
        print("hidden_layer_in")
        print(self.hidden_layer_in)
        print("hidden_layer_out")
        print(self.hidden_layer_out)
        print("hidden_layer_bias")
        print(self.hidden_layer_bias)
        print("output_layer_weight")
        print(self.output_layer_weight)
        print("output_layer_in")
        print(self.output_layer_in)
        print("output_layer_bias")
        print(self.output_layer_bias) 
    
    def setup_hidden_layer(self):
        self.hidden_layer_weight = np.random.random(self.input_length * self.hidden_layer_length).reshape(self.input_length, self.hidden_layer_length)
        
        self.hidden_layer_in = np.empty(self.hidden_layer_length)
        self.hidden_layer_out = np.empty(self.hidden_layer_length)

    def setup_output_layer(self):
        self.output_layer_weight = np.random.random(self.hidden_layer_length *  self.output_length).reshape(self.hidden_layer_length, self.output_length)
        self.output_layer_in = np.empty(self.hidden_layer_length)

    def set_biases(self):
        self.hidden_layer_bias = np.random.random(self.hidden_layer_length)
        self.output_layer_bias = np.random.random(self.output_length)

    def feed_forward(self, data):
        #self.show_structure()

        #print("data : ",data.shape)
        #print("self.hidden_layer_weight : ",self.hidden_layer_weight.shape)

        # z_in
        self.hidden_layer_in = np.matmul(data, self.hidden_layer_weight) + self.hidden_layer_bias
        #print("self.hidden_layer_in: ",self.hidden_layer_in.shape)
        #print("self.hidden_layer_bias: ",self.hidden_layer_bias)
        # z
        self.hidden_layer_out = self.activation_function(self.hidden_layer_in)
        #print("self.hidden_layer_out : ",self.hidden_layer_out.shape)
        #print("self.output_layer_weight : ",self.output_layer_weight.shape)
        # y_in
        # + 
        self.output_layer_in = np.matmul(self.hidden_layer_out, self.output_layer_weight) + self.output_layer_bias
        ##print("self.output_layer_in: ",self.output_layer_in.shape)
        # y
        output_layer_out = self.activation_function(self.output_layer_in)

        return output_layer_out
    
    def predict(self, data):

        output_layer_out = self.feed_forward(data)
        #print(len(output_layer_out))
        #print(output_layer_out)
        #for row in output_layer_out:
        print(np.round_(output_layer_out))
    

    def fit(self, x, t, max_epochs=float('inf'), max_error=0.5):
        '''
            x -> data
            t -> saída esperada
            trheshold -> limiar de comparação de erro para saída do laço
        '''
        epoch = 0
        erro_medio = float('inf')
        # lista com os erros totais médios para cada época
        erros_medios = []

        while erro_medio > max_error and epoch < max_epochs:
            epoch += 1

            squared_error = 0 
            if epoch % 100 == 0:
                print(f"epoch: {epoch}")
            #erros = [0 for x in range(len(x))]
            erros = np.zeros(len(x))

            for index in range(len(x)): 

                #Chama feedfoward
                y = self.feed_forward(x[index])

                #ENTENDEEEEEEEEEEEEEEEEEEEEER!!!!
                
                erro = np.linalg.norm(t[index] - y)
                erros[index] = erro

                #squared_error = squared_error + sum(pow(erros[index],2))

                #Chama backpropagation
                self.backpropagation(t[index], y, x[index])
            
                #squared_error = squared_error/ len(x)

            #Calcula erro médio
            erro_medio = erros.mean()
            erros_medios.append(erro_medio)

        print(f"epochs: {epoch}")
        # Printa o erro médio após fim do aprendizado.
        print(f"erro_medio: {erro_medio}")
        
        # Printa os pesos atribuidos às camadas.
        print(f"Pesos de entrada para a camada escondida após o: {self.hidden_layer_weight}")
        print(f"Pesos de entrada para a camada de saída após o treinamento: {self.output_layer_weight}")

    # trocar argumentos para : labels, output_layer_out, data
    def backpropagation(self, t, y, x):
        # z_in (hidden_layer_in)
        # z (hidden_layer_out)
        # w0 (output_layer_bias)
        # w (output_layer_weight)
        # v0 (hidden_layer_bias)
        # v  (hidden_layer_weight)
        # y_in (output_layer_in) = valores vindos dos neuronios da camada escondida
        # y (output_layer_out) = valor de y_in com a função de ativação
        
        '''
        '''
        '''
        Error = (t - y)
        Delta = Error * self.activation_function.prime(self.output_layer_in)
        Delta_output_layer_bias = self.alpha * Delta
        Delta_output_layer_weights = self.alpha * np.matmul(self.hidden_layer_out.reshape(-1, 1), Delta.reshape(1, -1))
        '''
        '''
        Delta_in_j = np.matmul(self.output_layer_weight, Delta)
        Delta_j = Delta_in_j * self.activation_function.prime(self.hidden_layer_in)
        
        # trocar nome
        Delta_v0_j = self.alpha * Delta_j
        Delta_vij = self.alpha * np.matmul(x.reshape(-1, 1), Delta_j.reshape(1, -1))
        
        self.output_layer_bias = self.output_layer_bias + Delta_j
        self.output_layer_weight = self.output_layer_weight + Delta_output_layer_weights

        self.hidden_layer_bias = self.hidden_layer_bias + Delta_v0_j
        self.hidden_layer_weight = self.hidden_layer_weight + Delta_vij
        '''
        Delta = (t - y) * self.activation_function.prime(self.output_layer_in)
        Delta_output_layer_bias = self.alpha * Delta
        Delta_output_layer_weights = self.alpha * np.matmul(self.hidden_layer_out.reshape(-1, 1), Delta.reshape(1, -1))
        '''
        '''
        Delta_in_j = np.matmul(self.output_layer_weight, Delta)
        Delta_j = Delta_in_j * self.activation_function.prime(self.hidden_layer_in)
        
        # trocar nome
        Delta_v0_j = self.alpha * Delta_j
        Delta_vij = self.alpha * np.matmul(x.reshape(-1, 1), Delta_j.reshape(1, -1))
        
        self.output_layer_bias = self.output_layer_bias + Delta_output_layer_bias
        self.output_layer_weight = self.output_layer_weight + Delta_output_layer_weights
        
        self.hidden_layer_bias = self.hidden_layer_bias + Delta_v0_j
        self.hidden_layer_weight = self.hidden_layer_weight + Delta_vij