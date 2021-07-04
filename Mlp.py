'''
# z_in (hidden_layer_in)
# z (hidden_layer_out)
# w0 (output_layer_bias)
# w (output_layer_weight)
# v0 (hidden_layer_bias)
# v  (hidden_layer_weight)
# y_in (output_layer_in) = valores vindos dos neuronios da camada escondida
# y (output_layer_out) = valor de y_in com a função de ativação
'''

# Importações de bibliotecas
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

class Mlp(object):
    """Classe utilizada para as definir as funcoes basicas do Multilayer Perceptron
       
    """
    def __init__(self, alpha, activation_function, hidden_layer_length, input_length, output_length):
        """Contrutor padrao que instacia o MlP.
    
            Recebe condicoes inicial para instaciar a Multilayes perceptron
            Seta variaveis globais e chama funcoes para setar condicoes 
            basicas de pesos e biases 

            Args:
                alpha : Float passado para representar o alpha que e 
                usado na hora do backpropagation (taxa de aprendizado)

                activation_function : Funcao que e usada no backpropagation 

                hidden_layer_length : Integer que representa o tamanho da camada 
                                    escondida

                input_length : Integer que representa tamanho da entrada
                output_length : Integer que representa tamanho da saida

        """
        # Seta todas variaveis globais utilizadas pra configurar a MLP
        
        self.alpha = alpha
        self.hidden_layer_length = hidden_layer_length
        self.activation_function = activation_function
        self.input_length = input_length
        self.output_length = output_length

        # Seta parte da hidden e outp
        self.setup_hidden_layer()
        self.setup_output_layer()
        
        # Seta todos os bias
        self.set_biases()

    def setup_hidden_layer(self):
        """Define condicoes iniciais da camada escondida.
    
           Cria vetores que representarao os pesos para camada escondida
           com a utilizacao da funcoes auxiliaries do numpy
           np.random define valores aleatorios
           reshape cria matriz com as dimencoes dadas

        """

        # Matriz contendo todos pesos 
        self.hidden_layer_weight = np.random.random(self.input_length * self.hidden_layer_length).reshape(self.input_length, self.hidden_layer_length)
        # Vetor de entrada da camada escondida com o tamanho dela mesmo
        self.hidden_layer_in = np.empty(self.hidden_layer_length)
        # Vetor de saida da camada escondida com o tamanho dela mesmo
        self.hidden_layer_out = np.empty(self.hidden_layer_length)

    def setup_output_layer(self):
        """Define condicoes iniciais da camada de saida.
    
           Cria vetores que representarao os pesos para camada escondida
           com a utilizacao da funcoes auxiliaries do numpy
           np.random define valores aleatorios
           reshape cria matriz com as dimencoes dadas
           empty cria um valor vazio

        """

        # Matriz com todos os pesos da camada de saida que tem como peso a
        self.output_layer_weight = np.random.random(self.hidden_layer_length *  self.output_length).reshape(self.hidden_layer_length, self.output_length)
        # Matriz com todos os pesos da camada de saida 
        self.output_layer_in = np.empty(self.hidden_layer_length)

    def set_biases(self):
        """Define os bias da camada escondida e de saída.
    
           Cria vetores que representarao os bias para camada escondida
           com a utilizacao da funcoes auxiliaries do numpy
           np.random define valores aleatorios e random valores float de [0.0,1.0)

        """
        
        # Matriz com todos os bias da camada escondida 
        self.hidden_layer_bias = np.random.random(self.hidden_layer_length)
        # Matriz com todos os bias da camada de saída
        self.output_layer_bias = np.random.random(self.output_length)

    def feed_forward(self, data):
        """Realiza o feedfoward para computação dos neurônios das camadas escondida e de saída.
    
           Computa e calcula os pesos de cada neurônio da camada escondida e de saída
           com a utilizacao da funcoes auxiliaries do numpy
           np.matmul que realiza multiplicação de matrizes

           Args:
               data : Vetor que representa os valores dos neurônios de entrada
           
           Returns:
                  output_layer_out: Vetor que representa os valores dos neurônios de saídas após a aplicação da função de ativação

        """
        # Nessa função, tomamos como base as equações apresentadas no algoritmo dos slides 
        # "Redes Neurais Artificiais - Perceptron Simples e Multilayer Perceptron"

        # Calcula os neurônios da camada escondida através dos pesos, os valores dos dados de entrada e bias 
        self.hidden_layer_in = np.matmul(data, self.hidden_layer_weight) + self.hidden_layer_bias

        # Aplica a função de ativação em cada neurônio da camada escondida
        self.hidden_layer_out = self.activation_function(self.hidden_layer_in)

        # # Calcula os neurônios da camada de saída através dos pesos, os valores dos neurônios da camada escondida após a aplicação da função de ativação e bias 
        self.output_layer_in = np.matmul(self.hidden_layer_out, self.output_layer_weight) + self.output_layer_bias

        # Aplica a função de ativação em cada neurônio da camada de saída
        output_layer_out = self.activation_function(self.output_layer_in)

        return output_layer_out
    
    #TODO : APAGAR
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
        print(f"Pesos de entrada para a camada escondida apos o: {self.hidden_layer_weight}")
        print(f"Pesos de entrada para a camada de saida apos o treinamento: {self.output_layer_weight}")

        # printa gráfico dos erros médios
        plt.plot(erros_medios)
        plt.show()

    def backpropagation(self, labels, output_layer_out, data):
        """Realiza a retropropagação do erro na rede neural.
    
        Calula as correções dos pesos dos neurônios e aplica a retropropagação do erro 
        através de diversas equações a fim de gerar um melhor desempenho no aprendizado.
        Utilizamos funções auxiliares do numpy
        np.matmul que realiza multiplicação de matrizes
        reshape cria matriz com as dimencoes dadas
            
        Args:
            labels : Vetor passado para representar as saídas esperadas

            output_layer_out :  Vetor que representa as saídas dos neurônios 
            da camada de saída com a função de ativação 

            data : Vetor que representa os valores dos neurônios de entrada

        """
        
        # Nessa função, tomamos como base as equações apresentadas no algoritmo dos slides 
        # "Redes Neurais Artificiais - Perceptron Simples e Multilayer Perceptron"
        
        # Esse bloco representa o passo 6 do algoritmo do slide, em que calculamos o delta_k e os deltas wjk e w0k
        # Respectivamente temos delta_wjk como delta_output_layer_weights e deltaw0k como delta_output_layer_bias

        # Calculo de erro (Resposta esperada - Resposta dada pelo algoritmo)
        error = (labels - output_layer_out)  
        # Calculo do termo de correção de erro 
        delta_k =  error * self.activation_function.prime(self.output_layer_in) 
        
        # Calculo da correção de bias de cada unidade de saída
        delta_output_layer_bias = self.alpha * delta_k  
        # Calculo da correção de pesos de cada unidade de saída
        delta_output_layer_weights = self.alpha * np.matmul(self.hidden_layer_out.reshape(-1, 1), delta_k.reshape(1, -1))  

        # Esse bloco representa o passo 7 do algoritmo do slide, em que calculamos o delta_in_j e o delta_j bem como os deltas vij e v0j
        # Respectivamente temos delta_vij como delta_output_layer_weights e deltav0j como delta_output_layer_bias
        
        # Calculo do termo de correção de erro utilizando o termo de correção da camada posterior
        delta_in_j = np.matmul(self.output_layer_weight, delta_k)  
        delta_j = delta_in_j * self.activation_function.prime(self.hidden_layer_in) 
        
        # Calculo da correção de bias de cada unidade escondida
        delta_hidden_layer_bias = self.alpha * delta_j   
        # Calculo da correção de pesos da camada de cada unidade escondida
        delta_hidden_layer_weight = self.alpha * np.matmul(data.reshape(-1, 1), delta_j.reshape(1, -1)) 

        # Esse bloco representa o passo 8 do algoritmo do slide, em que calculamos as alterações dos bias e pesos de cada unidadde de saída e cada unidade escondida
        
        # Atualização dos pesos e bias de cada unidade de saída
        self.output_layer_bias = self.output_layer_bias + delta_output_layer_bias
        self.output_layer_weight = self.output_layer_weight + delta_output_layer_weights
       
        # Atualização dos pesos e bias de cada unidade escondida
        self.hidden_layer_bias = self.hidden_layer_bias + delta_hidden_layer_bias
        self.hidden_layer_weight = self.hidden_layer_weight + delta_hidden_layer_weight

    def fit_demo(self, x, t, threshold = 0.1):
        squaredError = 2 * threshold
        counter = 0
        grafico = []
        erroFile = open("saida/erro-medio-quadrado.txt", "w")
        pesosEntradaFile = open("saida/pesos-iniciais.txt", "w")
        pesosSaidaFile = open("saida/pesos-finais.txt", "w")

        while (squaredError > threshold):
            squaredError = 0
            for index in range(len(x)):
                # pegando uma linha do dataset 
                Xp = x[index]
                # pegando o que desejo obter, pelo label 
                Yp = t[index]

                # Chama forward
                results = self.feed_forward(Xp)

                # Valor obtido
                Op = results

                # Calculando o erro 
                error =  np.subtract( Yp, Op )

                squaredError = squaredError + np.sum( np.power(error, 2) )

                self.backpropagation(Yp, Op, Xp)


            squaredError = squaredError / len(x)

            erroFile.write(f"Erro medio quadrado: {squaredError}\n")
            counter += 1
            grafico.append(squaredError)

        erroFile.write(f"Iteracoes necessarias: {counter}")
        erroFile.close()
        pesosEntradaFile.write(f"Pesos de entrada para a camada escondida apos o: \n{self.hidden_layer_weight}")
        pesosEntradaFile.close()
        pesosSaidaFile.write(f"Pesos de entrada para a camada de saída apos o treinamento: \n{self.output_layer_weight}")
        pesosSaidaFile.close()
        # printa gráfico dos erros médios
        plt.plot(grafico)
        plt.savefig('saida/grafico.png')

    def predict(self, data, name_of_file, y_true):
        """Realiza a predição dos resultados.
    
           Mostra a saída da predição dos resultados feitos pelo algortitmo
           com a utilizacao da funcoes auxiliaries do numpy
           np.round_ cria um vetor com decimais

           Args:
               data : Vetor que representa os valores dos neurônios de entrada
           
        """
        predicoes = open(f"saida/predicoes-{name_of_file}.txt", "w")
        # Aplica o feedfoward nos neurônios de entrada
        output_layer_out = self.feed_forward(data)
        y_pred = np.round_(output_layer_out)

        # Escreve o tamanho do test label
        len_test_data = len(data)
        predicoes.write(f"test labels: {len_test_data}\n")

        # Mostra a saída do resultado
        predicoes.write(str(np.round_(output_layer_out)))

        predicoes.write("\n\nMatrix de confusao:\n")
        predicoes.write(str(confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))))

        predicoes.write("\n\nRelatorio de classificacao:\n")
        target_names = ['A', 'B', 'C', 'D', 'E', 'J', 'K']
        predicoes.write(str(classification_report(y_true, y_pred, target_names=target_names)))

        predicoes.close()
