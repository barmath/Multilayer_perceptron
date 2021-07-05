# Importações de bibliotecas
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


class Mlp(object):
    """Classe utilizada para as definir as funções básicas do Multilayer Perceptron
       
    """
    def __init__(self, alpha, activation_function, hidden_layer_length, input_length, output_length):
        """Contrutor padrão que instacia o MlP.
    
            Recebe condições iniciais para instaciar a Multilayer perceptron
            Defini variáveis globais e chama funções para definir condições 
            básicas de pesos e biases 

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
        """Define condições iniciais da camada escondida.
    
           Cria vetores que representarão os pesos para camada escondida
           com a utilizacao da funcoes auxiliares do numpy
           np.random: define valores aleatorios
           reshape: cria matriz com as dimencoes dadas

        """

        # Matriz contendo todos pesos 
        self.hidden_layer_weight = np.random.random(self.input_length * self.hidden_layer_length).reshape(self.input_length, self.hidden_layer_length)

        # Vetor de entrada da camada escondida com o tamanho dela mesmo
        self.hidden_layer_in = np.empty(self.hidden_layer_length)

        # Vetor de saida da camada escondida com o tamanho dela mesmo
        self.hidden_layer_out = np.empty(self.hidden_layer_length)

    def setup_output_layer(self):
        """Define condições iniciais da camada de saída.
    
           Cria vetores que representarão os pesos para camada escondida
           com a utilização da funções auxiliares do numpy
           np.random: define valores aleatorios
           reshape: cria matriz com as dimencoes dadas
           empty: cria um vetor vazio

        """

        # Matriz com todos os pesos da camada de saida que tem como peso a
        self.output_layer_weight = np.random.random(self.hidden_layer_length *  self.output_length).reshape(self.hidden_layer_length, self.output_length)

        # Matriz com todos os pesos da camada de saida 
        self.output_layer_in = np.empty(self.hidden_layer_length)

    def set_biases(self):
        """Define os bias da camada escondida e de saída.
    
           Cria vetores que representarão os bias para camada escondida
           com a utilização da funções auxiliares do numpy
           np.random: cria um vetor com valores aleatórios
           random: valores float de [0.0,1.0)

        """
    
        # Matriz com todos os bias da camada escondida 
        self.hidden_layer_bias = np.random.random(self.hidden_layer_length)

        # Matriz com todos os bias da camada de saída
        self.output_layer_bias = np.random.random(self.output_length)

    def feed_forward(self, data):
        """Realiza o feedfoward para computação dos neurônios das camadas escondida e de saída.
    
           Computa e calcula os pesos de cada neurônio da camada escondida e de saída
           com a utilizacao da funcoes auxiliaries do numpy
           np.matmul: realiza multiplicação de matrizes

           Args:
               data: Vetor que representa os valores dos neurônios de entrada
           
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
    
    def backpropagation(self, labels, output_layer_out, data):
        """Realiza a retropropagação do erro na rede neural.
    
        Calula as correções dos pesos dos neurônios e aplica a retropropagação do erro 
        através de diversas equações a fim de gerar um melhor desempenho no aprendizado.
        Utilizamos funções auxiliares do numpy
        np.matmul: realiza multiplicação de matrizes
        reshape: cria matriz com as dimenções dadas
            
        Args:
            labels: Vetor passado para representar as saídas esperadas

            output_layer_out:  Vetor que representa as saídas dos neurônios 
            da camada de saída com a função de ativação 

            data: Vetor que representa os valores dos neurônios de entrada

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

    def fit(self, training_dataset, training_dataset_labels, threshold = 1):
        """Faz o trainamento baseado nos dados de treinamentos e seus labels.
    
        Atravez da condicao de parada do erro medio,
        fazemos o calculo deste e equanto este erro medio nao for maior que
        o threshold definido rodamos o treinamento chamando o feed forward e 
        o back propagation para os inputs passados 
            
        Args:
            x : Vetor passado para representar as saídas esperadas

            output_layer_out:  Vetor que representa as saídas dos neurônios 
            da camada de saída com a função de ativação 

            data: Vetor que representa os valores dos neurônios de entrada

        """

        squaredError = 2 * threshold
        counter = 0
        grafico = []

        # Inicializacao dos arquivos para escritas dos parametros de saida e logs
        erroFile = open("saida/erro-medio-quadrado.txt", "w")
        pesosEntradaFile = open("saida/pesos-iniciais.txt", "w")
        pesosSaidaFile = open("saida/pesos-finais.txt", "w")

        # Definicao da condicao de parada para a execucao do back propagation
        while (squaredError > threshold):
            squaredError = 0
            for index in range(len(training_dataset)):
                # pegando uma linha do dataset 
                Xp = training_dataset[index]
                # pegando o que desejo obter, pelo label 
                Yp = training_dataset_labels[index]

                # Chama forward
                results = self.feed_forward(Xp)

                # Valor obtido
                Op = results

                # Calculando o erro 
                error =  np.subtract( Yp, Op )
                squaredError = squaredError + np.sum( np.power(error, 2) )

                # Chama backpropagation passando resultados do feed_forward (Op), 
                # resultados passados do dataset (Yp)
                # e os resultados esperados (Xp)
                self.backpropagation(Yp, Op, Xp)


            squaredError = squaredError / len(training_dataset)

            print("Erro ao quadrado : ", squaredError)
            # Escrita do erro medio quadrado para monitorar como esta execucao 
            erroFile.write(f"Erro medio quadrado: {squaredError}\n")
            # Incrementar o numero de epocas 
            counter += 1
            # Alimenta dados para o grafico
            grafico.append(squaredError)

        # Escreve numero de iteracoes necessarias
        erroFile.write(f"Iteracoes necessarias: {counter}")
        erroFile.close()

        # Escreve Informacoes referentes aos pesos de estrada da camada escondida antes o treinamento
        pesosEntradaFile.write(f"Pesos de entrada para a camada escondida apos o: \n{self.hidden_layer_weight}")
        pesosEntradaFile.close()

        # Escreve Informacoes referentes aos pesos de estrada da camada saída apos o treinamento
        pesosSaidaFile.write(f"Pesos de entrada para a camada de saída apos o treinamento: \n{self.output_layer_weight}")
        pesosSaidaFile.close()

        # Cria imagem gráfico dos erros médios
        plt.plot(grafico)
        # Salva imagem gráfico dos erros médios
        plt.savefig('saida/grafico.png')

    def predict(self, data, name_of_file, y_true):
        """Realiza a predição dos resultados.
    
           Mostra a saída da predição dos resultados feitos pelo algoritmo
           com a utilizacao da funcoes auxiliaries do numpy, e criação de txt para apresentação
           dos resultados
           np.round_: cria um vetor com decimais

           Args:
               data : Vetor que representa os valores dos neurônios de entrada

               name_of_file: Nome do arquivo a ser lido

               y_true: Respostas esperadas para matriz de confusão
           
        """
        #predicoes = open(f"Saida/predicoes-{name_of_file}.txt", "w")

        # Aplica o feedfoward nos neurônios de entrada
        output_layer_out = self.feed_forward(data)

        return output_layer_out
