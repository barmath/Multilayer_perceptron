# Importações de bibliotecas
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt


class Mlp(object):
    """Classe utilizada para as definir as funções básicas do Multilayer Perceptron
       
    """
    def __init__(self, alpha, activation_function, hidden_layer_length, input_length, output_length):
        """Contrutor padrão que instancia o MlP.
    
            Recebe condições iniciais para instaciar a Multilayer perceptron
            Define variáveis globais e chama funções para definir condições 
            básicas de pesos e biases 

            Args:
                alpha : Float passado para representar o alpha que é 
                usado na hora do backpropagation (taxa de aprendizado)

                activation_function : Função que é usada no backpropagation e feedfoward 

                hidden_layer_length : Inteiro que representa o tamanho da camada 
                                      escondida

                input_length : Inteiro que representa tamanho da entrada

                output_length : Inteiro que representa tamanho da saida

        """

        # Define todas variaveis globais utilizadas pra configurar a MLP
        self.alpha = alpha
        self.hidden_layer_length = hidden_layer_length
        self.activation_function = activation_function
        self.input_length = input_length
        self.output_length = output_length

        # Define parte da hidden e output
        self.setup_hidden_layer()
        self.setup_output_layer()
        
        # Define todos os biases
        self.set_biases()

    def setup_hidden_layer(self):
        """Define condições iniciais da camada escondida.
    
           Cria vetores que representarão os pesos para camada escondida
           com a utilização da funções auxiliares do numpy
           np.random: define valores aleatórios
           reshape: cria matriz com as dimensões dadas

        """

        # Matriz contendo todos pesos que são definidos aleatoriamente (inicialmente)
        self.hidden_layer_weight = np.random.random(self.input_length * self.hidden_layer_length).reshape(self.input_length, self.hidden_layer_length)

        # Vetor de entrada da camada escondida com o tamanho dela mesma
        self.hidden_layer_in = np.empty(self.hidden_layer_length)

        # Vetor de saída da camada escondida com o tamanho dela mesma
        self.hidden_layer_out = np.empty(self.hidden_layer_length)

    def setup_output_layer(self):
        """Define condições iniciais da camada de saída.
    
           Cria vetores que representarão os pesos para camada escondida
           com a utilização da funções auxiliares do numpy
           np.random: define valores aleatórios
           reshape: cria matriz com as dimensões dadas
           empty: cria um vetor vazio

        """

        # Matriz com todos os pesos da camada de saida que são definidos aleatoriamente
        self.output_layer_weight = np.random.random(self.hidden_layer_length *  self.output_length).reshape(self.hidden_layer_length, self.output_length)

        # Matriz com todos os pesos da camada de saída 
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
           com a utilização da funções auxiliares do numpy
           np.matmul: realiza multiplicação de matrizes

           Args:
               data: Vetor que representa os valores dos neurônios de entrada
           
           Returns:
                  output_layer_out: Vetor que representa os valores dos neurônios de saída após a aplicação da função de ativação

        """
        # Nessa função, tomamos como base as equações apresentadas no algoritmo dos slides 
        # "Redes Neurais Artificiais - Perceptron Simples e Multilayer Perceptron"

        # Calcula os neurônios da camada escondida através dos pesos com os valores dos dados de entrada e bias 
        self.hidden_layer_in = np.matmul(data, self.hidden_layer_weight) + self.hidden_layer_bias

        # Aplica a função de ativação em cada neurônio da camada escondida
        self.hidden_layer_out = self.activation_function(self.hidden_layer_in)

        # Calcula os neurônios da camada de saída através dos pesos com os valores dos neurônios da camada escondida após a aplicação da função de ativação e bias 
        self.output_layer_in = np.matmul(self.hidden_layer_out, self.output_layer_weight) + self.output_layer_bias

        # Aplica a função de ativação em cada neurônio da camada de saída
        output_layer_out = self.activation_function(self.output_layer_in)

        # Retorna os valores preditos pelo cálculo de pesos
        return output_layer_out
    
    def backpropagation(self, labels, output_layer_out, data):
        """Realiza a retropropagação do erro na rede neural.
    
        Calcula as correções dos pesos dos neurônios e aplica a retropropagação do erro 
        através de diversas equações a fim de gerar um melhor desempenho no aprendizado.
        Utilizamos funções auxiliares do numpy
        np.matmul: realiza multiplicação de matrizes
        reshape: cria matriz com as dimensões dadas
            
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

        # Cálculo de erro (Resposta esperada - Resposta dada pelo algoritmo)
        error = (labels - output_layer_out)  
        # Cálculo do termo de correção de erro 
        delta_k =  error * self.activation_function.prime(self.output_layer_in) 
        
        # Cálculo da correção de bias de cada unidade de saída
        delta_output_layer_bias = self.alpha * delta_k  
        # Cálculo da correção de pesos de cada unidade de saída
        delta_output_layer_weights = self.alpha * np.matmul(self.hidden_layer_out.reshape(-1, 1), delta_k.reshape(1, -1))  

        # Esse bloco representa o passo 7 do algoritmo do slide, em que calculamos o delta_in_j e o delta_j bem como os deltas vij e v0j
        # Respectivamente temos delta_vij como delta_output_layer_weights e deltav0j como delta_output_layer_bias
        
        # Cálculo do termo de correção de erro utilizando o termo de correção da camada posterior
        delta_in_j = np.matmul(self.output_layer_weight, delta_k)  
        delta_j = delta_in_j * self.activation_function.prime(self.hidden_layer_in) 
        
        # Cálculo da correção de bias de cada unidade escondida
        delta_hidden_layer_bias = self.alpha * delta_j   
        # Cálculo da correção de pesos da camada de cada unidade escondida
        delta_hidden_layer_weight = self.alpha * np.matmul(data.reshape(-1, 1), delta_j.reshape(1, -1)) 

        # Esse bloco representa o passo 8 do algoritmo do slide, em que calculamos as alterações dos bias e pesos de cada unidadde de saída e cada unidade escondida
        
        # Atualização dos pesos e bias de cada unidade de saída
        self.output_layer_bias = self.output_layer_bias + delta_output_layer_bias
        self.output_layer_weight = self.output_layer_weight + delta_output_layer_weights
       
        # Atualização dos pesos e bias de cada unidade escondida
        self.hidden_layer_bias = self.hidden_layer_bias + delta_hidden_layer_bias
        self.hidden_layer_weight = self.hidden_layer_weight + delta_hidden_layer_weight

    def fit(self, training_dataset, training_dataset_labels, threshold = 0.1):
        """Faz o treinamento baseado nos dados de treinamentos e seus labels.
    
        Através da condição de parada do erro médio,
        fazemos o cálculo deste e enquanto este erro médio quadrado for maior que
        o threshold definido, rodamos o treinamento chamando o feed forward e 
        o backpropagation para os inputs passados 
            
        Args:
            training_dataset : Vetor passado para representar as saídas esperadas

            training_dataset_labels:  Vetor que representa as saídas dos neurônios 
            da camada de saída com a função de ativação 

            threshold: Float que representa um limiar para condição de parada

        """
        
        #Inicialização das variáveis de erro quadrado, épocas e plotagem para gráfico
        squaredError = 2 * threshold
        counter = 0
        grafico = []

        # Inicialização dos arquivos para escritas dos parâmetros de saida e logs
        erroFile = open("saida/erro-medio-quadrado.txt", "w")
        pesosEntradaFile = open("saida/pesos-iniciais.txt", "w")
        pesosSaidaFile = open("saida/pesos-finais.txt", "w")

        # Definição da condição de parada para a execução do treinamento da rede neural
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

            # Cálculo do erro quadrado médio
            squaredError = squaredError / len(training_dataset)
            print("Erro ao quadrado : ", squaredError)

            # Escrita do erro médio quadrado para monitorar como esta execucao 
            erroFile.write(f"Erro medio quadrado: {squaredError}\n")
            # Incrementar o numero de épocas 
            counter += 1
            # Alimenta dados para o grafico
            grafico.append(squaredError)

        # Escreve número de iterações necessárias
        erroFile.write(f"Iteracoes necessarias: {counter}")
        erroFile.close()

        # Escreve informações referentes aos pesos de entrada da camada escondida após o treinamento
        pesosEntradaFile.write(f"Pesos de entrada para a camada escondida apos o treinamento: \n{self.hidden_layer_weight}")
        pesosEntradaFile.close()

        # Escreve informações referentes aos pesos de entrada da camada saída após o treinamento
        pesosSaidaFile.write(f"Pesos de entrada para a camada de saida apos o treinamento: \n{self.output_layer_weight}")
        pesosSaidaFile.close()

        # Cria uma imagem do gráfico dos erros médios
        plt.plot(grafico)
        # Salva a imagem do gráfico dos erros médios
        plt.savefig('saida/grafico.png')

    def predict(self, data, name_of_file, y_true):
        """Realiza a predição dos resultados.
    
           Mostra a saída da predição dos resultados feitos pelo algoritmo
           com a utilização da funções auxiliares do numpy, e criação de txt para apresentação
           dos resultados
           np.round_: cria um vetor com decimais

           Args:
               data : Vetor que representa os valores dos neurônios de entrada

               name_of_file: Nome do arquivo a ser lido

               y_true: Respostas esperadas para matriz de confusão

            Returns:
                   output_layer_out: Vetor das respostas preditas
           
        """
       
        # Aplica o feedfoward nos neurônios de entrada
        output_layer_out = self.feed_forward(data)

        # Retorna o vetor com as respostas preditas
        return output_layer_out
