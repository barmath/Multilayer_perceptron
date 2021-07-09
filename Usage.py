# Importações de outras classes e bibliotecas
from Csv_manager import Csv_manager
from Activation_functions_lib import sigmoid
from characterPrint import characterPrint
from Mlp import Mlp
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

class Usage(object):
    """Classe Usage fornece condições mínimas para que a Mlp seja instanciada,
       bem como as funções que ajudam a organização dos dados e mostragem de 
       resultados 
    
    """

    def __init__(self, alpha, activation_function, hidden_layer_length, input_length, output_length):
        """Construtor padrão é responsável por criar as 
           condições mínimas para treinar e prever com base em dados.

           Deve ser passado como parâmetro os argumentos, para que em um primeiro momento
           a MLP seja instanciada
        
        Args:
            alpha : Float passado para representar o alpha que é
            usado na hora do backpropagation (taxa de aprendizado)

            activation_function : Função que é usada no backpropagation e feedfoward

            hidden_layer_length : Inteiro que representa o tamanho da camada 
                                  escondida

            input_length : Inteiro que representa o tamanho da entrada

            output_length : Inteiro que representa o tamanho da saída

        """
        # Criação de txt para print de parâmetros
        parametersFile = open("saida/parametros.txt", "w")
        parametersFile.write(f"Alpha: {alpha}\nFuncao de ativacao: sigmoid\nComprimento da camada escondida: {hidden_layer_length}\nComprimento da entrada: {input_length}\nComprimento da saida: {output_length}")
        parametersFile.close()

        # Criação de variáveis globais de tamanho de saída e instância da MLP
        self.output_length = output_length
        self.Mlp_instance = Mlp(alpha, activation_function, hidden_layer_length, input_length, output_length)
    
    def data_training(self, name_of_file, csv_num_line):
        """Recebe uma String com o nome do arquivo para ser usado no treinamento.
    
        Na primeira parte declara matrizes para os dados e rótulos 
        Em segundo passa o nome do arquivo para a função data_organizer
        responsável por separar os dados dos rótulos 

        Na segunda parte converte os conjuntos de dados e 
        rótulos para arranjos em numpy através do comando 
        "np.array" 

        E por fim na terceira parte chama a função fit do MLP para que 
        o algoritimo treine com base nos dados
        

        Args:
            name_of_file: String com o nome do arquivo que será usado para o treinamento

            csv_num_line: Inteiro que representa no número de linhas que o csv irá ler

        """

        # Cria os vetores para tamanho/atribuição de dados de entrada e dados de saída
        self.training_inputs = []
        self.labels = []
        training_inputs_complete = []

        # Organiza os dados de entrada e saída
        training_inputs_complete, self.labels = self.data_organizer(name_of_file)

        # Converte os vetores de entrada e saída para numpy array
        for i in range(0, csv_num_line):
            self.training_inputs.append(training_inputs_complete[i])

        self.training_inputs = np.array(self.training_inputs)
        self.labels = np.array(self.labels)
    
        # Chama a função da Mlp.py para treinamento da rede neural MLP
        self.Mlp_instance.fit(self.training_inputs, self.labels)
	
    def convert_negative_to_zero(self, training_inputs):
        """Utilizada para converter valores negativos em 0.
    
        Usando um laço aninhado, iteramos por todos elementos 
        da matriz passada, numa condição de:
        se encontrarmos um -1, trocamos por 0 na mesma posição 

        Args:
            training_inputs: Vetor do arquivo para ser convertido

        Returns:
               training_inputs: Vetor convertido

        """
        
        # Laço para troca de valores de -1 para 0
        for row in range(len(training_inputs)):
            for column in range(len(training_inputs[0])):
                if training_inputs[row][column] == -1 :
                    training_inputs[row][column] = 0

        # Retorno do vetor convertido
        return training_inputs

    
    def data_organizer(self,name_of_file):
        """Divisor de dados para treino e rótulos.
    
        Chamando o Csv_manager para abrir o arquivo 
        na sequência aloca duas matrizes: uma para as entradas
        (training_inputs) e a outra para os rótulos (labels)

        Na sequência divide os dados dos rótulos dentro do laço

        Por último popula as duas saídas com os rótulos e dados para
        treino os colocando dentro das matrizes e retornando-os

        Args:
            name_of_file: String com o nome do arquivo ( name_of_file ) para ser organizado

        Returns:
            training_inputs: Matriz com os dados de entrada
            labels: Matriz com os rótulos

        """

        # Lê o arquivo CSV para receber os dados
        training_data = Csv_manager.read_csv(name_of_file)

        # Separar dados de labels 
        training_inputs = [ [] for _ in range(len(training_data))]
        labels = [ [] for i in range(len(training_data))]

        for i in range(len(training_data)):
            training_inputs[i] = training_data[i][0:-self.output_length] 
            labels[i] = training_data[i][-self.output_length:]

        training_inputs = [ [ int(training_inputs[i][j]) for j in range(len(training_inputs[0])) ] for i in range(len(training_inputs)) ]
        labels = [ [ int(labels[i][j]) for j in range(len(labels[0])) ] for i in range(len(labels)) ]


        # Converter todos -1 em 0
        training_inputs = self.convert_negative_to_zero(training_inputs)

        # Retorna as matrizes de dados de entrada e rótulos
        return training_inputs, labels
    
    def predict(self, name_of_file):
        """Utilizada para fazer as predições.
    
        Primeiramente abre o arquivo usando o Csv_manager, 
        na sequência aloca matrizes para os conjuntos de dados 
        bem como os rótulos
        Separa o que é rótulo de dado
        E por ultimo chama a função de predição passando o conjunto de dados

        Args:
            name_of_file: O nome do arquivo ( name_of_file ) para ser predito
        """

        # Lê o arquivo CSV para atribuir os dados de entrada
        training_data = Csv_manager.read_csv(name_of_file)

        # Separar dados de labels 
        training_inputs = [ [] for _ in range(len(training_data))]
        labels = [ [] for i in range(len(training_data))]

        for i in range(len(training_data)):
            training_inputs[i] = training_data[i][0:-self.output_length] 
            labels[i] = training_data[i][-self.output_length:]
        test_data, test_labels = self.data_organizer(name_of_file)
        self.all_result_string = ""

        # Converte arrays de numpy para arranjo normais
        test_data = np.array(test_data)
        test_labels = np.array(test_labels)

        # Faz a predição dos dados 
        result = self.Mlp_instance.predict(test_data, name_of_file, test_labels)
        self.predictionOutputFormater(result, test_data, name_of_file, test_labels)

        # Inicia arquivo para escrever
        self.predicoes = open(f"Saida/predicoes-{name_of_file}.txt", "w")

        # Escreve o data do teste hora que o teste foi feito
        timeStamp = 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
        self.predicoes.write(f"Data de criacao : {timeStamp}\n")

        # Escreve o tamanho do test label
        len_test_data = len(test_data)
        self.predicoes.write(f"\nLabels testados : {len_test_data}\n")

        # Mostra a saída do resultado
        y_pred = np.round_(result)
        self.predicoes.write(str(np.round_(result)))

        # Monta visualizacao do dataset bem como o rótulo e o
        # resultados das predições
        self.predicoes.write("\n\nResultados das predicoes para o dataset:\n")
        self.predicoes.write(f"\n{self.all_result_string} ")

        # Cria matriz de confusão 
        self.predicoes.write("\n\nMatriz de confusao:\n")
        y_true = test_labels
        self.predicoes.write(str(confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))))

        # Cria relatório de classificação
        self.predicoes.write("\n\nRelatorio de classificacao:\n")
        target_names = ['A', 'B', 'C', 'D', 'E', 'J', 'K']
        self.predicoes.write(str(classification_report(y_true, y_pred, target_names=target_names, zero_division=0)))
        self.predicoes.close()

    def predictionOutputFormater(self, results, test_data, name_of_file,test_labels):
        """Utilizada para formatar o arquivo de saída de predições

        Args:
            results: Vetor que representa os resultados das predições

            test_data: Vetor que representa os valores de entrada

            name_of_file: O nome do arquivo ( name_of_file ) para ser predito

            test_labels: Vetor que representa os valores de saída
    
        """
        target_names = ['A', 'B', 'C', 'D', 'E', 'J', 'K']

        results = np.round_(results)

        results = np.array(results, dtype=int)
        test_labels = np.array(test_labels, dtype=int)

        numered_labels = self.labels_numerator(test_labels)
        results = self.separateResults(results)

        charDrawer = characterPrint()
        
        for index in range(len(numered_labels)):
            row_char_draw = ""
            target_index_a = numered_labels[index]
            correct_answer = target_names[target_index_a]
            if charDrawer.char_draws_string(index,name_of_file) is not None :
                row_char_draw = row_char_draw + charDrawer.char_draws_string(index,name_of_file) + "\n"
    
            data_of_prediction = "label : "+str(correct_answer)+" ,predicoes : "
            prediction_respective = ""
            for j in range(len(results[index])):
                target_index = results[index][j]
                prediction = target_names[target_index]
                prediction_respective = prediction_respective + " p"+str(j)+" : "+str(prediction)+" , "

            if len(prediction_respective) == 0:
                prediction_respective = "Nao ha predicoes"
            self.all_result_string = self.all_result_string +"Teste "+str(index+1)+") \n  Visualizacao dos dados : \n\n " +row_char_draw + " Resultados da predicao : "+data_of_prediction + prediction_respective +"\n\n"



    def labels_numerator(self,labels):
        """Utilizada para formatar o arquivo de saída de predições

        Args:
            labels: Vetor que representa os valores de saída
        
        Returns:
               numered_labels: Vetor que representa quais são as respostas
    
        """

        numered_labels = []

        for label in labels:
            for index in range(len(label)):
                if label[index] == 1 :
                    numered_labels.append(index)

        return numered_labels

    def separateResults(self, dataset_output):
        """Utilizada para separar os resultados

        Args:
            dataset_output: Vetor que representa os dataset de saída
        
        Returns:
               final_answer: Vetor que representa os resultados de resposta
    
        """
        result = []
        final_answer = []

        for dataset_row in dataset_output:
            temp = []

            for output_index in range(self.output_length):
                result = dataset_row[output_index]

                if result == 1 :
                    temp.append(output_index)
            final_answer.append(temp)

        return final_answer

if __name__ == '__main__':
    u = Usage(0.1, sigmoid, 15, 63, 7)
    u.data_training('caracteres-limpo.csv', 14)
    print("Predicoes para caracteres-limpo.csv")
    u.predict('caracteres-limpo.csv')
    print("Predicoes para caracteres-ruido.csv")
    u.predict('caracteres-ruido.csv')
    print("Predicoes para caracteres_ruido20.csv")
    u.predict('caracteres_ruido20.csv')

