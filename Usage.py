# Importações de outras classes e bibliotecas
from Csv_maneger import Csv_maneger
from Activation_functions_lib import sigmoid
from Mlp import Mlp
import numpy as np


class Usage(object):
    """Classe Usage fonece condições mínimas para que a Mlp seja instanciada,
       bem como as funções que ajudam a organização dos dados e mostragem de re-
       sultados 
    
    """

    def __init__(self, alpha, activation_function, hidden_layer_length, input_length, output_length):
        """Construtor padrão e responsável por criar as 
           condições mínimas para treinar e prever com base em dados.

           Deve ser passado como parametro argumetos para que em um primeiro momento
           A MLP seja instanciada
        
        Args:
            alpha : Float passado para representar o alpha que e 
            usado na hora do backpropagation 

            activation_function : Funcao que e usada no backpropagation 

            hidden_layer_length : Integer que representa o tamanho da camada 
                                  escondida

            input_length : Integer que representa tamanho da entrada
            output_length : Integer que representa tamanho da saida
        """
        parametersFile = open("saida/parametros.txt", "w")
        parametersFile.write(f"Alpha: {alpha}\nFuncao de ativacao: sigmoid\nComprimento da camada escondida: {hidden_layer_length}\nComprimento da entrada: {input_length}\nComprimento da saida: {output_length}")
        parametersFile.close()

        self.output_length = output_length
        self.Mlp_instance = Mlp(alpha, activation_function, hidden_layer_length, input_length, output_length)
    
    def data_training(self, name_of_file):
        """Recebe uma String com o nome do arquivo para ser usado no treinamento.
    
        Na primeira parte declara matrizes para os dados e rotulos 
        Em segundo passa o nome do arquivo para a funcao data_organizer
        responsavel por separar os dados dos rotulos 

        Na segunda parte converte os conjutos de dados e 
        rotulos para arranjos em numpy atravez do comando 
        "np.array" 

        E por fim na terceira parte chama a funcao fit do MLP para que 
        o algoritimo treine com base nos dados
        

        Args:
            Uma String com o nome do arquivo que sera usado para o treinamento

        """

        # Primera parte 
        self.training_inputs = []
        self.labels = []
        self.training_inputs, self.labels = self.data_organizer(name_of_file)

        # Segunda parte 
        self.training_inputs = np.array(self.training_inputs)
        self.labels = np.array(self.labels)

        # Terceira parte 
        #self.Mlp_instance.fit(self.training_inputs, self.labels, max_epochs= 1000000,max_error=0.1)
        # chamando backpropagation demo
        self.Mlp_instance.fit(self.training_inputs, self.labels)
	
    def convert_negative_to_zero(self, training_inputs):
        """Utilizada para converter valores negativos em 0.
    
        Usando um laco aninhado, iteramos por todos elementos 
        da matriz passada, numa condicao de se encontrarmos um -1
        trocamos para 0 na mesma posicao 

        Args:
            O arranjo do arquivo para ser convertido

        Returns:
            O arranjo convertido
        """

        for row in range(len(training_inputs)):
            for column in range(len(training_inputs[0])):
                if training_inputs[row][column] == -1 :
                    training_inputs[row][column] = 0

        return training_inputs

    
    def data_organizer(self,name_of_file):
        """Divisosr de dados para treino e rotulos.
    
        Chamando o Csv_maneger para abrir o arquivo 
        na sequencia aloca duas matrizes uma para as entradas
        (training_inputs) e a otra para os rotulos (labels).

        Na sequencia divide os dados dos rotulos dentro on laco

        por ultimo popula as duas saidas com os rotulos e dados para
        treino os colocando dentro das matrizes e retornando-os

        Args:
            String com o nome do arquivo ( name_of_file ) para ser organizado

        Returns:
            Duas matrizes com os dados e rotulos respectivamente
        """
        training_data = Csv_maneger.read_csv(name_of_file)

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

        return training_inputs, labels
    
    # TODO Funcao para mostrar resultados 
    def predict(self, name_of_file):
        """Utilizada para fazer as predicoes.
    
        Primeiramente abre o arquivo usando o Csv_maneger, 
        na sequencia aloca matrizes para os conjuntos de dados 
        bem como os rotulos.
        Separa o que e rotulo de dado.
        E por ultimo chama a funcao de predicao passando o conjunto de dados

        Args:
            O nome do arquivo ( name_of_file ) para ser predito
        """

        training_data = Csv_maneger.read_csv(name_of_file)

        # Separar dados de labels 
        training_inputs = [ [] for _ in range(len(training_data))]
        labels = [ [] for i in range(len(training_data))]

        for i in range(len(training_data)):
            training_inputs[i] = training_data[i][0:-self.output_length] 
            labels[i] = training_data[i][-self.output_length:]
        
        test_data, test_labels = self.data_organizer(name_of_file)

        test_data = np.array(test_data)
        test_labels = np.array(test_labels)
        self.Mlp_instance.predict(test_data, name_of_file, test_labels)


#TODO: Colocar 1 camada escondida e colocar os arquivos csv corretos.
u = Usage(0.1, sigmoid, 2, 63, 7)
u.data_training('caracteres-limpo.csv')
print("Predicoes para caracteres-limpo.csv")
u.predict('caracteres-limpo.csv')
print("Predicoes para caracteres-ruido2.csv")
u.predict('caracteres-ruido2.csv')
print("Predicoes para caracteres-ruido22.csv")
u.predict('caracteres-ruido22.csv')

