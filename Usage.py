from Csv_maneger import Csv_maneger
from Activation_functions_lib import sigmoid
from Mlp import Mlp

import numpy as np
from numpy import int64

class Usage(object):
    
    def __init__(self, alpha, activation_function, hidden_layer_length, input_length, output_length):
        self.output_length = output_length
        self.Mlp_parameters(alpha, activation_function, hidden_layer_length, input_length, output_length)
    
    def Mlp_parameters(self, alpha, activation_function, hidden_layer_length, input_length, output_length):
        self.Mlp_instance = Mlp(alpha, activation_function, hidden_layer_length, input_length, output_length)
    
    def data_training(self, name_of_file):
        self.training_inputs = []
        self.labels = []
        self.training_inputs, self.labels = self.data_organizer(name_of_file)
        input_length = len(self.training_inputs)
        output_length = len(self.labels)

        self.training_inputs = np.array(self.training_inputs)
        self.labels = np.array(self.labels)
        # TODO Chamar backpropagation 
        self.Mlp_instance.fit(self.training_inputs, self.labels, max_epochs= 1000000,max_error=0.1)
	
    def convert_negative_to_zero(self, training_inputs):

        for row in range(len(training_inputs)):
            for column in range(len(training_inputs[0])):
                if training_inputs[row][column] == -1 :
                    training_inputs[row][column] = 0

        return training_inputs

    # TROCAR -7 POR VARIAVEL OUTPUT LENGTH 
    def data_organizer(self,name_of_file):
        # Abrir CSV 
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
        # Abrir CSV 
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
        print("test labels")
        print(len(test_data))
        self.Mlp_instance.predict(test_data)
        


u = Usage(0.1, sigmoid, 2, 63, 7)
u.data_training('caracteres-limpo2.csv')
u.predict('caracteres-limpo2.csv')
