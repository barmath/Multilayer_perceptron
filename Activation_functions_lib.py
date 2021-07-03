
# Importações de bibliotecas
import numpy as np


# Função Sigmoid. É a função mais comum para uso em redes neurais,
# tendo um balanço entre um comportamento linear e não-linear.
# Também tem como característica ser derivável em todos os seus pontos.
def sigmoid(x):
    """Criação da função sigmoide: f(x) = 1/ 1 + exp(-x).
    
           Args:
               x: Vetor que será aplicado na função
            
            Returns:
                   1 / (1 + np.exp(-x)): Função sigmoide aplicada
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    """Criação da derivada da função sigmoide: f'(x) = f(x)*[1 - f(x)].
    
           Args:
               x: Vetor que será aplicado na função
            
            Returns:
                   sigmoid(x) * (1 - sigmoid(x)): Derivada da função sigmoide aplicada
    """
    return sigmoid(x) * (1 - sigmoid(x))

# Facilitação da chamada da função derivada em outras classes
sigmoid.prime = sigmoid_prime
