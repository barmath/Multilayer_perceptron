# Importação de biblioteca
import csv

class Csv_manager:
	"""Lê um arquivo CSV e o guarda em um vetor.

	   Essa classe abre, guarda e lê um arquivo CSV retornando um vetor de todas as linhas.
    """
	def read_csv(name_of_file):
		"""Lê, abre e formata um arquivo CSV.
    
		Basicamente checa se o arquivo não está vazio;
		Após isso, abre e passa por todas as linhas do arquivo CSV;
		e os guarda num vetor.

        Args:
            name_of_file: String com o nome do arquivo a ser aberto.

        Returns:
			name_of_file: Uma lista de inteiros com todos os datos do arquivo requerido.
        """

		# Verifica se o arquivo está vazio
		if name_of_file is None or len(name_of_file) == 0:
			return []

		# Cria um vetor para converter o arquivo CSV em um vetor
		converted = []

		# Converte o arquivo CSV em um vetor
		with open(name_of_file, encoding='utf-8-sig') as csv_file:
			csv_reader = csv.reader(csv_file, delimiter = ',')
			for row in csv_reader:
				converted.append(row)

		# Retorna o vetor com os dados do arquivo CSV
		return converted
