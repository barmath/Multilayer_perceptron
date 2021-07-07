from Csv_manager import Csv_manager

class characterPrint(object):
    def create_char_draw(self,linha,name_of_file):
        """Transforma os dados de uma linha dos arquivos csv numa matriz 7 por 9 

            para ficar melhor representado
            Os '1's viram asteriscos e o resto vira espaço em branco

            Args:
                String com o nome do arquivo ( name_of_file ) 

            Returns:
                matriz com representacao da letra no dataset
        """
        data = Csv_manager.read_csv(name_of_file)
        contador = 0
        desenho = []
        for j in range (9):
            arr = [0,0,0,0,0,0,0]
            for i in range(7):
                if(data[linha][contador] == "1"):
                  arr[i] = '*'
                else:
                    arr[i] = ' '
                contador += 1
            desenho.append(arr)
        return desenho

    def print_char_draw(self,linha,name_of_file):
        """Printa dados do dataset

            Args:
                String com o nome do arquivo ( name_of_file ) 
        """
        matriz = self.create_char_draw(linha,name_of_file)
        for j in range (9):
            for i in range(7):
                print(matriz[j][i],end="")
            print()
        print()

    def char_draws_string(self, linha,name_of_file):
        """Cria uma string da linha passada 

            Args:
                String com o nome do arquivo ( name_of_file ) 

            Returns:
                String representando a linha respectiva do dataset
        """
        all_draws = ""
        matriz = self.create_char_draw(linha,name_of_file)
        for j in range (9):
            line = "  "
            for i in range(7):
                line = line + str(matriz[j][i])
            all_draws = all_draws + line + "\n"
        return all_draws

#c = characterPrint()
#a = ""
#for i in range(0,4):
#    a = a + c.char_draws_string(i,'caracteres-limpo.csv')
#print(a)


    



        