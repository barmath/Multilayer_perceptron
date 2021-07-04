from Csv_manager import Csv_manager

class characterPrint:
    def draw(linha,name_of_file):
        #Transforma os dados de uma linha dos arquivos csv numa matriz 7 por 9 para ficar melhor representado
        #Os '1's viram asteriscos e o resto vira espaço em branco
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

    #Um exemplo de implementação para desenhar essa matriz seria:
    #matriz = draw(0,'caracteres-limpo.csv')
    #for j in range (9):
    #    for i in range(7):
    #        print(matriz[j][i],end="")
    #    print()

    

    



        