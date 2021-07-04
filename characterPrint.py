from Csv_maneger import Csv_maneger

class characterPrint:
    data = Csv_maneger.read_csv('caracteres-limpo.csv')
    
    for j in range(21):
        for i in range(63):
            if(i%7==0):
                print()
            if(data[j][i] == "1"):
                print('*', end = "")
            else:
                print(' ',end = "")
        print("\n--------")
        