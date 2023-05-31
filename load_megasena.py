import requests
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf

def aprendiz(sequencia):

    # Sequência de entrada e saída
    # sequencia = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sequencia_entrada = sequencia[:-1]
    sequencia_saida = sequencia[1:]

    print(sequencia)
    print(sequencia_entrada)
    print(sequencia_saida)

    # Transformando a sequência em um formato adequado para a entrada da RNN
    X = [[valor] for valor in sequencia_entrada]
    y = [[valor] for valor in sequencia_saida]

    # Criando o modelo de rede neural recorrente (LSTM)
    modelo = Sequential()
    modelo.add(LSTM(10, input_shape=(1, 1)))
    modelo.add(Dense(1))

    # Compilando o modelo
    modelo.compile(loss='mean_squared_error', optimizer='adam')

    # Treinando o modelo
    modelo.fit(X, y, epochs=20, batch_size=1, verbose=2)

    # Salvar o modelo
    modelo.save('modelo.h5')



def nextNumber(sequencia):

    modelo = load_model('modelo.h5')

    # Make predictions using the loaded model
    ultimo_numero = sequencia[-1]
    proximo_numero = modelo.predict([[ultimo_numero]])
    print("Próximo número na sequência:", int(proximo_numero))

def carregar_jogos_mega_sena():
    loteria = 'mega-sena'
    url = f"https://loteriascaixa-api.herokuapp.com/api/" + loteria
    response = requests.get(url)
    data = response.json()

    jogos = []

    for resultado in data:
        data_sorteio = resultado['data']
        numeros = resultado['dezenas']
        numeros = [int(num) for num in numeros]
        jogos.append((data_sorteio, numeros))

    todos = []
    
    # Exibindo os jogos
    for jogo in jogos:
        data, numeros = jogo
        # print(f"Data: {data}, Números: {numeros}")
        print(f"NumbesSequence: {numeros}")

        for n in numeros:
            print(n)
            todos.append(n)

    print(todos)

    return todos

# Testando a função
ALL_GAMES = carregar_jogos_mega_sena()

# Treinando
aprendiz(ALL_GAMES)

nextNumber(ALL_GAMES)