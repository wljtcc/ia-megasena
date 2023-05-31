from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Sequência de entrada e saída
sequencia = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
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
modelo.fit(X, y, epochs=100, batch_size=1, verbose=2)

# Prevendo a próxima sequência
proximo_numero = modelo.predict([[sequencia[-1]]])
print("Próximo número na sequência:", int(proximo_numero))
