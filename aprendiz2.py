import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Sequência de entrada e saída
sequencia = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Preparar os dados de treinamento
X = []
y = []
n_steps = 6  # Número de passos anteriores para usar na previsão
for i in range(len(sequencia) - n_steps):
    X.append(sequencia[i:i+n_steps])
    y.append(sequencia[i+n_steps:i+n_steps+6])

X = np.array(X)
y = np.array(y)

# Criar o modelo de rede neural recorrente (LSTM)
modelo = Sequential()
modelo.add(LSTM(32, input_shape=(n_steps, 1)))
modelo.add(Dense(6))

# Compilar o modelo
modelo.compile(loss='mean_squared_error', optimizer='adam')

# Treinar o modelo
modelo.fit(X, y, epochs=100, batch_size=1, verbose=2)

# Prever os próximos seis números
ultimos_numeros = sequencia[-n_steps:]
proximo_numeros = modelo.predict(np.array([ultimos_numeros]))
print("Próximos seis números na sequência:", proximo_numeros.flatten().astype(int).tolist())
