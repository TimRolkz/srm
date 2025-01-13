import numpy as np


np.random.seed(42)

# Генерація послідовності
n = 1000  # кількість елементів в послідовності
states = [0, 1, 2, 3]
p_k = [(k + 1) / 10 for k in states]  # початкові ймовірності [10%, 20%, 30% і 40%]

# Генерація станів
states_sequence = np.random.choice(states, size=n, p=p_k)

# Генерація випадкових величин
x_sequence = np.array([np.random.normal(loc=k, scale=1) for k in states_sequence])

print(x_sequence)