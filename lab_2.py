import numpy as np


np.random.seed(42)


def self_learning_algorithm(x_sequence, states, tol=0.001, max_iter=100):
    '''
    За допомогою алгоритму самонавчання отримати оцiнки ймовiрностей pK(k) i параметрiв ak,
    k ∈ {0, 1, 2, 3}. Умовою зупинки алгоритму вважати наступну: оцiнки параметрiв не змiни-
    лись, а оцiнки ймовiрностей змiнились менше нiж на 0.001. Алгоритм має працювати для
    довiльного n.
    '''
    iteration = -1
    # Початкові оцінки
    p_k_est = np.array([1 / len(states)] * len(states))
    a_k_est = np.array(states, dtype=float)  # a_k = k для кожного k

    for iteration in range(max_iter):
        # Очікуваний розподіл для кожного стану
        likelihoods = np.array([
            (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x_sequence - a_k) ** 2)
            for a_k in a_k_est
        ])

        # Нормалізація для отримання ймовірностей
        posterior_probs = likelihoods * p_k_est[:, np.newaxis]
        posterior_probs /= posterior_probs.sum(axis=0)

        # Оновлення параметрів
        new_p_k_est = posterior_probs.mean(axis=1)
        new_a_k_est = np.array([
            (posterior_probs[k, :] @ x_sequence) / posterior_probs[k, :].sum()
            for k in range(len(states))
        ])

        # Перевірка зміни параметрів
        if (
            np.max(np.abs(new_p_k_est - p_k_est)) < tol
            and np.max(np.abs(new_a_k_est - a_k_est)) < tol
        ):
            break

        p_k_est = new_p_k_est
        a_k_est = new_a_k_est

    return p_k_est, a_k_est, iteration + 1

if __name__ == '__main__':
    # Генерація послідовності
    n = 1000  # кількість елементів в послідовності
    states = [0, 1, 2, 3]
    p_k = [(k + 1) / 10 for k in states]  # початкові ймовірності [10%, 20%, 30% і 40%]

    # Генерація станів
    states_sequence = np.random.choice(states, size=n, p=p_k)

    # Генерація випадкових величин
    x_sequence = np.array([np.random.normal(loc=k, scale=1) for k in states_sequence])
    estimated_p_k, estimated_a_k, num_iterations = self_learning_algorithm(x_sequence, states)
    print("Оцінені ймовірності p_k:", estimated_p_k)
    print("Оцінені параметри a_k:", estimated_a_k)
    print("Кількість ітерацій:", num_iterations)
