import numpy as np
from tqdm import tqdm


def generate_random_vectors(N):
    """
    1. Згенерувати N тривимiрних незалежних випадкових векторiв, в яких координати незалежнi,
    першi двi координати мають показниковий розподiл з параметром 1, третя координата має
    стандартний нормальний розподiл.
    """
    # Перші дві координати: показниковий (експоненційний) розподіл з параметром λ=1
    first_two_coords = np.random.exponential(scale=1, size=(N, 2))
    # Третя координата: стандартний нормальний розподіл
    third_coord = np.random.normal(loc=0, scale=1, size=(N, 1))
    # Об'єднання координат
    vectors = np.hstack((first_two_coords, third_coord))
    return vectors

def kozinec_algorithm(vectors):
    """
    Реалізує алгоритм Козинця для пошуку розділяючого вектора.
    """
    # Крок 1: Знаходимо опуклу оболонку (з використанням np.cross для визначення орієнтації)
    n = len(vectors)
    separating_vector = None

    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Беремо три вектори
                p1, p2, p3 = vectors[i], vectors[j], vectors[k]
                # Знаходимо нормаль до площини, утвореної трьома векторами
                normal = np.cross(p2 - p1, p3 - p1)
                if np.linalg.norm(normal) == 0:
                    continue  # Вектори лежать на одній лінії

                # Перевіряємо, чи всі точки знаходяться по один бік від площини
                positive_side = None
                is_separating = True

                for vec in vectors:
                    if np.array_equal(vec, p1) or np.array_equal(vec, p2) or np.array_equal(vec, p3):
                        continue

                    # Обчислюємо скалярний добуток нормалі на вектор
                    side = np.dot(normal, vec - p1)

                    if positive_side is None:
                        positive_side = side > 0
                    elif (side > 0) != positive_side:
                        is_separating = False
                        break

                if is_separating:
                    separating_vector = normal
                    break

            if separating_vector is not None:
                break
        if separating_vector is not None:
            break

    return separating_vector



if __name__ == '__main__':
    num_vectors_ = 250 # кількість векторів
    vectors_ = generate_random_vectors(num_vectors_)
    separating_vector_ = kozinec_algorithm(vectors_)

    if separating_vector_ is not None:
        print("Розділяючий вектор знайдено:", separating_vector_)
    else:
        print("Розділяючий вектор не знайдено.")