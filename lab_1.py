import numpy as np

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
