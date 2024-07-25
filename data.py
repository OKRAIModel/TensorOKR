import numpy as np

def get_sample_data():
    # Tạo dữ liệu mẫu
    x_train = np.array([[0.0, 1.0], [1.0, 0.0]])
    y_train = np.array([1.0, 0.0])
    return x_train, y_train

def save_training_data(x_train, y_train):
    np.savez_compressed('training_data.npz', x_train=x_train, y_train=y_train)

def load_training_data():
    try:
        data = np.load('training_data.npz')
        x_train = data['x_train']
        y_train = data['y_train']
    except FileNotFoundError:
        x_train, y_train = get_sample_data()
    return x_train, y_train