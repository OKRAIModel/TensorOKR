import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model as keras_load_model, save_model as keras_save_model
import joblib

class ModelManager:
    def __init__(self):
        # Khởi tạo mô hình học máy (XGBoost) và bộ chuẩn hóa
        self.model = XGBRegressor()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.deep_model = None
        self.model_filename = 'model.pkl'
        self.deep_model_filename = 'deep_model.h5'

    def create_model(self):
        # Tạo mô hình học máy (XGBoost)
        return self.model

    def create_deep_model(self, input_shape):
        # Tạo mô hình học sâu
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_shape,)),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_model(self, x_train, y_train):
        # Chuẩn hóa dữ liệu và huấn luyện mô hình học máy
        x_train_scaled = self.scaler.fit_transform(x_train)
        self.model.fit(x_train_scaled, y_train)
        self.is_trained = True

    def train_deep_model(self, x_train, y_train):
        # Tạo và huấn luyện mô hình học sâu
        self.deep_model = self.create_deep_model(x_train.shape[1])
        self.deep_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    def predict(self, x_input):
        if not self.is_trained:
            raise ValueError("Mô hình chưa được huấn luyện")
        
        # Chuẩn hóa dữ liệu đầu vào và dự đoán
        x_input_scaled = self.scaler.transform(x_input)
        return self.model.predict(x_input_scaled)

    def predict_deep(self, x_input):
        if self.deep_model is None:
            raise ValueError("Mô hình học sâu chưa được huấn luyện")
        
        # Dự đoán với mô hình học sâu
        return self.deep_model.predict(x_input)

    def save_model(self):
        # Lưu mô hình học máy và bộ chuẩn hóa
        joblib.dump({'model': self.model, 'scaler': self.scaler}, self.model_filename)
        # Lưu mô hình học sâu
        if self.deep_model is not None:
            self.deep_model.save(self.deep_model_filename)

    def load_model(self):
        # Tải mô hình học máy và bộ chuẩn hóa
        data = joblib.load(self.model_filename)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_trained = True
        # Tải mô hình học sâu
        self.deep_model = keras_load_model(self.deep_model_filename)

# Hàm tạo và quản lý mô hình
def create_model_manager():
    return ModelManager()