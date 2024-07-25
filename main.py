from flask import Flask, request, jsonify
import numpy as np
from model import create_model_manager
from data import get_sample_data, save_training_data, load_training_data
from nlp_model import load_nlp_models, generate_text_gpt2, encode_text
import threading
import time

app = Flask(__name__)

# Khởi tạo mô hình quản lý
model_manager = create_model_manager()

# Khởi tạo các mô hình NLP
(gpt2_tokenizer, gpt2_model), (deberta_tokenizer, deberta_model), (roberta_tokenizer, roberta_model) = load_nlp_models()

def auto_train_model():
    while True:
        try:
            x_train, y_train = load_training_data()
            if x_train.size > 0 and y_train.size > 0:
                model_manager.train_model(x_train, y_train)
                model_manager.train_deep_model(x_train, y_train)
                model_manager.save_model()
        except FileNotFoundError:
            x_train, y_train = get_sample_data()
            save_training_data(x_train, y_train)
        time.sleep(60)  # Huấn luyện mỗi phút

@app.route('/create_okr', methods=['POST'])
def create_okr():
    data = request.json
    input_text = data['input']
    
    # Sinh văn bản mở rộng bằng GPT-2
    generated_text = generate_text_gpt2(gpt2_tokenizer, gpt2_model, input_text)
    
    # Mã hóa văn bản đầu vào bằng DeBERTa và RoBERTa
    deberta_encoding = encode_text(deberta_tokenizer, deberta_model, input_text)
    roberta_encoding = encode_text(roberta_tokenizer, roberta_model, input_text)
    
    # Kết hợp các mã hóa
    combined_encoding = np.concatenate([deberta_encoding.detach().numpy(), roberta_encoding.detach().numpy()], axis=1)
    
    # Dự đoán OKR từ mã hóa văn bản
    prediction = model_manager.predict(combined_encoding.reshape(1, -1))
    deep_prediction = model_manager.predict_deep(combined_encoding.reshape(1, -1))
    
    # Lưu dữ liệu mới cho huấn luyện trong tương lai
    x_train, y_train = load_training_data()
    x_train = np.append(x_train, [combined_encoding.flatten()], axis=0)
    y_train = np.append(y_train, prediction, axis=0)
    save_training_data(x_train, y_train)
    
    return jsonify({'okr': prediction.tolist(), 'deep_okr': deep_prediction.tolist(), 'generated_text': generated_text})

if __name__ == '__main__':
    threading.Thread(target=auto_train_model).start()
    app.run(host='0.0.0.0', port=8080)