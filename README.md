# Disease Prediction AI

Ứng dụng trí tuệ nhân tạo dự đoán bệnh từ triệu chứng sử dụng Machine Learning.

## 📁 Cấu trúc Project

```
disease-prediction-ai/
├── data/                          # Dữ liệu gốc
│   ├── disease_dataAI2.csv       # Dữ liệu bệnh - triệu chứng
│   └── disease_descriptionsAI.csv # Mô tả bệnh
├── src/                          # Source code
│   ├── __init__.py
│   ├── config.py                 # Cấu hình
│   ├── data_loader.py            # Xử lý dữ liệu
│   └── model.py                  # Model ML
├── models/                       # Model đã train
│   ├── disease_rf_model.pkl
│   ├── symptoms_list.pkl
│   └── disease_descriptions.pkl
├── requirements.txt              # Dependencies
├── train.py                      # Script train model
├── app.py                        # Streamlit app
└── README.md
```

## 🚀 Cài đặt

1. **Cài dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train model:**
   ```bash
   python train.py
   ```

3. **Chạy ứng dụng:**
   ```bash
   streamlit run app.py
   ```

## 📊 Cách sử dụng

1. Mở trình duyệt và truy cập địa chỉ hiển thị
2. Nhập các triệu chứng cách nhau bởi dấu phẩy
3. Nhấn "🔍 Dự đoán Bệnh"
4. Xem kết quả dự đoán và mô tả bệnh

## 🧠 Thuật toán

- **Model:** Random Forest Classifier
- **Features:** One-hot encoding của triệu chứng
- **Đánh giá:** Train/Validation/Test split (60/20/20)

## ⚠️ Lưu ý

- Kết quả chỉ mang tính tham khảo
- Không thay thế chẩn đoán của bác sĩ chuyên khoa
- Dữ liệu được sử dụng để train model có thể không đầy đủ

## 🔧 Phát triển

### Thêm model mới

1. Tạo class mới trong `src/model.py`
2. Implement các method: `train()`, `predict()`, `save()`, `load()`
3. Cập nhật `config.py` với parameters mới

### Thêm dữ liệu

1. Thêm dữ liệu vào `data/` folder
2. Cập nhật `src/data_loader.py` nếu cần
3. Re-train model

## 📈 Metrics

Sau khi train, model sẽ hiển thị accuracy trên các tập:
- Train accuracy
- Validation accuracy
- Test accuracy

## � Testing

Chạy unit tests để đảm bảo code hoạt động đúng:

```bash
# Cài pytest (nếu chưa có)
pip install pytest

# Chạy tất cả tests
python -m pytest tests/ -v

# Hoặc chạy trực tiếp
python tests/test_model.py
```

### Test Coverage
- ✅ Data loading và preprocessing
- ✅ Model training và loading
- ✅ Prediction với symptoms hợp lệ/không hợp lệ
- ✅ Data integrity checks