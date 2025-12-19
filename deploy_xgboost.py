import streamlit as st
import pandas as pd
import joblib

# 1. Load Model và Data
model = joblib.load('flight_price_model.pkl')

# Load data để lấy danh sách sân bay/hãng bay cho người dùng chọn
df = pd.read_csv('flight.csv')

# 2. Tạo giao diện Web
st.title("Dự đoán giá vé máy bay ✈️")
st.write("Nhập thông tin chuyến bay để nhận báo giá dự đoán.")

# Tạo các ô nhập liệu (Input)
col1, col2 = st.columns(2)

with col1:
    # Lấy danh sách hãng bay unique từ cột code_name
    airline = st.selectbox("Chọn hãng hàng không", df['code_name'].unique())
    # Lấy danh sách điểm đi
    source = st.selectbox("Điểm đi", df['from'].unique())

with col2:
    # Lấy danh sách điểm đến
    destination = st.selectbox("Điểm đến", df['to'].unique())
    flight_type = st.selectbox("Loại vé", df['type'].unique())

# 3. Xử lý khi bấm nút "Dự đoán"
if st.button("Dự đoán ngay"):
    # Tạo dataframe từ input của người dùng
    # LƯU Ý: Bạn cần đảm bảo các cột này khớp với lúc bạn train model
    input_data = pd.DataFrame({
        'code_name': [airline],
        'from': [source],
        'to': [destination],
        'type': [flight_type]
        # Thêm các cột khác cần thiết cho model của bạn (ví dụ giờ bay, ngày bay...)
    })

    # Vì model ML cần số, bạn cần dùng lại Encoder đã dùng lúc train để chuyển chữ sang số
    # Ở đây tôi giả định bạn đã xử lý, nếu chưa, cần thêm bước encode.

    # Dự đoán
    try:
        prediction = model.predict(input_data)
        st.success(f"Giá vé dự đoán là: {prediction[0]:,.0f} VND")
    except Exception as e:
        st.error(f"Có lỗi xảy ra: {e}. Hãy kiểm tra lại input data.")