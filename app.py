import streamlit as st
import pandas as pd
import joblib

#Load model
model = joblib.load('model.pkl')
df = pd.read_csv('flight.csv')

#tao gia dien
st.title("Dự đoán giá vé máy bay ✈")
st.write("Nhập thông tin chuyến bay để nhận báo giá dự đoán.")

#o nhap data
col1, col2 = st.columns(2)

with col1:
    #Lấy danh sách hãng bay unique từ cột code_name
    airline = st.selectbox("Chọn hãng hàng không", df['code_name'].unique())
    #Lấy danh sách điểm đi
    source = st.selectbox("Điểm đi", df['from'].unique())

with col2:
    #Lấy danh sách điểm đến
    destination = st.selectbox("Điểm đến", df['to'].unique())
    flight_type = st.selectbox("Loại vé", df['type'].unique())

#action button du doan
if st.button("Dự đoán ngay"):

    #cột cần khớp vơí cot lúc train
    input_data = pd.DataFrame({
        'code_name': [airline],
        'from': [source],
        'to': [destination],
        'type': [flight_type]
    })

    #predict
    try:
        prediction = model.predict(input_data)
        st.success(f"Giá vé dự đoán là: {prediction[0]:,.0f} VND")
    except Exception as e:
        st.error(f"Có lỗi xảy ra: {e}. Hãy kiểm tra lại input data.")