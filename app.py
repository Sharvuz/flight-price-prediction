import streamlit as st
import pandas as pd
import joblib
import datetime

# --- 1. LOAD MODEL VÀ DATA ---
# Load model đã train
model = joblib.load('model.pkl')

# Load data gốc để lấy cấu trúc cột (Columns)
# Việc này giúp đảm bảo input của web khớp hoàn toàn với lúc train
df_goc = pd.read_csv('flight.csv')


# --- Hàm xử lý tiền xử lý giống hệt file model.ipynb ---
def process_data(df):
    format_type = '%H:%M:%S %d/%m/%Y'
    # Chuyển đổi sang datetime nếu chưa phải (chỉ áp dụng cho df gốc lúc lấy cột)
    if df['f_time_from'].dtype == 'object':
        df['f_time_from'] = pd.to_datetime(df['f_time_from'], format=format_type)
        df['f_time_to'] = pd.to_datetime(df['f_time_to'], format=format_type)

    df['hour'] = df['f_time_from'].dt.hour
    df['day_of_week'] = df['f_time_from'].dt.day_of_week
    df['day'] = df['f_time_from'].dt.day
    df['month'] = df['f_time_from'].dt.month
    df['duration_minutes'] = (df['f_time_to'] - df['f_time_from']).dt.total_seconds() / 60

    return df


# Lấy danh sách cột chuẩn từ dữ liệu gốc (để dùng cho reindex sau này)
# Ta phải giả lập bước train để lấy đúng tên các cột One-Hot Encoding
df_temp = process_data(df_goc.copy())
features = ['code_name', 'from', 'to', 'type']
df_encoded_temp = pd.get_dummies(df_temp, columns=features, drop_first=True)
drop_cols = ['id', 'code', 'f_code', 'f_time_from', 'f_time_to',
             'f_price', 'fees', 'total_price', 'airport_from', 'airport_to']
# Đây là danh sách cột chuẩn mà Model mong muốn
model_columns = df_encoded_temp.drop(columns=drop_cols, errors='ignore').columns

# --- 2. GIAO DIỆN WEB ---
st.title("Dự đoán giá vé máy bay ✈️")

# Chia cột cho đẹp
col1, col2 = st.columns(2)

with col1:
    airline = st.selectbox("Hãng bay", df_goc['code_name'].unique())
    source = st.selectbox("Điểm đi", df_goc['from'].unique(), index=1)  # Default Hà Nội
    # Chọn Ngày và Giờ đi
    d_date = st.date_input("Ngày đi", datetime.date(2021, 5, 15))
    d_time = st.time_input("Giờ đi", datetime.time(8, 0))

with col2:
    flight_type = st.selectbox("Loại vé", df_goc['type'].unique())
    destination = st.selectbox("Điểm đến", df_goc['to'].unique(), index=0)  # Default TP HCM
    # Chọn Ngày và Giờ đến
    a_date = st.date_input("Ngày đến", datetime.date(2021, 5, 15))
    a_time = st.time_input("Giờ đến", datetime.time(10, 10))

# --- 3. XỬ LÝ DỰ ĐOÁN ---
if st.button("Dự đoán ngay", type="primary"):
    # 1. Ghép ngày và giờ thành datetime
    dep_datetime = pd.to_datetime(f"{d_date} {d_time}")
    arr_datetime = pd.to_datetime(f"{a_date} {a_time}")

    # Kiểm tra lỗi thời gian (Đến trước khi Đi)
    if arr_datetime <= dep_datetime:
        st.error("⚠️ Giờ đến phải sau Giờ đi!")
    else:
        # 2. Tạo DataFrame từ input người dùng
        input_data = pd.DataFrame({
            'code_name': [airline],
            'from': [source],
            'to': [destination],
            'type': [flight_type],
            'f_time_from': [dep_datetime],
            'f_time_to': [arr_datetime]
        })

        # 3. Tính toán Feature (Giờ, Ngày, Duration...) y hệt notebook
        # Code xử lý này khớp với hàm du_doan_gia_ve trong notebook của bạn
        input_data['hour'] = input_data['f_time_from'].dt.hour
        input_data['day_of_week'] = input_data['f_time_from'].dt.day_of_week
        input_data['day'] = input_data['f_time_from'].dt.day
        input_data['month'] = input_data['f_time_from'].dt.month
        input_data['duration_minutes'] = (input_data['f_time_to'] - input_data['f_time_from']).dt.total_seconds() / 60

        # 4. One-Hot Encoding và Reindex
        # Bước này cực quan trọng: Biến chữ thành số và sắp xếp đúng vị trí cột như lúc train
        input_encoded = pd.get_dummies(input_data, columns=['code_name', 'from', 'to', 'type'])

        # Tự động điền 0 vào các cột thiếu (ví dụ user chọn hãng A, nhưng model còn biết hãng B, C...)
        final_input = input_encoded.reindex(columns=model_columns, fill_value=0)

        # 5. Dự đoán
        try:
            prediction = model.predict(final_input)
            st.success(f"🎫 Giá vé dự đoán: **{prediction[0]:,.0f} VND**")
        except Exception as e:
            st.error(f"Có lỗi xảy ra: {e}")