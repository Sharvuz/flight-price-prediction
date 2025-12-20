import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from datetime import timedelta

# 1. CẤU HÌNH TRANG VÀ TẢI MÔ HÌNH
st.set_page_config(page_title="Dự Đoán Giá Vé Máy Bay Việt Nam", layout="wide")

@st.cache_resource
def load_models():
    # Giả định file model đã được lưu từ quá trình huấn luyện
    lr_model = joblib.load('linear_regression_model.pkl')
    xgb_model = joblib.load('xgboost_model.pkl')
    # Cần tải thêm danh sách cột mẫu để đồng bộ hóa (alignment)
    model_columns = joblib.load('model_columns.pkl') 
    return lr_model, xgb_model, model_columns

try:
    lr_model, xgb_model, model_columns = load_models()
except FileNotFoundError:
    st.error("Không tìm thấy file mô hình (.pkl). Vui lòng đảm bảo bạn đã huấn luyện và lưu model.")
    st.stop()

# 2. XÂY DỰNG DỮ LIỆU THAM CHIẾU (DURATION & FEES ENGINE)
# Dữ liệu từ phần phân tích 3.1 và 4.2
DURATION_MAP = {
    ("TP HCM", "Hà Nội"): 125, ("Hà Nội", "TP HCM"): 125,
    ("TP HCM", "Đà Nẵng"): 85, ("Đà Nẵng", "TP HCM"): 85,
    ("TP HCM", "Phú Quốc"): 65, ("Phú Quốc", "TP HCM"): 65,
    ("TP HCM", "Nha Trang"): 70, ("Nha Trang", "TP HCM"): 70,
    ("TP HCM", "Đà Lạt"): 55, ("Đà Lạt", "TP HCM"): 55,
    ("TP HCM", "Hải Phòng"): 120, ("Hải Phòng", "TP HCM"): 120,
    ("TP HCM", "Vinh"): 110, ("Vinh", "TP HCM"): 110,
    ("TP HCM", "Thanh Hóa"): 120, ("Thanh Hóa", "TP HCM"): 120,
    ("Hà Nội", "Đà Nẵng"): 80, ("Đà Nẵng", "Hà Nội"): 80,
    ("Hà Nội", "Phú Quốc"): 130, ("Phú Quốc", "Hà Nội"): 130,
    ("Hà Nội", "Nha Trang"): 115, ("Nha Trang", "Hà Nội"): 115,
    ("Hà Nội", "Đà Lạt"): 110, ("Đà Lạt", "Hà Nội"): 110,
    ("Hà Nội", "Cần Thơ"): 135, ("Cần Thơ", "Hà Nội"): 135,
    #... Bổ sung thêm các chặng khác nếu cần
}

FEE_MAP = {
    "Vietnam Airlines": 660000,
    "Vietjet": 650000,
    "Bamboo Airways": 657000,
    "Pacific Airlines": 655000,
    "Vietravel Airlines": 646000
}

# 3. GIAO DIỆN NHẬP LIỆU (SIDEBAR)
st.sidebar.header("Thông tin Chuyến bay")

airlines = list(FEE_MAP.keys())
cities =
ticket_types = # Cần khớp với dữ liệu huấn luyện

selected_airline = st.sidebar.selectbox("Hãng hàng không", airlines)
origin = st.sidebar.selectbox("Điểm đi", cities)
# Loại bỏ điểm đi khỏi danh sách điểm đến để tránh chọn trùng
dest_options = [c for c in cities if c!= origin]
destination = st.sidebar.selectbox("Điểm đến", dest_options)

dep_date = st.sidebar.date_input("Ngày bay")
dep_time = st.sidebar.time_input("Giờ khởi hành")
ticket_cls = st.sidebar.selectbox("Hạng vé", ticket_types)

# 4. LOGIC XỬ LÝ (BACKEND)
if st.sidebar.button("Dự đoán Giá Vé"):
    # 4.1 Tính toán Duration và Arrival Time
    duration_mins = DURATION_MAP.get((origin, destination), 120) # Mặc định 120p nếu không tìm thấy
    
    # Kết hợp ngày và giờ
    full_dep_datetime = pd.to_datetime(f"{dep_date} {dep_time}")
    full_arr_datetime = full_dep_datetime + timedelta(minutes=duration_mins)
    
    # 4.2 Trích xuất đặc trưng (Feature Extraction)
    # Lưu ý: Tên cột phải KHỚP CHÍNH XÁC với lúc train (ví dụ: 'day_of_week', 'month'...)
    input_data = {
        'f_price': 0, # Giá trị giả định, có thể không dùng nếu model predict total
        'fees': FEE_MAP.get(selected_airline, 650000),
        'duration_minutes': duration_mins,
        'day_of_week': full_dep_datetime.dayofweek, # 0=Monday, 6=Sunday
        'day': full_dep_datetime.day,
        'month': full_dep_datetime.month,
        'hour': full_dep_datetime.hour,
        # Các cột category sẽ được OHE bên dưới
        'code_name': selected_airline,
        'from': origin,
        'to': destination,
        'type': ticket_cls
    }
    
    # Tạo DataFrame ban đầu
    df_input = pd.DataFrame([input_data])
    
    # 4.3 One-Hot Encoding và Alignment
    # Thực hiện get_dummies cho dữ liệu nhập
    df_processed = pd.get_dummies(df_input)
    
    # CỰC KỲ QUAN TRỌNG: Reindex để khớp với cột của model
    # Thiếu bước này model sẽ báo lỗi shape hoặc predict sai
    df_final = df_processed.reindex(columns=model_columns, fill_value=0)
    
    # 5. DỰ ĐOÁN VÀ HIỂN THỊ
    st.title("Kết quả Dự đoán Giá Vé Máy Bay")
    st.write(f"✈️ **Hành trình:** {origin} ➝ {destination} | **Hãng:** {selected_airline}")
    st.write(f"🕒 **Thời gian bay dự kiến:** {duration_mins} phút")
    
    # Layout 2 cột cho 2 model
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Mô hình Hồi quy Tuyến tính (Linear Regression)")
        try:
            pred_lr = lr_model.predict(df_final)
            st.metric(label="Giá dự đoán", value=f"{pred_lr:,.0f} VND")
            st.info("Mô hình này hoạt động tốt với các xu hướng giá tuyến tính, ổn định.")
        except Exception as e:
            st.error(f"Lỗi dự đoán Linear: {e}")

    with col2:
        st.subheader("Mô hình XGBoost (Non-linear)")
        try:
            # XGBoost đôi khi yêu cầu input dạng DMatrix hoặc numpy array thuần tùy phiên bản
            pred_xgb = xgb_model.predict(df_final)
            st.metric(label="Giá dự đoán", value=f"{pred_xgb:,.0f} VND")
            st.success("Mô hình này nắm bắt tốt các biến động giá phức tạp (mùa vụ, giờ cao điểm).")
        except Exception as e:
            st.error(f"Lỗi dự đoán XGBoost: {e}")
            
    # Phân tích chênh lệch
    diff = abs(pred_lr - pred_xgb)
    st.write("---")
    st.write(f"💡 **Nhận định:** Hai mô hình có mức chênh lệch là {diff:,.0f} VND. "
             "Nếu chênh lệch thấp, độ tin cậy cao. Nếu chênh lệch lớn, chuyến bay có thể rơi vào các điều kiện đặc biệt (lễ, tết) mà XGBoost thường xử lý tốt hơn.")

else:
    st.info("Vui lòng chọn thông tin chuyến bay bên thanh trái và bấm 'Dự đoán Giá Vé'.")
