import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from datetime import datetime, timedelta

# ---------------------------------------------------------
# 1. CẤU HÌNH DỮ LIỆU GIẢ LẬP (HARDCODED DATA)
# ---------------------------------------------------------

# Từ điển thời gian bay trung bình (phút) giữa các thành phố lớn
# Dựa trên dữ liệu thực tế các chặng bay nội địa Việt Nam
DURATION_MAP = {
    # Trục Bắc - Nam
    ("Hà Nội", "TP HCM"): 130,
    ("TP HCM", "Hà Nội"): 130,
    ("Hà Nội", "Cần Thơ"): 135,
    ("Cần Thơ", "Hà Nội"): 135,
    ("Vinh", "TP HCM"): 110,
    ("TP HCM", "Vinh"): 110,
    ("Hải Phòng", "TP HCM"): 120,
    ("TP HCM", "Hải Phòng"): 120,
    ("Thanh Hóa", "TP HCM"): 120,
    ("TP HCM", "Thanh Hóa"): 120,

    # Miền Trung
    ("Hà Nội", "Đà Nẵng"): 80,
    ("Đà Nẵng", "Hà Nội"): 80,
    ("TP HCM", "Đà Nẵng"): 85,
    ("Đà Nẵng", "TP HCM"): 85,
    ("TP HCM", "Huế"): 85,
    ("Huế", "TP HCM"): 85,
    ("Hà Nội", "Huế"): 75,
    ("Huế", "Hà Nội"): 75,
    ("TP HCM", "Quy Nhơn"): 75,
    ("Quy Nhơn", "TP HCM"): 75,

    # Du lịch (Nha Trang, Phú Quốc, Đà Lạt)
    ("Hà Nội", "Nha Trang"): 115,
    ("Nha Trang", "Hà Nội"): 115,
    ("TP HCM", "Nha Trang"): 70,
    ("Nha Trang", "TP HCM"): 70,
    ("Hà Nội", "Phú Quốc"): 130,
    ("Phú Quốc", "Hà Nội"): 130,
    ("TP HCM", "Phú Quốc"): 60,
    ("Phú Quốc", "TP HCM"): 60,
    ("Hà Nội", "Đà Lạt"): 110,
    ("Đà Lạt", "Hà Nội"): 110,
    ("TP HCM", "Đà Lạt"): 50,
    ("Đà Lạt", "TP HCM"): 50,
}

# Từ điển ước tính thuế phí trung bình theo hãng (VND)
# Dùng để điền vào cột 'fees' nếu mô hình yêu cầu
FEE_MAP = {
    "Vietnam Airlines": 660000,
    "Vietjet": 650000,
    "Bamboo Airways": 657000,
    "Pacific Airlines": 655000,
    "Vietravel Airlines": 646000
}

# Danh sách các lựa chọn cho Dropdown
AIRLINES = list(FEE_MAP.keys())
CITIES =
# Các loại vé phổ biến (Cần khớp với dữ liệu lúc train)
TICKET_TYPES =

# ---------------------------------------------------------
# 2. HÀM XỬ LÝ MODEL VÀ DỮ LIỆU
# ---------------------------------------------------------

@st.cache_resource
def load_resources():
    """
    Tải 2 mô hình và danh sách cột mẫu.
    LƯU Ý: Bạn cần có file 'model_columns.pkl' chứa list(X_train.columns)
    để đảm bảo thứ tự cột khi dự đoán.
    """
    try:
        lr = joblib.load('linear_regression_model.pkl')
        xgb_mod = joblib.load('xgboost_model.pkl')
        # Danh sách cột này BẮT BUỘC phải khớp với lúc train (sau khi one-hot)
        cols = joblib.load('model_columns.pkl') 
        return lr, xgb_mod, cols
    except FileNotFoundError as e:
        return None, None, None

def process_input(airline, src, dst, date, time, ticket_type, model_cols):
    """Chuyển đổi input người dùng thành DataFrame đúng chuẩn model"""
    
    # 1. Tính toán thời gian bay tự động
    duration = DURATION_MAP.get((src, dst), 120) # Mặc định 120p nếu không tìm thấy
    
    # 2. Xử lý ngày giờ
    dep_dt = pd.to_datetime(f"{date} {time}")
    arr_dt = dep_dt + timedelta(minutes=duration)
    
    # 3. Tạo dictionary chứa dữ liệu thô
    data = {
        'duration_minutes': duration,
        'fees': FEE_MAP.get(airline, 650000), # Giả lập phí
        'day_of_week': dep_dt.dayofweek,      # 0=Monday
        'day': dep_dt.day,
        'month': dep_dt.month,
        'hour': dep_dt.hour,
        # Các cột Category (sẽ được One-Hot ngay sau đây)
        'code_name': airline,
        'from': src,
        'to': dst,
        'type': ticket_type
    }
    
    # 4. Tạo DataFrame và One-Hot Encoding
    df_raw = pd.DataFrame([data])
    df_encoded = pd.get_dummies(df_raw)
    
    # 5. ALIGNMENT (Bước quan trọng nhất)
    # Reindex để tạo các cột còn thiếu (với giá trị 0) và sắp xếp đúng thứ tự
    df_final = df_encoded.reindex(columns=model_cols, fill_value=0)
    
    return df_final, duration

# ---------------------------------------------------------
# 3. GIAO DIỆN STREAMLIT
# ---------------------------------------------------------

st.set_page_config(page_title="Dự Báo Giá Vé Máy Bay", layout="wide")

st.title("✈️ Ứng dụng Dự Đoán Giá Vé Máy Bay")
st.markdown("---")

# Tải model
lr_model, xgb_model, model_columns = load_resources()

if lr_model is None:
    st.error("⚠️ Không tìm thấy file model (.pkl). Hãy đảm bảo bạn đã upload file model và file 'model_columns.pkl'.")
    st.stop()

# Layout chia 2 phần: Input (Trái) và Kết quả (Phải)
col_input, col_result = st.columns([1, 2])

with col_input:
    st.subheader("Thông tin chuyến bay")
    
    # Input Form
    with st.form("flight_form"):
        airline = st.selectbox("Hãng hàng không", AIRLINES)
        col_src, col_dst = st.columns(2)
        with col_src:
            src = st.selectbox("Điểm đi", CITIES, index=1) # Default Hà Nội
        with col_dst:
            # Lọc điểm đến để không trùng điểm đi
            dst_opts =
            dst = st.selectbox("Điểm đến", dst_opts)
            
        col_date, col_time = st.columns(2)
        with col_date:
            d_date = st.date_input("Ngày bay", datetime.now())
        with col_time:
            d_time = st.time_input("Giờ bay", datetime.now())
            
        ticket_cls = st.selectbox("Hạng vé", TICKET_TYPES)
        
        submitted = st.form_submit_button("🔍 Dự đoán ngay")

# Xử lý khi bấm nút
if submitted:
    # Xử lý dữ liệu
    X_input, duration_mins = process_input(
        airline, src, dst, d_date, d_time, ticket_cls, model_columns
    )
    
    with col_result:
        st.subheader("Kết quả dự báo")
        
        # Hiển thị thông tin hành trình tự động tính toán
        st.info(f"⏱️ **Hệ thống tự động tính toán:** Chặng bay {src} - {dst} thường kéo dài **{duration_mins} phút**.")
        
        # Dự đoán
        try:
            pred_lr = lr_model.predict(X_input)
            pred_xgb = xgb_model.predict(X_input)
            
            # Hiển thị 2 model cạnh nhau để so sánh
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("### Linear Regression")
                st.metric(label="Giá dự kiến", value=f"{pred_lr:,.0f} VND")
                st.caption("Mô hình tuyến tính: Phù hợp xu hướng giá ổn định.")
                
            with c2:
                st.markdown("### XGBoost Model")
                st.metric(label="Giá dự kiến", value=f"{pred_xgb:,.0f} VND")
                st.caption("Mô hình phi tuyến: Bắt tốt các biến động giá phức tạp.")
            
            # So sánh độ lệch
            diff = abs(pred_lr - pred_xgb)
            st.warning(f"💡 **Phân tích:** Hai mô hình chênh lệch nhau **{diff:,.0f} VND**.")
            
        except Exception as e:
            st.error(f"Đã có lỗi xảy ra khi dự đoán: {str(e)}")
            st.write("Vui lòng kiểm tra lại sự tương thích giữa dữ liệu nhập và file model_columns.pkl")

else:
    with col_result:
        st.info("👈 Vui lòng nhập thông tin bên trái để bắt đầu dự đoán.")
