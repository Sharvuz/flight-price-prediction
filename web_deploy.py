import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from datetime import timedelta

# 1. CẤU HÌNH TRANG
st.set_page_config(page_title="Dự Đoán Giá Vé Máy Bay", layout="wide")

# 2. TẢI MÔ HÌNH VÀ DỮ LIỆU CẤU HÌNH
@st.cache_resource
def load_models():
    # Load models
    lr = joblib.load('linear_regression_model.pkl')
    xgb_mod = joblib.load('xgboost_model.pkl')
    # Load danh sách cột mẫu để đảm bảo one-hot encoding khớp 100%
    cols = joblib.load('model_columns.pkl') 
    return lr, xgb_mod, cols

try:
    lr_model, xgb_model, model_columns = load_models()
except Exception as e:
    st.error(f"Lỗi tải mô hình: {e}. Hãy đảm bảo bạn đã upload đủ file .pkl")
    st.stop()

# 3. DỮ LIỆU THAM CHIẾU (HARDCODED DATA)
# Danh sách thành phố trích xuất từ dữ liệu huấn luyện của bạn
CITIES = [
    'TP HCM', 'Hà Nội', 'Đà Nẵng', 'Phú Quốc', 'Nha Trang', 'Đà Lạt', 
    'Hải Phòng', 'Vinh', 'Thanh Hóa', 'Cần Thơ', 'Huế', 'Quy Nhơn'
]

# Danh sách hạng vé (Trích xuất từ các cột type_... trong notebook)
TICKET_TYPES = [
    'Eco', 'Eco Saver', 'Eco Smart', 'Eco Flex', 'SkyBoss', 
    'Buz Smart', 'Buz Flex',
    'Economy (EL)-P', 'Economy (EL)-A', 'Economy (EC)-T', 'Economy (EC)-R',
    'Economy (EC)-N', 'Economy (EC)-Q', 'Economy (EC)-L', 'Economy (EC)-E',
    'Economy (EF)-H', 'Economy (EF)-K', 'Economy (EF)-S', 'Economy (EG)-M',
    'Business (BC)-D', 'Business (BC)-I', 'Business (BF)-C', 'Business (BF)-J',
    'Promo1 (P) - Vé không hoàn', 
    'Aregow (A) - Vé không hoàn', 'Bregow (B) - Vé không hoàn',
    'Cregow (C) - Vé không hoàn', 'Dregow (D) - Vé không hoàn',
    'Eregow (E) - Vé không hoàn', 'Hregow (H) - Vé không hoàn',
    'Kregow (K) - Vé không hoàn', 'Lregow (L) - Vé không hoàn',
    'Mregow (M) - Vé không hoàn'
]

# Phí cơ bản của các hãng (Ước tính trung bình)
FEE_MAP = {
    "Vietnam Airlines": 650000,
    "Vietjet": 600000,
    "Bamboo Airways": 640000,
    "Pacific Airlines": 620000,
    "Vietravel Airlines": 610000
}

# Bản đồ thời gian bay trung bình (phút) cho các chặng phổ biến
# Key là tuple (Điểm đi, Điểm đến), Value là số phút
DURATION_MAP = {
    # Trục Bắc - Nam
    ("Hà Nội", "TP HCM"): 130, ("TP HCM", "Hà Nội"): 130,
    ("Hải Phòng", "TP HCM"): 120, ("TP HCM", "Hải Phòng"): 120,
    ("Vinh", "TP HCM"): 105, ("TP HCM", "Vinh"): 105,
    ("Thanh Hóa", "TP HCM"): 115, ("TP HCM", "Thanh Hóa"): 115,
    
    # Trục Miền Trung
    ("Hà Nội", "Đà Nẵng"): 85, ("Đà Nẵng", "Hà Nội"): 85,
    ("TP HCM", "Đà Nẵng"): 85, ("Đà Nẵng", "TP HCM"): 85,
    ("Hà Nội", "Huế"): 80, ("Huế", "Hà Nội"): 80,
    ("TP HCM", "Huế"): 90, ("Huế", "TP HCM"): 90,
    ("Hà Nội", "Quy Nhơn"): 100, ("Quy Nhơn", "Hà Nội"): 100,
    ("TP HCM", "Quy Nhơn"): 70, ("Quy Nhơn", "TP HCM"): 70,

    # Du lịch (Nha Trang, Đà Lạt, Phú Quốc)
    ("Hà Nội", "Nha Trang"): 115, ("Nha Trang", "Hà Nội"): 115,
    ("TP HCM", "Nha Trang"): 65, ("Nha Trang", "TP HCM"): 65,
    ("Hà Nội", "Đà Lạt"): 110, ("Đà Lạt", "Hà Nội"): 110,
    ("TP HCM", "Đà Lạt"): 50, ("Đà Lạt", "TP HCM"): 50,
    ("Hà Nội", "Phú Quốc"): 135, ("Phú Quốc", "Hà Nội"): 135,
    ("TP HCM", "Phú Quốc"): 60, ("Phú Quốc", "TP HCM"): 60,
    ("Cần Thơ", "Hà Nội"): 135, ("Hà Nội", "Cần Thơ"): 135,
}

# 4. GIAO DIỆN (UI)
st.title("✈️ Dự Đoán Giá Vé Máy Bay Việt Nam")
st.write("Nhập thông tin chuyến bay để hệ thống tự động tính toán và dự báo giá vé.")

col_ui_1, col_ui_2 = st.columns([1, 2])

with col_ui_1:
    st.subheader("Thông tin chuyến bay")
    
    # Chọn hãng
    airline = st.selectbox("Hãng hàng không", list(FEE_MAP.keys()))
    
    # Chọn điểm đi/đến
    origin = st.selectbox("Điểm đi", CITIES, index=0) # Mặc định TP HCM
    dest_options = [c for c in CITIES if c != origin]
    destination = st.selectbox("Điểm đến", dest_options, index=0) # Mặc định Hà Nội
    
    # Chọn ngày giờ
    col_d, col_t = st.columns(2)
    dep_date = col_d.date_input("Ngày bay")
    dep_time = col_t.time_input("Giờ bay")
    
    # Chọn hạng vé
    ticket_cls = st.selectbox("Hạng vé", TICKET_TYPES)
    
    predict_btn = st.button("🔍 Dự đoán ngay", use_container_width=True)

# 5. XỬ LÝ LOGIC (BACKEND)
with col_ui_2:
    if predict_btn:
        # --- BƯỚC 1: TỰ ĐỘNG TÍNH TOÁN DURATION ---
        route = (origin, destination)
        
        if route in DURATION_MAP:
            duration_mins = DURATION_MAP[route]
            is_estimated = False
        else:
            # Fallback nếu chặng bay lạ: Tính theo khoảng cách địa lý giả định hoặc trung bình
            duration_mins = 90 
            is_estimated = True
            st.warning(f"⚠️ Chặng bay {origin} - {destination} chưa có dữ liệu chính xác. Hệ thống sẽ dùng giá trị ước tính trung bình.")

        # Hiển thị thông tin máy tính toán cho user thấy
        st.subheader("Kết quả phân tích")
        
        info_col1, info_col2, info_col3 = st.columns(3)
        info_col1.metric("Thời gian bay", f"{duration_mins} phút")
        
        # Tính ngày giờ đến
        full_dep_datetime = pd.to_datetime(f"{dep_date} {dep_time}")
        full_arr_datetime = full_dep_datetime + timedelta(minutes=duration_mins)
        info_col2.metric("Giờ đến dự kiến", full_arr_datetime.strftime('%H:%M'))
        
        # Phí sân bay/hãng (Feature engineer)
        est_fees = FEE_MAP.get(airline, 650000)
        
        # --- BƯỚC 2: CHUẨN BỊ DỮ LIỆU CHO MODEL ---
        # Tạo dictionary input thô
        input_data = {
            'f_price': 0, # Placeholder
            'fees': est_fees,
            'duration_minutes': float(duration_mins),
            'day_of_week': full_dep_datetime.dayofweek,
            'day': full_dep_datetime.day,
            'month': full_dep_datetime.month,
            'hour': full_dep_datetime.hour,
            'code_name': airline,
            'from': origin,
            'to': destination,
            'type': ticket_cls
        }
        
        # Tạo DataFrame
        df_input = pd.DataFrame([input_data])
        
        # One-Hot Encoding
        # Quan trọng: Phải dùng pd.get_dummies giống hệt lúc train
        df_processed = pd.get_dummies(df_input)
        
        # --- BƯỚC 3: CĂN CHỈNH CỘT (ALIGNMENT) ---
        # Bắt buộc: Reindex để tạo ra các cột thiếu (với giá trị 0) và bỏ các cột thừa
        # giúp khớp hoàn toàn với model đã lưu.
        df_final = df_processed.reindex(columns=model_columns, fill_value=0)
        
        # --- BƯỚC 4: DỰ ĐOÁN ---
        try:
            # Linear Regression
            pred_lr = lr_model.predict(df_final)[0]
            
            # XGBoost
            pred_xgb = xgb_model.predict(df_final)[0]
            
            # Hiển thị kết quả
            st.divider()
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.success(f"💵 Dự báo (XGBoost): **{pred_xgb:,.0f} VND**")
                st.caption("Mô hình XGBoost thường chính xác hơn với các biến động phức tạp.")
                
            with res_col2:
                st.info(f"💵 Tham chiếu (Linear): **{pred_lr:,.0f} VND**")
                st.caption("Mô hình hồi quy tuyến tính cơ bản.")
            
            # Kiểm tra logic giá âm (nếu model dự đoán sai)
            if pred_xgb < 0 or pred_lr < 0:
                st.error("Lưu ý: Mô hình trả về giá trị âm, có thể do dữ liệu đầu vào (Hạng vé/Chặng bay) hiếm gặp trong tập huấn luyện.")
                
        except Exception as e:
            st.error(f"Lỗi trong quá trình dự đoán: {str(e)}")
            st.code(df_final.columns) # Debug info nếu cần

    else:
        st.info("👈 Vui lòng chọn thông tin và bấm nút dự đoán.")
