import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from datetime import datetime

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Dự Đoán Giá Vé Máy Bay", layout="wide")

# --- DANH SÁCH CÁC CỘT (FEATURES) TỪ QUÁ TRÌNH TRAINING ---
# Đây là danh sách chính xác thứ tự các cột mà mô hình yêu cầu (84 cột)
MODEL_COLUMNS = [
    'hour', 'day_of_week', 'day', 'month', 'duration_minutes',
    'code_name_Pacific Airlines', 'code_name_Vietjet', 'code_name_Vietnam Airlines', 'code_name_Vietravel Airlines',
    'from_Cần Thơ', 'from_Huế', 'from_Hà Nội', 'from_Hải Phòng', 'from_Nha Trang', 'from_Phú Quốc',
    'from_Quy Nhơn', 'from_TP HCM', 'from_Thanh Hóa', 'from_Vinh', 'from_Đà Lạt', 'from_Đà Nẵng',
    'to_Cần Thơ', 'to_Huế', 'to_Hà Nội', 'to_Hải Phòng', 'to_Nha Trang', 'to_Phú Quốc',
    'to_Quy Nhơn', 'to_TP HCM', 'to_Thanh Hóa', 'to_Vinh', 'to_Đà Lạt', 'to_Đà Nẵng',
    'type_Bregow (B) - Vé không hoàn', 'type_Business (BC)-D', 'type_Business (BC)-I', 'type_Business (BF)-C', 'type_Business (BF)-J',
    'type_Buz Flex', 'type_Buz smart', 'type_Cregow (C) - Vé không hoàn', 'type_Dregow (D) - Vé không hoàn',
    'type_Eco', 'type_Eco Flex', 'type_Eco Saver', 'type_Eco Saver max', 'type_Eco Smart',
    'type_Economy (EC)-E', 'type_Economy (EC)-L', 'type_Economy (EC)-N', 'type_Economy (EC)-Q',
    'type_Economy (EC)-R', 'type_Economy (EC)-T', 'type_Economy (EF)-H', 'type_Economy (EF)-K',
    'type_Economy (EF)-S', 'type_Economy (EG)-M', 'type_Economy (EL)-A', 'type_Economy (EL)-P',
    'type_Eregow (E) - Vé không hoàn', 'type_Hregow (H) - Vé không hoàn', 'type_Kregow (K) - Vé không hoàn',
    'type_Lregow (L) - Vé không hoàn', 'type_Mregow (M) - Vé không hoàn', 'type_Nfleow (N) - Vé được hoàn',
    'type_Ofleow (O) - Vé được hoàn', 'type_Promo1 (P) - Vé không hoàn', 'type_Qfleow (Q) - Vé được hoàn',
    'type_Rfleow (R) - Vé được hoàn', 'type_Sfleow (S) - Vé được hoàn', 'type_SkyBoss',
    'type_Tfleow (T) - Vé được hoàn', 'type_Vfleow (V) - Vé được hoàn', 'type_Yfleow (Y) - Vé được hoàn'
]

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    try:
        lin_reg = joblib.load('linear_regression_model.pkl')
        xgb_model = joblib.load('xgboost_model.pkl')
        return lin_reg, xgb_model
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        return None, None

lin_reg, xgb_model = load_models()

# --- GIAO DIỆN NGƯỜI DÙNG ---
st.title("✈️ Dự Đoán Giá Vé Máy Bay Việt Nam")
st.markdown("So sánh kết quả giữa mô hình **Linear Regression** và **XGBoost**.")

# Tạo 2 cột cho form nhập liệu
col1, col2 = st.columns(2)

with col1:
    st.subheader("Thông tin chuyến bay")
    
    # 1. Hãng bay (Bamboo Airways là reference category nên không có trong list cột, ta thêm vào UI để xử lý logic)
    airline_options = ['Bamboo Airways', 'Pacific Airlines', 'Vietjet', 'Vietnam Airlines', 'Vietravel Airlines']
    airline = st.selectbox("Hãng hàng không", airline_options)

    # 2. Điểm đi và đến
    # Lấy danh sách thành phố từ tên cột (bỏ tiền tố 'from_' hoặc 'to_')
    city_options = sorted(list(set([c.replace('from_', '') for c in MODEL_COLUMNS if c.startswith('from_')])))
    # Thêm tùy chọn "Khác" cho các thành phố bị ẩn do drop_first=True (Reference Category)
    city_options.append("Khác (Thành phố khác)")
    
    source = st.selectbox("Điểm đi", city_options, index=city_options.index('Hà Nội') if 'Hà Nội' in city_options else 0)
    destination = st.selectbox("Điểm đến", city_options, index=city_options.index('TP HCM') if 'TP HCM' in city_options else 0)

    # 3. Thời gian bay (thay vì nhập giờ hạ cánh)
    duration = st.number_input("Thời gian bay dự kiến (phút)", min_value=30, max_value=300, value=120, step=5, help="Ví dụ: Bay Hà Nội - Sài Gòn khoảng 120 phút")

with col2:
    st.subheader("Chi tiết vé & Thời gian")
    
    # 4. Loại vé
    type_options = sorted([c.replace('type_', '') for c in MODEL_COLUMNS if c.startswith('type_')])
    ticket_type = st.selectbox("Hạng vé", type_options, index=type_options.index('Eco') if 'Eco' in type_options else 0)

    # 5. Ngày giờ khởi hành
    dep_date = st.date_input("Ngày khởi hành", datetime.now())
    dep_time = st.time_input("Giờ khởi hành", datetime.now())

# --- XỬ LÝ DỮ LIỆU ĐẦU VÀO ---
def preprocess_input(airline, source, destination, ticket_type, dep_date, dep_time, duration):
    # Tạo vector đầu vào với toàn số 0
    input_data = pd.DataFrame(np.zeros((1, len(MODEL_COLUMNS))), columns=MODEL_COLUMNS)
    
    # 1. Điền các biến số học
    # Ghép ngày và giờ
    flight_datetime = datetime.combine(dep_date, dep_time)
    
    input_data['hour'] = flight_datetime.hour
    input_data['day_of_week'] = flight_datetime.weekday()
    input_data['day'] = flight_datetime.day
    input_data['month'] = flight_datetime.month
    input_data['duration_minutes'] = duration

    # 2. One-Hot Encoding (Điền số 1 vào các cột tương ứng)
    # Lưu ý: Nếu chọn Bamboo Airways hoặc thành phố "Khác", tất cả các cột liên quan sẽ giữ nguyên là 0 (đúng logic drop_first)
    
    # Hãng bay
    if f'code_name_{airline}' in MODEL_COLUMNS:
        input_data[f'code_name_{airline}'] = 1
        
    # Điểm đi
    if f'from_{source}' in MODEL_COLUMNS:
        input_data[f'from_{source}'] = 1
        
    # Điểm đến
    if f'to_{destination}' in MODEL_COLUMNS:
        input_data[f'to_{destination}'] = 1
        
    # Loại vé
    if f'type_{ticket_type}' in MODEL_COLUMNS:
        input_data[f'type_{ticket_type}'] = 1
        
    return input_data

# --- NÚT DỰ ĐOÁN ---
if st.button("🔍 Dự đoán giá vé", use_container_width=True):
    if lin_reg and xgb_model:
        # Xử lý dữ liệu
        X_input = preprocess_input(airline, source, destination, ticket_type, dep_date, dep_time, duration)
        
        # Dự đoán
        try:
            price_lr = lin_reg.predict(X_input)[0]
            price_xgb = xgb_model.predict(X_input)[0]
            
            # Hiển thị kết quả
            st.markdown("---")
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.info("🤖 **Linear Regression**")
                st.metric(label="Giá dự đoán", value=f"{price_lr:,.0f} VNĐ")
            
            with res_col2:
                st.success("🚀 **XGBoost (Thường chính xác hơn)**")
                st.metric(label="Giá dự đoán", value=f"{price_xgb:,.0f} VNĐ")
                
            # So sánh
            diff = abs(price_lr - price_xgb)
            st.caption(f"Chênh lệch giữa 2 mô hình: {diff:,.0f} VNĐ")
            
        except Exception as e:
            st.error(f"Có lỗi xảy ra khi dự đoán: {e}")
            st.write("Vui lòng kiểm tra lại dữ liệu đầu vào hoặc file model.")
    else:
        st.warning("Chưa tải được file model. Vui lòng kiểm tra file .pkl trong thư mục.")

# --- FOOTER ---
st.markdown("---")
st.markdown("*Lưu ý: Giá vé chỉ mang tính chất tham khảo dựa trên dữ liệu lịch sử.*")
