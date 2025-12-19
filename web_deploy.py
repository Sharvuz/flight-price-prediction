import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# 1. CẤU HÌNH & LOAD DỮ LIỆU
# ==========================================
st.set_page_config(page_title="Dự đoán giá vé máy bay", page_icon="✈️", layout="wide")

# Danh sách 74 cột model yêu cầu (Cố định để tránh lỗi lệch cột)
MODEL_COLUMNS = [
    'hour', 'day_of_week', 'day', 'month', 'duration_minutes', 
    'code_name_Pacific Airlines', 'code_name_Vietjet', 'code_name_Vietnam Airlines', 'code_name_Vietravel Airlines', 
    'from_Cần Thơ', 'from_Huế', 'from_Hà Nội', 'from_Hải Phòng', 'from_Nha Trang', 'from_Phú Quốc', 'from_Quy Nhơn', 'from_TP HCM', 'from_Thanh Hóa', 'from_Vinh', 'from_Đà Lạt', 'from_Đà Nẵng', 
    'to_Cần Thơ', 'to_Huế', 'to_Hà Nội', 'to_Hải Phòng', 'to_Nha Trang', 'to_Phú Quốc', 'to_Quy Nhơn', 'to_TP HCM', 'to_Thanh Hóa', 'to_Vinh', 'to_Đà Lạt', 'to_Đà Nẵng', 
    'type_Bregow (B) - Vé không hoàn', 'type_Business (BC)-D', 'type_Business (BC)-I', 'type_Business (BF)-C', 'type_Business (BF)-J', 'type_Buz Flex', 'type_Buz smart', 'type_Cregow (C) - Vé không hoàn', 'type_Dregow (D) - Vé không hoàn', 'type_Eco', 'type_Eco Flex', 'type_Eco Saver', 'type_Eco Saver max', 'type_Eco Smart', 'type_Economy (EC)-E', 'type_Economy (EC)-L', 'type_Economy (EC)-N', 'type_Economy (EC)-Q', 'type_Economy (EC)-R', 'type_Economy (EC)-T', 'type_Economy (EF)-H', 'type_Economy (EF)-K', 'type_Economy (EF)-S', 'type_Economy (EG)-M', 'type_Economy (EL)-A', 'type_Economy (EL)-P', 'type_Eregow (E) - Vé không hoàn', 'type_Hregow (H) - Vé không hoàn', 'type_Kregow (K) - Vé không hoàn', 'type_Lregow (L) - Vé không hoàn', 'type_Mregow (M) - Vé không hoàn', 'type_Nfleow (N) - Vé được hoàn', 'type_Ofleow (O) - Vé được hoàn', 'type_Promo1 (P) - Vé không hoàn', 'type_Qfleow (Q) - Vé được hoàn', 'type_Rfleow (R) - Vé được hoàn', 'type_Sfleow (S) - Vé được hoàn', 'type_SkyBoss', 'type_Tfleow (T) - Vé được hoàn', 'type_Vfleow (V) - Vé được hoàn', 'type_Yfleow (Y) - Vé được hoàn'
]

@st.cache_resource
def load_resources():
    try:
        lr_model = joblib.load('linear_regression_model.pkl')
        xgb_model = joblib.load('xgboost_model.pkl')
        
        # Load và xử lý data để tính duration
        df = pd.read_csv('flight.csv') 
        df['f_time_from'] = pd.to_datetime(df['f_time_from'], format='%H:%M:%S %d/%m/%Y')
        df['f_time_to'] = pd.to_datetime(df['f_time_to'], format='%H:%M:%S %d/%m/%Y')
        df['duration_minutes'] = (df['f_time_to'] - df['f_time_from']).dt.total_seconds() / 60
        
        # Tạo map: (Điểm đi, Điểm đến) -> Thời gian bay trung bình
        route_map = df.groupby(['from', 'to'])['duration_minutes'].mean().to_dict()
        
        # Tính trung bình toàn bộ data để backup nếu gặp chặng lạ
        global_mean_duration = df['duration_minutes'].mean()
        
        return lr_model, xgb_model, df, route_map, global_mean_duration
    except Exception as e:
        st.error(f"Lỗi khởi tạo: {e}")
        return None, None, None, None, 120

lr_model, xgb_model, df_org, route_map, global_avg = load_resources()

# ==========================================
# 2. LOGIC TÍNH THỜI GIAN THÔNG MINH
# ==========================================
def get_smart_duration(source, dest, route_map, global_avg):
    # 1. Tìm đúng chặng
    if (source, dest) in route_map:
        return route_map[(source, dest)]
    
    # 2. Nếu không có, thử tìm chặng ngược lại (Vd: đi A->B không có, thì tìm B->A)
    # Vì thời gian bay về thường tương đương bay đi
    elif (dest, source) in route_map:
        return route_map[(dest, source)]
    
    # 3. Nếu vẫn không có, lấy trung bình toàn sàn
    else:
        return global_avg

# ==========================================
# 3. GIAO DIỆN (BỎ FORM ĐỂ TƯƠNG TÁC NGAY)
# ==========================================
st.sidebar.title("⚙️ Cấu hình")
model_option = st.sidebar.radio("Chọn Model:", ("XGBoost (Khuyên dùng)", "Linear Regression"))

st.title("✈️ Dự đoán giá vé máy bay AI")

if df_org is not None:
    # --- INPUT ---
    # Không dùng st.form để dữ liệu cập nhật tức thì
    col1, col2, col3 = st.columns(3)
    
    with col1:
        airline = st.selectbox("Hãng bay", sorted(df_org['code_name'].unique()))
        source = st.selectbox("Điểm đi", sorted(df_org['from'].unique()))
    
    with col2:
        ticket_type = st.selectbox("Loại vé", sorted(df_org['type'].unique()))
        # Lọc điểm đến khác điểm đi
        dest_list = [d for d in sorted(df_org['to'].unique()) if d != source]
        if not dest_list: dest_list = sorted(df_org['to'].unique())
        destination = st.selectbox("Điểm đến", dest_list)
        
    with col3:
        # Thời gian
        d_date = st.date_input("Ngày bay", datetime.now())
        d_time = st.time_input("Giờ bay", datetime.now().time())

    # --- TÍNH TOÁN REAL-TIME ---
    # Bước này chạy ngay lập tức mỗi khi bạn chỉnh giờ/địa điểm
    avg_duration = get_smart_duration(source, destination, route_map, global_avg)
    
    dep_dt = datetime.combine(d_date, d_time)
    arr_dt = dep_dt + timedelta(minutes=avg_duration)
    
    # Hiển thị thông tin hành trình ngay lập tức
    st.info(
        f"⏱️ **Thời gian bay:** {int(avg_duration)} phút  |  "
        f"🛫 **Khởi hành:** {d_time.strftime('%H:%M')}  ➡  "
        f"🛬 **Hạ cánh:** {arr_dt.strftime('%H:%M')} (Hôm sau: {'Có' if arr_dt.date() > d_date else 'Không'})"
    )

    # Nút bấm dự đoán giá
    if st.button("💰 Dự đoán giá vé ngay", type="primary"):
        try:
            # 1. Tạo input chuẩn 74 cột (Toàn bộ là 0)
            input_df = pd.DataFrame(0, index=[0], columns=MODEL_COLUMNS)
            
            # 2. Điền thông tin số
            input_df['hour'] = d_time.hour
            input_df['day_of_week'] = d_date.weekday()
            input_df['day'] = d_date.day
            input_df['month'] = d_date.month
            input_df['duration_minutes'] = avg_duration
            
            # 3. Điền thông tin One-Hot (Đánh dấu 1)
            # Tạo các tên cột cần tìm
            cols_to_active = [
                f'code_name_{airline}',
                f'from_{source}',
                f'to_{destination}',
                f'type_{ticket_type}'
            ]
            
            found_cols = []
            for col in cols_to_active:
                if col in input_df.columns:
                    input_df[col] = 1
                    found_cols.append(col)
            
            # Debug: In ra để kiểm tra
            # st.write("Các đặc trưng được kích hoạt:", found_cols)

            # 4. Dự đoán
            if model_option == "Linear Regression":
                price = lr_model.predict(input_df)[0]
            else:
                price = xgb_model.predict(input_df)[0]

            # Hiển thị kết quả to đẹp
            st.success(f"### 💵 Giá vé dự đoán: {price:,.0f} VND")
            
        except Exception as e:
            st.error(f"Có lỗi xảy ra: {e}")
