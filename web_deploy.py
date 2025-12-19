import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta

#CẤU HÌNH & DANH SÁCH CỘT (QUAN TRỌNG)
st.set_page_config(page_title="Dự đoán giá vé máy bay", page_icon="✈️", layout="wide")

# Đây là danh sách 74 cột chính xác mà Model của bạn yêu cầu (lấy từ log lỗi)
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
        
        # Load data để lấy danh sách dropdown
        df = pd.read_csv('flight.csv') 
        # Format datetime
        df['f_time_from'] = pd.to_datetime(df['f_time_from'], format='%H:%M:%S %d/%m/%Y')
        df['f_time_to'] = pd.to_datetime(df['f_time_to'], format='%H:%M:%S %d/%m/%Y')
        df['duration_minutes'] = (df['f_time_to'] - df['f_time_from']).dt.total_seconds() / 60
        
        route_map = df.groupby(['from', 'to'])['duration_minutes'].mean().to_dict()
        return lr_model, xgb_model, df, route_map
    except Exception as e:
        st.error(f"Lỗi khởi tạo: {e}")
        return None, None, None, None

lr_model, xgb_model, df_org, route_map = load_resources()


#CLIENT NGƯỜI DÙNG
st.sidebar.title("⚙️ Cấu hình")
model_option = st.sidebar.radio("Chọn Model:", ("XGBoost (Khuyên dùng)", "Linear Regression"))

st.title("✈️ Dự đoán giá vé máy bay AI")

if df_org is not None:
    with st.form("flight_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            airline = st.selectbox("Hãng bay", df_org['code_name'].unique())
            source = st.selectbox("Điểm đi", df_org['from'].unique())
        with col2:
            ticket_type = st.selectbox("Loại vé", df_org['type'].unique())
            dest_list = [d for d in df_org['to'].unique() if d != source]
            destination = st.selectbox("Điểm đến", dest_list if dest_list else df_org['to'].unique())
        with col3:
            d_date = st.date_input("Ngày bay", datetime.now())
            d_time = st.time_input("Giờ bay", datetime.now().time())

        submitted = st.form_submit_button("🔍 Dự đoán ngay")

    if submitted:
        #1. Tính toán thời gian
        avg_duration = route_map.get((source, destination), 120)
        dep_dt = datetime.combine(d_date, d_time)
        arr_dt = dep_dt + timedelta(minutes=avg_duration)
        
        st.success(f"⏱️ Thời gian bay: {int(avg_duration)} phút | 🛬 Hạ cánh: {arr_dt.strftime('%H:%M')}")

        #2. XỬ LÝ ONE-HOT ENCODING (PHẦN SỬA LỖI QUAN TRỌNG)
        try:
            # Tạo một DataFrame chỉ có 1 dòng, chứa tất cả các cột model cần, giá trị mặc định là 0
            input_df = pd.DataFrame(0, index=[0], columns=MODEL_COLUMNS)
            
            # Điền các giá trị số
            input_df['hour'] = d_time.hour
            input_df['day_of_week'] = d_date.weekday()
            input_df['day'] = d_date.day
            input_df['month'] = d_date.month
            input_df['duration_minutes'] = avg_duration
            
            #Điền các giá trị One-Hot (Đánh dấu 1 vào cột tương ứng)
            #Ví dụ: Nếu chọn 'Vietjet', cột 'code_name_Vietjet' sẽ bằng 1
            
            #Danh sách các prefix tương ứng với logic get_dummies của bạn
            cat_mapping = {
                f'code_name_{airline}': 1,
                f'from_{source}': 1,
                f'to_{destination}': 1,
                f'type_{ticket_type}': 1
            }
            
            for col_name, val in cat_mapping.items():
                if col_name in input_df.columns:
                    input_df[col_name] = val
                else:
                    #Trường hợp hiếm: Dữ liệu nhập vào không có trong lúc train (ví dụ sân bay mới)
                    pass 

            # 3. DỰ ĐOÁN
            if model_option == "Linear Regression":
                price = lr_model.predict(input_df)[0]
            else:
                price = xgb_model.predict(input_df)[0]

            st.header(f"💰 Giá vé dự đoán: {price:,.0f} VND")
            
        except Exception as e:
            st.error(f"Lỗi dự đoán: {e}")
