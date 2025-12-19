import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# 1. DANH SÁCH CỘT CỐ ĐỊNH (FIX CỨNG ĐỂ TRÁNH LỖI)
# ==========================================
# Đây là danh sách 74 cột chính xác mà Model XGBoost/Linear của bạn đã học
# Chúng ta phải tạo ra đúng thứ tự này thì model mới chạy được.
MODEL_COLUMNS = [
    'hour', 'day_of_week', 'day', 'month', 'duration_minutes', 
    'code_name_Pacific Airlines', 'code_name_Vietjet', 'code_name_Vietnam Airlines', 'code_name_Vietravel Airlines', 
    'from_Cần Thơ', 'from_Huế', 'from_Hà Nội', 'from_Hải Phòng', 'from_Nha Trang', 'from_Phú Quốc', 'from_Quy Nhơn', 'from_TP HCM', 'from_Thanh Hóa', 'from_Vinh', 'from_Đà Lạt', 'from_Đà Nẵng', 
    'to_Cần Thơ', 'to_Huế', 'to_Hà Nội', 'to_Hải Phòng', 'to_Nha Trang', 'to_Phú Quốc', 'to_Quy Nhơn', 'to_TP HCM', 'to_Thanh Hóa', 'to_Vinh', 'to_Đà Lạt', 'to_Đà Nẵng', 
    'type_Bregow (B) - Vé không hoàn', 'type_Business (BC)-D', 'type_Business (BC)-I', 'type_Business (BF)-C', 'type_Business (BF)-J', 'type_Buz Flex', 'type_Buz smart', 'type_Cregow (C) - Vé không hoàn', 'type_Dregow (D) - Vé không hoàn', 'type_Eco', 'type_Eco Flex', 'type_Eco Saver', 'type_Eco Saver max', 'type_Eco Smart', 'type_Economy (EC)-E', 'type_Economy (EC)-L', 'type_Economy (EC)-N', 'type_Economy (EC)-Q', 'type_Economy (EC)-R', 'type_Economy (EC)-T', 'type_Economy (EF)-H', 'type_Economy (EF)-K', 'type_Economy (EF)-S', 'type_Economy (EG)-M', 'type_Economy (EL)-A', 'type_Economy (EL)-P', 'type_Eregow (E) - Vé không hoàn', 'type_Hregow (H) - Vé không hoàn', 'type_Kregow (K) - Vé không hoàn', 'type_Lregow (L) - Vé không hoàn', 'type_Mregow (M) - Vé không hoàn', 'type_Nfleow (N) - Vé được hoàn', 'type_Ofleow (O) - Vé được hoàn', 'type_Promo1 (P) - Vé không hoàn', 'type_Qfleow (Q) - Vé được hoàn', 'type_Rfleow (R) - Vé được hoàn', 'type_Sfleow (S) - Vé được hoàn', 'type_SkyBoss', 'type_Tfleow (T) - Vé được hoàn', 'type_Vfleow (V) - Vé được hoàn', 'type_Yfleow (Y) - Vé được hoàn'
]

# ==========================================
# 2. HÀM LOAD DỮ LIỆU VÀ MODEL
# ==========================================
st.set_page_config(page_title="Dự đoán giá vé máy bay", page_icon="✈️", layout="wide")

@st.cache_resource
def load_resources():
    try:
        # Load 2 model đã train
        lr_model = joblib.load('linear_regression_model.pkl')
        xgb_model = joblib.load('xgboost_model.pkl')
        
        # Load data csv để lấy thông tin dropdown và tính giờ bay
        df = pd.read_csv('flight.csv') 
        
        # Xử lý datetime chuẩn xác
        df['f_time_from'] = pd.to_datetime(df['f_time_from'], format='%H:%M:%S %d/%m/%Y', errors='coerce')
        df['f_time_to'] = pd.to_datetime(df['f_time_to'], format='%H:%M:%S %d/%m/%Y', errors='coerce')
        
        # Tính thời gian bay (phút) cho từng dòng
        df['duration_minutes'] = (df['f_time_to'] - df['f_time_from']).dt.total_seconds() / 60
        
        # Tạo bảng tra cứu thời gian bay trung bình: (Nơi đi, Nơi đến) -> Phút
        # Ví dụ: ('Hà Nội', 'TP HCM') -> 125.0
        route_map = df.groupby(['from', 'to'])['duration_minutes'].mean().to_dict()
        
        # Tính trung bình toàn bộ để backup nếu gặp chặng lạ
        global_avg = df['duration_minutes'].mean()
        
        return lr_model, xgb_model, df, route_map, global_avg
    except FileNotFoundError as e:
        st.error(f"❌ Không tìm thấy file: {e}")
        return None, None, None, None, 120
    except Exception as e:
        st.error(f"❌ Lỗi khởi tạo: {e}")
        return None, None, None, None, 120

# Gọi hàm load
lr_model, xgb_model, df_org, route_map, global_avg = load_resources()

# ==========================================
# 3. HÀM TÍNH TOÁN THÔNG MINH
# ==========================================
def get_smart_duration(source, dest):
    """Tìm thời gian bay dựa trên lịch sử"""
    if route_map is None: return 120
    
    # 1. Tìm chính xác chiều đi
    if (source, dest) in route_map:
        return route_map[(source, dest)]
    # 2. Nếu không có, tìm chiều về (thường thời gian bay tương đương)
    elif (dest, source) in route_map:
        return route_map[(dest, source)]
    # 3. Không có nữa thì lấy trung bình chung
    else:
        return global_avg

# ==========================================
# 4. GIAO DIỆN NGƯỜI DÙNG
# ==========================================
# Sidebar chọn model
st.sidebar.title("⚙️ Cấu hình")
model_option = st.sidebar.radio("Thuật toán dự đoán:", ("XGBoost (Khuyên dùng)", "Linear Regression"))
st.sidebar.info("💡 **Mẹo:** XGBoost thường chính xác hơn cho các bài toán giá cả phức tạp.")

st.title("✈️ Dự đoán giá vé máy bay AI")
st.markdown("---")

if df_org is not None:
    # --- KHU VỰC NHẬP LIỆU (LIVE UPDATE) ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Sắp xếp danh sách cho dễ tìm
        airline_list = sorted(df_org['code_name'].dropna().unique())
        source_list = sorted(df_org['from'].dropna().unique())
        
        airline = st.selectbox("Hãng hàng không", airline_list)
        source = st.selectbox("Điểm đi (Nơi xuất phát)", source_list)
    
    with col2:
        type_list = sorted(df_org['type'].dropna().unique())
        # Lọc điểm đến khác điểm đi
        dest_list = [d for d in sorted(df_org['to'].dropna().unique()) if d != source]
        
        ticket_type = st.selectbox("Loại vé / Hạng ghế", type_list)
        destination = st.selectbox("Điểm đến", dest_list if dest_list else ["Không có điểm đến"])
        
    with col3:
        d_date = st.date_input("Ngày bay", datetime.now())
        d_time = st.time_input("Giờ bay", datetime.now().time())

    # --- TÍNH TOÁN & HIỂN THỊ THỜI GIAN (TỰ ĐỘNG) ---
    # Code chạy ngay khi người dùng thay đổi bất kỳ ô nào ở trên
    avg_duration = get_smart_duration(source, destination)
    
    dep_dt = datetime.combine(d_date, d_time)
    arr_dt = dep_dt + timedelta(minutes=avg_duration)
    
    # Box thông tin hành trình
    st.info(
        f"📅 **Hành trình dự kiến:**\n\n"
        f"🛫 **{source}** ({d_time.strftime('%H:%M')})  ➡  "
        f"🛬 **{destination}** ({arr_dt.strftime('%H:%M')})\n\n"
        f"⏱️ Thời gian bay: **{int(avg_duration)} phút** "
        f"({ 'Bay qua đêm' if arr_dt.date() > d_date else 'Trong ngày' })"
    )

    # --- NÚT DỰ ĐOÁN & XỬ LÝ MODEL ---
    if st.button("💰 Dự đoán giá vé ngay", type="primary", use_container_width=True):
        if lr_model is None or xgb_model is None:
            st.error("Chưa load được model!")
        else:
            try:
                # 1. TẠO DATAFRAME RỖNG 74 CỘT (TOÀN SỐ 0)
                # Đây là bước quan trọng nhất để fix lỗi lệch cột
                input_df = pd.DataFrame(0, index=[0], columns=MODEL_COLUMNS)
                
                # 2. ĐIỀN DỮ LIỆU SỐ
                input_df['hour'] = d_time.hour
                input_df['day_of_week'] = d_date.weekday()
                input_df['day'] = d_date.day
                input_df['month'] = d_date.month
                input_df['duration_minutes'] = avg_duration
                
                # 3. ĐIỀN DỮ LIỆU CATEGORY (ONE-HOT ENCODING)
                # Tạo các tên cột cần bật lên số 1
                # Lưu ý: Các prefix này phải khớp với cách pd.get_dummies đặt tên
                cols_to_active = [
                    f'code_name_{airline}',
                    f'from_{source}',
                    f'to_{destination}',
                    f'type_{ticket_type}'
                ]
                
                # Duyệt qua các cột cần bật, nếu có trong MODEL_COLUMNS thì gán = 1
                # Nếu không có (ví dụ Bamboo Airways bị drop do drop_first=True), thì giữ nguyên là 0
                for col in cols_to_active:
                    if col in input_df.columns:
                        input_df[col] = 1
                
                # 4. CHẠY PREDICT
                if model_option == "Linear Regression":
                    pred_price = lr_model.predict(input_df)[0]
                else:
                    pred_price = xgb_model.predict(input_df)[0]
                
                # 5. HIỂN THỊ KẾT QUẢ
                st.success(f"### 💵 Giá vé dự đoán: {pred_price:,.0f} VNĐ")
                st.balloons()
                
            except Exception as e:
                st.error(f"Có lỗi xảy ra khi dự đoán: {e}")
                # Debug chi tiết nếu cần thiết
                # st.write("Input Data:", input_df)

else:
    st.warning("Đang tải dữ liệu flight.csv...")
