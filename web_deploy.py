import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta

# ==========================================
# 1. CẤU HÌNH & LOAD DỮ LIỆU
# ==========================================
st.set_page_config(page_title="Dự đoán giá vé máy bay", page_icon="✈️", layout="wide")


@st.cache_resource
def load_resources():
    # Load models
    try:
        lr_model = joblib.load('linear_regression_model.pkl')
        xgb_model = joblib.load('xgboost_model.pkl')
    except Exception as e:
        st.error(f"Lỗi load model: {e}")
        return None, None, None, None

    # Load data để tính toán thời gian bay trung bình
    try:
        df = pd.read_csv('flight.csv')  # Hoặc flight_v2.csv

        # Xử lý datetime để tính duration
        # Format trong file csv của bạn là: HH:MM:SS dd/mm/yyyy
        df['f_time_from'] = pd.to_datetime(df['f_time_from'], format='%H:%M:%S %d/%m/%Y')
        df['f_time_to'] = pd.to_datetime(df['f_time_to'], format='%H:%M:%S %d/%m/%Y')

        # Tính thời lượng bay (phút)
        df['duration_minutes'] = (df['f_time_to'] - df['f_time_from']).dt.total_seconds() / 60

        # Tạo từ điển thời gian bay trung bình: {(Điểm đi, Điểm đến): Phút}
        route_duration_map = df.groupby(['from', 'to'])['duration_minutes'].mean().to_dict()

        return lr_model, xgb_model, df, route_duration_map

    except Exception as e:
        st.error(f"Lỗi xử lý dữ liệu csv: {e}")
        return None, None, None, None


lr_model, xgb_model, df_org, route_map = load_resources()

# ==========================================
# 2. SIDEBAR - CẤU HÌNH
# ==========================================
st.sidebar.title("⚙️ Cấu hình")
model_option = st.sidebar.radio(
    "Chọn Model dự đoán:",
    ("XGBoost (Khuyên dùng)", "Linear Regression")
)
st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Hệ thống tự động:**\n"
    "Dựa trên dữ liệu lịch sử, hệ thống sẽ tự tính toán thời gian bay và giờ hạ cánh dự kiến."
)

# ==========================================
# 3. GIAO DIỆN CHÍNH
# ==========================================
st.title("✈️ Dự đoán giá vé máy bay AI")

if df_org is not None:
    # Lấy danh sách cho dropdown
    airlines = df_org['code_name'].unique()
    sources = df_org['from'].unique()
    destinations = df_org['to'].unique()
    ticket_types = df_org['type'].unique()

    with st.form("flight_form"):
        st.subheader("Thông tin chuyến bay")
        col1, col2, col3 = st.columns(3)

        with col1:
            airline = st.selectbox("Hãng hàng không", airlines)
            source = st.selectbox("Điểm đi", sources)

        with col2:
            ticket_type = st.selectbox("Loại vé", ticket_types)
            # Logic: Điểm đến không được trùng điểm đi (đơn giản hóa hiển thị)
            remain_dest = [d for d in destinations if d != source]
            destination = st.selectbox("Điểm đến", remain_dest if remain_dest else destinations)

        with col3:
            d_date = st.date_input("Ngày khởi hành", datetime.now())
            d_time = st.time_input("Giờ khởi hành", datetime.now().time())

        submit_btn = st.form_submit_button("🔍 Dự đoán Giá & Giờ đến")

    # ==========================================
    # 4. XỬ LÝ KHI BẤM NÚT
    # ==========================================
    if submit_btn:
        # --- A. TÍNH TOÁN THỜI GIAN ---
        # Lấy thời gian bay trung bình từ dữ liệu quá khứ
        # Mặc định 120 phút nếu là chặng bay mới chưa có trong data
        avg_duration = route_map.get((source, destination), 120)

        # Tính giờ đến dự kiến
        departure_datetime = datetime.combine(d_date, d_time)
        arrival_datetime = departure_datetime + timedelta(minutes=avg_duration)

        # Hiển thị thông tin hành trình cho người dùng xem
        st.success(f"⏱️ Thời gian bay dự kiến: **{int(avg_duration)} phút**")
        st.info(
            f"🛫 Khởi hành: {departure_datetime.strftime('%H:%M %d/%m/%Y')}  ➡  🛬 Hạ cánh (Dự kiến): **{arrival_datetime.strftime('%H:%M %d/%m/%Y')}**")

        # --- B. CHUẨN BỊ DỮ LIỆU CHO MODEL (Encoding) ---
        # Model của bạn cần input là số (Label Encoding), không phải chữ.
        # Ta cần map dữ liệu input về số dựa trên logic lúc train.
        # (Lý tưởng nhất là load encoder.pkl, ở đây ta dùng mapping từ data frame gốc)

        try:
            # Tạo mapping dynamic từ dataframe gốc
            airline_encoder = {val: i for i, val in enumerate(sorted(df_org['code_name'].unique()))}
            source_encoder = {val: i for i, val in enumerate(sorted(df_org['from'].unique()))}
            dest_encoder = {val: i for i, val in enumerate(sorted(df_org['to'].unique()))}
            # Lưu ý: Cột 'type' lúc train bạn dùng cột nào để encode? Kiểm tra kỹ lại notebook.
            # Giả sử bạn encode cột 'type'
            type_encoder = {val: i for i, val in enumerate(sorted(df_org['type'].unique()))}

            # Tạo input vector (cấu trúc cột phải KHỚP 100% với lúc train model)
            # Dựa trên notebook của bạn, tôi thấy bạn có các cột:
            # [code, from, to, type, f_time_from(xử lý ra hour, day...), duration...]

            input_data = pd.DataFrame({
                'code': [airline_encoder.get(airline, 0)],
                # Cần check lại tên cột trong notebook là 'code' hay 'code_name'
                'from': [source_encoder.get(source, 0)],
                'to': [dest_encoder.get(destination, 0)],
                'type': [type_encoder.get(ticket_type, 0)],

                # Các feature thời gian
                'hour': [d_time.hour],
                'day_of_week': [d_date.weekday()],  # 0=Monday
                'day': [d_date.day],
                'month': [d_date.month],
                'duration_minutes': [avg_duration]
            })

            # --- C. DỰ ĐOÁN ---
            if model_option == "Linear Regression":
                pred_price = lr_model.predict(input_data)[0]
            else:
                pred_price = xgb_model.predict(input_data)[0]

            # Hiển thị giá tiền đẹp
            st.header(f"💰 Giá vé dự đoán: {pred_price:,.0f} VND")

            # Debug: Hiện bảng input để bạn kiểm tra xem mapping đúng chưa
            with st.expander("Xem chi tiết dữ liệu đầu vào Model"):
                st.write(input_data)

        except Exception as e:
            st.error(f"Lỗi trong quá trình dự đoán: {e}")
            st.warning(
                "Gợi ý: Kiểm tra lại tên các cột (features) trong DataFrame input có khớp với tên cột lúc train model không?")
else:
    st.stop()