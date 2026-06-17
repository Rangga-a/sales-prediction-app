import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

# ===================================================
# CONFIG
# ===================================================

st.set_page_config(
    page_title="Sales Prediction",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    #MainMenu, footer { visibility: hidden; }

    .chip {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 4px 4px 0 0;
    }
    .chip-on  { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
    .chip-off { background: #f1f5f9; color: #94a3b8; border: 1px solid #e2e8f0; }

    .result-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
        border-radius: 16px;
        padding: 28px 32px;
        color: white;
        text-align: center;
    }
    .result-card .label { font-size: 0.85rem; opacity: 0.7; margin-bottom: 6px; }
    .result-card .value { font-size: 2.6rem; font-weight: 700; letter-spacing: -1px; }
</style>
""", unsafe_allow_html=True)

# ===================================================
# LOAD MODEL
# ===================================================

@st.cache_resource
def load_model():
    return joblib.load("model_xgb.pkl")

try:
    loaded = load_model()
    model = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.info("Pastikan file `model_xgb.pkl` ada di folder yang sama dengan `app.py`.")
    st.stop()

# ===================================================
# HEADER
# ===================================================

st.title("📊 Sales Prediction Dashboard")

st.markdown("""
Prediksi penjualan toko menggunakan **XGBoost Regression**

**Model Performance**

✅ R² = 0.95 &nbsp;&nbsp; ✅ MAPE = 9.35% &nbsp;&nbsp; ✅ RMSE = €904.24
""")

st.divider()

# ===================================================
# SIDEBAR — INPUT (semua manual)
# ===================================================

st.sidebar.header("⚙️ Input Data")

store = st.sidebar.number_input(
    "Store ID", min_value=1, max_value=1115, value=1
)

day_of_week = st.sidebar.selectbox(
    "Day Of Week", [1, 2, 3, 4, 5, 6, 7],
    format_func=lambda x: ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"][x-1]
)

promo = st.sidebar.selectbox(
    "Promo", [1, 0],
    format_func=lambda x: "Ya" if x else "Tidak"
)

state_hol = st.sidebar.selectbox(
    "State Holiday", ["0", "a", "b", "c"],
    format_func=lambda x: {
        "0": "Tidak Ada", "a": "Public Holiday",
        "b": "Easter", "c": "Christmas"
    }[x]
)

school_hol = st.sidebar.selectbox(
    "School Holiday", [0, 1],
    format_func=lambda x: "Ya" if x else "Tidak"
)

month = st.sidebar.selectbox(
    "Month", list(range(1, 13)),
    format_func=lambda x: ["Jan","Feb","Mar","Apr","Mei","Jun",
                            "Jul","Agu","Sep","Okt","Nov","Des"][x-1]
)

# Is Weekend: otomatis dari Day Of Week (read-only, ditampilkan saja)
is_weekend = 1 if day_of_week in [6, 7] else 0
st.sidebar.selectbox(
    "Is Weekend (auto dari Day Of Week)",
    [is_weekend],
    format_func=lambda x: "Ya" if x else "Tidak",
    disabled=True
)

is_payday_period = st.sidebar.selectbox(
    "Is Payday Period", [1, 0],
    format_func=lambda x: "Ya" if x else "Tidak"
)

store_avg_customer = st.sidebar.number_input(
    "Store Avg Customer", min_value=0.0, value=600.0, step=10.0
)

# ===================================================
# DERIVED (auto dari tanggal hari ini, one-hot StateHoliday)
# ===================================================

today = pd.Timestamp.now()
day_of_month = int(today.day)
week_of_year = int(today.isocalendar().week)

state_hol_a = 1 if state_hol == "a" else 0
state_hol_b = 1 if state_hol == "b" else 0
state_hol_c = 1 if state_hol == "c" else 0

nama_hari = ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"][day_of_week - 1]

# ===================================================
# DATAFRAME — urutan & tipe HARUS sama dengan X_train
# ===================================================

input_data = pd.DataFrame([{
    "Store": store,
    "DayOfWeek": day_of_week,
    "Promo": promo,
    "SchoolHoliday": school_hol,
    "month": month,
    "day_of_month": day_of_month,
    "week_of_year": week_of_year,
    "is_weekend": is_weekend,
    "is_payday_period": is_payday_period,
    "store_avg_customer": store_avg_customer,
    "StateHoliday_a": state_hol_a,
    "StateHoliday_b": state_hol_b,
    "StateHoliday_c": state_hol_c
}])

try:
    expected_cols = model.get_booster().feature_names
    if expected_cols is not None:
        input_data = input_data[expected_cols]
except Exception:
    pass

# ===================================================
# BUTTON — PREDIKSI
# ===================================================

if st.button("🚀 Prediksi Sales", use_container_width=True, type="primary"):

    try:
        pred = model.predict(input_data)[0]
    except Exception as e:
        st.error(f"❌ Gagal prediksi: {e}")
        st.info("Cek kembali urutan kolom & tipe data dibanding X_train di notebook.")
        st.stop()

    target = 20000.0
    persen = min(pred / target * 100, 100)

    col_left, col_right = st.columns([1, 1.3], gap="large")

    # ── Kiri: hasil prediksi + chip kondisi ──
    with col_left:
        st.markdown(f"""
        <div class="result-card">
            <div class="label">ESTIMASI SALES — {nama_hari.upper()}</div>
            <div class="value">€ {pred:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Kondisi Input**")

        chips = ""
        chips += f'<span class="chip {"chip-on" if promo else "chip-off"}">{"✅" if promo else "▫️"} Promo</span>'
        chips += f'<span class="chip {"chip-on" if is_weekend else "chip-off"}">{"✅" if is_weekend else "▫️"} Weekend</span>'
        chips += f'<span class="chip {"chip-on" if is_payday_period else "chip-off"}">{"✅" if is_payday_period else "▫️"} Payday Period</span>'
        chips += f'<span class="chip {"chip-on" if school_hol else "chip-off"}">{"✅" if school_hol else "▫️"} School Holiday</span>'
        st.markdown(chips, unsafe_allow_html=True)

    # ── Kanan: gauge chart sederhana ──
    with col_right:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            number={"prefix": "€ ", "valueformat": ",.0f"},
            title={"text": "Pencapaian Target"},
            gauge={
                "axis": {"range": [0, target * 1.2]},
                "bar": {"color": "#2563eb"},
            }
        ))
        fig.update_layout(height=240, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

        st.caption(f"Prediksi mencapai **{persen:.1f}%** dari target sales €{target:,.0f}.")

    st.divider()

    with st.expander("📋 Ringkasan Input"):
        st.write(f"""
        **Store ID:** {store}

        **Day Of Week:** {nama_hari} &nbsp;|&nbsp; **Day Of Month:** {day_of_month}

        **Month:** {month} &nbsp;|&nbsp; **Week Of Year:** {week_of_year}

        **Promo:** {"Ya" if promo else "Tidak"}

        **Is Weekend (auto):** {"Ya" if is_weekend else "Tidak"}

        **Is Payday Period:** {"Ya" if is_payday_period else "Tidak"}

        **School Holiday:** {"Ya" if school_hol else "Tidak"}

        **State Holiday:** {state_hol}

        **Store Avg Customer:** {store_avg_customer:,.0f}
        """)

        st.dataframe(input_data, use_container_width=True)

else:
    st.markdown("""
    <div style="border: 2px dashed #cbd5e1; border-radius: 16px; padding: 48px 32px;
                text-align: center; color: #94a3b8; margin-top: 8px;">
        <div style="font-size: 2.5rem; margin-bottom: 12px">📊</div>
        <div style="font-weight: 600; font-size: 1rem; margin-bottom: 6px">
            Hasil prediksi akan muncul di sini
        </div>
        <div style="font-size: 0.83rem">
            Lengkapi input di sidebar, lalu klik <strong>Prediksi Sales</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.caption("📊 Sales Prediction Dashboard | XGBoost Regression")