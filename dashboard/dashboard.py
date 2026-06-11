import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# KONFIGURASI PATH
# ============================================================
# Gunakan path relatif terhadap lokasi script ini
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, 'dataset',
                           'Filedata Indeks Standar Pencemaran Udara ISPU Tahun 2022.csv')
MODEL_DIR   = os.path.join(BASE_DIR, 'model')
MODEL_PATH  = os.path.join(MODEL_DIR, 'best_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
LE_PATH     = os.path.join(MODEL_DIR, 'label_encoder.pkl')
INFO_PATH   = os.path.join(MODEL_DIR, 'model_info.json')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title='Dashboard ISPU DKI Jakarta 2022',
    page_icon=None,
    layout='wide',
    initial_sidebar_state='expanded'
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*='css'] {
        font-family: 'Inter', sans-serif;
    }

    .main { background-color: #0f1117; }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #252d3d 100%);
        border: 1px solid #2d3561;
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .metric-card h3 { color: #8892b0; font-size: 0.8rem; font-weight: 500;
                       text-transform: uppercase; letter-spacing: 1px; margin: 0 0 0.4rem; }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #64ffda; margin: 0; }
    .metric-card .sub { font-size: 0.75rem; color: #8892b0; margin: 0.3rem 0 0; }

    /* Section Headers */
    .section-title {
        font-size: 1.1rem; font-weight: 600; color: #ccd6f6;
        border-left: 4px solid #64ffda; padding-left: 0.8rem;
        margin: 1.5rem 0 1rem;
    }

    /* Category Badges */
    .badge-baik     { background: #1a4731; color: #55d68a; border-radius: 8px; padding: 2px 10px; font-weight: 600; }
    .badge-sedang   { background: #4a3a00; color: #fbbf24; border-radius: 8px; padding: 2px 10px; font-weight: 600; }
    .badge-tidaksehat { background: #4a1515; color: #f87171; border-radius: 8px; padding: 2px 10px; font-weight: 600; }

    /* Prediction Box */
    .pred-box {
        background: linear-gradient(135deg, #1e2130, #2d3561);
        border: 2px solid #64ffda;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .pred-box .pred-label { font-size: 0.85rem; color: #8892b0; margin-bottom: 0.5rem; }
    .pred-box .pred-value { font-size: 3rem; font-weight: 700; margin: 0; }
    .pred-box .pred-conf  { font-size: 0.9rem; color: #8892b0; margin-top: 0.5rem; }

    /* Sidebar */
    .css-1d391kg { background: #0d1117; }
    section[data-testid='stSidebar'] { background: linear-gradient(180deg, #0d1117 0%, #161b27 100%); }

    /* Tabs */
    .stTabs [data-baseweb='tab-list'] { gap: 8px; }
    .stTabs [data-baseweb='tab-list'] > div[role='presentation'] { display: none; }
    .stTabs [data-baseweb='tab'] {
        background: #1e2130; border-radius: 10px;
        color: #8892b0; padding: 0.5rem 1.2rem;
        border: 1px solid #2d3561; font-weight: 500;
    }
    .stTabs [aria-selected='true'] {
        background: linear-gradient(135deg, #233554, #2d3561) !important;
        color: #64ffda !important;
        border-color: #64ffda !important;
    }
    /* Hilangkan garis merah/highlight di bawah tab aktif */
    .stTabs [data-baseweb='tab-highlight'] { display: none !important; }
    .stTabs [data-baseweb='tab-border']    { display: none !important; }

    /* Divider */
    hr { border-color: #2d3561; }

    /* Info box */
    .info-box {
        background: #131b2e; border: 1px solid #2d3561;
        border-radius: 12px; padding: 1rem 1.2rem; margin: 0.8rem 0;
    }
    .info-box p { color: #8892b0; margin: 0; font-size: 0.88rem; }

    /* Table */
    .stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA & MODEL
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Fix anomali tanggal Februari (tahun 2020 → 2022)
    mask_feb = df['periode_data'] == 202202
    df.loc[mask_feb, 'tanggal'] = (
        df.loc[mask_feb, 'tanggal']
        .astype(str)
        .str.replace('2020-02', '2022-02', regex=False)
    )

    # Fix Excel serial date
    def fix_serial(val):
        try:
            v = float(str(val))
            if v > 40000:
                from datetime import datetime, timedelta
                return (datetime(1899, 12, 30) + timedelta(days=v)).strftime('%Y-%m-%d')
            return str(val)
        except Exception:
            return str(val)

    df['tanggal'] = df['tanggal'].astype(str).apply(fix_serial)
    df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')

    # Fix missing values
    df['lokasi_spku'] = df['lokasi_spku'].replace('0', np.nan).replace(0, np.nan)
    df['critical']    = df['critical'].fillna(df['critical'].mode()[0])
    df['lokasi_spku'] = df['lokasi_spku'].fillna(df['lokasi_spku'].mode()[0])

    num_cols = ['pm_10', 'pm_duakomalima', 'so2', 'co', 'o3', 'no2', 'max']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

    # Feature engineering
    df['bulan']     = df['tanggal'].dt.month
    df['bulan_str'] = df['tanggal'].dt.strftime('%B').fillna('Unknown')
    df['bulan_num'] = df['periode_data'].astype(str).str[-2:].astype(int, errors='ignore')

    return df


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, None, None
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le     = joblib.load(LE_PATH)
    info   = json.load(open(INFO_PATH)) if os.path.exists(INFO_PATH) else {}
    return model, scaler, le, info


# ============================================================
# COLOR PALETTE
# ============================================================
CAT_COLORS = {
    'BAIK'        : '#55d68a',
    'SEDANG'      : '#fbbf24',
    'TIDAK SEHAT' : '#f87171'
}
CAT_BG = {
    'BAIK'        : '#1a4731',
    'SEDANG'      : '#4a3a00',
    'TIDAK SEHAT' : '#4a1515'
}
POL_COLORS = {
    'pm_10'          : '#e74c3c',
    'pm_duakomalima' : '#e67e22',
    'so2'            : '#3498db',
    'co'             : '#9b59b6',
    'o3'             : '#2ecc71',
    'no2'            : '#1abc9c'
}
POL_LABELS = {
    'pm_10'          : 'PM10',
    'pm_duakomalima' : 'PM2.5',
    'so2'            : 'SO2',
    'co'             : 'CO',
    'o3'             : 'O3',
    'no2'            : 'NO2'
}
BULAN_LABELS = {
    1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'Mei', 6:'Jun',
    7:'Jul', 8:'Agu', 9:'Sep', 10:'Okt', 11:'Nov', 12:'Des'
}

# ============================================================
# LOAD
# ============================================================
df_full          = load_data()
model, scaler, le, model_info = load_model()
model_loaded     = model is not None

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:1.2rem; font-weight:700; color:#64ffda;'>ISPU Dashboard</div>
        <div style='font-size:0.75rem; color:#8892b0;'>DKI Jakarta 2022</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("**Filter Periode**")
    bulan_opts = sorted(df_full['bulan'].dropna().unique().tolist())
    bulan_sel  = st.multiselect(
        'Pilih Bulan',
        options=bulan_opts,
        default=bulan_opts,
        format_func=lambda x: BULAN_LABELS.get(int(x), str(x))
    )

    st.markdown("**Filter Lokasi SPKU**")
    lokasi_opts = sorted(df_full['lokasi_spku'].dropna().unique().tolist())
    lokasi_sel  = st.multiselect('Pilih Lokasi', options=lokasi_opts, default=lokasi_opts)

    st.divider()

    # Model Info
    if model_loaded and model_info:
        st.markdown("**Model Aktif**")
        st.markdown(f"""
        <div class='info-box'>
            <p><b style='color:#64ffda;'>{model_info.get('best_model_name','—')}</b></p>
            <p>Accuracy: <b style='color:#64ffda;'>{model_info.get('accuracy',0)*100:.1f}%</b></p>
            <p>F1-Score: <b style='color:#64ffda;'>{model_info.get('f1_macro',0):.4f}</b></p>
            <p style='font-size:0.75rem;'>SMOTE: Aktif | Classes: BAIK / SEDANG / TIDAK SEHAT</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning('Model belum dilatih. Jalankan notebook.ipynb terlebih dahulu.')

    st.divider()
    st.markdown("<p style='color:#8892b0; font-size:0.75rem; text-align:center;'>Data Mining ISPU 2022<br>Kerangka: CRISP-DM</p>", unsafe_allow_html=True)

# ============================================================
# APPLY FILTER
# ============================================================
df = df_full[
    (df_full['bulan'].isin(bulan_sel)) &
    (df_full['lokasi_spku'].isin(lokasi_sel))
].copy()

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<h1 style='color:#ccd6f6; font-weight:700; margin-bottom:0.2rem;'>
    Dashboard Analisis ISPU DKI Jakarta 2022
</h1>
<p style='color:#8892b0; margin-top:0; font-size:0.9rem;'>
    Indeks Standar Pencemaran Udara — CRISP-DM Analysis Dashboard
</p>
""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    'Overview',
    'Tren Temporal',
    'Prediksi Real-time',
    'Analisis Lanjutan'
])

# ==============================================================
# TAB 1: OVERVIEW
# ==============================================================
with tab1:
    # KPI Metrics
    cat_counts = df['categori'].value_counts()
    n_total    = len(df)
    n_baik     = int(cat_counts.get('BAIK', 0))
    n_sedang   = int(cat_counts.get('SEDANG', 0))
    n_tidak    = int(cat_counts.get('TIDAK SEHAT', 0))

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Total Data</h3>
            <p class='value'>{n_total}</p>
            <p class='sub'>hari pemantauan</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>🟢 BAIK</h3>
            <p class='value' style='color:#55d68a;'>{n_baik}</p>
            <p class='sub'>{n_baik/n_total*100:.1f}% dari total</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>🟡 SEDANG</h3>
            <p class='value' style='color:#fbbf24;'>{n_sedang}</p>
            <p class='sub'>{n_sedang/n_total*100:.1f}% dari total</p>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>🔴 TIDAK SEHAT</h3>
            <p class='value' style='color:#f87171;'>{n_tidak}</p>
            <p class='sub'>{n_tidak/n_total*100:.1f}% dari total</p>
        </div>""", unsafe_allow_html=True)
    with c5:
        pm25_mean = df['pm_duakomalima'].mean()
        pm25_color = '#f87171' if pm25_mean > 100 else ('#fbbf24' if pm25_mean > 50 else '#55d68a')
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Rata-rata PM2.5</h3>
            <p class='value' style='color:{pm25_color};'>{pm25_mean:.0f}</p>
            <p class='sub'>nilai ISPU</p>
        </div>""", unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # Distribusi kategori + polutan dominan
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("<div class='section-title'>Distribusi Kategori Udara</div>", unsafe_allow_html=True)
        fig_pie = go.Figure(go.Pie(
            labels=cat_counts.index.tolist(),
            values=cat_counts.values.tolist(),
            marker_colors=[CAT_COLORS.get(c, '#888') for c in cat_counts.index],
            hole=0.45,
            textinfo='label+percent',
            textfont=dict(size=13, family='Inter'),
            pull=[0.05] * len(cat_counts)
        ))
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#ccd6f6',
            showlegend=True,
            legend=dict(font=dict(size=12), bgcolor='rgba(0,0,0,0)'),
            margin=dict(t=30, b=10, l=10, r=10),
            height=320
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.markdown("<div class='section-title'>Rata-rata Nilai Polutan</div>", unsafe_allow_html=True)
        pol_means = {POL_LABELS[k]: df[k].mean() for k in POL_LABELS}
        fig_bar = go.Figure(go.Bar(
            x=list(pol_means.keys()),
            y=list(pol_means.values()),
            marker_color=[POL_COLORS[k] for k in POL_LABELS],
            text=[f'{v:.1f}' for v in pol_means.values()],
            textposition='outside',
            textfont=dict(size=12, color='#ccd6f6')
        ))
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#ccd6f6',
            xaxis=dict(gridcolor='#2d3561', title='Polutan'),
            yaxis=dict(gridcolor='#2d3561', title='Nilai ISPU'),
            margin=dict(t=30, b=10, l=10, r=10),
            height=320
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Polutan kritis terbanyak
    st.markdown("<div class='section-title'>Polutan Kritis Dominan per Bulan</div>", unsafe_allow_html=True)
    critical_monthly = (
        df.groupby(['bulan', 'critical'])
        .size()
        .reset_index(name='count')
    )
    critical_monthly['bulan_str'] = critical_monthly['bulan'].map(BULAN_LABELS)
    fig_critical = px.bar(
        critical_monthly,
        x='bulan_str', y='count', color='critical',
        barmode='stack',
        color_discrete_map={
            'PM2,5' : '#e67e22',
            'O3'    : '#2ecc71',
            'SO2'   : '#3498db',
            'CO'    : '#9b59b6',
            'NO2'   : '#1abc9c',
            'PM10'  : '#e74c3c'
        },
        labels={'bulan_str': 'Bulan', 'count': 'Jumlah Hari', 'critical': 'Polutan Kritis'}
    )
    fig_critical.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#ccd6f6',
        xaxis=dict(gridcolor='#2d3561'),
        yaxis=dict(gridcolor='#2d3561'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        margin=dict(t=30, b=10, l=10, r=10),
        height=350
    )
    st.plotly_chart(fig_critical, use_container_width=True)

    # Data tabel ringkasan
    st.markdown("<div class='section-title'>Ringkasan Statistik per Lokasi SPKU</div>", unsafe_allow_html=True)
    summary_lokasi = df.groupby('lokasi_spku').agg(
        Jumlah_Data=('tanggal', 'count'),
        PM10_mean=('pm_10', 'mean'),
        PM25_mean=('pm_duakomalima', 'mean'),
        Tidak_Sehat_days=('categori', lambda x: (x == 'TIDAK SEHAT').sum())
    ).round(1).reset_index()
    summary_lokasi.columns = ['Lokasi', 'Jumlah Data', 'PM10 Rata-rata', 'PM2.5 Rata-rata', 'Hari Tidak Sehat']
    st.dataframe(
        summary_lokasi,
        use_container_width=True,
        hide_index=True
    )

# ==============================================================
# TAB 2: TREN TEMPORAL
# ==============================================================
with tab2:
    st.markdown("<div class='section-title'>Tren Nilai Polutan per Bulan</div>", unsafe_allow_html=True)

    pol_sel = st.multiselect(
        'Pilih Polutan',
        options=list(POL_LABELS.keys()),
        default=['pm_duakomalima', 'o3', 'no2'],
        format_func=lambda x: POL_LABELS.get(x, x),
        key='trend_pol'
    )

    if pol_sel:
        monthly_avg = (
            df.groupby('bulan')[pol_sel]
            .mean()
            .reset_index()
        )
        monthly_avg['bulan_str'] = monthly_avg['bulan'].map(BULAN_LABELS)

        fig_trend = go.Figure()
        for pol in pol_sel:
            fig_trend.add_trace(go.Scatter(
                x=monthly_avg['bulan_str'],
                y=monthly_avg[pol],
                mode='lines+markers',
                name=POL_LABELS[pol],
                line=dict(color=POL_COLORS[pol], width=3),
                marker=dict(size=9, symbol='circle',
                            line=dict(width=2, color='white')),
                fill='tozeroy',
                fillcolor=POL_COLORS[pol].replace(')', ', 0.07)').replace('rgb', 'rgba')
            ))

        fig_trend.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#ccd6f6',
            xaxis=dict(gridcolor='#2d3561', title='Bulan'),
            yaxis=dict(gridcolor='#2d3561', title='Rata-rata Nilai ISPU'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=13)),
            margin=dict(t=20, b=20, l=10, r=10),
            height=420,
            hovermode='x unified'
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # Distribusi kategori per bulan
    st.markdown("<div class='section-title'>Distribusi Kategori Udara per Bulan</div>", unsafe_allow_html=True)
    cat_monthly = (
        df.groupby(['bulan', 'categori'])
        .size()
        .reset_index(name='count')
    )
    cat_monthly['bulan_str'] = cat_monthly['bulan'].map(BULAN_LABELS)

    fig_cat_monthly = px.bar(
        cat_monthly,
        x='bulan_str', y='count', color='categori',
        barmode='stack',
        color_discrete_map=CAT_COLORS,
        labels={'bulan_str': 'Bulan', 'count': 'Jumlah Hari', 'categori': 'Kategori'}
    )
    fig_cat_monthly.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#ccd6f6',
        xaxis=dict(gridcolor='#2d3561'),
        yaxis=dict(gridcolor='#2d3561'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        margin=dict(t=20, b=20, l=10, r=10),
        height=350
    )
    st.plotly_chart(fig_cat_monthly, use_container_width=True)

    # Heatmap bulanan per polutan
    st.markdown("<div class='section-title'>Heatmap Rata-rata Polutan per Bulan</div>", unsafe_allow_html=True)
    heat_data = df.groupby('bulan')[list(POL_LABELS.keys())].mean().round(1)
    heat_data.index = [BULAN_LABELS.get(i, i) for i in heat_data.index]
    heat_data.columns = [POL_LABELS[c] for c in heat_data.columns]

    fig_heat = go.Figure(go.Heatmap(
        z=heat_data.values.T,
        x=heat_data.index.tolist(),
        y=heat_data.columns.tolist(),
        colorscale='RdYlGn_r',
        text=heat_data.values.T,
        texttemplate='%{text:.0f}',
        textfont=dict(size=12, color='white'),
        hovertemplate='Bulan: %{x}<br>Polutan: %{y}<br>Nilai: %{z:.1f}<extra></extra>'
    ))
    fig_heat.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#ccd6f6',
        xaxis=dict(title='Bulan'),
        yaxis=dict(title='Polutan'),
        margin=dict(t=20, b=20, l=10, r=10),
        height=320
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ==============================================================
# TAB 3: PREDIKSI REAL-TIME
# ==============================================================
with tab3:
    st.markdown("""
    <div class='info-box'>
        <p><b style='color:#64ffda;'>Prediksi Kategori Udara Real-time</b><br>
        Geser nilai 6 parameter polutan di bawah ini — prediksi akan diperbarui secara otomatis.</p>
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.error('Model belum tersedia. Silakan jalankan notebook.ipynb terlebih dahulu untuk melatih model.')
    else:
        col_in, col_out = st.columns([1.2, 1])

        with col_in:
            st.markdown("<div class='section-title'>Input Nilai Polutan</div>", unsafe_allow_html=True)

            # Tampilkan statistik referensi
            with st.expander('Referensi Nilai Dataset (min / rata-rata / max)'):
                ref_data = []
                for pol, label in POL_LABELS.items():
                    ref_data.append({
                        'Polutan'  : label,
                        'Min'      : f"{df_full[pol].min():.0f}",
                        'Rata-rata': f"{df_full[pol].mean():.0f}",
                        'Max'      : f"{df_full[pol].max():.0f}"
                    })
                st.dataframe(pd.DataFrame(ref_data), hide_index=True, use_container_width=True)

            # Input sliders
            input_vals = {}
            slider_cfg = {
                'pm_10'          : ('PM10 (Partikel ≤10µm)',   0,  300, int(df_full['pm_10'].mean())),
                'pm_duakomalima' : ('PM2.5 (Partikel ≤2.5µm)', 0,  300, int(df_full['pm_duakomalima'].mean())),
                'so2'            : ('SO2 (Sulfur Dioksida)',    0,  150, int(df_full['so2'].mean())),
                'co'             : ('CO (Karbon Monoksida)',    0,  100, int(df_full['co'].mean())),
                'o3'             : ('O3 (Ozon)',                0,  250, int(df_full['o3'].mean())),
                'no2'            : ('NO2 (Nitrogen Dioksida)',  0,  150, int(df_full['no2'].mean()))
            }

            for key, (label, vmin, vmax, vdef) in slider_cfg.items():
                input_vals[key] = st.slider(
                    label, min_value=vmin, max_value=vmax,
                    value=vdef, step=1, key=f'slider_{key}'
                )

        with col_out:
            st.markdown("<div class='section-title'>Hasil Prediksi</div>", unsafe_allow_html=True)

            if True:  # Selalu tampilkan prediksi (reaktif terhadap slider)
                # Buat input array
                feat_order = ['pm_10', 'pm_duakomalima', 'so2', 'co', 'o3', 'no2']
                X_input    = np.array([[input_vals[k] for k in feat_order]])
                X_scaled   = scaler.transform(X_input)

                pred_code  = model.predict(X_scaled)[0]
                pred_label = le.inverse_transform([pred_code])[0]
                pred_proba = model.predict_proba(X_scaled)[0]

                # Warna prediksi
                pred_color  = CAT_COLORS.get(pred_label, '#64ffda')
                pred_marker = {'BAIK': '[BAIK]', 'SEDANG': '[SEDANG]', 'TIDAK SEHAT': '[TIDAK SEHAT]'}.get(pred_label, '')

                st.markdown(f"""
                <div class='pred-box'>
                    <p class='pred-label'>Hasil Prediksi Kualitas Udara</p>
                    <p class='pred-value' style='color:{pred_color};'>{pred_label}</p>
                    <p class='pred-conf'>Berdasarkan model {model_info.get('best_model_name', 'ML')}</p>
                </div>
                """, unsafe_allow_html=True)

                # Confidence chart
                st.markdown("**Probabilitas per Kategori:**")
                classes = le.classes_
                prob_fig = go.Figure(go.Bar(
                    x=list(classes),
                    y=pred_proba * 100,
                    marker_color=[CAT_COLORS.get(c, '#888') for c in classes],
                    text=[f'{p*100:.1f}%' for p in pred_proba],
                    textposition='outside',
                    textfont=dict(size=13, color='#ccd6f6')
                ))
                prob_fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#ccd6f6',
                    xaxis=dict(gridcolor='#2d3561'),
                    yaxis=dict(gridcolor='#2d3561', title='Probabilitas (%)', range=[0, 115]),
                    margin=dict(t=20, b=10, l=10, r=10),
                    height=280
                )
                st.plotly_chart(prob_fig, use_container_width=True)



# ==============================================================
# TAB 4: ANALISIS LANJUTAN
# ==============================================================
with tab4:
    col_a, col_b = st.columns(2)

    with col_a:
        # Heatmap korelasi
        st.markdown("<div class='section-title'>Matriks Korelasi Polutan</div>", unsafe_allow_html=True)
        pol_cols  = list(POL_LABELS.keys()) + ['max']
        corr_mat  = df[pol_cols].corr().round(2)
        pol_ticks = [POL_LABELS.get(c, c.upper()) for c in pol_cols]

        fig_corr = go.Figure(go.Heatmap(
            z=corr_mat.values,
            x=pol_ticks, y=pol_ticks,
            colorscale='RdYlGn',
            zmin=-1, zmax=1,
            text=corr_mat.values,
            texttemplate='%{text:.2f}',
            textfont=dict(size=11),
            hovertemplate='%{x} ↔ %{y}<br>Korelasi: %{z:.2f}<extra></extra>'
        ))
        fig_corr.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#ccd6f6',
            margin=dict(t=20, b=10, l=10, r=10),
            height=380
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with col_b:
        # Feature importance (jika model adalah RF atau DT)
        st.markdown("<div class='section-title'>Feature Importance Model</div>", unsafe_allow_html=True)
        if model_loaded and hasattr(model, 'feature_importances_'):
            features_used = model_info.get('features', list(POL_LABELS.keys()))
            fi_vals  = model.feature_importances_
            fi_df    = pd.DataFrame({
                'Polutan'    : [POL_LABELS.get(f, f) for f in features_used],
                'Importance' : fi_vals
            }).sort_values('Importance', ascending=True)

            fig_fi = go.Figure(go.Bar(
                x=fi_df['Importance'],
                y=fi_df['Polutan'],
                orientation='h',
                marker_color=[
                    '#e74c3c' if v == fi_df['Importance'].max() else '#3498db'
                    for v in fi_df['Importance']
                ],
                text=[f'{v:.3f}' for v in fi_df['Importance']],
                textposition='outside',
                textfont=dict(size=12, color='#ccd6f6')
            ))
            fig_fi.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#ccd6f6',
                xaxis=dict(gridcolor='#2d3561', title='Importance Score'),
                yaxis=dict(gridcolor='#2d3561'),
                margin=dict(t=20, b=10, l=10, r=10),
                height=380
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info('Feature importance hanya tersedia untuk tree-based models (Random Forest, Decision Tree).')



# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown("""
<div style='text-align:center; color:#8892b0; font-size:0.78rem; padding: 0.5rem 0;'>
    Dashboard ISPU DKI Jakarta 2022 | Analisis Data Mining CRISP-DM |
    Model: Decision Tree · Random Forest · KNN | Augmentasi: SMOTE
</div>
""", unsafe_allow_html=True)
