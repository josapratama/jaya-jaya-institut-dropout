"""
Jaya Jaya Institut - Student Dropout Prediction System
Prototype Machine Learning menggunakan Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Jaya Jaya Institut - Dropout Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .main-header h1 { font-size: 2.2rem; margin: 0; }
    .main-header p  { font-size: 1rem; opacity: 0.85; margin: 0.4rem 0 0; }

    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .label { font-size: 0.8rem; color: #666; text-transform: uppercase; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #0f3460; }

    .risk-high   { background:#ffeaea; border-left:5px solid #e74c3c; padding:1rem; border-radius:8px; }
    .risk-low    { background:#eafbea; border-left:5px solid #2ecc71; padding:1rem; border-radius:8px; }
    .risk-medium { background:#fff8ea; border-left:5px solid #f39c12; padding:1rem; border-radius:8px; }

    .section-title {
        font-size: 1.15rem; font-weight: 700;
        color: #0f3460; border-bottom: 2px solid #0f3460;
        padding-bottom: 0.3rem; margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1.2rem;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = Path("model/dropout_model.joblib")
    features_path = Path("model/features.json")
    if not model_path.exists():
        st.error("Model tidak ditemukan! Pastikan file model/dropout_model.joblib tersedia.")
        st.stop()
    model = joblib.load(model_path)
    with open(features_path) as f:
        features = json.load(f)
    return model, features

model, features = load_model()

# ─── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data.csv", delimiter=";")

df = load_data()

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎓 Jaya Jaya Institut</h1>
    <p>Sistem Prediksi Dropout Mahasiswa — Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/graduation-cap.png", width=80)
    st.markdown("## 📌 Navigasi")
    page = st.radio(
        "",
        ["🏠 Dashboard Overview", "🔍 Prediksi Dropout", "📊 Analisis Data", "ℹ️ Tentang Sistem"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Model:** Random Forest  \n**Akurasi:** ~86%  \n**Dataset:** 4,424 mahasiswa")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard Overview":
    st.markdown("### 📈 Ringkasan Performa Mahasiswa")

    # KPI Row
    total = len(df)
    n_dropout  = (df["Status"] == "Dropout").sum()
    n_graduate = (df["Status"] == "Graduate").sum()
    n_enrolled = (df["Status"] == "Enrolled").sum()
    dropout_rate = n_dropout / total * 100

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, color in [
        (c1, "Total Mahasiswa", f"{total:,}", "#0f3460"),
        (c2, "Dropout ⚠️",     f"{n_dropout:,} ({dropout_rate:.1f}%)", "#e74c3c"),
        (c3, "Graduate ✅",    f"{n_graduate:,}", "#2ecc71"),
        (c4, "Masih Enrolled 📚", f"{n_enrolled:,}", "#3498db"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value" style="color:{color}">{value}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row 1
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Distribusi Status Mahasiswa</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#e74c3c", "#2ecc71", "#3498db"]
        counts = df["Status"].value_counts()
        wedges, texts, autotexts = ax.pie(
            counts.values, labels=counts.index,
            autopct="%1.1f%%", colors=colors,
            startangle=90, textprops={"fontsize": 11}
        )
        for at in autotexts:
            at.set_fontweight("bold")
        ax.set_title("Status Mahasiswa", fontsize=13, fontweight="bold")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown('<div class="section-title">Dropout Rate per Program Studi (Top 10)</div>', unsafe_allow_html=True)
        course_map = {
            33: "Biofuel Production", 171: "Animation & Multimedia",
            8014: "Social Service (eve)", 9003: "Agronomy",
            9070: "Communication Design", 9085: "Veterinary Nursing",
            9119: "Informatics Eng.", 9130: "Equinculture",
            9147: "Management", 9238: "Social Service",
            9254: "Tourism", 9500: "Nursing",
            9556: "Oral Hygiene", 9670: "Adv. & Marketing",
            9773: "Journalism", 9853: "Basic Education",
            9991: "Management (eve)"
        }
        df_c = df.copy()
        df_c["Course_name"] = df_c["Course"].map(course_map).fillna("Other")
        dropout_by_course = (
            df_c.groupby("Course_name")
            .apply(lambda x: (x["Status"] == "Dropout").sum() / len(x) * 100)
            .sort_values(ascending=False)
            .head(10)
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        bar_colors = ["#e74c3c" if v > 35 else "#f39c12" if v > 25 else "#2ecc71"
                      for v in dropout_by_course.values]
        ax.barh(dropout_by_course.index[::-1], dropout_by_course.values[::-1],
                color=bar_colors[::-1], edgecolor="white")
        ax.axvline(dropout_by_course.mean(), color="navy", linestyle="--", lw=1.5,
                   label=f"Avg {dropout_by_course.mean():.1f}%")
        ax.set_xlabel("Dropout Rate (%)")
        ax.set_title("Dropout Rate per Prodi (Top 10)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Charts row 2
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-title">Pengaruh Status SPP terhadap Dropout</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        tuition_df = df.groupby(["Tuition_fees_up_to_date", "Status"]).size().unstack(fill_value=0)
        tuition_df.index = ["Tidak Lunas", "Lunas"]
        tuition_df.plot(kind="bar", ax=ax,
                        color={"Dropout": "#e74c3c", "Graduate": "#2ecc71", "Enrolled": "#3498db"},
                        edgecolor="white", width=0.6)
        ax.set_title("Status SPP vs Status Mahasiswa", fontsize=12, fontweight="bold")
        ax.set_xlabel("Status SPP"); ax.set_ylabel("Jumlah")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Status")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col4:
        st.markdown('<div class="section-title">Distribusi Usia saat Enrollment</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        for status, color in [("Dropout", "#e74c3c"), ("Graduate", "#2ecc71"), ("Enrolled", "#3498db")]:
            ax.hist(df[df["Status"] == status]["Age_at_enrollment"],
                    bins=20, alpha=0.55, label=status, color=color, edgecolor="white")
        ax.set_title("Distribusi Usia per Status", fontsize=12, fontweight="bold")
        ax.set_xlabel("Usia"); ax.set_ylabel("Frekuensi")
        ax.legend(title="Status")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: PREDIKSI DROPOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Prediksi Dropout":
    st.markdown("### 🔍 Prediksi Risiko Dropout Mahasiswa")
    st.info("Masukkan data mahasiswa di bawah ini untuk mendapatkan prediksi risiko dropout.")

    with st.form("prediction_form"):
        st.markdown('<div class="section-title">📋 Data Demografis</div>', unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        with d1:
            age = st.number_input("Usia saat Enrollment", min_value=17, max_value=70, value=20)
            gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki")
            marital = st.selectbox("Status Pernikahan", options=[1, 2, 3, 4, 5, 6],
                                   format_func=lambda x: {1:"Single",2:"Menikah",3:"Janda/Duda",
                                                           4:"Cerai",5:"Kohabitas",6:"Pisah Hukum"}[x])
        with d2:
            displaced = st.selectbox("Pindahan (Displaced)", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
            international = st.selectbox("Mahasiswa Internasional", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
            edu_special = st.selectbox("Kebutuhan Pendidikan Khusus", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
        with d3:
            debtor = st.selectbox("Status Debitur (Hutang)", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
            tuition = st.selectbox("SPP Lunas Tepat Waktu", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
            scholarship = st.selectbox("Penerima Beasiswa", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

        st.markdown('<div class="section-title">🎓 Data Akademik</div>', unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3)
        with a1:
            app_mode = st.selectbox("Mode Pendaftaran", [1,2,5,7,10,15,16,17,18,26,27,39,42,43,44,51,53,57],
                                    format_func=lambda x: {1:"1st phase-general",17:"2nd phase-general",
                                                            18:"3rd phase-general",39:"Over 23 yrs",
                                                            42:"Transfer",43:"Change of course"}.get(x, str(x)))
            app_order = st.slider("Urutan Pilihan (0=1st choice)", 0, 9, 1)
            prev_qual = st.selectbox("Kualifikasi Sebelumnya", [1,2,3,4,5,6,9,10,12,14,15,19,38,39,40,42,43],
                                     format_func=lambda x: {1:"SMA",2:"S1",3:"Degree",4:"S2",5:"S3",6:"Kuliah tdk selesai"}.get(x, str(x)))
        with a2:
            prev_grade = st.number_input("Nilai Kualifikasi Sebelumnya (0-200)", 0.0, 200.0, 130.0, step=0.5)
            admission_grade = st.number_input("Nilai Masuk (0-200)", 0.0, 200.0, 130.0, step=0.5)
            course = st.selectbox("Program Studi", [33,171,8014,9003,9070,9085,9119,9130,9147,9238,9254,9500,9556,9670,9773,9853,9991],
                                  format_func=lambda x: {33:"Biofuel",171:"Animation&Multimedia",9003:"Agronomy",
                                                          9119:"Informatics Eng.",9147:"Management",9500:"Nursing",
                                                          9853:"Basic Education"}.get(x, str(x)))
        with a3:
            attendance = st.selectbox("Waktu Kuliah", [1, 0], format_func=lambda x: "Siang" if x == 1 else "Malam")
            nationality = st.number_input("Kewarganegaraan (kode)", 1, 109, 1)
            mothers_qual = st.number_input("Pendidikan Ibu (kode 1-44)", 1, 44, 19)
        
        a4, a5 = st.columns(2)
        with a4:
            fathers_qual = st.number_input("Pendidikan Ayah (kode 1-44)", 1, 44, 19)
            mothers_occ  = st.number_input("Pekerjaan Ibu (kode)", 0, 194, 5)
        with a5:
            fathers_occ  = st.number_input("Pekerjaan Ayah (kode)", 0, 194, 9)

        st.markdown('<div class="section-title">📚 Performa Semester 1</div>', unsafe_allow_html=True)
        s1a, s1b, s1c = st.columns(3)
        with s1a:
            cu1_credited  = st.number_input("Sem 1 - Unit Dikreditkan", 0, 20, 0)
            cu1_enrolled  = st.number_input("Sem 1 - Unit Diambil", 0, 30, 6)
        with s1b:
            cu1_evals     = st.number_input("Sem 1 - Jumlah Evaluasi", 0, 45, 6)
            cu1_approved  = st.number_input("Sem 1 - Unit Disetujui", 0, 30, 5)
        with s1c:
            cu1_grade     = st.number_input("Sem 1 - Nilai Rata-rata (0-20)", 0.0, 20.0, 12.0, step=0.1)
            cu1_no_eval   = st.number_input("Sem 1 - Unit Tanpa Evaluasi", 0, 20, 0)

        st.markdown('<div class="section-title">📚 Performa Semester 2</div>', unsafe_allow_html=True)
        s2a, s2b, s2c = st.columns(3)
        with s2a:
            cu2_credited  = st.number_input("Sem 2 - Unit Dikreditkan", 0, 20, 0)
            cu2_enrolled  = st.number_input("Sem 2 - Unit Diambil", 0, 30, 6)
        with s2b:
            cu2_evals     = st.number_input("Sem 2 - Jumlah Evaluasi", 0, 45, 6)
            cu2_approved  = st.number_input("Sem 2 - Unit Disetujui", 0, 30, 5)
        with s2c:
            cu2_grade     = st.number_input("Sem 2 - Nilai Rata-rata (0-20)", 0.0, 20.0, 12.0, step=0.1)
            cu2_no_eval   = st.number_input("Sem 2 - Unit Tanpa Evaluasi", 0, 20, 0)

        st.markdown('<div class="section-title">🌍 Kondisi Ekonomi Makro</div>', unsafe_allow_html=True)
        e1, e2, e3 = st.columns(3)
        with e1:
            unemp_rate = st.number_input("Tingkat Pengangguran (%)", 0.0, 25.0, 11.0, step=0.1)
        with e2:
            inf_rate   = st.number_input("Tingkat Inflasi (%)", -5.0, 15.0, 1.0, step=0.1)
        with e3:
            gdp        = st.number_input("GDP", -5.0, 5.0, 1.0, step=0.01)

        submitted = st.form_submit_button("🔮 Prediksi Sekarang", use_container_width=True, type="primary")

    if submitted:
        # Hitung fitur turunan
        approval_rate_1 = cu1_approved / cu1_enrolled if cu1_enrolled > 0 else 0
        approval_rate_2 = cu2_approved / cu2_enrolled if cu2_enrolled > 0 else 0
        total_approved  = cu1_approved + cu2_approved
        avg_grade       = (cu1_grade + cu2_grade) / 2

        input_data = pd.DataFrame([[
            marital, app_mode, app_order, course, attendance, prev_qual,
            prev_grade, nationality, mothers_qual, fathers_qual,
            mothers_occ, fathers_occ, admission_grade,
            displaced, edu_special, debtor, tuition, gender, scholarship,
            age, international,
            cu1_credited, cu1_enrolled, cu1_evals, cu1_approved, cu1_grade, cu1_no_eval,
            cu2_credited, cu2_enrolled, cu2_evals, cu2_approved, cu2_grade, cu2_no_eval,
            unemp_rate, inf_rate, gdp,
            approval_rate_1, approval_rate_2, total_approved, avg_grade
        ]], columns=features)

        prediction  = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        dropout_prob = probability[1] * 100

        st.markdown("---")
        st.markdown("### 🎯 Hasil Prediksi")

        res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
        with res_col2:
            if dropout_prob >= 60:
                risk_class = "risk-high"
                risk_label = "⚠️ RISIKO TINGGI DROPOUT"
                recommendation = "Segera lakukan intervensi! Mahasiswa ini memerlukan bimbingan akademik intensif dan konseling."
            elif dropout_prob >= 35:
                risk_class = "risk-medium"
                risk_label = "🔔 RISIKO SEDANG DROPOUT"
                recommendation = "Pantau perkembangan mahasiswa ini secara berkala dan berikan dukungan tambahan."
            else:
                risk_class = "risk-low"
                risk_label = "✅ RISIKO RENDAH DROPOUT"
                recommendation = "Mahasiswa ini berada pada jalur yang baik. Tetap pantau dan berikan motivasi."

            st.markdown(f"""
            <div class="{risk_class}">
                <h3 style="margin:0;">{risk_label}</h3>
                <p style="font-size:1.8rem; font-weight:700; margin:0.5rem 0;">
                    Probabilitas Dropout: <span style="color:#e74c3c">{dropout_prob:.1f}%</span>
                </p>
                <p style="margin:0;">{recommendation}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge-like bar
        fig, ax = plt.subplots(figsize=(8, 1.2))
        bar_color = "#e74c3c" if dropout_prob >= 60 else "#f39c12" if dropout_prob >= 35 else "#2ecc71"
        ax.barh(["Risiko Dropout"], [dropout_prob], color=bar_color, height=0.5)
        ax.barh(["Risiko Dropout"], [100 - dropout_prob], left=[dropout_prob],
                color="#ecf0f1", height=0.5)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probabilitas (%)")
        ax.axvline(35, color="#f39c12", linestyle="--", lw=1.5, alpha=0.7)
        ax.axvline(60, color="#e74c3c", linestyle="--", lw=1.5, alpha=0.7)
        ax.text(dropout_prob / 2, 0, f"{dropout_prob:.1f}%",
                ha="center", va="center", fontweight="bold", fontsize=12, color="white")
        ax.set_title("Dropout Risk Gauge", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Faktor risiko
        st.markdown("#### 🔎 Faktor Risiko yang Teridentifikasi")
        risk_factors = []
        if cu1_approved == 0 and cu1_enrolled > 0:
            risk_factors.append("❌ Tidak lulus satupun mata kuliah di Semester 1")
        if cu2_approved == 0 and cu2_enrolled > 0:
            risk_factors.append("❌ Tidak lulus satupun mata kuliah di Semester 2")
        if cu1_grade < 10:
            risk_factors.append(f"📉 Nilai Semester 1 sangat rendah ({cu1_grade:.1f}/20)")
        if cu2_grade < 10:
            risk_factors.append(f"📉 Nilai Semester 2 sangat rendah ({cu2_grade:.1f}/20)")
        if tuition == 0:
            risk_factors.append("💸 SPP tidak terbayar tepat waktu")
        if debtor == 1:
            risk_factors.append("⚠️ Mahasiswa memiliki hutang kepada institusi")
        if scholarship == 0 and debtor == 1:
            risk_factors.append("💰 Tidak mendapat beasiswa dan memiliki hutang")
        if age > 25:
            risk_factors.append(f"👤 Usia enrollment di atas rata-rata ({age} tahun)")
        if approval_rate_1 < 0.5 and cu1_enrolled > 0:
            risk_factors.append(f"📊 Tingkat kelulusan Sem 1 rendah ({approval_rate_1*100:.0f}%)")

        if risk_factors:
            for rf in risk_factors:
                st.warning(rf)
        else:
            st.success("✅ Tidak teridentifikasi faktor risiko signifikan.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: ANALISIS DATA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analisis Data":
    st.markdown("### 📊 Analisis Mendalam Data Mahasiswa")

    tab1, tab2, tab3 = st.tabs(["📈 Visualisasi EDA", "🔢 Statistik", "🎯 Feature Importance"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-title">Korelasi Nilai Sem 1 vs Sem 2</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            for status, color in [("Dropout","#e74c3c"),("Graduate","#2ecc71"),("Enrolled","#3498db")]:
                sub = df[df["Status"] == status]
                ax.scatter(sub["Curricular_units_1st_sem_grade"],
                           sub["Curricular_units_2nd_sem_grade"],
                           c=color, label=status, alpha=0.3, s=15)
            ax.set_xlabel("Nilai Semester 1"); ax.set_ylabel("Nilai Semester 2")
            ax.set_title("Nilai Sem 1 vs Sem 2", fontsize=12, fontweight="bold")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col2:
            st.markdown('<div class="section-title">Unit Disetujui: Sem 1 vs Sem 2</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            avg_approved = df.groupby("Status")[["Curricular_units_1st_sem_approved",
                                                   "Curricular_units_2nd_sem_approved"]].mean()
            avg_approved.columns = ["Semester 1", "Semester 2"]
            avg_approved.plot(kind="bar", ax=ax,
                              color=["#3498db","#2ecc71"], edgecolor="white", width=0.6)
            ax.set_title("Rata-rata Unit Disetujui per Status", fontsize=12, fontweight="bold")
            ax.set_xlabel(""); ax.set_ylabel("Rata-rata Unit")
            ax.tick_params(axis="x", rotation=0)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<div class="section-title">Pengaruh Beasiswa terhadap Status</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            scholar = df.groupby(["Scholarship_holder","Status"]).size().unstack(fill_value=0)
            scholar.index = ["Non-Beasiswa","Beasiswa"]
            scholar_pct = scholar.div(scholar.sum(axis=1), axis=0) * 100
            scholar_pct.plot(kind="bar", ax=ax, stacked=True,
                             color={"Dropout":"#e74c3c","Graduate":"#2ecc71","Enrolled":"#3498db"},
                             edgecolor="white")
            ax.set_title("Proporsi Status berdasarkan Beasiswa (%)", fontsize=11, fontweight="bold")
            ax.set_xlabel(""); ax.set_ylabel("Persentase (%)")
            ax.tick_params(axis="x", rotation=0)
            ax.legend(title="Status")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col4:
            st.markdown('<div class="section-title">Distribusi Nilai Masuk</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            for status, color in [("Dropout","#e74c3c"),("Graduate","#2ecc71"),("Enrolled","#3498db")]:
                ax.hist(df[df["Status"]==status]["Admission_grade"],
                        bins=25, alpha=0.55, label=status, color=color, edgecolor="white")
            ax.set_title("Distribusi Nilai Masuk per Status", fontsize=12, fontweight="bold")
            ax.set_xlabel("Nilai Masuk"); ax.set_ylabel("Frekuensi")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    with tab2:
        st.markdown('<div class="section-title">Statistik Deskriptif per Status</div>', unsafe_allow_html=True)
        stat_cols = ["Age_at_enrollment","Admission_grade","Curricular_units_1st_sem_grade",
                     "Curricular_units_2nd_sem_grade","Curricular_units_1st_sem_approved",
                     "Curricular_units_2nd_sem_approved"]
        stat_df = df.groupby("Status")[stat_cols].mean().round(2)
        stat_df.columns = ["Usia","Nilai Masuk","Nilai Sem 1","Nilai Sem 2","Unit Sem 1","Unit Sem 2"]
        st.dataframe(stat_df.style.background_gradient(cmap="RdYlGn", axis=0), use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-title">Preview Data Mahasiswa</div>', unsafe_allow_html=True)
        filter_status = st.selectbox("Filter berdasarkan Status:", ["Semua"] + df["Status"].unique().tolist())
        shown = df if filter_status == "Semua" else df[df["Status"] == filter_status]
        st.dataframe(shown.head(50), use_container_width=True)
        st.caption(f"Menampilkan {min(50, len(shown))} dari {len(shown)} baris")

    with tab3:
        st.markdown('<div class="section-title">Feature Importance - Random Forest Model</div>', unsafe_allow_html=True)
        rf_clf = model.named_steps["clf"]
        importances = pd.Series(rf_clf.feature_importances_, index=features).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 10))
        colors_fi = ["#e74c3c" if i >= len(importances) - 5 else "#3498db"
                     for i in range(len(importances))]
        importances.plot(kind="barh", ax=ax, color=colors_fi, edgecolor="white")
        ax.set_title("Feature Importance (seluruh fitur)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: TENTANG SISTEM
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ Tentang Sistem":
    st.markdown("### ℹ️ Tentang Sistem Prediksi Dropout")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### 🎯 Tujuan Sistem
        Sistem ini dikembangkan untuk membantu **Jaya Jaya Institut** dalam:
        - Mendeteksi dini mahasiswa berisiko dropout
        - Memahami faktor-faktor penyebab dropout
        - Memonitor performa mahasiswa secara real-time
        - Memberikan dasar untuk pengambilan kebijakan berbasis data

        #### 🤖 Model Machine Learning
        | Komponen | Detail |
        |----------|--------|
        | Algoritma | Random Forest Classifier |
        | Jumlah Tree | 200 estimator |
        | Max Depth | 12 |
        | Fitur | 35 fitur |
        | Akurasi | ~86% |
        | F1 Score | ~0.83 |
        | ROC-AUC | ~0.92 |
        """)

    with col2:
        st.markdown("""
        #### 📊 Dataset
        | Keterangan | Detail |
        |------------|--------|
        | Sumber | Dicoding Academy / UCI ML Repository |
        | Jumlah Data | 4,424 mahasiswa |
        | Fitur | 36 kolom |
        | Target | Dropout / Graduate / Enrolled |
        | Missing Values | Tidak ada |

        #### 🚀 Cara Menjalankan Lokal
        ```bash
        # Install dependencies
        pip install -r requirements.txt

        # Jalankan Streamlit
        streamlit run app.py
        ```

        #### 🔗 Link
        - **Streamlit Cloud:** [Lihat App](https://share.streamlit.io)
        - **GitHub:** Repository Submission
        - **Dataset:** [Dicoding Dataset](https://github.com/dicodingacademy/dicoding_dataset)
        """)

    st.markdown("---")
    st.markdown("""
    #### ⚠️ Action Items untuk Jaya Jaya Institut

    Berdasarkan hasil analisis data, berikut rekomendasi yang dapat diimplementasikan:

    1. **🚨 Early Warning System** — Implementasikan model ini sebagai sistem peringatan dini. Setiap semester, jalankan prediksi untuk seluruh mahasiswa aktif dan identifikasi yang berisiko tinggi dropout.

    2. **📚 Program Bimbingan Intensif** — Mahasiswa dengan nilai semester 1 di bawah 10/20 atau yang gagal lebih dari 50% mata kuliah harus segera mendapatkan bimbingan akademik intensif dari dosen wali.

    3. **💰 Bantuan Finansial Proaktif** — Identifikasi mahasiswa dengan tunggakan SPP sebelum semester baru dimulai. Tawarkan skema cicilan fleksibel atau rekomendasikan program beasiswa. Data menunjukkan mahasiswa dengan SPP tidak lunas memiliki risiko dropout 3x lebih tinggi.

    4. **🎓 Evaluasi Program Studi Bermasalah** — Program studi dengan dropout rate > 35% perlu evaluasi menyeluruh: review kurikulum, beban akademik, dan kualitas pengajaran.

    5. **📈 Dashboard Monitoring Rutin** — Gunakan dashboard yang telah dibuat untuk rapat evaluasi bulanan. Pantau tren dropout per semester dan per program studi.

    6. **🤝 Konseling Mahasiswa Baru** — Mahasiswa yang mendaftar di usia > 25 tahun atau mahasiswa pindahan (displaced) perlu program orientasi dan konseling khusus untuk adaptasi.
    """)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; font-size:0.85rem;'>"
    "Jaya Jaya Institut — Student Dropout Prediction System | "
    "Dibuat untuk Submission Dicoding Data Science"
    "</p>",
    unsafe_allow_html=True,
)
