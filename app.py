"""
Jaya Jaya Institut - Student Dropout Prediction System
Prototype Machine Learning menggunakan Streamlit

Model dilatih pada data Dropout vs Graduate:
  - 1 = Dropout
  - 0 = Graduate
Data Enrolled disimpan terpisah untuk prediksi masa depan.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Jaya Jaya Institut - Dropout Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;
    color: white; text-align: center;
}
.main-header h1 { font-size: 2.2rem; margin: 0; }
.main-header p  { font-size: 1rem; opacity: 0.85; margin: 0.4rem 0 0; }
.metric-card {
    background: #f8f9fa; border: 1px solid #e0e0e0;
    border-radius: 10px; padding: 1rem 1.2rem; text-align: center;
}
.metric-card .label { font-size: 0.8rem; color: #666; text-transform: uppercase; }
.metric-card .value { font-size: 1.8rem; font-weight: 700; color: #0f3460; }
.risk-high   { background:#ffeaea; border-left:5px solid #e74c3c; padding:1rem; border-radius:8px; }
.risk-low    { background:#eafbea; border-left:5px solid #2ecc71; padding:1rem; border-radius:8px; }
.risk-medium { background:#fff8ea; border-left:5px solid #f39c12; padding:1rem; border-radius:8px; }
.info-box    { background:#eaf4ff; border-left:5px solid #3498db; padding:1rem; border-radius:8px; margin-bottom:1rem; }
.section-title {
    font-size: 1.1rem; font-weight: 700; color: #0f3460;
    border-bottom: 2px solid #0f3460; padding-bottom: 0.3rem; margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path    = Path("model/dropout_model.joblib")
    features_path = Path("model/features.json")
    if not model_path.exists():
        st.error("Model tidak ditemukan! Pastikan file model/dropout_model.joblib tersedia.")
        st.stop()
    model = joblib.load(model_path)
    with open(features_path) as f:
        features = json.load(f)
    return model, features

model, features = load_model()

# ── Load Data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv", delimiter=";")
    # Data untuk EDA (semua status)
    df_all = df.copy()
    # Data untuk analisis model (Dropout + Graduate only)
    df_dg = df[df["Status"].isin(["Dropout", "Graduate"])].copy()
    return df_all, df_dg

df_all, df_dg = load_data()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎓 Jaya Jaya Institut</h1>
    <p>Sistem Prediksi Dropout Mahasiswa — Powered by Random Forest (Accuracy: 92.56%)</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📌 Navigasi")
    page = st.radio(
        "",
        ["🏠 Dashboard Overview", "🔍 Prediksi Dropout", "📊 Analisis Data", "ℹ️ Tentang Sistem"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    **Model Info:**
    - Algoritma: Random Forest
    - Data training: Dropout + Graduate
    - Target: 1=Dropout, 0=Graduate
    - Accuracy: **92.56%**
    - F1 Score: **0.9043**
    - ROC-AUC: **0.9716**
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard Overview":
    st.markdown("### 📈 Ringkasan Performa Mahasiswa")

    st.markdown("""
    <div class="info-box">
    ℹ️ <b>Catatan Model:</b> Model ML dilatih hanya pada mahasiswa berstatus
    <b>Dropout</b> dan <b>Graduate</b>. Mahasiswa <b>Enrolled</b> belum diikutsertakan
    karena outcome mereka belum diketahui — mereka masih aktif kuliah.
    </div>
    """, unsafe_allow_html=True)

    total      = len(df_all)
    n_dropout  = (df_all["Status"] == "Dropout").sum()
    n_graduate = (df_all["Status"] == "Graduate").sum()
    n_enrolled = (df_all["Status"] == "Enrolled").sum()

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, color in [
        (c1, "Total Mahasiswa",        f"{total:,}",                          "#0f3460"),
        (c2, "Dropout ⚠️",            f"{n_dropout:,} ({n_dropout/total*100:.1f}%)", "#e74c3c"),
        (c3, "Graduate ✅",            f"{n_graduate:,}",                     "#2ecc71"),
        (c4, "Enrolled (aktif) 📚",   f"{n_enrolled:,}",                     "#3498db"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value" style="color:{color}">{value}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Distribusi Status Mahasiswa (Semua Data)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df_all["Status"].value_counts()
        colors = ["#e74c3c", "#2ecc71", "#3498db"]
        wedges, texts, autotexts = ax.pie(
            counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=colors, startangle=90, textprops={"fontsize": 11})
        for at in autotexts: at.set_fontweight("bold")
        ax.set_title("Status Mahasiswa (n=4,424)", fontsize=12, fontweight="bold")
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        st.markdown('<div class="section-title">Dropout vs Graduate (Data Training Model)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        dg_counts = df_dg["Status"].value_counts()
        ax.bar(dg_counts.index, dg_counts.values,
               color=["#e74c3c", "#2ecc71"], edgecolor="white", linewidth=1.5)
        ax.set_title("Dropout vs Graduate (n=3,630)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Jumlah Mahasiswa")
        for i, v in enumerate(dg_counts.values):
            ax.text(i, v + 20, str(v), ha="center", fontweight="bold", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-title">Pengaruh Status SPP terhadap Dropout</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        t = df_all.groupby(["Tuition_fees_up_to_date","Status"]).size().unstack(fill_value=0)
        t.index = ["Tidak Lunas", "Lunas"]
        colors_s = {"Dropout": "#e74c3c", "Graduate": "#2ecc71", "Enrolled": "#3498db"}
        t.plot(kind="bar", ax=ax,
               color=[colors_s.get(c, "gray") for c in t.columns],
               edgecolor="white", width=0.6)
        ax.set_title("Status SPP vs Status Mahasiswa", fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=0); ax.legend(title="Status")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with col4:
        st.markdown('<div class="section-title">Dropout Rate per Program Studi (Top 8)</div>', unsafe_allow_html=True)
        course_map = {
            33:"Biofuel", 171:"Animation", 8014:"Social Svc(eve)", 9003:"Agronomy",
            9070:"Comm Design", 9085:"Vet Nursing", 9119:"Informatics Eng",
            9130:"Equinculture", 9147:"Management", 9238:"Social Service",
            9254:"Tourism", 9500:"Nursing", 9556:"Oral Hygiene",
            9670:"Advertising", 9773:"Journalism", 9853:"Basic Edu", 9991:"Mgmt(eve)"
        }
        df_c = df_all.copy()
        df_c["Course_name"] = df_c["Course"].map(course_map).fillna("Other")
        dbc = (df_c.groupby("Course_name")
               .apply(lambda x: (x["Status"]=="Dropout").sum()/len(x)*100)
               .sort_values(ascending=False).head(8))
        fig, ax = plt.subplots(figsize=(6, 4))
        bar_colors = ["#e74c3c" if v>35 else "#f39c12" if v>25 else "#2ecc71" for v in dbc.values]
        ax.barh(dbc.index[::-1], dbc.values[::-1], color=bar_colors[::-1], edgecolor="white")
        ax.axvline(dbc.mean(), color="navy", linestyle="--", lw=1.5,
                   label=f"Avg {dbc.mean():.1f}%")
        ax.set_xlabel("Dropout Rate (%)"); ax.legend(fontsize=9)
        ax.set_title("Top 8 Dropout Rate per Prodi", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: PREDIKSI DROPOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Prediksi Dropout":
    st.markdown("### 🔍 Prediksi Risiko Dropout Mahasiswa")

    st.markdown("""
    <div class="info-box">
    ℹ️ Sistem ini memprediksi apakah mahasiswa akan <b>Dropout</b> atau <b>Graduate</b>
    berdasarkan data akademik dan demografis. Masukkan data mahasiswa di bawah ini.
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown('<div class="section-title">📋 Data Demografis & Finansial</div>', unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        with d1:
            age      = st.number_input("Usia saat Enrollment", 17, 70, 20)
            gender   = st.selectbox("Gender", [0,1], format_func=lambda x: "Perempuan" if x==0 else "Laki-laki")
            marital  = st.selectbox("Status Pernikahan", [1,2,3,4,5,6],
                                    format_func=lambda x: {1:"Single",2:"Menikah",3:"Janda/Duda",
                                                            4:"Cerai",5:"Kohabitas",6:"Pisah Hukum"}[x])
        with d2:
            displaced  = st.selectbox("Displaced (Pindahan)", [0,1], format_func=lambda x: "Tidak" if x==0 else "Ya")
            intl       = st.selectbox("Mahasiswa Internasional", [0,1], format_func=lambda x: "Tidak" if x==0 else "Ya")
            edu_spec   = st.selectbox("Kebutuhan Pendidikan Khusus", [0,1], format_func=lambda x: "Tidak" if x==0 else "Ya")
        with d3:
            debtor     = st.selectbox("Status Debitur (Hutang)", [0,1], format_func=lambda x: "Tidak" if x==0 else "Ya")
            tuition    = st.selectbox("SPP Lunas Tepat Waktu", [0,1], format_func=lambda x: "Tidak" if x==0 else "Ya")
            scholarship= st.selectbox("Penerima Beasiswa", [0,1], format_func=lambda x: "Tidak" if x==0 else "Ya")

        st.markdown('<div class="section-title">🎓 Data Akademik Latar Belakang</div>', unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3)
        with a1:
            app_mode   = st.selectbox("Mode Pendaftaran", [1,17,18,39,42,43,51],
                                      format_func=lambda x: {1:"1st phase-general",17:"2nd phase",
                                                              18:"3rd phase",39:"Over 23 yrs",
                                                              42:"Transfer",43:"Change of course",51:"Change institution"}.get(x,str(x)))
            app_order  = st.slider("Urutan Pilihan (0=1st choice)", 0, 9, 1)
        with a2:
            prev_qual  = st.selectbox("Kualifikasi Sebelumnya", [1,2,3,4,5,6,9,10],
                                      format_func=lambda x: {1:"SMA",2:"S1",3:"Degree",4:"S2",
                                                              5:"S3",6:"Kuliah tdk selesai",9:"12th not completed",10:"11th not completed"}.get(x,str(x)))
            prev_grade = st.number_input("Nilai Kualifikasi Sebelumnya (0-200)", 0.0, 200.0, 130.0, step=0.5)
        with a3:
            admission  = st.number_input("Nilai Masuk (0-200)", 0.0, 200.0, 130.0, step=0.5)
            course     = st.selectbox("Program Studi",
                                      [171,9003,9085,9119,9147,9238,9254,9500,9556,9670,9773,9853,9991],
                                      format_func=lambda x: {171:"Animation",9003:"Agronomy",
                                                              9085:"Vet Nursing",9119:"Informatics Eng",
                                                              9147:"Management",9500:"Nursing",
                                                              9853:"Basic Education"}.get(x,str(x)))

        a4, a5 = st.columns(2)
        with a4:
            attendance = st.selectbox("Waktu Kuliah", [1,0], format_func=lambda x: "Siang" if x==1 else "Malam")
            nationality= st.number_input("Kewarganegaraan (kode)", 1, 109, 1)
        with a5:
            mothers_q  = st.number_input("Pendidikan Ibu (kode)", 1, 44, 19)
            fathers_q  = st.number_input("Pendidikan Ayah (kode)", 1, 44, 19)

        m_occ = st.number_input("Pekerjaan Ibu (kode)", 0, 194, 5)
        f_occ = st.number_input("Pekerjaan Ayah (kode)", 0, 194, 9)

        st.markdown('<div class="section-title">📚 Performa Semester 1</div>', unsafe_allow_html=True)
        s1a, s1b, s1c = st.columns(3)
        with s1a:
            cu1_crd  = st.number_input("Sem1 - Unit Dikreditkan", 0, 20, 0)
            cu1_enr  = st.number_input("Sem1 - Unit Diambil", 0, 30, 6)
        with s1b:
            cu1_eval = st.number_input("Sem1 - Jumlah Evaluasi", 0, 45, 6)
            cu1_app  = st.number_input("Sem1 - Unit Disetujui ✅", 0, 30, 5)
        with s1c:
            cu1_grd  = st.number_input("Sem1 - Nilai Rata-rata (0-20)", 0.0, 20.0, 12.0, step=0.1)
            cu1_noe  = st.number_input("Sem1 - Unit Tanpa Evaluasi", 0, 20, 0)

        st.markdown('<div class="section-title">📚 Performa Semester 2</div>', unsafe_allow_html=True)
        s2a, s2b, s2c = st.columns(3)
        with s2a:
            cu2_crd  = st.number_input("Sem2 - Unit Dikreditkan", 0, 20, 0)
            cu2_enr  = st.number_input("Sem2 - Unit Diambil", 0, 30, 6)
        with s2b:
            cu2_eval = st.number_input("Sem2 - Jumlah Evaluasi", 0, 45, 6)
            cu2_app  = st.number_input("Sem2 - Unit Disetujui ✅", 0, 30, 5)
        with s2c:
            cu2_grd  = st.number_input("Sem2 - Nilai Rata-rata (0-20)", 0.0, 20.0, 12.0, step=0.1)
            cu2_noe  = st.number_input("Sem2 - Unit Tanpa Evaluasi", 0, 20, 0)

        st.markdown('<div class="section-title">🌍 Kondisi Ekonomi Makro</div>', unsafe_allow_html=True)
        e1, e2, e3 = st.columns(3)
        with e1: unemp = st.number_input("Tingkat Pengangguran (%)", 0.0, 25.0, 11.0, step=0.1)
        with e2: inf   = st.number_input("Tingkat Inflasi (%)", -5.0, 15.0, 1.0, step=0.1)
        with e3: gdp   = st.number_input("GDP", -5.0, 5.0, 1.0, step=0.01)

        submitted = st.form_submit_button("🔮 Prediksi Sekarang", use_container_width=True, type="primary")

    if submitted:
        # Hitung fitur turunan
        apr1 = cu1_app / cu1_enr if cu1_enr > 0 else 0.0
        apr2 = cu2_app / cu2_enr if cu2_enr > 0 else 0.0
        tot_app = cu1_app + cu2_app
        avg_grd = (cu1_grd + cu2_grd) / 2

        input_df = pd.DataFrame([[
            age, gender, marital, displaced, intl,
            app_mode, app_order, prev_qual, prev_grade, admission, course, attendance,
            debtor, tuition, scholarship, edu_spec,
            cu1_enr, cu1_app, cu1_grd, cu1_eval, cu1_noe,
            cu2_enr, cu2_app, cu2_grd, cu2_eval, cu2_noe,
            unemp, inf, gdp,
            apr1, apr2, tot_app, avg_grd
        ]], columns=features)

        prediction   = model.predict(input_df)[0]
        probability  = model.predict_proba(input_df)[0]
        dropout_prob = probability[1] * 100  # prob class 1 = Dropout

        st.markdown("---")
        st.markdown("### 🎯 Hasil Prediksi")

        _, res_col, _ = st.columns([1, 2, 1])
        with res_col:
            if dropout_prob >= 60:
                risk_class = "risk-high"
                risk_label = "⚠️ DIPREDIKSI: DROPOUT"
                rec = "Segera lakukan intervensi! Mahasiswa ini memerlukan bimbingan akademik intensif dan konseling."
            elif dropout_prob >= 35:
                risk_class = "risk-medium"
                risk_label = "🔔 RISIKO SEDANG DROPOUT"
                rec = "Pantau perkembangan mahasiswa ini dan berikan dukungan tambahan."
            else:
                risk_class = "risk-low"
                risk_label = "✅ DIPREDIKSI: GRADUATE"
                rec = "Mahasiswa ini berada pada jalur yang baik menuju kelulusan."

            st.markdown(f"""
            <div class="{risk_class}">
                <h3 style="margin:0">{risk_label}</h3>
                <p style="font-size:1.8rem; font-weight:700; margin:0.5rem 0">
                    Probabilitas Dropout: <span style="color:#e74c3c">{dropout_prob:.1f}%</span>
                    &nbsp;|&nbsp;
                    Probabilitas Graduate: <span style="color:#2ecc71">{probability[0]*100:.1f}%</span>
                </p>
                <p style="margin:0">{rec}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability bar
        fig, ax = plt.subplots(figsize=(8, 1.2))
        bar_color = "#e74c3c" if dropout_prob >= 60 else "#f39c12" if dropout_prob >= 35 else "#2ecc71"
        ax.barh([""], [dropout_prob], color=bar_color, height=0.5)
        ax.barh([""], [100 - dropout_prob], left=[dropout_prob], color="#ecf0f1", height=0.5)
        ax.set_xlim(0, 100); ax.set_xlabel("Probabilitas (%)")
        ax.axvline(35, color="#f39c12", linestyle="--", lw=1.5, alpha=0.7)
        ax.axvline(60, color="#e74c3c", linestyle="--", lw=1.5, alpha=0.7)
        ax.text(dropout_prob / 2, 0, f"Dropout {dropout_prob:.1f}%",
                ha="center", va="center", fontweight="bold", fontsize=11, color="white")
        ax.set_title("Dropout Risk Gauge (Merah=Dropout, Hijau=Graduate)", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

        # Faktor risiko
        st.markdown("#### 🔎 Faktor Risiko yang Teridentifikasi")
        risks = []
        if cu1_app == 0 and cu1_enr > 0:
            risks.append("❌ Tidak lulus satupun mata kuliah di Semester 1")
        if cu2_app == 0 and cu2_enr > 0:
            risks.append("❌ Tidak lulus satupun mata kuliah di Semester 2")
        if cu1_grd < 10 and cu1_enr > 0:
            risks.append(f"📉 Nilai Semester 1 sangat rendah ({cu1_grd:.1f}/20)")
        if cu2_grd < 10 and cu2_enr > 0:
            risks.append(f"📉 Nilai Semester 2 sangat rendah ({cu2_grd:.1f}/20)")
        if tuition == 0:
            risks.append("💸 SPP tidak terbayar tepat waktu")
        if debtor == 1:
            risks.append("⚠️ Mahasiswa memiliki hutang kepada institusi")
        if age > 25:
            risks.append(f"👤 Usia enrollment di atas rata-rata ({age} tahun)")
        if apr1 < 0.5 and cu1_enr > 0:
            risks.append(f"📊 Tingkat kelulusan Sem 1 rendah ({apr1*100:.0f}%)")

        if risks:
            for r in risks: st.warning(r)
        else:
            st.success("✅ Tidak teridentifikasi faktor risiko signifikan.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: ANALISIS DATA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analisis Data":
    st.markdown("### 📊 Analisis Mendalam Data Mahasiswa")

    tab1, tab2, tab3 = st.tabs(["📈 Visualisasi EDA", "🔢 Statistik", "🎯 Feature Importance"])

    with tab1:
        st.markdown("""
        <div class="info-box">
        Analisis dilakukan pada data <b>Dropout vs Graduate</b> (3,630 mahasiswa).
        Data Enrolled tidak diikutsertakan dalam analisis model.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-title">Nilai Sem 1 vs Sem 2 (Dropout vs Graduate)</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            for status, color in [("Dropout","#e74c3c"), ("Graduate","#2ecc71")]:
                sub = df_dg[df_dg["Status"]==status]
                ax.scatter(sub["Curricular_units_1st_sem_grade"],
                           sub["Curricular_units_2nd_sem_grade"],
                           c=color, label=status, alpha=0.3, s=15)
            ax.set_xlabel("Nilai Semester 1"); ax.set_ylabel("Nilai Semester 2")
            ax.set_title("Korelasi Nilai Semester", fontsize=12, fontweight="bold")
            ax.legend(); plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

        with col2:
            st.markdown('<div class="section-title">Rata-rata Unit Disetujui per Status</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            avg_app = df_dg.groupby("Status")[
                ["Curricular_units_1st_sem_approved", "Curricular_units_2nd_sem_approved"]
            ].mean()
            avg_app.columns = ["Semester 1", "Semester 2"]
            avg_app.plot(kind="bar", ax=ax, color=["#3498db","#2ecc71"], edgecolor="white", width=0.6)
            ax.set_title("Rata-rata Unit Disetujui", fontsize=12, fontweight="bold")
            ax.tick_params(axis="x", rotation=0); ax.legend()
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<div class="section-title">Pengaruh Beasiswa terhadap Status</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            sch = df_dg.groupby(["Scholarship_holder","Status"]).size().unstack(fill_value=0)
            sch.index = ["Non-Beasiswa","Beasiswa"]
            sch_pct = sch.div(sch.sum(axis=1), axis=0) * 100
            sch_pct.plot(kind="bar", ax=ax, stacked=True,
                         color={"Dropout":"#e74c3c","Graduate":"#2ecc71"},
                         edgecolor="white")
            ax.set_title("Beasiswa vs Status (%)", fontsize=11, fontweight="bold")
            ax.tick_params(axis="x", rotation=0); ax.legend(title="Status")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

        with col4:
            st.markdown('<div class="section-title">Distribusi Nilai Masuk</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            for status, color in [("Dropout","#e74c3c"),("Graduate","#2ecc71")]:
                ax.hist(df_dg[df_dg["Status"]==status]["Admission_grade"],
                        bins=25, alpha=0.6, label=status, color=color, edgecolor="white")
            ax.set_title("Distribusi Nilai Masuk (DO vs GR)", fontsize=11, fontweight="bold")
            ax.set_xlabel("Nilai Masuk"); ax.legend()
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

    with tab2:
        st.markdown('<div class="section-title">Statistik Deskriptif: Dropout vs Graduate</div>', unsafe_allow_html=True)
        stat_cols = ["Age_at_enrollment","Admission_grade",
                     "Curricular_units_1st_sem_grade","Curricular_units_2nd_sem_grade",
                     "Curricular_units_1st_sem_approved","Curricular_units_2nd_sem_approved"]
        stat_df = df_dg.groupby("Status")[stat_cols].mean().round(2)
        stat_df.columns = ["Usia","Nilai Masuk","Nilai Sem1","Nilai Sem2","Unit Sem1","Unit Sem2"]
        st.dataframe(stat_df.style.background_gradient(cmap="RdYlGn", axis=0), use_container_width=True)

        st.markdown("---")
        filter_s = st.selectbox("Filter Status:", ["Semua (3,630)"] + ["Dropout","Graduate"])
        shown = df_dg if filter_s.startswith("Semua") else df_dg[df_dg["Status"]==filter_s]
        st.dataframe(shown.head(50), use_container_width=True)
        st.caption(f"Menampilkan {min(50,len(shown))} dari {len(shown)} baris")

    with tab3:
        st.markdown('<div class="section-title">Feature Importance — Random Forest (Dropout vs Graduate)</div>', unsafe_allow_html=True)
        rf_clf = model.named_steps["clf"]
        imp = pd.Series(rf_clf.feature_importances_, index=features).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 10))
        colors_fi = ["#e74c3c" if i >= len(imp)-5 else "#3498db" for i in range(len(imp))]
        imp.plot(kind="barh", ax=ax, color=colors_fi, edgecolor="white")
        ax.set_title("Feature Importance (merah = 5 teratas)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: TENTANG SISTEM
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ Tentang Sistem":
    st.markdown("### ℹ️ Tentang Sistem Prediksi Dropout")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### 🎯 Tujuan Sistem
        Sistem ini membantu **Jaya Jaya Institut** untuk:
        - Mendeteksi dini mahasiswa berisiko dropout
        - Memahami faktor-faktor penyebab dropout
        - Memonitor performa mahasiswa secara berkala
        - Pengambilan kebijakan berbasis data

        #### 🤖 Model Machine Learning
        | Komponen | Detail |
        |----------|--------|
        | Algoritma | Random Forest Classifier |
        | Data training | Dropout + Graduate (3,630 sampel) |
        | Data excluded | Enrolled (794 sampel) |
        | Target | **1=Dropout, 0=Graduate** |
        | Fitur | 33 fitur |
        | Akurasi | **92.56%** |
        | F1 Score | **0.9043** |
        | ROC-AUC | **0.9716** |
        """)

    with col2:
        st.markdown("""
        #### 📊 Dataset
        | Keterangan | Detail |
        |------------|--------|
        | Sumber | Dicoding / UCI ML Repository |
        | Total data | 4,424 mahasiswa |
        | Graduate | 2,209 (49.9%) |
        | Dropout | 1,421 (32.1%) |
        | Enrolled | 794 (17.9%) |

        #### 🚀 Cara Menjalankan
        ```bash
        # Anaconda
        conda create --name main-ds python=3.9
        conda activate main-ds
        pip install -r requirements.txt
        streamlit run app.py

        # Pipenv
        pip install pipenv
        pipenv install
        pipenv shell
        streamlit run app.py
        ```
        """)

    st.markdown("---")
    st.markdown("""
    #### ⚠️ Action Items untuk Jaya Jaya Institut

    1. **🚨 Early Warning System** — Implementasikan model ini setiap awal semester untuk seluruh mahasiswa aktif (Enrolled). Identifikasi yang berisiko tinggi dropout untuk intervensi segera.
    2. **📚 Bimbingan Akademik Intensif** — Mahasiswa dengan nilai semester 1 di bawah 10/20 atau gagal > 50% mata kuliah harus langsung mendapat mentor.
    3. **💰 Bantuan Finansial Proaktif** — Identifikasi tunggakan SPP sebelum semester baru. Tawarkan cicilan atau beasiswa.
    4. **🎓 Evaluasi Prodi Bermasalah** — Prodi dengan dropout rate > 35% perlu review kurikulum menyeluruh.
    5. **📈 Monitoring Dashboard Rutin** — Gunakan dashboard Metabase untuk rapat evaluasi bulanan.
    6. **🤝 Program Orientasi Khusus** — Mahasiswa usia > 25 tahun dan pindahan mendapat pendampingan ekstra.
    """)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; font-size:0.85rem;'>"
    "Jaya Jaya Institut — Student Dropout Prediction | "
    "Model: Random Forest | Target: 1=Dropout, 0=Graduate | "
    "Dicoding Data Science Final Project"
    "</p>",
    unsafe_allow_html=True,
)
