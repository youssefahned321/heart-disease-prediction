import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns



st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Mono', monospace; }
    .stApp { background: #0f0f0f; color: #e0e0e0; }
    div[data-testid="stSidebar"] { background: #141414; border-right: 1px solid #222; }
    .card {
        background: #1a1a1a; border: 1px solid #2a2a2a;
        border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center;
    }
    .card-label { font-size: 0.75rem; color: #777; letter-spacing: 0.1em; text-transform: uppercase; }
    .card-value { font-size: 1.7rem; font-weight: 700; font-family: 'Space Mono', monospace; }
    .result-risk {
        background: linear-gradient(135deg, #2a0808, #1a0000);
        border: 1px solid #7a1515; border-radius: 16px; padding: 2rem; text-align: center;
    }
    .result-ok {
        background: linear-gradient(135deg, #082a12, #001a08);
        border: 1px solid #156b30; border-radius: 16px; padding: 2rem; text-align: center;
    }
    .stButton > button {
        background: #b03030; color: white; border: none; border-radius: 8px;
        padding: 0.7rem 1rem; font-family: 'Space Mono', monospace;
        font-weight: 700; width: 100%; font-size: 1rem;
    }
    .stButton > button:hover { background: #d44040; }
</style>
""", unsafe_allow_html=True)
##تحميل الموديل
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    columns = joblib.load("columns.pkl")  
    return model, columns

model, columns = load_model()

with st.sidebar:
    st.markdown("## 🫀 Patient Data")
    st.markdown("---")
    st.markdown("**Demographics**")
    age = st.slider("Age", 20, 80, 52)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")

    st.markdown("**Clinical Measurements**")
    trestbps = st.number_input("Resting Blood Pressure (mmHg)", 80, 200, 130)
    chol     = st.number_input("Cholesterol (mg/dl)", 100, 600, 250)
    thalach  = st.number_input("Max Heart Rate Achieved", 60, 220, 170)
    oldpeak  = st.number_input("ST Depression (Oldpeak)", 0.0, 7.0, 1.2, step=0.1)

    st.markdown("**Diagnostic Results**")
    cp      = st.selectbox("Chest Pain Type (cp)", [0,1,2,3])
    fbs     = st.selectbox("Fasting Blood Sugar > 120 (fbs)", [0,1], format_func=lambda x: f"{'Yes' if x else 'No'} ({x})")
    restecg = st.selectbox("Resting ECG (restecg)", [0,1,2])
    exang   = st.selectbox("Exercise Induced Angina (exang)", [0,1], format_func=lambda x: f"{'Yes' if x else 'No'} ({x})")
    slope   = st.selectbox("Slope of ST Segment", [0,1,2])
    ca      = st.selectbox("Major Vessels (ca)", [0,1,2,3])
    thal    = st.selectbox("Thalassemia (thal)", [0,1,2,3],
                           format_func=lambda x: ["Normal","Fixed Defect","Reversible Defect","Unknown"][x])

    predict_btn = st.button("🔍 Run Prediction")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🫀 Heart Disease Predictor")
st.markdown("*Data Mining Project — Egyptian Chinese University, Spring 2026*")
st.markdown("---")

# ── Prediction ─────────────────────────────────────────────────────────────────
if predict_btn:

    # Build input — same column order as training
    input_values = [age, sex, cp, trestbps, chol, fbs,
                    restecg, thalach, exang, oldpeak, slope, ca, thal]

    # columns.pkl = X.columns (13 original features + age_group after engineering)
    original_cols = [c for c in columns if c != 'age_group']
    data = pd.DataFrame([input_values], columns=original_cols)

    # Feature Engineering — same as notebook
    data['age_group'] = pd.cut(data['age'], bins=[20,40,60,80], labels=[0,1,2]).astype(int)
    data = data[columns]   # enforce training column order

    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0, 1]

    # ── Result Card ────────────────────────────────────────────────────────────
    col_res, col_gauge = st.columns([1, 1])

    with col_res:
        if pred == 1:
            st.markdown(f"""
            <div class="result-risk">
                <div style="font-family:'Space Mono',monospace;font-size:1.5rem;font-weight:700;color:#e05555;">⚠ AT RISK</div>
                <div style="font-size:3rem;font-weight:700;font-family:'Space Mono',monospace;color:#e05555;">{prob:.1%}</div>
                <p style="color:#bbb;margin-top:0.5rem;">High probability of heart disease detected.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-ok">
                <div style="font-family:'Space Mono',monospace;font-size:1.5rem;font-weight:700;color:#3ecf6e;">✓ HEALTHY</div>
                <div style="font-size:3rem;font-weight:700;font-family:'Space Mono',monospace;color:#3ecf6e;">{1-prob:.1%}</div>
                <p style="color:#bbb;margin-top:0.5rem;">Low probability of heart disease detected.</p>
            </div>""", unsafe_allow_html=True)

    with col_gauge:
        fig, ax = plt.subplots(figsize=(4, 3.5), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        theta = np.linspace(0, np.pi, 200)
        ax.plot(theta, [1]*200, color='#2a2a2a', linewidth=18, solid_capstyle='round')
        fill = np.linspace(0, prob * np.pi, 200)
        color = '#e05555' if prob > 0.5 else '#3ecf6e'
        ax.plot(fill, [1]*200, color=color, linewidth=18, solid_capstyle='round')
        needle = prob * np.pi
        ax.annotate('', xy=(needle, 0.95), xytext=(needle, 0.1),
                    arrowprops=dict(arrowstyle='->', color='white', lw=2))
        ax.set_ylim(0, 1.3)
        ax.set_theta_zero_location('W')
        ax.set_theta_direction(-1)
        ax.set_xlim(0, np.pi)
        ax.axis('off')
        ax.text(np.pi/2, 0.28, f"{prob:.0%}", ha='center', va='center',
                fontsize=22, fontweight='bold', color='white', fontfamily='monospace')
        ax.text(np.pi/2, -0.1, "Risk Score", ha='center', va='center', fontsize=9, color='#888')
        st.pyplot(fig)
        plt.close()

    # ── Patient Metrics ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Patient Profile")
    m1, m2, m3, m4, m5 = st.columns(5)
    for col_widget, label, val, unit in zip(
        [m1, m2, m3, m4, m5],
        ["Age", "Blood Pressure", "Cholesterol", "Max Heart Rate", "ST Depression"],
        [age, int(trestbps), int(chol), int(thalach), f"{oldpeak:.1f}"],
        ["yrs", "mmHg", "mg/dl", "bpm", ""]
    ):
        with col_widget:
            st.markdown(f"""
            <div class="card">
                <div class="card-label">{label}</div>
                <div class="card-value">{val}<span style="font-size:0.85rem;color:#666"> {unit}</span></div>
            </div>""", unsafe_allow_html=True)

    # ── Dashboards ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Visual Dashboard")

    dash1, dash2 = st.columns(2)

    # Chart 1: Feature Importance
    with dash1:
        st.markdown("**Feature Importance**")
        try:
            importances = model.feature_importances_
        except AttributeError:
            try:
                importances = np.abs(model.named_steps['model'].coef_[0])
            except Exception:
                importances = None

        if importances is not None:
            fi = pd.Series(importances, index=columns).sort_values()
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#1a1a1a')
            ax.set_facecolor('#1a1a1a')
            bar_colors = ['#e05555' if v == fi.max() else '#3d5a80' for v in fi.values]
            fi.plot(kind='barh', ax=ax, color=bar_colors, edgecolor='none')
            ax.set_xlabel("Importance", color='#aaa')
            ax.tick_params(colors='#aaa')
            for spine in ax.spines.values():
                spine.set_color('#2a2a2a')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Chart 2: Patient vs Population Radar
    with dash2:
        st.markdown("**Patient vs Population Average**")
        feat_labels = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        pop_avg  = [54.4, 131.6, 246.0, 149.1, 1.07]
        pat_vals = [age, trestbps, chol, thalach, oldpeak]
        max_vals = [80, 200, 600, 220, 7.0]

        pop_norm = [v/m for v, m in zip(pop_avg, max_vals)]
        pat_norm = [v/m for v, m in zip(pat_vals, max_vals)]
        angles   = np.linspace(0, 2*np.pi, len(feat_labels), endpoint=False).tolist()
        pop_norm += pop_norm[:1]; pat_norm += pat_norm[:1]; angles += angles[:1]

        fig, ax = plt.subplots(figsize=(5, 4), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        ax.plot(angles, pop_norm, 'o-', color='#3d5a80', linewidth=2, label='Population Avg')
        ax.fill(angles, pop_norm, alpha=0.15, color='#3d5a80')
        ax.plot(angles, pat_norm, 'o-', color='#e05555', linewidth=2, label='This Patient')
        ax.fill(angles, pat_norm, alpha=0.2, color='#e05555')
        ax.set_thetagrids(np.degrees(angles[:-1]), feat_labels, color='#aaa', fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(color='#2a2a2a')
        ax.spines['polar'].set_color('#2a2a2a')
        ax.tick_params(colors='#333')
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1),
                  facecolor='#1a1a1a', edgecolor='#333', labelcolor='white', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    dash3, dash4 = st.columns(2)

    # Chart 3: Risk Factor Status
    with dash3:
        st.markdown("**Key Risk Factor Status**")
        factors = {
            "Blood Pressure": (trestbps, 120, 140, "mmHg"),
            "Cholesterol":    (chol,     200, 240, "mg/dl"),
            "Max Heart Rate": (thalach,  100, 140, "bpm"),
        }
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        for i, (name, (val, low, high, unit)) in enumerate(factors.items()):
            color = '#e05555' if val > high else ('#f39c12' if val > low else '#3ecf6e')
            ax.barh(i, val, color=color, height=0.5, edgecolor='none')
            ax.axvline(low,  color='#f39c12', linestyle='--', linewidth=0.8, alpha=0.6)
            ax.axvline(high, color='#e05555', linestyle='--', linewidth=0.8, alpha=0.6)
            ax.text(val + 2, i, f"{val:.0f} {unit}", va='center', color='#ccc', fontsize=8)
        ax.set_yticks(range(len(factors)))
        ax.set_yticklabels(list(factors.keys()), color='#aaa', fontsize=8)
        ax.tick_params(colors='#aaa')
        for spine in ax.spines.values():
            spine.set_color('#2a2a2a')
        ax.set_title("Green=Normal  Orange=Borderline  Red=High", fontsize=7, color='#777', pad=6)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Chart 4: Input Summary Table
    with dash4:
        st.markdown("**Input Summary**")
        summary_data = {
            "Feature": ["Sex", "Chest Pain", "Fasting BS", "Exercise Angina", "Thalassemia", "Age Group"],
            "Value": [
                "Male" if sex == 1 else "Female",
                f"Type {cp}",
                "High (>120)" if fbs == 1 else "Normal",
                "Yes" if exang == 1 else "No",
                ["Normal","Fixed Defect","Reversible","Unknown"][thal],
                ["Young (20-40)","Middle (40-60)","Senior (60-80)"][
                    0 if age <= 40 else (1 if age <= 60 else 2)]
            ]
        }
        df_summary = pd.DataFrame(summary_data)
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        ax.axis('off')
        tbl = ax.table(cellText=df_summary.values, colLabels=df_summary.columns,
                       cellLoc='center', loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor('#1f1f1f' if r % 2 == 0 else '#242424')
            cell.set_edgecolor('#2a2a2a')
            cell.set_text_props(color='#ddd')
            if r == 0:
                cell.set_facecolor('#2d2d2d')
                cell.set_text_props(color='white', fontweight='bold')
        tbl.scale(1, 1.6)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

else:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color: #555;">
        <div style="font-size: 5rem;">🫀</div>
        <h2 style="font-family:'Space Mono',monospace; color:#777;">Enter Patient Data</h2>
        <p>Fill in the patient details in the sidebar, then press <b>Run Prediction</b>.</p>
    </div>
    """, unsafe_allow_html=True)
