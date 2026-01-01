# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')   # for headless environments to save PNGs
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import tempfile
import uuid

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

# -------------------------
# Helpers
# -------------------------
@st.cache_resource
def load_model(path='model/diabetes_pipeline_v1.pkl'):
    """Load the saved pipeline dict: {'pipeline': ..., 'features': [...] }"""
    assert os.path.exists(path), f"Model file not found at {path}"
    data = joblib.load(path)
    pipeline = data['pipeline']
    features = data['features']
    return pipeline, features

@st.cache_resource
def load_feature_importance(path='model/feature_importance_coefficients.csv'):
    """
    Robust loader for feature importance CSV.
    Expected formats:
      feature,coefficient
      age,0.312
    or a file with index and one column (no header).
    This forces 'coefficient' to numeric.
    """
    if not os.path.exists(path):
        return None

    # Try reading the CSV in a few common forms
    try:
        df = pd.read_csv(path)
        # If the file had no header and read produced single-column unnamed data,
        # try reading with index_col=0
        if df.shape[1] == 1 and 'coefficient' not in df.columns and 'feature' not in df.columns:
            df2 = pd.read_csv(path, index_col=0, header=None)
            if df2.shape[1] == 1:
                df = df2.reset_index()
                df.columns = ['feature', 'coefficient']
        # If columns exist but names differ attempt to standardize
        if 'feature' not in df.columns and df.shape[1] >= 2:
            # assume first column is feature name and second is coefficient
            df = df.rename(columns={df.columns[0]: 'feature', df.columns[1]: 'coefficient'})[['feature', 'coefficient']]
        # force numeric
        df["coefficient"] = pd.to_numeric(df["coefficient"], errors="coerce")
        return df.set_index("feature")["coefficient"]
    except Exception:
        # Last-resort attempt: read as index and single column
        try:
            df2 = pd.read_csv(path, index_col=0, header=None)
            if df2.shape[1] == 1:
                df = df2.reset_index()
                df.columns = ['feature', 'coefficient']
                df["coefficient"] = pd.to_numeric(df["coefficient"], errors="coerce")
                return df.set_index("feature")["coefficient"]
        except Exception:
            return None
    return None

def parse_feature_groups(feature_list):
    """
    Fixed parsing:
      - First classify known binary/mapped features
      - Then handle one-hot (dummies) by '_' pattern
      - Remaining are continuous-like
    Returns: continuous_like, binary_flags, dummies (real groups), mapped
    """
    dummies = {}
    binary_flags = []
    continuous_like = []
    mapped = {}

    # known exact-name binary features
    known_binaries = {'family_history_diabetes', 'hypertension_history', 'cardiovascular_history'}

    # known mapped (ordinal) features
    known_mapped = {'education_level', 'income_level'}

    for f in feature_list:
        if f in known_binaries:
            binary_flags.append(f)
            continue
        if f in known_mapped:
            if f == 'education_level':
                mapped['education_level'] = {
                    0: 'No formal',
                    1: 'Highschool',
                    2: 'Graduate',
                    3: 'Postgraduate'
                }
            elif f == 'income_level':
                mapped['income_level'] = {
                    0: 'Low',
                    1: 'Lower-Middle',
                    2: 'Middle',
                    3: 'Upper-Middle',
                    4: 'High'
                }
            continue
        # now handle one-hot patterns
        if '_' in f:
            prefix = f.split('_', 1)[0]
            dummies.setdefault(prefix, []).append(f)
        else:
            continuous_like.append(f)

    # Only keep dummies that are real groups (length>1)
    real_dummies = {k: v for k, v in dummies.items() if len(v) > 1}
    # Any singleton dummy columns should be treated as continuous-like
    for k, v in dummies.items():
        if len(v) == 1:
            continuous_like.extend(v)

    return continuous_like, binary_flags, real_dummies, mapped


def build_input_row(features, cont_vals, bin_flags, dummy_choices, mapped_choices):
    # start zeros
    row = pd.Series(0, index=features, dtype=float)
    # continuous
    for k, v in cont_vals.items():
        if k in row.index:
            row[k] = v
    # binary flags (0/1)
    for k, v in bin_flags.items():
        if k in row.index:
            row[k] = 1 if v else 0
    # dummies: set selected category to 1
    for prefix, selected in dummy_choices.items():
        cols = [c for c in features if c.startswith(prefix + '_')]
        for c in cols:
            if c == selected:
                row[c] = 1
            else:
                row[c] = 0
    # mapped (education/income)
    for k, v in mapped_choices.items():
        if k in row.index:
            row[k] = int(v)
    return pd.DataFrame([row], columns=features)

def compute_feature_contributions(pipeline, X_row_df):
    """
    Compute contribution to model logit per original feature using scaler + linear coef:
    contribution = coef * (x - mean) / scale
    """
    scaler = pipeline.named_steps.get('scaler', None)
    model = pipeline.named_steps['model']
    coef = model.coef_[0]
    intercept = model.intercept_[0]
    features = X_row_df.columns.tolist()

    # If scaler exists, use its mean & scale, otherwise assume identity
    if scaler is not None:
        means = scaler.mean_
        scales = scaler.scale_
    else:
        # fallback: zero mean, unit scale
        means = np.zeros(len(features))
        scales = np.ones(len(features))

    x = X_row_df.values.reshape(-1, len(features))[0]
    scaled = (x - means) / scales
    contrib = coef * scaled
    contrib_series = pd.Series(contrib, index=features)
    return contrib_series, intercept

def _save_series_bar(series, title, path, top_n=12, figsize=(6,4)):
    """
    Save a horizontal bar chart for a pandas Series (index: label, values numeric).
    """
    s = series.copy().dropna()
    s = s.sort_values(ascending=True).tail(top_n)  # keep top
    plt.clf()
    fig, ax = plt.subplots(figsize=figsize)
    s.plot(kind='barh', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def generate_pdf_report(pdf_path,
                        model_name,
                        roc_auc=None,
                        feat_imp_series=None,
                        contribs_series=None,
                        input_row=None,
                        proba=None):
    """
    Create a professional-looking PDF report.
    - pdf_path: destination filepath (str)
    - model_name: string
    - roc_auc: optional float (displayed)
    - feat_imp_series: pd.Series indexed by feature name (coefficients or importance)
    - contribs_series: pd.Series of contributions for this instance (signed)
    - input_row: pd.DataFrame single-row with input values
    - proba: predicted probability (0..1)
    """
    styles = getSampleStyleSheet()
    normal = styles['Normal']
    heading = styles['Heading1']
    small = ParagraphStyle('small', parent=styles['Normal'], fontSize=9, leading=11)

    # Temporary images directory
    tmpdir = tempfile.gettempdir()
    uid = uuid.uuid4().hex[:8]
    fi_png = os.path.join(tmpdir, f"fi_{uid}.png")
    contrib_png = os.path.join(tmpdir, f"contrib_{uid}.png")

    # Save charts
    if feat_imp_series is not None:
        # prepare a copy showing absolute sorted importance
        _save_series_bar(feat_imp_series.sort_values(key=abs, ascending=False).head(12),
                         "Top Features (by importance)", fi_png, top_n=12)

    if contribs_series is not None:
        # show top positive & top negative combined (signed)
        contrib_sorted = contribs_series.sort_values()
        # for readability show the 12 largest magnitude contributors (mix of neg/pos)
        contrib_to_plot = contrib_sorted.abs().sort_values(ascending=False).head(12)
        # create a signed series for display with original signs, keeping those names
        contrib_display = contribs_series.loc[contrib_to_plot.index].sort_values()
        _save_series_bar(contrib_display, "Top Feature Contributions (this person)", contrib_png, top_n=12)

    # Build PDF
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                            rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)

    elems = []
    elems.append(Paragraph("Diabetes Risk Report", heading))
    elems.append(Spacer(1, 6))
    elems.append(Paragraph(f"Model: {model_name}", normal))
    if roc_auc is not None:
        elems.append(Paragraph(f"Cross-validated ROC-AUC: {roc_auc:.3f}", normal))
    elems.append(Spacer(1, 12))

    # Executive summary (one short paragraph)
    summary_text = ("This report summarizes the model prediction and the top features that influenced the"
                    " result for the supplied input. The feature contributions show how individual inputs"
                    " shifted the model score for this person. These are explanatory cues, not clinical diagnoses.")
    elems.append(Paragraph(summary_text, small))
    elems.append(Spacer(1, 12))

    # Prediction box
    if proba is not None:
        pct = proba * 100
        risk = "Low" if proba < 0.33 else ("Medium" if proba < 0.66 else "High")
        elems.append(Paragraph(f"<b>Predicted probability of diabetes:</b> {pct:.2f}%  &nbsp;&nbsp; <b>Risk:</b> {risk}", normal))
        elems.append(Spacer(1, 12))

    # Add feature importance image (if exists)
    if feat_imp_series is not None and os.path.exists(fi_png):
        elems.append(Paragraph("Feature importance (top features):", styles['Heading3']))
        elems.append(Spacer(1,6))
        elems.append(RLImage(fi_png, width=6.5*inch, height=3.5*inch))
        elems.append(Spacer(1,12))

    # Add contribution image
    if contribs_series is not None and os.path.exists(contrib_png):
        elems.append(Paragraph("Feature contributions for this person:", styles['Heading3']))
        elems.append(Spacer(1,6))
        elems.append(RLImage(contrib_png, width=6.5*inch, height=3.5*inch))
        elems.append(Spacer(1,12))

    # Input values table
    if input_row is not None:
        elems.append(Paragraph("Input values used for prediction:", styles['Heading3']))
        elems.append(Spacer(1,6))
        # Convert input_row to table data
        tdata = [["Feature", "Value"]]
        for k, v in input_row.squeeze().items():
            tdata.append([str(k), f"{v}"])
        table = Table(tdata, colWidths=[3.2*inch, 3.2*inch])
        table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f0f0f0')),
            ('ALIGN',(0,0),(-1,-1),'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ]))
        elems.append(table)
        elems.append(Spacer(1,12))

    # Footnote and references
    elems.append(Paragraph("Notes: Feature contributions are computed from the trained logistic model by scaling inputs and multiplying by coefficients. These are approximate explanations intended to increase transparency.", small))
    elems.append(Spacer(1,6))
    elems.append(Paragraph("Generated by Diabetes Risk Predictor", small))

    # Build
    doc.build(elems)

    # Clean up PNGs (optional)
    try:
        if os.path.exists(fi_png):
            os.remove(fi_png)
        if os.path.exists(contrib_png):
            os.remove(contrib_png)
    except Exception:
        pass

    return pdf_path


# -------------------------
# Load model & metadata
# -------------------------
pipeline, FEATURES = load_model('model/diabetes_pipeline_v1.pkl')
feat_imp_series = load_feature_importance('model/feature_importance_coefficients.csv')

# -------------------------
# Parse features to build UI
# -------------------------
cont_features, bin_flags, dummy_groups, mapped = parse_feature_groups(FEATURES)

# We'll define sensible UI order: continuous first, mapped, binary, dummies
st.title("Diabetes Risk Predictor — Demo")
st.markdown("Simple, interpretable logistic regression pipeline. This app performs inference only (no retraining).")

with st.expander("Model info", expanded=False):
    st.write("Loaded model pipeline from `model/diabetes_pipeline_v1.pkl`")
    st.write("Number of features:", len(FEATURES))
    if feat_imp_series is not None:
        st.write("Top important features (from saved coefficients):")
        # safe sorting using numeric series; feat_imp_series dtype should be numeric now
        st.dataframe(feat_imp_series.sort_values(key=abs, ascending=False).head(10))

st.sidebar.header("Input features")

# -------------------------
# Sidebar inputs for continuous features
# -------------------------
st.sidebar.subheader("Continuous features")
cont_values = {}
# sensible defaults and ranges for known features (fallback generic)
# Provide mapping for known continuous features to ranges/defaults
ranges = {
    'age': (18, 100, 40),
    'alcohol_consumption_per_week': (0, 50, 2),
    'physical_activity_minutes_per_week': (0, 2000, 150),
    'diet_score': (0.0, 10.0, 5.0),
    'sleep_hours_per_day': (0.0, 16.0, 7.0),
    'screen_time_hours_per_day': (0.0, 20.0, 3.0),
    'bmi': (10.0, 60.0, 24.0),
    'waist_to_hip_ratio': (0.3, 1.5, 0.85),
    'systolic_bp': (70, 220, 120),
    'diastolic_bp': (40, 140, 80),
    'heart_rate': (30, 200, 70),
    'cholesterol_total': (80, 400, 180),
    'hdl_cholesterol': (10, 200, 50),
    'ldl_cholesterol': (10, 300, 100),
    'triglycerides': (10, 1000, 100)
}

# -------------------------
# Sidebar form for inputs (clean hover help, no captions)
# -------------------------
HELP_TEXTS = {
    'age': "Age in years. Older age increases diabetes risk on average.",
    'alcohol_consumption_per_week': "Approx. standard drinks per week (one standard drink ≈ 14g pure alcohol). Typical: low 0 — avg 2 — high 14+.",
    'physical_activity_minutes_per_week': "Minutes of moderate+ physical activity per week. WHO recommends 150 min/week. Typical: low 0 — avg 150 — high 300+.",
    'diet_score': "0 (poor) → 10 (excellent). Measures how healthy the person's diet is (fruits, veg, whole grains, low sugar/fat). Typical: low 2 — avg 5 — high 8.",
    'sleep_hours_per_day': "Average sleep per day. Adults usually aim 7–9 hours. Typical: low 4 — avg 7 — high 10.",
    'screen_time_hours_per_day': "Non-work screen time per day (hours). High sedentary screen time links to worse metabolic health. Typical: low 0 — avg 3 — high 8+.",
    'bmi': "Body Mass Index = weight(kg)/height(m)². Underweight <18.5, healthy 18.5–24.9, overweight 25–29.9, obese 30+. Typical: low 18 — avg 24 — high 35.",
    'waist_to_hip_ratio': "Waist ÷ hip. Higher ratios indicate more belly fat. For men >0.90 and women >0.85 often higher risk. Typical: low 0.7 — avg 0.85 — high 1.0+.",
    'systolic_bp': "Top blood pressure number (mmHg). Normal ~120. High is 140+. Typical: low 100 — avg 120 — high 160+.",
    'diastolic_bp': "Bottom blood pressure number (mmHg). Normal ~80. High is 90+. Typical: low 60 — avg 80 — high 100+.",
    'heart_rate': "Resting heart beats per minute. Typical adult resting: 60–100 bpm. Typical: low 50 — avg 70 — high 100.",
    'cholesterol_total': "Total cholesterol (mg/dL). Lower is usually better: desirable <200 mg/dL. Typical: low 150 — avg 180 — high 240+.",
    'hdl_cholesterol': "HDL = 'good' cholesterol (mg/dL). Higher is protective. Typical: low 30 — avg 50 — high 70+.",
    'ldl_cholesterol': "LDL = 'bad' cholesterol (mg/dL). Lower is better. Typical: low 70 — avg 100 — high 160+.",
    'triglycerides': "Triglycerides are fats in your blood. High levels often come from sugary foods, alcohol, or lack of exercise. Blood triglycerides (mg/dL). Normal <150, borderline 150–199, high 200+. Typical: low 70 — avg 120 — high 250+.",
}

def clean_label(name):
    name = name.replace('status_', '').replace('_', ' ').title()
    name = name.replace('Hdl', 'HDL').replace('Ldl', 'LDL').replace('Bmi', 'BMI')
    return name

with st.sidebar.form("input_form", clear_on_submit=False):
    st.header("Input features")

    # Continuous features
    st.subheader("Personal / Health")
    cont_values = {}
    for f in cont_features:
        lo, hi, default = ranges.get(f, (0.0, 100.0, 0.0))
        label = clean_label(f)
        help_text = HELP_TEXTS.get(f, "")
        if isinstance(default, int):
            cont_values[f] = st.slider(
                label,
                int(lo),
                int(hi),
                int(default),
                help=help_text
            )
        else:
            cont_values[f] = st.slider(
                label,
                float(lo),
                float(hi),
                float(default),
                step=0.1,
                help=help_text
            )

    # Mapped ordinal features (education/income) - PERSONAL income wording
    st.subheader("Socioeconomic")
    mapped_choices = {}
    if 'education_level' in mapped:
        edu_map = mapped['education_level']
        labels = [edu_map[k] for k in sorted(edu_map.keys())]
        sel = st.selectbox("Education level", labels, index=min(1, len(labels)-1),
                           help="Highest completed education level.")
        inv = {v:k for k,v in edu_map.items()}
        mapped_choices['education_level'] = inv[sel]

    if 'income_level' in mapped:
        inc_map = mapped['income_level']
        labels = [inc_map[k] for k in sorted(inc_map.keys())]
        # Clarify this is individual's income level
        sel = st.selectbox("Personal income level", labels, index=min(2, len(labels)-1),
                           help="Estimated personal monthly/annual income level (used as a socioeconomic indicator).")
        inv = {v:k for k,v in inc_map.items()}
        mapped_choices['income_level'] = inv[sel]

    # Binary flags
    st.subheader("History / Flags")
    bin_inputs = {}
    if bin_flags:
        for b in bin_flags:
            label = clean_label(b)
            help_text = ""
            if b == 'family_history_diabetes':
                help_text = "Tick if a parent or sibling has been diagnosed with diabetes."
            elif b == 'hypertension_history':
                help_text = "Tick if you have been previously diagnosed with high blood pressure."
            elif b == 'cardiovascular_history':
                help_text = "Tick if you have a history of heart disease (heart attack, angina, etc.)."
            bin_inputs[b] = st.checkbox(label, value=False, help=help_text)

    # Dummied categories (clean labels)
    st.subheader("Categories")
    dummy_choices = {}
    dummy_display_maps = {}
    if dummy_groups:
        for prefix, cols in dummy_groups.items():
            display_opts = []
            display_to_col = {}
            for c in cols:
                opt = c.split(prefix + "_", 1)[1]
                opt_clean = opt.replace('status_', '').replace('_', ' ').title()
                display_opts.append(opt_clean)
                display_to_col[opt_clean] = c
            dummy_display_maps[prefix] = display_to_col
            sel_display = st.selectbox(clean_label(prefix), display_opts, index=0,
                                       help=f"Select {prefix.replace('_', ' ')}")
            selected_col = display_to_col.get(sel_display, cols[0])
            dummy_choices[prefix] = selected_col

    st.markdown("---")
    submit = st.form_submit_button("Predict risk")

# -------------------------
# Prediction action
# -------------------------
if submit:
    # Build input row (DataFrame with columns in FEATURES order)
    input_df = build_input_row(FEATURES, cont_values, bin_inputs, dummy_choices, mapped_choices)
    # Ensure column order and dtype
    input_df = input_df.reindex(columns=FEATURES, fill_value=0)

    # Run prediction
    proba = pipeline.predict_proba(input_df)[:, 1][0]
    pct = proba * 100.0
    # risk labeling (simple)
    if proba < 0.33:
        risk_lbl = "Low"
        color = "green"
    elif proba < 0.66:
        risk_lbl = "Medium"
        color = "orange"
    else:
        risk_lbl = "High"
        color = "red"

    st.subheader("Prediction")
    st.markdown(f"**Predicted probability of diabetes:** {pct:.2f}%")
    st.markdown(f"**Risk category:** :{color}[{risk_lbl}]")

    # Show contributions (top positive / negative)
    try:
        contribs, intercept = compute_feature_contributions(pipeline, input_df)
        # sort and show top contributors
        top_pos = contribs.sort_values(ascending=False).head(8)
        top_neg = contribs.sort_values(ascending=True).head(8)

        st.markdown("**What are these contributions?**")
        st.write(
            "Each bar shows how much that feature *moved the model's score* for this person. "
            "Positive contributions increase predicted risk; negative contributions reduce it. "
            "These are approximate linear contributions from the logistic model (scaled feature × coefficient)."
        )
        with st.expander("Why this matters (short):"):
            st.write(
                "- Positive top contributors: features that pushed the model toward predicting diabetes for this person.\n"
                "- Negative top contributors: features that helped push risk down.\n"
                "- Use these as *explanatory* signals — not medical diagnoses. Consult a professional for health advice."
            )

        st.subheader("Top positive contributions (increase risk)")
        st.bar_chart(top_pos)

        st.subheader("Top negative contributions (decrease risk)")
        st.bar_chart(top_neg)

        # also show raw input
        with st.expander("Show input data"):
            st.write(input_df.T)
    except Exception as e:
        st.write("Could not compute contributions:", e)
        st.write("Model prediction (prob):", proba)

    # Option to download single prediction
    outdf = input_df.copy()
    outdf['predicted_proba'] = proba
    csv = outdf.to_csv(index=False)
    st.download_button("Download prediction as CSV", csv, file_name="prediction.csv")

    # Generate PDF and provide download
    try:
        # create a unique temporary pdf path
        tmp_pdf = os.path.join(tempfile.gettempdir(), f"diabetes_report_{uuid.uuid4().hex[:8]}.pdf")
        pdf_path = generate_pdf_report(
            pdf_path=tmp_pdf,
            model_name="LogisticRegression(StandardScaler)",
            roc_auc=None,  # optional, set if available
            feat_imp_series=feat_imp_series if feat_imp_series is not None else None,
            contribs_series=contribs if 'contribs' in locals() else None,
            input_row=input_df,
            proba=proba
        )
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        st.download_button("Download PDF report", data=pdf_bytes, file_name="diabetes_report.pdf", mime="application/pdf")
    except Exception as e:
        st.write("Could not create PDF report:", e)

# -------------------------
# Footer notes
# -------------------------
st.markdown("---")
st.caption("Notes: This app performs inference only using a saved pipeline. It is for demonstration purposes. "
           "Interpretations are approximate. For production use, consider calibration and robust validation.")
# Data source: Kaggle Playground Series S5E12 (for model training)
