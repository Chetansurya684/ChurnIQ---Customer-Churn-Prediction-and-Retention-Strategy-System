import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

BASE = Path(__file__).resolve().parent
sys.path.append(str(BASE / "src"))

from model_utils import load_data, load_artifacts, risk_level, retention_strategy, FI_PATH

st.set_page_config(
    page_title="ChurnIQ | Churn Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- UI STYLING ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family: Inter, sans-serif;}
.stApp {background:#f4f7fb;color:#172033;}
[data-testid="stHeader"] {background:rgba(244,247,251,.92);}
[data-testid="stAppViewContainer"] > .main .block-container {
    max-width:1500px;padding-top:1.7rem;padding-bottom:2rem;
}
[data-testid="stSidebar"] {background:#111c32;border-right:1px solid #26334a;}
[data-testid="stSidebar"] * {color:#e7edf7;}
[data-testid="stSidebar"] .block-container {padding-top:1.2rem;}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding:7px 10px;border-radius:9px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background:rgba(255,255,255,.07);
}
.brand {padding:8px 4px 20px;margin-bottom:18px;border-bottom:1px solid #ffffff20;}
.brand-mark {
    display:inline-flex;width:39px;height:39px;align-items:center;justify-content:center;
    border-radius:12px;background:linear-gradient(135deg,#38bdf8,#4f46e5);
    margin-right:10px;vertical-align:middle;font-size:20px;
}
.brand-title {color:#fff;font-size:1.38rem;font-weight:800;letter-spacing:-.045em;}
.brand-sub {color:#9cabc2;font-size:.72rem;margin:8px 0 0 49px;}
.side-label {color:#71819c;font-size:.65rem;font-weight:800;letter-spacing:.15em;margin:8px 0 7px 4px;}
.hero {
    background:linear-gradient(115deg,#14233d 0%,#1b3153 62%,#243e68 100%);
    border:1px solid #243b5e;border-radius:20px;padding:29px 31px;margin-bottom:23px;
    box-shadow:0 12px 32px #1627431a;
}
.eyebrow {color:#7dd3fc;text-transform:uppercase;font-size:.67rem;font-weight:800;letter-spacing:.16em;margin-bottom:9px;}
.hero h1 {color:#fff;margin:0;font-size:2.05rem;line-height:1.16;letter-spacing:-.045em;}
.hero p {margin:10px 0 0;color:#c1cee1;font-size:.91rem;max-width:850px;line-height:1.6;}
.section-title {color:#172033;font-size:1.08rem;font-weight:750;margin:24px 0 12px;letter-spacing:-.025em;}
.metric-card,.info-card {
    background:#fff;border:1px solid #e3e9f2;border-radius:15px;padding:17px;
    box-shadow:0 4px 14px #18274b09;
}
.metric-card {min-height:112px;}
.metric-label {color:#64748b;font-size:.72rem;font-weight:650;}
.metric-value {color:#14213a;font-size:1.55rem;font-weight:800;margin-top:8px;overflow-wrap:anywhere;}
.metric-note {color:#8793a6;font-size:.67rem;margin-top:5px;line-height:1.35;}
.info-card h4 {color:#172033;margin:0 0 8px;font-size:1rem;}
.info-card p {color:#64748b;font-size:.83rem;line-height:1.65;}
.small-muted {color:#8390a3;font-size:.74rem;}
.footer-note {text-align:center;color:#8995a7;font-size:.68rem;margin-top:30px;padding-top:16px;border-top:1px solid #e3e9f2;}
.status-line {color:#b8c5d8;font-size:.72rem;margin-top:14px;}
.pill {display:inline-block;padding:4px 9px;border-radius:999px;background:#7dd3fc1f;color:#7dd3fc;font-size:.65rem;font-weight:700;margin:5px 4px 0 0;}
.stButton>button,.stDownloadButton>button {border-radius:10px;font-weight:700;border:1px solid #d7e0ec;min-height:42px;}
.stButton>button[kind="primary"] {background:linear-gradient(135deg,#2563eb,#4f46e5);border:0;}
[data-testid="stDataFrame"] {border:1px solid #e1e8f1;border-radius:12px;overflow:hidden;background:#fff;}
[data-testid="stAlert"] {border-radius:12px;}
.stTabs [data-baseweb="tab-list"] {gap:8px;border-bottom:1px solid #e1e8f1;}
.stTabs [data-baseweb="tab"] {background:#edf2f8;border-radius:9px 9px 0 0;padding:9px 14px;color:#475569;}
.stTabs [aria-selected="true"] {background:#dbeafe;color:#1d4ed8;}
footer {visibility:hidden;}
@media(max-width:768px){.hero{padding:21px}.hero h1{font-size:1.55rem}.metric-value{font-size:1.25rem}}
</style>
""", unsafe_allow_html=True)


# --------------------------- LOAD MODEL AND DATA ---------------------------
@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_artifacts():
    return load_artifacts()


try:
    df = get_data()
    model, meta = get_artifacts()
except Exception as exc:
    st.error(f"Could not load the dataset or model: {exc}")
    st.stop()

features = meta["features"]
missing_features = [feature for feature in features if feature not in df.columns]
if missing_features:
    st.error("The saved model expects columns missing from the dataset: " + ", ".join(missing_features))
    st.info("Check that the dataset and model artifacts were generated using the same preprocessing pipeline.")
    st.stop()

try:
    scored = df.copy()
    scored["churn_probability"] = model.predict_proba(df[features])[:, 1]
    scored["risk_level"] = scored["churn_probability"].apply(risk_level)
except Exception as exc:
    st.error(f"Could not score the dataset: {exc}")
    st.stop()


# --------------------------- HELPERS ---------------------------
def style_chart(fig, height=None):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(family="Inter, sans-serif", color="#334155"),
        margin=dict(l=12, r=12, t=58, b=12),
        legend_title_text="",
    )
    if height:
        fig.update_layout(height=height)
    return fig


def safe_percent(numerator, denominator):
    return float(numerator) / float(denominator) * 100 if denominator else 0.0


def hero(eyebrow, title, subtitle):
    st.markdown(
        f"""<div class="hero"><div class="eyebrow">{eyebrow}</div>
        <h1>{title}</h1><p>{subtitle}</p></div>""",
        unsafe_allow_html=True,
    )


def metric_card(label, value, note=""):
    st.markdown(
        f"""<div class="metric-card"><div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>""",
        unsafe_allow_html=True,
    )


def download_csv(dataframe, filename, label):
    st.download_button(
        label,
        dataframe.to_csv(index=False).encode("utf-8"),
        filename,
        "text/csv",
    )


# --------------------------- SIDEBAR ---------------------------
st.sidebar.markdown("""
<div class="brand">
  <span class="brand-mark">📊</span><span class="brand-title">ChurnIQ</span>
  <div class="brand-sub">Customer Intelligence Platform</div>
  <div class="status-line">🟢 Analytics engine ready</div>
</div>
<div class="side-label">WORKSPACE</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "WORKSPACE",
    [
        "Dashboard",
        "Predict Churn",
        "Customer Risk Center",
        "Model Performance",
        "Retention Strategies",
        "Retention & Growth Studio",
        "Analyze Your Dataset",
        "Project Info",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div class='footer-note'>ML-powered decision support<br>Customer analytics • Risk scoring • Retention</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<span class='pill'>Python</span><span class='pill'>ML</span><span class='pill'>Campaigns</span>",
    unsafe_allow_html=True,
)


# --------------------------- USER DATASET ANALYZER ---------------------------
if page == "Analyze Your Dataset":
    hero(
        "Self-Service Business Intelligence",
        "Analyze Your Customer Dataset",
        "Upload your customer data to generate churn-risk predictions, business insights, retention recommendations and downloadable reports.",
    )
    st.markdown("""
    <div class="info-card">
      <h4>Before you upload</h4>
      <p>For predictions, your file must contain every input feature expected by the trained ChurnIQ model, with compatible column names, data types, categories and units. Extra columns are allowed. The model cannot reliably score unrelated datasets without model retraining.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload customer dataset", type=["csv", "xlsx", "xls"],
                                help="Supported formats: CSV and Excel. Avoid uploading names, phone numbers, email addresses or other unnecessary personal information.")
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith('.csv'):
                user_df = pd.read_csv(uploaded)
            else:
                user_df = pd.read_excel(uploaded)
            if user_df.empty:
                st.error("The uploaded file contains no rows.")
            else:
                st.success(f"Dataset loaded: {len(user_df):,} rows and {len(user_df.columns):,} columns.")
                st.markdown("<div class='section-title'>1. Dataset quality check</div>", unsafe_allow_html=True)
                q1, q2, q3, q4 = st.columns(4)
                with q1: metric_card("Rows", f"{len(user_df):,}", "Customer records")
                with q2: metric_card("Columns", f"{len(user_df.columns):,}", "Available fields")
                with q3: metric_card("Missing Cells", f"{int(user_df.isna().sum().sum()):,}", "Values requiring attention")
                with q4: metric_card("Duplicate Rows", f"{int(user_df.duplicated().sum()):,}", "Exact duplicate records")
                st.dataframe(user_df.head(10), width="stretch", hide_index=True)

                missing_features_user = [f for f in features if f not in user_df.columns]
                extra_features_user = [c for c in user_df.columns if c not in features]
                st.markdown("<div class='section-title'>2. Model compatibility</div>", unsafe_allow_html=True)
                if missing_features_user:
                    st.error("This file is missing model input columns: " + ", ".join(missing_features_user))
                    st.warning("ChurnIQ cannot make valid predictions for this file yet. Rename or add the missing fields using the same definitions and units used to train the model. Do not fill unknown business fields with arbitrary values.")
                    st.caption("Expected model columns: " + ", ".join(features))
                else:
                    st.success("All required model input columns are present. Extra columns will be retained in the downloadable report.")
                    if extra_features_user:
                        st.caption("Additional columns detected (not used by the model): " + ", ".join(map(str, extra_features_user[:25])))

                    # Score only the exact trained feature set, in the model's expected order.
                    try:
                        scoring_input = user_df[features].copy()
                        user_scored = user_df.copy()
                        user_scored["churn_probability"] = model.predict_proba(scoring_input)[:, 1]
                        user_scored["risk_level"] = user_scored["churn_probability"].apply(risk_level)
                        user_scored["churn_probability_pct"] = (user_scored["churn_probability"] * 100).round(2)

                        # Allow the user to identify the customer key and business revenue field.
                        col_a, col_b = st.columns(2)
                        with col_a:
                            id_options = ["(row number)"] + list(user_df.columns)
                            id_col = st.selectbox("Customer identifier column", id_options, index=(id_options.index("customer_id") if "customer_id" in id_options else 0))
                        with col_b:
                            numeric_options = ["(not available)"] + [c for c in user_df.select_dtypes(include="number").columns if c not in features or c in ["monthly_charges", "MonthlyCharges"]]
                            default_rev = "monthly_charges" if "monthly_charges" in numeric_options else ("MonthlyCharges" if "MonthlyCharges" in numeric_options else "(not available)")
                            revenue_col = st.selectbox("Monthly revenue / customer value column (optional)", numeric_options, index=numeric_options.index(default_rev))

                        # Attach clear recommendations to every scored customer.
                        def make_actions(row):
                            try:
                                return " | ".join(retention_strategy(row.to_dict(), float(row["churn_probability"])))
                            except Exception:
                                level = row["risk_level"]
                                if level == "HIGH":
                                    return "Contact customer promptly; investigate service issues; consider a tailored retention offer."
                                if level == "MEDIUM":
                                    return "Send a relevant engagement message; review satisfaction; monitor risk."
                                return "Maintain service quality; continue normal engagement and periodic monitoring."

                        user_scored["recommended_actions"] = user_scored.apply(make_actions, axis=1)
                        high_u = user_scored[user_scored["risk_level"] == "HIGH"]
                        medium_u = user_scored[user_scored["risk_level"] == "MEDIUM"]
                        low_u = user_scored[user_scored["risk_level"] == "LOW"]
                        actual_col = next((c for c in ["churn", "Churn", "Exited", "is_churn"] if c in user_scored.columns), None)

                        st.markdown("<div class='section-title'>3. Churn prediction summary</div>", unsafe_allow_html=True)
                        k1, k2, k3, k4 = st.columns(4)
                        with k1: metric_card("Customers Analyzed", f"{len(user_scored):,}", "Successfully scored records")
                        with k2: metric_card("High Risk", f"{len(high_u):,}", "Priority intervention")
                        with k3: metric_card("Medium Risk", f"{len(medium_u):,}", "Proactive engagement")
                        with k4: metric_card("Low Risk", f"{len(low_u):,}", "Routine monitoring")

                        chart_left, chart_right = st.columns(2)
                        with chart_left:
                            risk_counts = user_scored["risk_level"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"]).fillna(0).reset_index()
                            risk_counts.columns = ["Risk Level", "Customers"]
                            fig = px.pie(risk_counts, names="Risk Level", values="Customers", hole=0.58, title="Predicted Customer Risk Distribution")
                            st.plotly_chart(style_chart(fig), width="stretch")
                        with chart_right:
                            fig = px.histogram(user_scored, x="churn_probability_pct", nbins=20, title="Distribution of Predicted Churn Probability", labels={"churn_probability_pct": "Predicted churn probability (%)"})
                            st.plotly_chart(style_chart(fig), width="stretch")

                        if actual_col:
                            st.markdown("<div class='section-title'>4. Historical churn overview</div>", unsafe_allow_html=True)
                            vals = user_scored[actual_col].astype(str).str.strip().str.lower()
                            yes_vals = {"yes", "1", "true", "churn", "churned", "left", "exited"}
                            no_vals = {"no", "0", "false", "stay", "stayed", "retained", "not churned"}
                            mapped = vals.map(lambda x: 1 if x in yes_vals else (0 if x in no_vals else pd.NA))
                            if mapped.notna().any():
                                actual_rate = float(mapped.dropna().mean() * 100)
                                metric_card("Observed Churn Rate", f"{actual_rate:.1f}%", f"Calculated from recognized values in '{actual_col}'")
                            else:
                                st.info(f"A possible churn label column '{actual_col}' was found, but its values were not recognized. Historical churn rate was not calculated.")
                        else:
                            st.info("No recognized historical churn label column was found. Predictions are still available, but actual churn rate and model accuracy cannot be calculated from this file.")

                        if revenue_col != "(not available)":
                            revenue_values = pd.to_numeric(user_scored[revenue_col], errors="coerce").fillna(0).clip(lower=0)
                            weighted_risk = float((revenue_values * user_scored["churn_probability"]).sum())
                            st.markdown("<div class='section-title'>5. Revenue exposure estimate</div>", unsafe_allow_html=True)
                            metric_card("Estimated Revenue at Risk", f"{weighted_risk:,.2f}", f"Sum of {revenue_col} × predicted churn probability; currency/unit follows your uploaded data")
                        else:
                            st.info("Select a monthly revenue/customer-value column to estimate revenue exposed to churn. No currency is assumed by ChurnIQ.")

                        st.markdown("<div class='section-title'>6. Customer-level recommendations</div>", unsafe_allow_html=True)
                        risk_filter = st.multiselect("Show risk groups", ["HIGH", "MEDIUM", "LOW"], ["HIGH", "MEDIUM", "LOW"], key="upload_risk_filter")
                        filtered = user_scored[user_scored["risk_level"].isin(risk_filter)].copy().sort_values("churn_probability", ascending=False)
                        if id_col != "(row number)":
                            filtered.insert(0, "Customer Reference", filtered[id_col].astype(str))
                        else:
                            filtered.insert(0, "Customer Reference", [str(i + 1) for i in filtered.index])
                        report_cols = ["Customer Reference", "churn_probability_pct", "risk_level", "recommended_actions"]
                        if revenue_col != "(not available)": report_cols.insert(3, revenue_col)
                        st.dataframe(filtered[report_cols], width="stretch", hide_index=True)
                        st.download_button("Download complete prediction & recommendation report", user_scored.to_csv(index=False).encode("utf-8"), "churniq_customer_analysis.csv", "text/csv", type="primary")

                        st.markdown("<div class='section-title'>7. Retention campaign planner</div>", unsafe_allow_html=True)
                        camp_a, camp_b = st.columns(2)
                        with camp_a:
                            campaign_segment = st.selectbox("Campaign target", ["HIGH", "MEDIUM", "HIGH + MEDIUM"], key="upload_campaign_segment")
                        with camp_b:
                            discount_pct = st.slider("Proposed discount (%)", 0, 50, 10, 5, key="upload_discount")
                        selected_campaign = user_scored[user_scored["risk_level"].isin(["HIGH", "MEDIUM"] if campaign_segment == "HIGH + MEDIUM" else [campaign_segment])].copy()
                        st.write(f"**Customers selected:** {len(selected_campaign):,}")
                        if not selected_campaign.empty:
                            if revenue_col != "(not available)":
                                selected_value = pd.to_numeric(selected_campaign[revenue_col], errors="coerce").fillna(0).clip(lower=0)
                                success = st.slider("Assumed campaign success rate (%)", 5, 80, 25, 5, key="upload_success")
                                protected = float((selected_value * selected_campaign["churn_probability"]).sum()) * success / 100
                                cost = float(selected_value.sum()) * discount_pct / 100
                                net = protected - cost
                                a, b, c = st.columns(3)
                                with a: metric_card("Potential Value Protected", f"{protected:,.2f}", "Scenario estimate, not guaranteed")
                                with b: metric_card("Estimated Discount Cost", f"{cost:,.2f}", "One-period discount assumption")
                                with c: metric_card("Estimated Net Impact", f"{net:,.2f}", "Protected value minus discount cost")
                            else:
                                st.caption("Add/select a revenue column to calculate campaign economics. You can still export the target customer list.")
                            st.download_button("Download selected campaign list", selected_campaign.to_csv(index=False).encode("utf-8"), "churniq_targeted_campaign.csv", "text/csv")
                        else:
                            st.info("No customers match this campaign segment.")
                        st.caption("Important: these are model-based estimates and rule-based recommendations. Validate model fit, data definitions and business assumptions before contacting customers or offering incentives. Uploaded data is analyzed for this session and is not saved to a permanent customer database by this page.")
                    except Exception as exc:
                        st.error(f"The uploaded data could not be scored by the current model: {exc}")
                        st.info("Check that feature names, category values, data types and units match the dataset used to train ChurnIQ. If your business uses a different schema, the model needs a separate training/preprocessing workflow.")
        except Exception as exc:
            st.error(f"Could not read this file: {exc}")
            st.info("Please upload a valid CSV or Excel file.")


# --------------------------- DASHBOARD ---------------------------
elif page == "Dashboard":
    hero(
        "Customer Intelligence",
        "Customer Churn Prediction & Retention Strategy System",
        "A professional analytics workspace for identifying churn risk and prioritizing retention actions.",
    )
    st.markdown("<div class='small-muted'>Overview · Predictive risk · Retention intelligence</div>", unsafe_allow_html=True)

    total = len(df)
    churn = int(df["churn"].sum()) if "churn" in df.columns else 0
    churn_rate = safe_percent(churn, total)
    high = int((scored["risk_level"] == "HIGH").sum())
    avg_monthly = float(df["monthly_charges"].mean()) if "monthly_charges" in df.columns else 0.0

    cols = st.columns(5)
    values = [
        ("Total Customers", f"{total:,}", "Customers analyzed"),
        ("Actual Churned", f"{churn:,}", "Historical churn"),
        ("Churn Rate", f"{churn_rate:.1f}%", "Actual dataset rate"),
        ("High Risk", f"{high:,}", "Priority customers"),
        ("Avg. Monthly", f"${avg_monthly:,.2f}", "Average monthly charge; currency not verified"),
    ]
    for col, (label, value, note) in zip(cols, values):
        with col:
            metric_card(label, value, note)

    st.markdown("<div class='section-title'>Business Overview</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if {"contract", "churn"}.issubset(df.columns):
            x = df.groupby("contract", as_index=False)["churn"].mean()
            x["rate"] = x["churn"] * 100
            fig = px.bar(x, x="contract", y="rate", text=x["rate"].round(1),
                         title="Churn Rate by Contract",
                         labels={"rate": "Churn Rate (%)", "contract": "Contract"})
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            st.plotly_chart(style_chart(fig), width="stretch")
        else:
            st.info("Contract or churn column is unavailable in this dataset.")

    with c2:
        x = scored["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).fillna(0).reset_index()
        x.columns = ["risk_level", "customers"]
        fig = px.pie(x, names="risk_level", values="customers", hole=0.58,
                     title="Predicted Customer Risk Distribution")
        st.plotly_chart(style_chart(fig), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        if {"internet_service", "churn"}.issubset(df.columns):
            x = df.groupby("internet_service", as_index=False)["churn"].mean()
            x["rate"] = x["churn"] * 100
            fig = px.bar(x, x="internet_service", y="rate", title="Churn Rate by Internet Service",
                         labels={"rate": "Churn Rate (%)", "internet_service": "Internet Service"})
            st.plotly_chart(style_chart(fig), width="stretch")
        else:
            st.info("Internet service or churn column is unavailable.")

    with c4:
        required = {"tenure", "monthly_charges", "customer_id"}
        if required.issubset(scored.columns):
            sample = scored.sample(min(1800, len(scored)), random_state=42)
            hover_cols = [col for col in ["customer_id", "contract", "churn_probability"] if col in sample.columns]
            fig = px.scatter(sample, x="tenure", y="monthly_charges", color="risk_level",
                             hover_data=hover_cols, title="Customer Risk Map",
                             labels={"tenure": "Tenure (months)", "monthly_charges": "Monthly Charges"})
            st.plotly_chart(style_chart(fig), width="stretch")
        else:
            st.info("Required columns for the customer risk map are unavailable.")

    st.markdown("<div class='section-title'>Top Priority Customers</div>", unsafe_allow_html=True)
    top = scored.nlargest(10, "churn_probability").copy()
    top["churn_probability"] = (top["churn_probability"] * 100).round(1).astype(str) + "%"
    top = top.rename(columns={
        "customer_id": "Customer ID", "tenure": "Tenure",
        "monthly_charges": "Monthly Charges", "contract": "Contract",
        "tech_support": "Tech Support", "churn_probability": "Churn Risk",
        "risk_level": "Risk",
    })
    desired = ["Customer ID", "Tenure", "Monthly Charges", "Contract", "Tech Support", "Churn Risk", "Risk"]
    available = [col for col in desired if col in top.columns]
    st.dataframe(top[available], width="stretch", hide_index=True)


# --------------------------- PREDICT CHURN ---------------------------
elif page == "Predict Churn":
    hero("Predictive Analytics", "Predict Individual Churn Risk",
         "Enter customer characteristics to estimate churn probability and generate a targeted retention plan.")

    with st.form("prediction_form"):
        st.markdown("<div class='section-title'>Customer Profile</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            cid = st.text_input("Customer ID", "NEW-CUSTOMER")
            tenure = st.number_input("Tenure (months)", 0, 120, 12)
            monthly = st.number_input("Monthly Charges", 0.0, 500.0, 70.0)
            senior = st.selectbox("Senior Citizen", [0, 1])
        with c2:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            support = st.selectbox("Tech Support", ["Yes", "No"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer", "Credit card"
            ])
        with c3:
            partner = st.selectbox("Partner", ["Yes", "No"])
            paper = st.selectbox("Paperless Billing", ["Yes", "No"])
            st.markdown(
                "<div class='info-card'><h4>How it works</h4><p>The trained classification model estimates churn probability. The rule engine maps the score to HIGH, MEDIUM or LOW risk.</p></div>",
                unsafe_allow_html=True,
            )
        submitted = st.form_submit_button("🚀 Analyze Customer Risk", type="primary", width="stretch")

    if submitted:
        submitted_profile = {
            "tenure": tenure, "monthly_charges": monthly, "contract": contract,
            "internet_service": internet, "tech_support": support,
            "payment_method": payment, "senior_citizen": senior,
            "partner": partner, "paperless_billing": paper,
        }
        base_row = {}
        for col in df.columns:
            if col == "churn":
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                base_row[col] = df[col].median() if df[col].notna().any() else 0
            else:
                modes = df[col].dropna().mode()
                base_row[col] = modes.iloc[0] if not modes.empty else ""
        base_row.update({key: value for key, value in submitted_profile.items() if key in df.columns})
        row = pd.DataFrame([base_row])
        missing = [feature for feature in features if feature not in row.columns]
        if missing:
            st.error("Cannot score this profile because model features are missing: " + ", ".join(missing))
        else:
            try:
                probability = float(model.predict_proba(row[features])[0, 1])
                risk = risk_level(probability)
                actions = retention_strategy(row.iloc[0].to_dict(), probability)
                st.session_state["prediction_result"] = {
                    "cid": cid, "p": probability, "risk": risk, "actions": actions,
                }
            except Exception as exc:
                st.error(f"Prediction could not be generated: {exc}")
                st.caption("Check that the form values use the same encoding and preprocessing as the training pipeline.")

    result = st.session_state.get("prediction_result")
    if result:
        if st.button("Clear current result"):
            st.session_state.pop("prediction_result", None)
            st.rerun()

        st.markdown("<div class='section-title'>Prediction Result</div>", unsafe_allow_html=True)
        probability, risk = result["p"], result["risk"]
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Churn Probability", f"{probability * 100:.1f}%", "Model-estimated probability")
        with c2:
            metric_card("Risk Level", risk, "Priority classification")
        with c3:
            metric_card("Customer", result["cid"], "Prediction reference")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=probability * 100, number={"suffix": "%"},
            title={"text": "Churn Risk Score"},
            gauge={
                "axis": {"range": [0, 100]}, "bar": {"thickness": 0.32},
                "steps": [{"range": [0, 30]}, {"range": [30, 60]}, {"range": [60, 100]}],
                "threshold": {"line": {"width": 4}, "value": probability * 100},
            },
        ))
        gauge.update_layout(template="plotly_white", height=300,
                            paper_bgcolor="rgba(255,255,255,0)",
                            font=dict(color="#334155"), margin=dict(l=20, r=20, t=60, b=10))
        st.plotly_chart(gauge, width="stretch")

        st.markdown("<div class='section-title'>Recommended Retention Actions</div>", unsafe_allow_html=True)
        for index, action in enumerate(result["actions"], 1):
            st.markdown(f"**{index}.** {action}")

        if st.button("💾 Save Prediction to Session History"):
            if "prediction_history" not in st.session_state:
                st.session_state["prediction_history"] = []
            st.session_state["prediction_history"].append({
                "Customer ID": result["cid"],
                "Churn Probability (%)": round(result["p"] * 100, 2),
                "Risk Level": result["risk"],
                "Model": meta.get("best_model", "Unknown"),
                "Recommended Actions": " | ".join(result["actions"]),
            })
            st.success("Prediction saved to this session's history.")

    if st.session_state.get("prediction_history"):
        st.markdown("<div class='section-title'>Prediction History</div>", unsafe_allow_html=True)
        history_df = pd.DataFrame(st.session_state["prediction_history"])
        st.dataframe(history_df, width="stretch", hide_index=True)
        download_csv(history_df, "churniq_prediction_history.csv", "Download Prediction History")


# --------------------------- CUSTOMER RISK CENTER ---------------------------
elif page == "Customer Risk Center":
    hero("Risk Operations", "Customer Risk Center",
         "Filter, prioritize and export customers who need retention attention.")

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        threshold = st.slider("Minimum churn probability", 0.0, 1.0, 0.30, 0.05)
    with c2:
        risks = st.multiselect("Risk Level", ["HIGH", "MEDIUM", "LOW"], ["HIGH", "MEDIUM"])
    with c3:
        search = st.text_input("Search Customer ID", placeholder="e.g. CUST-001")

    view = scored[(scored["churn_probability"] >= threshold) & (scored["risk_level"].isin(risks))]
    if search and "customer_id" in view.columns:
        view = view[view["customer_id"].astype(str).str.contains(search, case=False, na=False)]
    view = view.copy().sort_values("churn_probability", ascending=False)
    view["churn_probability_pct"] = (view["churn_probability"] * 100).round(2)

    k1, k2, k3 = st.columns(3)
    with k1:
        metric_card("Matching Customers", f"{len(view):,}", "Customers passing current filters")
    with k2:
        mean_risk = f"{view['churn_probability'].mean() * 100:.1f}%" if len(view) else "—"
        metric_card("Mean Predicted Risk", mean_risk, "Average among filtered customers")
    with k3:
        high_count = int((view["risk_level"] == "HIGH").sum())
        metric_card("High-Risk Segment", f"{high_count:,}", "High-risk customers in this view")

    st.caption("Results are ordered from highest to lowest predicted churn probability.")
    display_cols = [
        "customer_id", "tenure", "monthly_charges", "contract", "internet_service",
        "tech_support", "payment_method", "churn_probability_pct", "risk_level",
    ]
    display_cols = [col for col in display_cols if col in view.columns]
    display = view[display_cols].rename(columns={
        "customer_id": "Customer ID", "tenure": "Tenure",
        "monthly_charges": "Monthly Charges", "contract": "Contract",
        "internet_service": "Internet Service", "tech_support": "Tech Support",
        "payment_method": "Payment Method", "churn_probability_pct": "Churn Probability %",
        "risk_level": "Risk",
    })
    st.dataframe(display, width="stretch", hide_index=True)

    report = view.copy()
    report["recommended_actions"] = report.apply(
        lambda row: " | ".join(retention_strategy(row.to_dict(), row["churn_probability"])),
        axis=1,
    )
    download_csv(report, "customer_risk_report.csv", "⬇️ Download Risk & Retention Report")


# --------------------------- MODEL PERFORMANCE ---------------------------
elif page == "Model Performance":
    hero("Machine Learning", "Model Performance",
         "Transparent comparison of classification models used for churn prediction.")

    metrics = pd.DataFrame(meta.get("metrics", []))
    if metrics.empty:
        st.warning("No model-comparison metrics were found in saved metadata.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Selected Model", meta.get("best_model", "Unknown"), "Highest validation ROC-AUC")
    with c2:
        mean_auc, std_auc = meta.get("cv_roc_auc_mean"), meta.get("cv_roc_auc_std")
        metric_card("5-Fold ROC-AUC", f"{mean_auc:.3f}" if mean_auc is not None else "N/A",
                    f"± {std_auc:.3f}" if std_auc is not None else "Cross-validation")
    with c3:
        metric_card("Models Compared", str(len(metrics)), "Classification algorithms")

    st.markdown("<div class='section-title'>Comparison Matrix</div>", unsafe_allow_html=True)
    shown = metrics.copy()
    shown.columns = [column.replace("_", " ").title() for column in shown.columns]
    st.dataframe(shown.round(4), width="stretch", hide_index=True)

    metric_cols = [col for col in ["accuracy", "precision", "recall", "f1", "roc_auc"] if col in metrics.columns]
    if "model" not in metrics.columns or not metric_cols:
        st.warning("The metrics file does not contain the expected model/metric columns.")
        st.stop()

    chart_data = metrics.melt(id_vars="model", value_vars=metric_cols,
                              var_name="metric", value_name="score")
    fig = px.bar(chart_data, x="model", y="score", color="metric",
                 barmode="group", title="Model Metric Comparison")
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(style_chart(fig), width="stretch")

    if FI_PATH.exists():
        importance = pd.read_csv(FI_PATH)
        if {"feature", "importance"}.issubset(importance.columns):
            importance = importance.head(15).sort_values("importance")
            fig = px.bar(importance, x="importance", y="feature", orientation="h",
                         title="Top Predictive Features")
            st.plotly_chart(style_chart(fig), width="stretch")


# --------------------------- RETENTION STRATEGIES ---------------------------
elif page == "Retention Strategies":
    hero("Decision Support", "Retention Strategy Engine",
         "Convert predicted risk into practical, customer-focused retention actions.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='info-card'><h4 style='color:#e11d48'>HIGH RISK</h4><p>Proactive contact, personalized incentive, service review and long-term plan conversion.</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='info-card'><h4 style='color:#b77908'>MEDIUM RISK</h4><p>Targeted engagement, personalized offers and timely follow-up before the risk increases.</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='info-card'><h4 style='color:#059669'>LOW RISK</h4><p>Maintain service quality, regular engagement and lightweight monitoring.</p></div>", unsafe_allow_html=True)

    example = scored.nlargest(1, "churn_probability").iloc[0]
    st.markdown("<div class='section-title'>Highest-Risk Customer Example</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        metric_card("Customer", str(example.get("customer_id", "Unknown")), f"Risk: {example['risk_level']}")
    with c2:
        metric_card("Churn Probability", f"{example['churn_probability'] * 100:.1f}%", "Priority retention case")
    for action in retention_strategy(example.to_dict(), example["churn_probability"]):
        st.markdown(f"• {action}")


# --------------------------- RETENTION & GROWTH STUDIO ---------------------------
elif page == "Retention & Growth Studio":
    hero("Business Growth Intelligence", "Retention & Growth Studio",
         "Build targeted retention campaigns, explore business scenarios and identify revenue exposed to customer churn.")

    if "monthly_charges" not in scored.columns:
        st.warning("Monthly charges are unavailable in this dataset.")
        st.stop()

    total_at_risk = float((scored["monthly_charges"] * scored["churn_probability"]).sum())
    high_risk = scored[scored["risk_level"] == "HIGH"]
    medium_risk = scored[scored["risk_level"] == "MEDIUM"]

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Estimated Monthly Revenue at Risk", f"${total_at_risk:,.2f}",
                    "Probability-weighted estimate; not guaranteed loss")
    with c2:
        metric_card("High-Risk Customers", f"{len(high_risk):,}", "Customers requiring priority attention")
    with c3:
        metric_card("Medium-Risk Customers", f"{len(medium_risk):,}", "Customers for proactive engagement")

    st.markdown("<div class='section-title'>1. Retention Campaign Builder</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        segment = st.selectbox("Choose target customer segment", ["HIGH", "MEDIUM", "LOW", "HIGH + MEDIUM"])
    with c2:
        discount = st.slider("Proposed one-month discount (%)", 0, 50, 10, 5)

    if segment == "HIGH + MEDIUM":
        campaign = scored[scored["risk_level"].isin(["HIGH", "MEDIUM"])].copy()
    else:
        campaign = scored[scored["risk_level"] == segment].copy()

    st.write(f"**Customers selected for campaign:** {len(campaign):,}")
    if not campaign.empty:
        campaign["recommended_actions"] = campaign.apply(
            lambda row: " | ".join(retention_strategy(row.to_dict(), row["churn_probability"])),
            axis=1,
        )
        campaign["churn_probability_pct"] = (campaign["churn_probability"] * 100).round(2)
        campaign["proposed_discount_pct"] = discount
        campaign_cols = [
            "customer_id", "risk_level", "churn_probability_pct", "monthly_charges",
            "proposed_discount_pct", "recommended_actions",
        ]
        campaign_cols = [col for col in campaign_cols if col in campaign.columns]
        st.dataframe(campaign[campaign_cols], width="stretch", hide_index=True)
        download_csv(campaign, "churniq_retention_campaign.csv", "Download Retention Campaign CSV")
    else:
        st.info("No customers were found in this segment.")

    st.markdown("<div class='section-title'>2. What-If Retention Simulator</div>", unsafe_allow_html=True)
    st.caption("Change the assumptions to explore a hypothetical one-month retention campaign. These are scenario estimates, not model predictions.")
    success_rate = st.slider("Assumed campaign success rate (%)", 5, 80, 25, 5)

    if not campaign.empty:
        revenue_exposure = float((campaign["monthly_charges"] * campaign["churn_probability"]).sum())
        estimated_protected = revenue_exposure * success_rate / 100
        discount_cost = float(campaign["monthly_charges"].sum() * discount / 100)
        estimated_net = estimated_protected - discount_cost

        a, b, c = st.columns(3)
        with a:
            metric_card("Potential Revenue Protected", f"${estimated_protected:,.2f}",
                        "Based on the assumed campaign success rate")
        with b:
            metric_card("Estimated Discount Cost", f"${discount_cost:,.2f}",
                        "One-month discount for all selected customers")
        with c:
            metric_card("Estimated Net Impact", f"${estimated_net:,.2f}",
                        "Potential revenue protected minus discount cost")

        if estimated_net > 0:
            st.success("Under these assumptions, the campaign has a positive estimated net impact.")
        else:
            st.warning("Under these assumptions, the campaign cost is higher than the estimated revenue protected.")

    st.markdown("<div class='section-title'>3. Customer Retention Insights</div>", unsafe_allow_html=True)
    if {"contract", "churn_probability"}.issubset(scored.columns):
        contract_insights = scored.groupby("contract").agg(
            customers=("churn_probability", "size"),
            average_churn_risk=("churn_probability", "mean"),
        ).reset_index()
        contract_insights["average_churn_risk"] *= 100

        fig = px.bar(
            contract_insights, x="contract", y="average_churn_risk",
            text=contract_insights["average_churn_risk"].round(1),
            title="Average Predicted Churn Risk by Contract",
            labels={"contract": "Contract Type", "average_churn_risk": "Average Predicted Risk (%)"},
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(style_chart(fig), width="stretch")
        st.dataframe(
            contract_insights.rename(columns={
                "contract": "Contract", "customers": "Customers",
                "average_churn_risk": "Average Risk (%)",
            }).round(2),
            width="stretch", hide_index=True,
        )


# --------------------------- PROJECT INFO ---------------------------
else:
    hero("Project Documentation", "Customer Churn Prediction & Retention Strategy System",
         "An academic machine-learning application combining prediction, risk segmentation and retention decision support.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="info-card">
        <h4>Problem Statement</h4>
        <p>Businesses may lose customers without early warning. Manual analysis is slow and does not consistently prioritize customers who need intervention.</p>
        <h4>Objectives</h4>
        <p>• Preprocess customer data<br>
        • Compare classification models<br>
        • Predict individual churn probability<br>
        • Classify customer risk<br>
        • Generate retention recommendations<br>
        • Create retention campaign reports</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="info-card">
        <h4>Technology Stack</h4>
        <p>Python • Pandas • NumPy • Scikit-learn • Logistic Regression • Random Forest • HistGradientBoosting • Streamlit • Plotly • Joblib • Campaign Analytics</p>
        <h4>System Flow</h4>
        <p><b>Customer Data → Preprocessing → Model Comparison → Best Model → Churn Probability → Risk Level → Retention Strategy → Dashboard / Campaign Reports</b></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Project Highlights</div>", unsafe_allow_html=True)
    st.markdown("""
    - **Predictive analytics:** estimates the probability that a customer will churn.
    - **Model comparison:** evaluates classifiers using Accuracy, Precision, Recall, F1 and ROC-AUC.
    - **Risk segmentation:** categorizes customers into LOW, MEDIUM and HIGH risk.
    - **Decision support:** generates rule-based retention recommendations.
    - **Campaign management:** exports customer-level retention recommendations and prediction reports.
    - **Scenario analysis:** estimates potential campaign outcomes using user-defined assumptions.
    """)

st.markdown(
    "<div class='footer-note'>ChurnIQ · Decision-support analytics · Predictions should be reviewed before customer action</div>",
    unsafe_allow_html=True,
)
