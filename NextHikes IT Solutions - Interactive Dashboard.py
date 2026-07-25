"""
Retail Transaction Analytics Dashboard
NextHikes IT Solutions — Internship Project
Interactive companion to the EDA notebook (trial2__3_.ipynb)
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from datetime import timedelta

try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score,
                                  precision_score, recall_score, confusion_matrix)
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Transaction Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Earthy palette — matches the notebook's visual identity
# ─────────────────────────────────────────────────────────────
RUST, BURNT_SIENNA, DUSTY_ROSE = "#B5451B", "#C4683A", "#C4907A"
POUPON, CANVAS = "#C9A84C", "#E0D5BB"
SAGE, FERN, GORGE = "#7D9E7A", "#5A7A52", "#3B4F35"
PALETTE = [RUST, BURNT_SIENNA, POUPON, SAGE, FERN, GORGE, DUSTY_ROSE]
SEG_COLORS = {"New": POUPON, "Regular": FERN, "Premium": GORGE}
RETURN_COLORS = {"No": FERN, "Yes": RUST}

st.markdown(f"""
<style>
    .stApp {{ background-color: #F8F9FA; }}
    [data-testid="stMetricValue"] {{ color: {GORGE}; font-weight: 700; }}
    [data-testid="stMetricLabel"] {{ color: {GORGE}; }}
    h1, h2, h3 {{ color: {GORGE}; }}
    .stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
</style>
""", unsafe_allow_html=True)

LAYOUT_KWARGS = dict(
    paper_bgcolor="#F8F9FA", plot_bgcolor="#F8F9FA",
    font=dict(color=GORGE, family="sans-serif"),
    title_font=dict(size=16, color=GORGE),
    colorway=PALETTE,
)

NUM_COLS = ["age", "product_price", "quantity", "discount_percentage",
            "final_price", "delivery_days", "revenue_per_unit", "discount_impact"]
CAT_COLS = ["gender", "customer_segment", "product_category",
            "payment_method", "shipping_type", "price_bucket", "return_status"]
GEO_COLS = ["state", "city"]
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ─────────────────────────────────────────────────────────────
# Statistical-rigor helpers (Benjamini-Hochberg, bootstrap effect sizes,
# ANOVA power / minimum detectable effect) — no statsmodels dependency
# ─────────────────────────────────────────────────────────────
def benjamini_hochberg(pvals):
    """Return BH-adjusted p-values (same order as input), no statsmodels needed."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * n / (np.arange(1, n + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]  # enforce monotonicity
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out


def bootstrap_cramers_v(codes1, codes2, n_boot=200, seed=42):
    """Vectorized bootstrap CI for Cramér's V between two label-encoded arrays."""
    n = len(codes1)
    k1, k2 = codes1.max() + 1, codes2.max() + 1
    rng = np.random.RandomState(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        c1, c2 = codes1[idx], codes2[idx]
        flat = c1 * k2 + c2
        table = np.bincount(flat, minlength=k1 * k2).reshape(k1, k2).astype(float)
        row_sums, col_sums = table.sum(1, keepdims=True), table.sum(0, keepdims=True)
        total = table.sum()
        expected = row_sums @ col_sums / total
        chi2 = np.sum((table - expected) ** 2 / np.where(expected == 0, 1, expected))
        vals[i] = np.sqrt(chi2 / (total * (min(k1, k2) - 1)))
    return vals


def bootstrap_eta_squared(codes, num_vals, n_boot=200, seed=42):
    """Vectorized bootstrap CI for eta-squared (one-way ANOVA effect size)."""
    n = len(codes)
    k = codes.max() + 1
    rng = np.random.RandomState(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        c, x = codes[idx], num_vals[idx]
        counts = np.bincount(c, minlength=k)
        sums = np.bincount(c, weights=x, minlength=k)
        means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        grand_mean = x.mean()
        ss_between = np.sum(counts * (means - grand_mean) ** 2)
        ss_total = np.sum((x - grand_mean) ** 2)
        vals[i] = ss_between / ss_total if ss_total > 0 else 0
    return vals


def anova_power(cohens_f, n, k, alpha=0.05):
    """Achieved power of a one-way ANOVA via the noncentral F distribution."""
    df1, df2 = k - 1, n - k
    ncp = cohens_f ** 2 * n
    f_crit = stats.f.ppf(1 - alpha, df1, df2)
    return 1 - stats.ncf.cdf(f_crit, df1, df2, ncp)


def anova_minimum_detectable_effect(n, k, alpha=0.05, target_power=0.8):
    """Smallest Cohen's f detectable at target_power via root-finding."""
    from scipy.optimize import brentq

    def gap(f):
        return anova_power(f, n, k, alpha) - target_power

    try:
        return brentq(gap, 1e-5, 2.0)
    except ValueError:
        return np.nan


def load_data(path="retail_large_dataset.csv"):
    raw = pd.read_csv(path, parse_dates=["order_date"])
    df = raw.copy()
    df["returned"] = (df["return_status"] == "Yes").astype(int)
    df["age_group"] = pd.cut(df["age"], bins=[17, 30, 45, 65],
                              labels=["18–30", "30–45", "45–65"])
    df["disc_bucket"] = pd.cut(df["discount_percentage"], bins=[-1, 10, 20, 30],
                                labels=["Low (0–10%)", "Mid (11–20%)", "High (21–30%)"])
    df["order_month"] = df["order_date"].dt.to_period("M").astype(str)
    df["order_quarter"] = df["order_date"].dt.to_period("Q").astype(str)
    df["order_year"] = df["order_date"].dt.year
    df["order_dow"] = df["order_date"].dt.day_name()
    df["order_dom"] = df["order_date"].dt.day
    df["high_value"] = (df["final_price"] > df["final_price"].quantile(0.90)).astype(int)
    df["revenue_per_unit"] = df["final_price"] / df["quantity"]
    df["discount_impact"] = df["product_price"] * df["quantity"] - df["final_price"]
    df["price_bucket"] = pd.qcut(df["product_price"], 4,
                                  labels=["Low", "Mid", "High", "Premium"])
    return df, raw

try:
    with st.spinner("Loading transaction data..."):
        df_full, raw_full = load_data()
except FileNotFoundError:
    st.error(
        "Couldn't find **retail_large_dataset.csv**. Make sure the file sits in the "
        "same folder as this script (or pass its path to `load_data()`), then rerun."
    )
    st.stop()
except Exception as e:
    st.error(f"Something went wrong while loading the data: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────
# Sidebar filters
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🛒 Filters")


def _reset_filters():
    for k in ("f_date", "f_segment", "f_category", "f_payment",
              "f_shipping", "f_gender", "f_age"):
        st.session_state.pop(k, None)


st.sidebar.button("↺ Reset all filters", on_click=_reset_filters, use_container_width=True)

date_min, date_max = df_full["order_date"].min(), df_full["order_date"].max()
date_range = st.sidebar.date_input(
    "Order date range", value=(date_min, date_max),
    min_value=date_min, max_value=date_max, key="f_date"
)

age_min, age_max = int(df_full["age"].min()), int(df_full["age"].max())
age_range = st.sidebar.slider(
    "Customer age", min_value=age_min, max_value=age_max,
    value=(age_min, age_max), key="f_age"
)

segment_sel = st.sidebar.multiselect(
    "Customer segment", options=sorted(df_full["customer_segment"].unique()),
    default=sorted(df_full["customer_segment"].unique()), key="f_segment"
)
category_sel = st.sidebar.multiselect(
    "Product category", options=sorted(df_full["product_category"].unique()),
    default=sorted(df_full["product_category"].unique()), key="f_category"
)
payment_sel = st.sidebar.multiselect(
    "Payment method", options=sorted(df_full["payment_method"].unique()),
    default=sorted(df_full["payment_method"].unique()), key="f_payment"
)
shipping_sel = st.sidebar.multiselect(
    "Shipping type", options=sorted(df_full["shipping_type"].unique()),
    default=sorted(df_full["shipping_type"].unique()), key="f_shipping"
)
gender_sel = st.sidebar.multiselect(
    "Gender", options=sorted(df_full["gender"].unique()),
    default=sorted(df_full["gender"].unique()), key="f_gender"
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data note: this dataset shows near-uniform distributions across every "
    "categorical variable — a pattern consistent with synthetically generated "
    "data. See the Data Quality tab and the notebook's limitations section."
)

# ─────────────────────────────────────────────────────────────
# Apply filters
# ─────────────────────────────────────────────────────────────
if len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start, end = date_min, date_max


def apply_filters(source, start, end, age_lo, age_hi, segs, cats, pays, ships, genders):
    return source[
        (source["order_date"] >= start) & (source["order_date"] <= end) &
        (source["age"] >= age_lo) & (source["age"] <= age_hi) &
        (source["customer_segment"].isin(segs)) &
        (source["product_category"].isin(cats)) &
        (source["payment_method"].isin(pays)) &
        (source["shipping_type"].isin(ships)) &
        (source["gender"].isin(genders))
    ]


df = apply_filters(df_full, start, end, age_range[0], age_range[1],
                    segment_sel, category_sel, payment_sel, shipping_sel, gender_sel)

st.sidebar.caption(f"Showing **{len(df):,}** of {len(df_full):,} transactions")
st.sidebar.download_button(
    "⬇️ Download filtered data (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_transactions.csv",
    mime="text/csv",
    use_container_width=True,
)

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.title("🛒 Retail Transaction Analytics Dashboard")
st.caption("NextHikes IT Solutions Internship Project — interactive companion to the EDA notebook")

if df.empty:
    st.warning("No transactions match the current filters. Try widening your selection.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# KPI row — with delta vs. the immediately preceding period of equal length
# ─────────────────────────────────────────────────────────────
period_len = end - start
prev_start = start - period_len - timedelta(days=1)
prev_end = start - timedelta(days=1)
df_prev = apply_filters(df_full, prev_start, prev_end, age_range[0], age_range[1],
                         segment_sel, category_sel, payment_sel, shipping_sel, gender_sel)


def pct_delta(curr, prev):
    if prev in (None, 0) or pd.isna(prev):
        return None
    return (curr - prev) / prev * 100


has_prev = len(df_prev) > 0
prev_revenue = df_prev["final_price"].sum() if has_prev else None
prev_orders = len(df_prev) if has_prev else None
prev_aov = df_prev["final_price"].mean() if has_prev else None
prev_return = df_prev["returned"].mean() * 100 if has_prev else None
prev_delivery = df_prev["delivery_days"].mean() if has_prev else None

cur_revenue = df["final_price"].sum()
cur_orders = len(df)
cur_aov = df["final_price"].mean()
cur_return = df["returned"].mean() * 100
cur_delivery = df["delivery_days"].mean()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Revenue", f"₹{cur_revenue/1e7:.2f} Cr",
          delta=f"{pct_delta(cur_revenue, prev_revenue):+.1f}%" if has_prev else None)
k2.metric("Total Orders", f"{cur_orders:,}",
          delta=f"{pct_delta(cur_orders, prev_orders):+.1f}%" if has_prev else None)
k3.metric("Avg Order Value", f"₹{cur_aov:,.0f}",
          delta=f"{pct_delta(cur_aov, prev_aov):+.1f}%" if has_prev else None)
k4.metric("Return Rate", f"{cur_return:.2f}%",
          delta=f"{cur_return - prev_return:+.2f} pp" if has_prev else None,
          delta_color="inverse")
k5.metric("Avg Delivery Days", f"{cur_delivery:.2f}",
          delta=f"{cur_delivery - prev_delivery:+.2f}" if has_prev else None,
          delta_color="inverse")
if has_prev:
    st.caption(f"Δ vs. preceding {period_len.days + 1}-day period "
               f"({prev_start.date()} → {prev_end.date()})")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
(tab1, tab2, tab3, tab4, tab_geo, tab5, tab6, tab_ts, tab7, tab_reg,
 tab8, tab_biz, tab_stat, tab9) = st.tabs([
    "📈 Overview & Trends", "📊 Univariate", "🔗 Bivariate & Correlation",
    "🧩 Multivariate", "🌍 Geography", "🎯 Outliers", "🧹 Data Quality",
    "📆 Time Series", "👥 Customer / RFM", "🤖 Revenue Prediction",
    "🔮 Returns Model", "🎯 Business Questions", "🔬 Statistical Rigor",
    "💡 Key Insights"
])

# ── TAB 1: Overview & Trends ───────────────────────────────────
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        monthly = (df.groupby("order_month")
                     .agg(revenue=("final_price", "sum"),
                          orders=("order_id", "count"))
                     .reset_index().sort_values("order_month"))
        fig = px.line(monthly, x="order_month", y="revenue",
                      title="Monthly Revenue Trend", markers=True)
        fig.update_traces(line_color=RUST)
        fig.update_layout(**LAYOUT_KWARGS, xaxis_title="Month", yaxis_title="Revenue (₹)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        monthly_ret = (df.groupby("order_month")["returned"].mean().reset_index()
                         .sort_values("order_month"))
        monthly_ret["returned"] *= 100
        fig = px.line(monthly_ret, x="order_month", y="returned",
                      title="Monthly Return Rate (%)", markers=True)
        fig.update_traces(line_color=GORGE)
        fig.update_layout(**LAYOUT_KWARGS, xaxis_title="Month", yaxis_title="Return Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        seg = df["customer_segment"].value_counts().reset_index()
        seg.columns = ["Segment", "Count"]
        fig = px.pie(seg, names="Segment", values="Count", hole=0.45,
                     title="Orders by Customer Segment",
                     color="Segment", color_discrete_map=SEG_COLORS)
        fig.update_layout(**LAYOUT_KWARGS)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        cat = df["product_category"].value_counts().reset_index()
        cat.columns = ["Category", "Count"]
        fig = px.bar(cat, x="Count", y="Category", orientation="h",
                     title="Orders by Product Category", color="Category",
                     color_discrete_sequence=PALETTE)
        fig.update_layout(**LAYOUT_KWARGS, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Average order value trend, with 95% confidence interval")
    aov_month = (df.groupby("order_month")["final_price"]
                   .agg(mean="mean", std="std", n="count").reset_index()
                   .sort_values("order_month"))
    aov_month["sem"] = aov_month["std"] / np.sqrt(aov_month["n"])
    aov_month["ci95"] = 1.96 * aov_month["sem"].fillna(0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=aov_month["order_month"], y=aov_month["mean"], mode="lines+markers",
        line=dict(color=RUST), name="Avg order value",
        error_y=dict(type="data", array=aov_month["ci95"], visible=True, color=BURNT_SIENNA)
    ))
    fig.update_layout(**LAYOUT_KWARGS, xaxis_title="Month", yaxis_title="Avg Final Price (₹)",
                       title="Monthly AOV ± 95% CI")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Error bars show the 95% confidence interval of the monthly mean "
               "(mean ± 1.96 × standard error). Wide bands mean the month's average "
               "isn't precisely estimated — usually because that month has fewer orders.")

    st.subheader("Time patterns: day-of-week & day-of-month")
    c1, c2 = st.columns(2)
    with c1:
        dow = (df.groupby("order_dow")["final_price"].sum()
                 .reindex(WEEKDAY_ORDER).reset_index())
        dow.columns = ["Day", "Revenue"]
        fig = px.bar(dow, x="Day", y="Revenue", color="Day",
                     color_discrete_sequence=PALETTE,
                     title="Revenue by Day of Week")
        fig.update_layout(**LAYOUT_KWARGS, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        dom = df.groupby("order_dom")["final_price"].sum().reset_index()
        dom.columns = ["Day of Month", "Revenue"]
        fig = px.bar(dom, x="Day of Month", y="Revenue",
                     color_discrete_sequence=[SAGE],
                     title="Revenue by Day of Month")
        fig.update_layout(**LAYOUT_KWARGS)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Trend vs. weekday seasonality (daily revenue)")
    daily = df.groupby(df["order_date"].dt.date)["final_price"].sum().reset_index()
    daily.columns = ["date", "revenue"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")
    daily["trend_7d"] = daily["revenue"].rolling(7, center=True, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["revenue"], mode="lines",
                              line=dict(color=CANVAS, width=1), name="Daily revenue"))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["trend_7d"], mode="lines",
                              line=dict(color=RUST, width=2.5), name="7-day rolling trend"))
    fig.update_layout(**LAYOUT_KWARGS, xaxis_title="Date", yaxis_title="Revenue (₹)",
                       title="Daily Revenue with 7-Day Rolling Trend")
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 2: Univariate ───────────────────────────────────────────
with tab2:
    st.subheader("Distribution explorer")
    sel_col = st.selectbox("Choose a numerical variable", NUM_COLS, index=4)

    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.histogram(df, x=sel_col, nbins=40, marginal="box",
                            color_discrete_sequence=[SAGE],
                            title=f"Distribution of {sel_col}")
        fig.add_vline(x=df[sel_col].mean(), line_dash="dash", line_color=RUST,
                      annotation_text="Mean")
        fig.add_vline(x=df[sel_col].median(), line_color=GORGE,
                      annotation_text="Median")
        fig.update_layout(**LAYOUT_KWARGS)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Summary statistics**")
        s = df[sel_col]
        stats_tbl = pd.DataFrame({
            "Metric": ["Mean", "Median", "Std Dev", "Skewness", "Kurtosis", "Min", "Max"],
            "Value": [f"{s.mean():,.2f}", f"{s.median():,.2f}", f"{s.std():,.2f}",
                      f"{s.skew():+.4f}", f"{s.kurtosis():+.4f}",
                      f"{s.min():,.2f}", f"{s.max():,.2f}"]
        })
        st.dataframe(stats_tbl, hide_index=True, use_container_width=True)

    st.markdown("**Normality tests**")
    sample_for_shapiro = s.sample(min(5000, len(s)), random_state=42)
    sh_stat, sh_p = stats.shapiro(sample_for_shapiro)
    da_stat, da_p = stats.normaltest(s)
    n1, n2 = st.columns(2)
    n1.metric("Shapiro–Wilk p-value", f"{sh_p:.4g}",
              help="Tested on a random sample of up to 5,000 rows (Shapiro-Wilk "
                   "loses reliability on very large samples).")
    n2.metric("D'Agostino K² p-value", f"{da_p:.4g}",
              help="Uses skewness + kurtosis; reliable on the full sample.")
    if sh_p < 0.05 or da_p < 0.05:
        st.caption(f"Both tests reject normality at α = 0.05 for **{sel_col}** "
                    "(p < 0.05 → not normally distributed). This matters if you're "
                    "planning to use Pearson correlation or t-tests/ANOVA on this "
                    "variable — consider Spearman correlation or a non-parametric "
                    "test (Kruskal–Wallis) instead.")
    else:
        st.caption(f"Neither test rejects normality for **{sel_col}** at α = 0.05 — "
                    "parametric tests (Pearson, ANOVA, t-tests) are reasonably safe to use.")

    st.subheader("Categorical frequency explorer")
    sel_cat = st.selectbox("Choose a categorical variable", CAT_COLS, index=1)
    vc = df[sel_cat].value_counts(normalize=True).mul(100).round(2).reset_index()
    vc.columns = [sel_cat, "Pct (%)"]
    fig = px.bar(vc, x=sel_cat, y="Pct (%)", color=sel_cat,
                 color_discrete_sequence=PALETTE,
                 title=f"{sel_cat} — Frequency Distribution (%)")
    fig.update_layout(**LAYOUT_KWARGS, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 3: Bivariate & Correlation ──────────────────────────────
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        corr_method = st.radio("Correlation method", ["pearson", "spearman"],
                                horizontal=True,
                                help="Spearman is more robust if a variable is skewed "
                                     "or the relationship isn't linear.")
        corr = df[NUM_COLS].corr(method=corr_method).round(3)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdYlGn",
                         zmin=-1, zmax=1,
                         title=f"{corr_method.title()} Correlation Matrix")
        fig.update_layout(**LAYOUT_KWARGS)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        x_axis = st.selectbox("X variable", NUM_COLS, index=1, key="scx")
        y_axis = st.selectbox("Y variable", NUM_COLS, index=4, key="scy")
        sample = df.sample(min(4000, len(df)), random_state=42)
        r, p = stats.pearsonr(df[x_axis], df[y_axis])
        sig = "significant" if p < 0.05 else "not significant"
        fig = px.scatter(sample, x=x_axis, y=y_axis,
                          opacity=0.35, color_discrete_sequence=[SAGE],
                          title=f"{x_axis} vs {y_axis}  (r = {r:+.3f}, {sig})")
        # Manual regression line (avoids needing the statsmodels package)
        m, b = np.polyfit(df[x_axis], df[y_axis], 1)
        x_line = np.linspace(df[x_axis].min(), df[x_axis].max(), 100)
        fig.add_trace(go.Scatter(x=x_line, y=m * x_line + b, mode="lines",
                                  line=dict(color=RUST, width=2.5), name="Trend"))
        fig.update_layout(**LAYOUT_KWARGS)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Pearson r = {r:+.3f}, p = {p:.4f} (n = {len(df):,})")

    st.subheader("Multicollinearity check (VIF)")

    def compute_vif(data, cols):
        """Variance Inflation Factor via least-squares R², no statsmodels required."""
        X = data[cols].astype(float).values
        out = {}
        for i, col in enumerate(cols):
            y_i = X[:, i]
            X_others = np.delete(X, i, axis=1)
            X_others = np.column_stack([np.ones(len(X_others)), X_others])
            coef, _, _, _ = np.linalg.lstsq(X_others, y_i, rcond=None)
            y_pred = X_others @ coef
            ss_res = np.sum((y_i - y_pred) ** 2)
            ss_tot = np.sum((y_i - y_i.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            out[col] = 1 / (1 - r2) if r2 < 1 else np.inf
        return pd.Series(out, name="VIF").sort_values(ascending=False)

    vif_series = compute_vif(df, NUM_COLS)
    vif_df = vif_series.reset_index()
    vif_df.columns = ["Variable", "VIF"]

    def flag(v):
        if v >= 10:
            return "🔴 High"
        elif v >= 5:
            return "🟡 Moderate"
        return "🟢 OK"

    vif_df["Flag"] = vif_df["VIF"].apply(flag)
    st.dataframe(vif_df.style.format({"VIF": "{:.2f}"}), hide_index=True, use_container_width=True)
    st.caption("VIF < 5: low multicollinearity. 5–10: moderate, worth watching. "
               "> 10: high — that variable is largely explained by the others, "
               "which can destabilize regression coefficients if used together as predictors.")

    st.subheader("Spend by category (numerical vs categorical)")
    cat_choice = st.selectbox(
        "Group by", ["customer_segment", "product_category", "payment_method",
                     "shipping_type", "gender"], index=0
    )
    grp = (df.groupby(cat_choice)["final_price"].agg(mean="mean", median="median",
                                                       std="std", count="count")
             .reset_index())
    grp["sem"] = grp["std"] / np.sqrt(grp["count"])
    grp["ci95"] = 1.96 * grp["sem"].fillna(0)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grp[cat_choice], y=grp["mean"],
        error_y=dict(type="data", array=grp["ci95"], visible=True),
        marker_color=PALETTE[:len(grp)], name="Avg final price"
    ))
    fig.add_hline(y=df["final_price"].mean(), line_dash="dash", line_color=RUST,
                  annotation_text="Overall avg")
    fig.update_layout(**LAYOUT_KWARGS, showlegend=False,
                       title=f"Avg Final Price by {cat_choice} (± 95% CI)",
                       xaxis_title=cat_choice, yaxis_title="Avg Final Price (₹)")
    st.plotly_chart(fig, use_container_width=True)

    # One-way ANOVA: does final_price actually differ across these groups?
    groups = [g["final_price"].values for _, g in df.groupby(cat_choice)]
    f_stat, p_val = stats.f_oneway(*groups)
    verdict = "a statistically significant" if p_val < 0.05 else "no statistically significant"
    st.caption(
        f"One-way ANOVA: F = {f_stat:.2f}, p = {p_val:.4f} — {cat_choice} has "
        f"{verdict} effect on final_price at α = 0.05."
    )

    st.subheader("Return rate by category")
    ret_grp = (df.groupby(cat_choice)["returned"].mean().mul(100).round(2).reset_index())
    fig = px.bar(ret_grp, x=cat_choice, y="returned", color=cat_choice,
                 color_discrete_sequence=PALETTE,
                 title=f"Return Rate (%) by {cat_choice}")
    fig.update_layout(**LAYOUT_KWARGS, showlegend=False, yaxis_title="Return Rate (%)")
    st.plotly_chart(fig, use_container_width=True)

    # Chi-square test of independence: is return status related to this category?
    contingency = pd.crosstab(df[cat_choice], df["return_status"])
    chi2, p_chi, _, _ = stats.chi2_contingency(contingency)
    verdict_chi = "a statistically significant" if p_chi < 0.05 else "no statistically significant"
    st.caption(
        f"Chi-square test: χ² = {chi2:.2f}, p = {p_chi:.4f} — {cat_choice} has "
        f"{verdict_chi} association with return status at α = 0.05."
    )

# ── TAB 4: Multivariate ─────────────────────────────────────────
with tab4:
    st.subheader("Segment × Age Group — Avg Spend")
    pivot1 = df.pivot_table(values="final_price", index="age_group",
                             columns="customer_segment", aggfunc="mean",
                             observed=True).round(0)
    fig = px.imshow(pivot1, text_auto=".0f", color_continuous_scale="YlOrBr",
                     title="Avg Final Price (₹): Age Group × Customer Segment")
    fig.update_layout(**LAYOUT_KWARGS)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Category × Discount Level — Avg Revenue")
    pivot2 = df.pivot_table(values="final_price", index="product_category",
                             columns="disc_bucket", aggfunc="mean",
                             observed=True).round(0)
    fig = px.imshow(pivot2, text_auto=".0f", color_continuous_scale="YlOrBr",
                     title="Avg Final Price (₹): Product Category × Discount Bucket")
    fig.update_layout(**LAYOUT_KWARGS)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("High-value order profile (top 10% vs rest)")
    hv = (df.groupby("high_value")
            .agg(avg_final_price=("final_price", "mean"),
                 avg_quantity=("quantity", "mean"),
                 avg_discount=("discount_percentage", "mean"),
                 return_rate=("returned", "mean"),
                 count=("final_price", "count"))
            .rename(index={0: "Regular (bottom 90%)", 1: "High-Value (top 10%)"}))
    hv["return_rate"] = (hv["return_rate"] * 100).round(2)
    st.dataframe(hv.style.format({
        "avg_final_price": "₹{:,.0f}", "avg_quantity": "{:.2f}",
        "avg_discount": "{:.1f}%", "return_rate": "{:.2f}%", "count": "{:,}"
    }), use_container_width=True)

    st.markdown("---")
    st.subheader("3D view: Product Price × Quantity × Discount → Final Price")
    st.caption("Mirrors the notebook's extended multivariate view — the same "
               "product_price/quantity/discount relationship that drives final_price, "
               "seen in three dimensions at once instead of pairwise.")
    sample_3d = df.sample(min(2000, len(df)), random_state=42)
    fig = px.scatter_3d(sample_3d, x="product_price", y="quantity", z="discount_percentage",
                         color="final_price", opacity=0.6,
                         color_continuous_scale="YlOrBr",
                         title="Price × Quantity × Discount, colored by Final Price")
    fig.update_layout(**LAYOUT_KWARGS, scene=dict(
        xaxis_title="Product Price (₹)", yaxis_title="Quantity", zaxis_title="Discount (%)"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Final price (color) brightens almost purely along the product_price axis — "
               "visual confirmation that price, not quantity or discount alone, dominates.")

    st.subheader("Bubble view: Product Subcategory Performance")
    subcat = (df.groupby("product_subcategory")
                .agg(revenue=("final_price", "sum"), orders=("order_id", "count"),
                     return_rate=("returned", "mean"))
                .reset_index())
    subcat["return_rate"] = subcat["return_rate"] * 100
    fig = px.scatter(subcat, x="orders", y="revenue", size="orders", color="return_rate",
                      hover_name="product_subcategory", color_continuous_scale="RdYlGn_r",
                      title="Subcategory: Order Volume vs Revenue (bubble size = orders, color = return rate %)",
                      labels={"orders": "Order Count", "revenue": "Total Revenue (₹)",
                              "return_rate": "Return Rate (%)"})
    fig.update_layout(**LAYOUT_KWARGS)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{df['product_subcategory'].nunique()} subcategories shown — revenue tracks "
               "order volume almost linearly, consistent with the 'no category is special' "
               "finding holding even at this finer grain.")

# ── TAB GEO: Geography ──────────────────────────────────────────
with tab_geo:
    st.caption("Mirrors the notebook's Geography section — city & state vs. revenue "
               "and returns, including the state-level artifact flagged in Section 10.")

    state_rev = (df.groupby("state")["final_price"].sum().sort_values(ascending=False))
    state_share = (state_rev / state_rev.sum() * 100).round(2)
    top_state = state_share.index[0]
    top_share = state_share.iloc[0]
    n_cities_top_state = df.loc[df["state"] == top_state, "city"].nunique()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(state_share.reset_index(), x="state", y="final_price",
                     color="state", color_discrete_sequence=PALETTE,
                     title="Revenue Share by State (%)",
                     labels={"final_price": "Revenue Share (%)", "state": "State"})
        fig.add_hline(y=100 / df["state"].nunique(), line_dash="dash", line_color=GORGE,
                      annotation_text="Even split")
        fig.update_layout(**LAYOUT_KWARGS, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        city_rev = df.groupby("city")["final_price"].sum().sort_values(ascending=False)
        city_share = (city_rev / city_rev.sum() * 100).round(2)
        fig = px.bar(city_share.reset_index(), x="city", y="final_price",
                     color="city", color_discrete_sequence=PALETTE,
                     title="Revenue Share by City (%)",
                     labels={"final_price": "Revenue Share (%)", "city": "City"})
        fig.add_hline(y=100 / df["city"].nunique(), line_dash="dash", line_color=GORGE,
                      annotation_text="Even split")
        fig.update_layout(**LAYOUT_KWARGS, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    city_spread = ((city_share - city_share.mean()).abs().max())
    if n_cities_top_state > 1:
        st.warning(
            f"⚠️ **Artifact, not a real leader:** {top_state} tops the state chart at "
            f"{top_share:.1f}% share only because it's the sole state mapped to "
            f"{n_cities_top_state} cities ({', '.join(df.loc[df['state'] == top_state, 'city'].unique())}) "
            f"while every other state maps to exactly 1 city. By city, all "
            f"{df['city'].nunique()} cities sit within **±{city_spread:.1f} percentage points** "
            "of an even split — there's no genuine geographic winner in this data."
        )
    else:
        st.info(f"Top state by revenue: **{top_state}** ({top_share:.1f}% share).")

    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        geo_choice = st.radio("Return rate by", ["state", "city"], horizontal=True, key="geo_ret")
        ret_geo = (df.groupby(geo_choice)["returned"].mean().mul(100).round(2)
                     .sort_values(ascending=False).reset_index())
        fig = px.bar(ret_geo, x=geo_choice, y="returned", color=geo_choice,
                     color_discrete_sequence=PALETTE,
                     title=f"Return Rate (%) by {geo_choice.title()}")
        fig.update_layout(**LAYOUT_KWARGS, showlegend=False, yaxis_title="Return Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
        contingency = pd.crosstab(df[geo_choice], df["return_status"])
        chi2, p_chi, _, _ = stats.chi2_contingency(contingency)
        verdict = "a statistically significant" if p_chi < 0.05 else "no statistically significant"
        st.caption(f"Chi-square: χ² = {chi2:.2f}, p = {p_chi:.4f} — {geo_choice} has "
                   f"{verdict} association with return status.")

    with c4:
        st.subheader("State × Category revenue heatmap")
        pivot_geo = df.pivot_table(values="final_price", index="state",
                                    columns="product_category", aggfunc="sum",
                                    observed=True)
        fig = px.imshow(pivot_geo, color_continuous_scale="YlOrBr", aspect="auto",
                         title="Total Revenue (₹): State × Product Category")
        fig.update_layout(**LAYOUT_KWARGS)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Avg final price by state, with 95% CI")
    grp_state = (df.groupby("state")["final_price"].agg(mean="mean", std="std", count="count")
                   .reset_index())
    grp_state["ci95"] = 1.96 * grp_state["std"] / np.sqrt(grp_state["count"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=grp_state["state"], y=grp_state["mean"],
                          error_y=dict(type="data", array=grp_state["ci95"], visible=True),
                          marker_color=PALETTE[0]))
    fig.add_hline(y=df["final_price"].mean(), line_dash="dash", line_color=RUST,
                  annotation_text="Overall avg")
    fig.update_layout(**LAYOUT_KWARGS, title="Avg Final Price by State (± 95% CI)",
                       yaxis_title="Avg Final Price (₹)")
    st.plotly_chart(fig, use_container_width=True)
    f_stat, p_val = stats.f_oneway(*[g["final_price"].values for _, g in df.groupby("state")])
    st.caption(f"One-way ANOVA across states: F = {f_stat:.2f}, p = {p_val:.4f} — "
               f"{'a real' if p_val < 0.05 else 'no real'} state-level effect on order value.")

# ── TAB 5: Outliers ──────────────────────────────────────────────
with tab5:
    sel_out = st.selectbox("Choose variable for outlier check", NUM_COLS, index=4, key="outlier_col")
    method = st.radio("Detection method", ["IQR (1.5×)", "Z-score (|z| > 3)"], horizontal=True)

    if method.startswith("IQR"):
        Q1, Q3 = df[sel_out].quantile(0.25), df[sel_out].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outlier_mask = (df[sel_out] < lower) | (df[sel_out] > upper)
        fence_labels = ("Lower Fence", "Upper Fence")
    else:
        z = (df[sel_out] - df[sel_out].mean()) / df[sel_out].std()
        outlier_mask = z.abs() > 3
        lower, upper = df[sel_out].mean() - 3 * df[sel_out].std(), df[sel_out].mean() + 3 * df[sel_out].std()
        fence_labels = ("Lower Bound (−3σ)", "Upper Bound (+3σ)")

    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.box(df, y=sel_out, points="outliers",
                     color_discrete_sequence=[BURNT_SIENNA],
                     title=f"Boxplot — {sel_out}")
        fig.update_layout(**LAYOUT_KWARGS)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.metric("Outlier Count", f"{outlier_mask.sum():,}")
        st.metric("Outlier %", f"{outlier_mask.mean()*100:.2f}%")
        st.metric(fence_labels[0], f"{lower:,.1f}")
        st.metric(fence_labels[1], f"{upper:,.1f}")

    if outlier_mask.sum() > 0:
        st.markdown("**Sample of flagged outlier rows**")
        st.dataframe(df[outlier_mask].head(20), use_container_width=True)
    else:
        st.info("No outliers flagged by this method for the current filters.")

    st.markdown("---")
    st.subheader("Sensitivity check — does the IQR fence width choice matter?")
    st.caption("Mirrors the notebook's Section 4 sensitivity check — re-running IQR "
               "outlier detection at several fence widths to see if conclusions "
               "(which rows get flagged, how many) are stable or fence-dependent.")
    fence_widths = [1.0, 1.5, 2.0, 2.5, 3.0]
    sens_rows = []
    Q1s, Q3s = df[sel_out].quantile(0.25), df[sel_out].quantile(0.75)
    IQRs = Q3s - Q1s
    for k in fence_widths:
        lo, hi = Q1s - k * IQRs, Q3s + k * IQRs
        mask_k = (df[sel_out] < lo) | (df[sel_out] > hi)
        sens_rows.append({"Fence width (×IQR)": k, "Lower fence": round(lo, 1),
                           "Upper fence": round(hi, 1), "Outliers flagged": int(mask_k.sum()),
                           "Outlier %": round(mask_k.mean() * 100, 2)})
    sens_df = pd.DataFrame(sens_rows)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(sens_df, hide_index=True, use_container_width=True)
    with c2:
        fig = px.line(sens_df, x="Fence width (×IQR)", y="Outlier %", markers=True,
                       title=f"Outlier % vs Fence Width — {sel_out}")
        fig.update_traces(line_color=RUST)
        fig.update_layout(**LAYOUT_KWARGS)
        st.plotly_chart(fig, use_container_width=True)
    spread = sens_df["Outlier %"].max() - sens_df["Outlier %"].min()
    st.caption(f"Outlier % ranges from {sens_df['Outlier %'].min():.2f}% (widest fence, 3.0×IQR) "
               f"to {sens_df['Outlier %'].max():.2f}% (tightest fence, 1.0×IQR) — a "
               f"{spread:.2f} percentage-point spread. The standard 1.5×IQR choice sits in the "
               "middle of this range, and the qualitative conclusion (a small minority of rows "
               "are flagged) is stable across all five widths tested.")

# ── TAB 6: Data Quality ─────────────────────────────────────────
with tab6:
    st.caption("This tab reports on the **full, unfiltered** dataset — data quality "
               "is a property of the source file, not of whatever slice you're viewing.")

    dq1, dq2, dq3, dq4 = st.columns(4)
    dq1.metric("Total Rows", f"{len(raw_full):,}")
    dq1.metric("Total Columns", f"{raw_full.shape[1]}")
    dup_count = raw_full.duplicated().sum()
    dq2.metric("Duplicate Rows", f"{dup_count:,}")
    dq2.metric("Duplicate %", f"{dup_count/len(raw_full)*100:.3f}%")
    total_missing = raw_full.isna().sum().sum()
    dq3.metric("Missing Cells", f"{total_missing:,}")
    dq3.metric("Missing %", f"{total_missing/raw_full.size*100:.4f}%")
    dq4.metric("Fully Complete Rows", f"{(~raw_full.isna().any(axis=1)).sum():,}")
    dq4.metric("Date Span", f"{(df_full['order_date'].max() - df_full['order_date'].min()).days:,} days")

    st.subheader("Missing values by column")
    miss = raw_full.isna().sum()
    miss = miss[miss > 0]
    if miss.empty:
        st.success("No missing values in any column. ✅")
    else:
        miss_df = miss.reset_index()
        miss_df.columns = ["Column", "Missing Count"]
        miss_df["Missing %"] = (miss_df["Missing Count"] / len(raw_full) * 100).round(3)
        fig = px.bar(miss_df, x="Column", y="Missing %", color_discrete_sequence=[RUST],
                     title="Missing Values by Column (%)")
        fig.update_layout(**LAYOUT_KWARGS)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Column types & cardinality")
    dtype_df = pd.DataFrame({
        "Column": raw_full.columns,
        "Dtype": raw_full.dtypes.astype(str).values,
        "Unique Values": [raw_full[c].nunique() for c in raw_full.columns],
        "Missing": [raw_full[c].isna().sum() for c in raw_full.columns],
    })
    st.dataframe(dtype_df, hide_index=True, use_container_width=True)

    st.subheader("Numeric range sanity check")
    raw_num_cols = [c for c in NUM_COLS if c in raw_full.columns]
    range_df = raw_full[raw_num_cols].agg(["min", "max", "mean", "std"]).T.reset_index()
    range_df.columns = ["Column", "Min", "Max", "Mean", "Std"]
    st.dataframe(range_df.style.format({"Min": "{:.2f}", "Max": "{:.2f}",
                                         "Mean": "{:.2f}", "Std": "{:.2f}"}),
                 hide_index=True, use_container_width=True)
    neg_price = (raw_full["final_price"] < 0).sum() if "final_price" in raw_full else 0
    bad_discount = ((raw_full["discount_percentage"] < 0) | (raw_full["discount_percentage"] > 100)).sum() \
        if "discount_percentage" in raw_full else 0
    st.caption(f"Sanity flags: {neg_price:,} negative final_price rows, "
               f"{bad_discount:,} out-of-range discount_percentage rows.")

    st.subheader("Categorical uniformity check")
    st.caption("A near-flat distribution across every category (roughly equal bar heights) "
               "is one of the signals that this dataset was synthetically generated "
               "rather than collected from real transactions.")
    uniform_cols = st.multiselect("Columns to check", CAT_COLS,
                                   default=["customer_segment", "product_category"])
    for c in uniform_cols:
        vc = raw_full[c].value_counts(normalize=True).mul(100).round(2).reset_index()
        vc.columns = [c, "Pct (%)"]
        fig = px.bar(vc, x=c, y="Pct (%)", color=c, color_discrete_sequence=PALETTE,
                     title=f"{c} distribution (full dataset)")
        fig.update_layout(**LAYOUT_KWARGS, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ── TAB TS: Time Series Deep-Dive ────────────────────────────────
with tab_ts:
    st.caption("Mirrors the notebook's Section 8 — formal trend testing, a manual "
               "seasonal decomposition, and category/state-level trend breakdowns.")

    monthly_rev = (df.groupby("order_month")["final_price"].sum()
                     .reset_index().sort_values("order_month"))
    x_idx = np.arange(len(monthly_rev))
    if len(monthly_rev) >= 3:
        slope, intercept, r_val, p_trend, se = stats.linregress(x_idx, monthly_rev["final_price"])
        trend_verdict = "a statistically significant" if p_trend < 0.05 else "no statistically significant"
        st.info(f"**Overall trend test** (linear regression of monthly revenue vs. time): "
                f"slope = ₹{slope:,.0f}/month, r² = {r_val**2:.4f}, p = {p_trend:.4f} — "
                f"{trend_verdict} trend across the period.")

    st.subheader("Manual seasonal decomposition (additive: trend + seasonal + residual)")
    st.caption("Trend = 12-month centered rolling mean. Seasonal = average deviation from "
               "trend for each calendar month, repeated across years. Residual = what's left.")
    ts = df.groupby(pd.PeriodIndex(df["order_date"], freq="M"))["final_price"].sum()
    ts.index = ts.index.to_timestamp()
    if len(ts) >= 13:
        trend_component = ts.rolling(12, center=True, min_periods=6).mean()
        detrended = ts - trend_component
        seasonal_avg = detrended.groupby(ts.index.month).mean()
        seasonal_component = ts.index.month.map(seasonal_avg)
        seasonal_component = pd.Series(seasonal_component, index=ts.index)
        residual_component = ts - trend_component - seasonal_component

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name="Observed",
                                  line=dict(color=CANVAS, width=1)))
        fig.add_trace(go.Scatter(x=ts.index, y=trend_component.values, name="Trend",
                                  line=dict(color=RUST, width=2.5)))
        fig.update_layout(**LAYOUT_KWARGS, title="Observed vs Trend (12-month rolling)",
                           xaxis_title="Month", yaxis_title="Revenue (₹)")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(x=seasonal_avg.index, y=seasonal_avg.values,
                         color_discrete_sequence=[SAGE],
                         title="Seasonal Component by Calendar Month",
                         labels={"x": "Month", "y": "Avg deviation from trend (₹)"})
            fig.update_layout(**LAYOUT_KWARGS)
            st.plotly_chart(fig, use_container_width=True)
            lowest_month = seasonal_avg.idxmin()
            month_name = pd.Timestamp(2024, lowest_month, 1).strftime("%B")
            st.caption(f"Lowest average revenue month across the period: **{month_name}**.")
        with c2:
            fig = px.scatter(x=ts.index, y=residual_component.values,
                              color_discrete_sequence=[BURNT_SIENNA],
                              title="Residual Component (should look like noise)",
                              labels={"x": "Month", "y": "Residual (₹)"})
            fig.add_hline(y=0, line_dash="dash", line_color=GORGE)
            fig.update_layout(**LAYOUT_KWARGS)
            st.plotly_chart(fig, use_container_width=True)

        seasonal_strength = max(0, 1 - residual_component.var(skipna=True) /
                                 (residual_component + seasonal_component).var(skipna=True))
        st.caption(f"Seasonal strength (1 − Var(residual) / Var(residual+seasonal)) ≈ "
                   f"**{seasonal_strength:.2f}**. A trend p-value near 1 combined with a "
                   "non-trivial seasonal strength is the nuanced read: no long-run growth "
                   "or decline, but a repeating within-year shape.")
    else:
        st.info("Not enough months in the current filter for a 12-month seasonal decomposition — "
                "widen the date range.")

    st.markdown("---")
    st.subheader("Trend test by product category")
    cat_trend_rows = []
    for cat, g in df.groupby("product_category"):
        m = g.groupby("order_month")["final_price"].sum().reset_index().sort_values("order_month")
        if len(m) >= 3:
            s, i, r, p, se = stats.linregress(np.arange(len(m)), m["final_price"])
            cat_trend_rows.append({"Category": cat, "Slope (₹/mo)": round(s, 0),
                                    "p-value": round(p, 4),
                                    "Significant?": "Yes" if p < 0.05 else "No"})
    cat_trend_df = pd.DataFrame(cat_trend_rows)
    st.dataframe(cat_trend_df, hide_index=True, use_container_width=True)
    n_sig_cat = (cat_trend_df["Significant?"] == "Yes").sum() if len(cat_trend_df) else 0
    st.caption(f"{n_sig_cat} of {len(cat_trend_df)} categories show a significant monthly trend "
               "at α = 0.05.")

    st.subheader("Trend test by state")
    state_trend_rows = []
    for st_name, g in df.groupby("state"):
        m = g.groupby("order_month")["final_price"].sum().reset_index().sort_values("order_month")
        if len(m) >= 3:
            s, i, r, p, se = stats.linregress(np.arange(len(m)), m["final_price"])
            state_trend_rows.append({"State": st_name, "Slope (₹/mo)": round(s, 0),
                                      "p-value": round(p, 4),
                                      "Significant?": "Yes" if p < 0.05 else "No"})
    state_trend_df = pd.DataFrame(state_trend_rows)
    st.dataframe(state_trend_df, hide_index=True, use_container_width=True)
    n_sig_state = (state_trend_df["Significant?"] == "Yes").sum() if len(state_trend_df) else 0
    st.caption(f"{n_sig_state} of {len(state_trend_df)} states show a significant monthly trend "
               "at α = 0.05.")

# ── TAB 7: Customer / RFM ───────────────────────────────────────
with tab7:
    id_candidates = [c for c in df_full.columns
                      if c.lower() in ("customer_id", "cust_id", "customerid", "client_id")]

    # A customer_id column existing isn't enough for real RFM — every customer needs
    # more than one order, or "frequency" and "recency" are meaningless (this matches
    # the notebook's Section 11 finding: "RFM: Can Not Be Done — requires repeat customers").
    max_orders_per_customer = df_full[id_candidates[0]].value_counts().max() if id_candidates else 0
    repeat_customers_exist = max_orders_per_customer > 1

    if id_candidates and repeat_customers_exist:
        cust_col = id_candidates[0]
        st.subheader("RFM Analysis (Recency, Frequency, Monetary)")
        snapshot_date = df["order_date"].max() + timedelta(days=1)
        rfm = (df.groupby(cust_col).agg(
                    recency=("order_date", lambda x: (snapshot_date - x.max()).days),
                    frequency=("order_id", "count"),
                    monetary=("final_price", "sum"))
                 .reset_index())
        rfm["R_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
        rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5,
                                  labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
        rfm["M_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
        rfm["RFM_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

        def rfm_tier(score):
            if score >= 12:
                return "Champions"
            elif score >= 9:
                return "Loyal"
            elif score >= 6:
                return "At Risk"
            return "Lost"

        rfm["Tier"] = rfm["RFM_score"].apply(rfm_tier)

        c1, c2 = st.columns(2)
        with c1:
            tier_counts = rfm["Tier"].value_counts().reset_index()
            tier_counts.columns = ["Tier", "Customers"]
            fig = px.pie(tier_counts, names="Tier", values="Customers", hole=0.45,
                         title="Customer Tiers", color_discrete_sequence=PALETTE)
            fig.update_layout(**LAYOUT_KWARGS)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(rfm, x="frequency", y="monetary", color="Tier",
                              size="recency", opacity=0.6,
                              color_discrete_sequence=PALETTE,
                              title="Frequency vs Monetary (bubble size = recency, days)")
            fig.update_layout(**LAYOUT_KWARGS)
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(rfm.sort_values("RFM_score", ascending=False).head(20),
                     use_container_width=True)
    elif id_candidates:
        n_cust = df_full[id_candidates[0]].nunique()
        st.info(
            f"A `{id_candidates[0]}` column exists, but it doesn't support real RFM: "
            f"all **{n_cust:,}** customer IDs appear **exactly once** each across "
            f"{len(df_full):,} transactions — nobody in this dataset has a repeat order. "
            "Recency and Frequency would both collapse to the same trivial value for "
            "every row, so a genuine RFM segmentation can't be built here (it needs "
            "repeat-purchase history). This is a real dataset limitation, not a chart "
            "you're missing — see Section 11 of the notebook. Showing a segment-level "
            "substitute instead."
        )
    else:
        st.info(
            "No customer identifier column (e.g. `customer_id`) was found in this dataset, "
            "so a true per-customer RFM analysis isn't possible — each row here is an "
            "independent transaction with no way to link repeat purchases to the same "
            "person. Showing a segment-level substitute instead."
        )
        st.subheader("Customer segment mix over time")
        seg_month = (df.groupby(["order_month", "customer_segment"]).size()
                       .reset_index(name="orders"))
        seg_month_pct = seg_month.pivot(index="order_month", columns="customer_segment",
                                         values="orders").fillna(0)
        seg_month_pct = seg_month_pct.div(seg_month_pct.sum(axis=1), axis=0) * 100
        seg_month_pct = seg_month_pct.reset_index().melt(id_vars="order_month",
                                                           var_name="Segment", value_name="Share (%)")
        fig = px.area(seg_month_pct.sort_values("order_month"), x="order_month", y="Share (%)",
                       color="Segment", color_discrete_map=SEG_COLORS,
                       title="Customer Segment Share of Orders by Month")
        fig.update_layout(**LAYOUT_KWARGS)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Segment-level spend summary (proxy for Monetary)")
        seg_summary = (df.groupby("customer_segment")
                          .agg(orders=("order_id", "count"),
                               total_spend=("final_price", "sum"),
                               avg_spend=("final_price", "mean"),
                               return_rate=("returned", "mean"))
                          .reset_index())
        seg_summary["return_rate"] = (seg_summary["return_rate"] * 100).round(2)
        st.dataframe(seg_summary.style.format({
            "total_spend": "₹{:,.0f}", "avg_spend": "₹{:,.0f}",
            "return_rate": "{:.2f}%", "orders": "{:,}"
        }), hide_index=True, use_container_width=True)

# ── TAB REG: Revenue Prediction (Regression Baseline + Random Forest) ──
with tab_reg:
    st.subheader("How much of final_price is actually predictable?")
    st.caption("Mirrors the notebook's Section 7 (OLS baseline) and Section 9 "
               "(Random Forest) — since final_price = product_price × quantity × "
               "(1 − discount/100), this should (and does) solve almost perfectly.")

    if not SKLEARN_AVAILABLE:
        st.warning("`scikit-learn` isn't installed, so the regression models can't run.")
    elif len(df) < 200:
        st.warning("Not enough rows in the current filter to train/test a model reliably. "
                   "Try widening your filters.")
    else:
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score, mean_absolute_error

        feat_cols = ["product_price", "quantity", "discount_percentage"]

        @st.cache_data(show_spinner="Training regression models...")
        def train_revenue_models(data):
            X = data[feat_cols]
            y = data["final_price"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42)

            lr = LinearRegression().fit(X_train, y_train)
            lr_pred = lr.predict(X_test)

            rf = RandomForestRegressor(n_estimators=150, max_depth=14,
                                        random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)

            return dict(
                lr_r2=r2_score(y_test, lr_pred), lr_mae=mean_absolute_error(y_test, lr_pred),
                rf_r2=r2_score(y_test, rf_pred), rf_mae=mean_absolute_error(y_test, rf_pred),
                lr_coefs=pd.Series(lr.coef_, index=feat_cols),
                rf_importance=pd.Series(rf.feature_importances_, index=feat_cols).sort_values(),
                y_test=y_test.values, lr_pred=lr_pred, rf_pred=rf_pred,
                n_train=len(X_train), n_test=len(X_test),
            )

        result = train_revenue_models(df)

        st.markdown("**Linear Regression (OLS baseline) vs. Random Forest**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Linear R²", f"{result['lr_r2']:.4f}")
        m2.metric("Linear MAE", f"₹{result['lr_mae']:,.0f}")
        m3.metric("Random Forest R²", f"{result['rf_r2']:.4f}")
        m4.metric("Random Forest MAE", f"₹{result['rf_mae']:,.0f}")
        st.caption(f"Trained on {result['n_train']:,} rows, tested on {result['n_test']:,} "
                   f"held-out rows. Features: {', '.join(feat_cols)}.")

        c1, c2 = st.columns(2)
        with c1:
            sample_idx = np.random.RandomState(42).choice(len(result["y_test"]),
                                                            min(3000, len(result["y_test"])),
                                                            replace=False)
            fig = px.scatter(x=result["y_test"][sample_idx], y=result["rf_pred"][sample_idx],
                              opacity=0.35, color_discrete_sequence=[SAGE],
                              title=f"Random Forest: Actual vs Predicted (R²={result['rf_r2']:.4f})",
                              labels={"x": "Actual final_price (₹)", "y": "Predicted final_price (₹)"})
            lims = [result["y_test"].min(), result["y_test"].max()]
            fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                                      line=dict(color=RUST, dash="dash"), name="Perfect prediction"))
            fig.update_layout(**LAYOUT_KWARGS)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(result["rf_importance"], orientation="h",
                         color_discrete_sequence=[FERN],
                         title="Random Forest Feature Importance")
            fig.update_layout(**LAYOUT_KWARGS, showlegend=False,
                               xaxis_title="Importance", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Linear regression coefficients (standardized scale not applied — raw units)**")
        st.dataframe(result["lr_coefs"].reset_index().rename(
            columns={"index": "Feature", 0: "Coefficient"}), hide_index=True,
            use_container_width=True)

        gap = result["rf_r2"] - result["lr_r2"]
        if result["rf_r2"] > 0.99:
            st.success(
                f"Random Forest R² of {result['rf_r2']:.4f} confirms `final_price` is "
                "essentially solved by these 3 fields — consistent with it being a "
                "**mathematically derived** quantity (product_price × quantity × "
                "(1 − discount%)), not something driven by real-world market dynamics."
            )
        st.caption(f"Random Forest improves {gap:.4f} R² over the linear baseline, "
                   "reflecting the small multiplicative (non-additive) interaction between "
                   "price, quantity, and discount that a linear model can't fully capture.")

        st.markdown("---")
        st.subheader("Residual diagnostics — is the unexplained variance structured or random?")
        st.caption("Mirrors the notebook's Section 7 residual diagnostics, run on the "
                   "Linear Regression baseline (the model with meaningful unexplained "
                   "variance — Random Forest leaves almost none to diagnose).")
        residuals = result["y_test"] - result["lr_pred"]
        c1, c2 = st.columns(2)
        with c1:
            sample_idx2 = np.random.RandomState(7).choice(len(residuals),
                                                            min(3000, len(residuals)), replace=False)
            fig = px.scatter(x=result["lr_pred"][sample_idx2], y=residuals.values[sample_idx2] if hasattr(residuals, "values") else residuals[sample_idx2],
                              opacity=0.35, color_discrete_sequence=[GORGE],
                              title="Residuals vs Fitted",
                              labels={"x": "Fitted final_price (₹)", "y": "Residual (₹)"})
            fig.add_hline(y=0, line_dash="dash", line_color=RUST)
            fig.update_layout(**LAYOUT_KWARGS)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            osm, osr = stats.probplot(residuals, dist="norm", fit=False)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
                                      marker=dict(color=SAGE, opacity=0.5), name="Residuals"))
            line_x = np.array([osm.min(), osm.max()])
            fig.add_trace(go.Scatter(x=line_x, y=line_x * residuals.std() + residuals.mean(),
                                      mode="lines", line=dict(color=RUST, dash="dash"),
                                      name="Normal reference"))
            fig.update_layout(**LAYOUT_KWARGS, title="Normal Q-Q Plot of Residuals",
                               xaxis_title="Theoretical quantiles", yaxis_title="Ordered residuals")
            st.plotly_chart(fig, use_container_width=True)

        resid_skew, resid_kurt = stats.skew(residuals), stats.kurtosis(residuals)
        fitted_median = np.median(result["lr_pred"])
        low_group = residuals[result["lr_pred"] <= fitted_median]
        high_group = residuals[result["lr_pred"] > fitted_median]
        lev_stat, lev_p = stats.levene(low_group, high_group)
        het_ratio = np.var(high_group) / np.var(low_group) if np.var(low_group) > 0 else np.nan

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Residual skew", f"{resid_skew:+.3f}")
        m2.metric("Residual kurtosis", f"{resid_kurt:+.3f}")
        m3.metric("Heteroscedasticity ratio", f"{het_ratio:.2f}",
                  help="Variance of residuals in the high-fitted-value half ÷ low half. "
                       "Near 1.0 = homoscedastic (constant variance).")
        m4.metric("Levene's test p-value", f"{lev_p:.4f}",
                  help="Tests whether residual variance differs between low- and "
                       "high-fitted-value halves.")
        het_verdict = "roughly constant (homoscedastic)" if lev_p >= 0.05 else "not constant (heteroscedastic)"
        shape_verdict = "close to normal" if abs(resid_skew) < 0.5 and abs(resid_kurt) < 1 else "notably non-normal"
        st.caption(
            f"Residuals are {shape_verdict} in shape (skew {resid_skew:+.3f}, kurtosis "
            f"{resid_kurt:+.3f}) and their spread is {het_verdict} across the range of "
            f"predictions (Levene p = {lev_p:.4f}). Read together: the ~11% of variance "
            "the linear model doesn't capture looks like genuine structure from the "
            "multiplicative price×quantity×discount interaction (which Random Forest "
            "picks up), not random noise."
        )

# ── TAB 8: Returns Prediction Model ─────────────────────────────
with tab8:
    st.subheader("Can we actually predict a return?")
    st.caption("This tab backs up the 'returns can't be predicted' claim in Key Insights "
               "with a real model on the current filtered data, instead of just asserting it.")

    if df["returned"].nunique() < 2 or len(df) < 200:
        st.warning("Not enough data / class variety in the current filter selection "
                   "to train a model. Try widening your filters.")
    elif not SKLEARN_AVAILABLE:
        st.warning(
            "`scikit-learn` isn't installed in this environment, so the logistic "
            "regression model can't run. Install it with `pip install scikit-learn` "
            "to enable this section. Showing simple point-biserial correlations "
            "with the return flag as a fallback."
        )
        rows = []
        for c in NUM_COLS:
            corr, p = stats.pointbiserialr(df["returned"], df[c])
            rows.append({"Variable": c, "Correlation with returned": round(corr, 4), "p-value": round(p, 4)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        @st.cache_data(show_spinner="Training logistic regression model...")
        def train_return_model(data):
            feat_num = ["age", "product_price", "quantity", "discount_percentage",
                        "final_price", "delivery_days"]
            feat_cat = ["gender", "customer_segment", "product_category",
                        "payment_method", "shipping_type"]
            X = pd.get_dummies(data[feat_num + feat_cat], columns=feat_cat, drop_first=True)
            y = data["returned"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train_s, y_train)
            probs = model.predict_proba(X_test_s)[:, 1]
            preds = (probs >= 0.5).astype(int)
            return dict(
                auc=roc_auc_score(y_test, probs),
                fpr_tpr=roc_curve(y_test, probs)[:2],
                acc=accuracy_score(y_test, preds),
                prec=precision_score(y_test, preds, zero_division=0),
                rec=recall_score(y_test, preds, zero_division=0),
                cm=confusion_matrix(y_test, preds),
                coefs=pd.Series(model.coef_[0], index=X.columns).sort_values(),
                n_train=len(X_train), n_test=len(X_test),
                base_rate=y.mean(),
            )

        result = train_return_model(df)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("AUC-ROC", f"{result['auc']:.3f}",
                  help="0.5 = no better than a coin flip. 1.0 = perfect separation.")
        m2.metric("Accuracy", f"{result['acc']*100:.1f}%")
        m3.metric("Precision", f"{result['prec']*100:.1f}%")
        m4.metric("Recall", f"{result['rec']*100:.1f}%")
        st.caption(f"Trained on {result['n_train']:,} rows, tested on {result['n_test']:,} "
                   f"held-out rows. Base return rate in this slice: {result['base_rate']*100:.2f}%.")

        c1, c2 = st.columns(2)
        with c1:
            fpr, tpr = result["fpr_tpr"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                      line=dict(color=RUST, width=2.5),
                                      name=f"Model (AUC={result['auc']:.3f})"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                      line=dict(color=GORGE, dash="dash"), name="Random guess"))
            fig.update_layout(**LAYOUT_KWARGS, title="ROC Curve",
                               xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            cm = result["cm"]
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="YlOrBr",
                             x=["Pred: No Return", "Pred: Return"],
                             y=["Actual: No Return", "Actual: Return"],
                             title="Confusion Matrix (test set)")
            fig.update_layout(**LAYOUT_KWARGS)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Feature importance (standardized logistic regression coefficients)")
        coefs = result["coefs"]
        top = pd.concat([coefs.head(8), coefs.tail(8)]).drop_duplicates().sort_values()
        fig = px.bar(top, orientation="h",
                     color=top.values > 0,
                     color_discrete_map={True: RUST, False: FERN},
                     title="Strongest Predictors of Return (± coefficient weight)")
        fig.update_layout(**LAYOUT_KWARGS, showlegend=False,
                           xaxis_title="Coefficient (standardized)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        if result["auc"] < 0.6:
            st.info(
                f"AUC of {result['auc']:.3f} is close to 0.5, confirming the claim in "
                "Key Insights: the available features carry very little signal for "
                "predicting returns. A production model would need richer features "
                "(product defect rate, review sentiment, complaint codes) to do better."
            )
        else:
            st.success(
                f"AUC of {result['auc']:.3f} suggests these features carry more predictive "
                "signal on the current filtered slice than the 'unpredictable' claim implies — "
                "worth re-checking that claim against this filter selection."
            )

# ── TAB BIZ: Business Questions ──────────────────────────────────
with tab_biz:
    st.caption("Mirrors the notebook's Section 10 — 10 direct business questions "
               "answered on the current filtered slice, each with its own caveat.")

    total_rev = df["final_price"].sum()

    with st.expander("🚀 1. Which product category brings in the most total revenue?", expanded=True):
        cat_rev = df.groupby("product_category")["final_price"].sum().sort_values(ascending=False)
        cat_loss = df.groupby("product_category").apply(
            lambda g: g.loc[g["returned"] == 1, "final_price"].sum() / g["final_price"].sum() * 100
            if g["final_price"].sum() > 0 else 0)
        fig = px.bar(x=cat_rev.index, y=cat_rev.values / 1e7, color=cat_rev.index,
                     color_discrete_sequence=PALETTE, title="Revenue by Category (₹ Cr)",
                     labels={"x": "Category", "y": "Revenue (₹ Cr)"})
        fig.update_layout(**LAYOUT_KWARGS, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        top_cat = cat_rev.index[0]
        st.markdown(f"**Answer:** {top_cat} leads with **{cat_rev.iloc[0]/total_rev*100:.2f}%** "
                    f"revenue share.")
        st.caption(f"**Caveat:** highest loss-rate category is "
                   f"**{cat_loss.idxmax()}** ({cat_loss.max():.2f}%) — leading on volume "
                   "doesn't mean leading on efficiency.")

    with st.expander("🛍️ 2. Total revenue vs. average order value by customer segment"):
        seg_stats = df.groupby("customer_segment")["final_price"].agg(
            total="sum", aov="mean").reset_index()
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(seg_stats, x="customer_segment", y="total", color="customer_segment",
                         color_discrete_map=SEG_COLORS, title="Total Revenue by Segment")
            fig.update_layout(**LAYOUT_KWARGS, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(seg_stats, x="customer_segment", y="aov", color="customer_segment",
                         color_discrete_map=SEG_COLORS, title="Average Order Value by Segment")
            fig.update_layout(**LAYOUT_KWARGS, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        rev_leader = seg_stats.loc[seg_stats["total"].idxmax(), "customer_segment"]
        aov_leader = seg_stats.loc[seg_stats["aov"].idxmax(), "customer_segment"]
        st.markdown(f"**Answer:** **{rev_leader}** leads total revenue; **{aov_leader}** "
                    "leads average order value.")
        st.caption("**Caveat:** the ranking flips depending which metric you ask about — "
                   "segments are close to identical on a per-order basis.")

    with st.expander("🧾 3. Monthly/quarterly revenue trend — is there real seasonality?"):
        st.write("See the **📆 Time Series** tab for the full trend test and seasonal "
                 "decomposition.")
        m = df.groupby("order_month")["final_price"].sum().reset_index().sort_values("order_month")
        if len(m) >= 3:
            s, i, r, p, se = stats.linregress(np.arange(len(m)), m["final_price"])
            st.markdown(f"**Answer:** Trend p = {p:.3f} → "
                        f"{'a real long-run trend' if p < 0.05 else 'no real long-run trend'}. "
                        "Seasonality (within-year shape) is a more nuanced 'maybe' — see Time Series tab.")

    with st.expander("💰 4. Which state/city generates the most revenue?"):
        st.write("See the **🌍 Geography** tab for the full breakdown and the state-level artifact.")
        state_share = (df.groupby("state")["final_price"].sum() / total_rev * 100).sort_values(ascending=False)
        st.markdown(f"**Answer:** {state_share.index[0]} 'leads' at {state_share.iloc[0]:.1f}% by state — "
                    "flagged as an artifact of city-to-state mapping, not a real geographic effect.")

    with st.expander("💳 5. Revenue and return-related loss by payment method"):
        pay_rev = df.groupby("payment_method")["final_price"].sum().sort_values(ascending=False)
        pay_loss = df.groupby("payment_method").apply(
            lambda g: g.loc[g["returned"] == 1, "final_price"].sum() / g["final_price"].sum() * 100)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(x=pay_rev.index, y=pay_rev.values / 1e7, color=pay_rev.index,
                         color_discrete_sequence=PALETTE, title="Revenue by Payment Method (₹ Cr)")
            fig.update_layout(**LAYOUT_KWARGS, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(x=pay_loss.index, y=pay_loss.values, color=pay_loss.index,
                         color_discrete_sequence=PALETTE, title="Return-Loss Rate by Payment Method (%)")
            fig.update_layout(**LAYOUT_KWARGS, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        contingency = pd.crosstab(df["payment_method"], df["return_status"])
        chi2, p_chi, _, _ = stats.chi2_contingency(contingency)
        st.markdown(f"**Answer:** {pay_rev.idxmax()} leads revenue; {pay_loss.idxmax()} has "
                    "the highest loss rate.")
        st.caption(f"**Caveat:** χ² = {chi2:.2f}, p = {p_chi:.4f} — "
                   f"{'a real' if p_chi < 0.05 else 'not a statistically real'} difference.")

    with st.expander("💰 6. Does quantity affect unit price?"):
        r, p = stats.pearsonr(df["quantity"], df["product_price"])
        fig = px.scatter(df.sample(min(4000, len(df)), random_state=42), x="quantity",
                          y="product_price", opacity=0.35, color_discrete_sequence=[SAGE],
                          title=f"Quantity vs Product Price (r = {r:+.4f})")
        fig.update_layout(**LAYOUT_KWARGS)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Answer:** r = {r:+.4f}, p = {p:.4f} — "
                    f"{'yes, a real relationship' if p < 0.05 else 'no, essentially no relationship'} "
                    "(no bulk-pricing effect).")

    with st.expander("👥 7. Average discount by category"):
        disc_cat = df.groupby("product_category")["discount_percentage"].mean().sort_values(ascending=False)
        fig = px.bar(x=disc_cat.index, y=disc_cat.values, color=disc_cat.index,
                     color_discrete_sequence=PALETTE, title="Avg Discount % by Category")
        fig.add_hline(y=df["discount_percentage"].mean(), line_dash="dash", line_color=RUST,
                      annotation_text="Overall avg")
        fig.update_layout(**LAYOUT_KWARGS, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        f_stat, p_val = stats.f_oneway(*[g["discount_percentage"].values
                                          for _, g in df.groupby("product_category")])
        st.markdown(f"**Answer:** range {disc_cat.min():.2f}%–{disc_cat.max():.2f}% — essentially flat.")
        st.caption(f"**Caveat:** ANOVA F = {f_stat:.2f}, p = {p_val:.4f} — "
                   f"{'a real' if p_val < 0.05 else 'no real'} category effect on discount level.")

    with st.expander("📅 8. Revenue lost to returns, by category"):
        lost = df.loc[df["returned"] == 1].groupby("product_category")["final_price"].sum()
        lost_pct = (lost / total_rev * 100).sort_values(ascending=False)
        fig = px.bar(x=lost_pct.index, y=lost_pct.values, color=lost_pct.index,
                     color_discrete_sequence=PALETTE, title="Revenue Lost to Returns, % of Total, by Category")
        fig.update_layout(**LAYOUT_KWARGS, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        total_lost = df.loc[df["returned"] == 1, "final_price"].sum()
        st.markdown(f"**Answer:** ₹{total_lost/1e7:.2f} Cr lost overall "
                    f"({total_lost/total_rev*100:.2f}% of revenue); highest single category: "
                    f"**{lost_pct.idxmax()}** ({lost_pct.max():.2f}% of total revenue).")
        st.caption("**Caveat:** loss tends to track category size (bigger category → bigger "
                   "absolute loss), not concentrated disproportionately in any one category.")

    with st.expander("💳 9. Category × segment interaction"):
        pivot_cs = df.pivot_table(values="final_price", index="product_category",
                                   columns="customer_segment", aggfunc="mean", observed=True)
        fig = px.imshow(pivot_cs, text_auto=".0f", color_continuous_scale="YlOrBr",
                         title="Avg Final Price: Category × Segment")
        fig.update_layout(**LAYOUT_KWARGS)
        st.plotly_chart(fig, use_container_width=True)
        groups = [g["final_price"].values for _, g in df.groupby(["product_category", "customer_segment"])]
        f_stat, p_val = stats.f_oneway(*groups)
        st.markdown(f"**Answer:** F = {f_stat:.2f}, p = {p_val:.4f} — "
                    f"{'a real' if p_val < 0.05 else 'no real'} interaction effect, confirming "
                    "the Multivariate tab at a finer grain.")

    with st.expander("🐋 10. Whale-transaction concentration"):
        def whale_share(g):
            thresh = g["final_price"].quantile(0.9)
            top = g.loc[g["final_price"] >= thresh, "final_price"].sum()
            return top / g["final_price"].sum() * 100 if g["final_price"].sum() > 0 else 0

        overall_whale = whale_share(df)
        by_cat_whale = df.groupby("product_category").apply(whale_share)
        fig = px.bar(x=by_cat_whale.index, y=by_cat_whale.values, color=by_cat_whale.index,
                     color_discrete_sequence=PALETTE,
                     title="Top-10%-of-Orders Revenue Share, by Category (%)")
        fig.add_hline(y=overall_whale, line_dash="dash", line_color=RUST,
                      annotation_text=f"Overall: {overall_whale:.1f}%")
        fig.update_layout(**LAYOUT_KWARGS, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Answer:** top 10% of orders drive **{overall_whale:.1f}%** of revenue "
                    "overall — and this is nearly identical across every category slice "
                    f"(range {by_cat_whale.min():.1f}%–{by_cat_whale.max():.1f}%), so there's no "
                    "segment where whale transactions matter disproportionately more.")

# ── TAB STAT: Statistical Rigor ──────────────────────────────────
with tab_stat:
    st.caption("Closes the remaining gap vs. the notebook: goodness-of-fit for uniformity, "
               "non-parametric robustness tests, a confounding/partial-correlation check, "
               "bootstrap CIs on effect sizes, formal statistical power / minimum-detectable-"
               "effect analysis, and Benjamini-Hochberg FDR correction across many tests at once.")

    all_cat_cols = CAT_COLS[:-1] + GEO_COLS + ["product_subcategory"]  # drop return_status (it's the target elsewhere)

    # ── 1. Goodness-of-fit for uniformity ──────────────────────
    with st.expander("① Goodness-of-fit — are categorical variables really uniform?", expanded=True):
        st.caption("A formal chi-square goodness-of-fit test against a uniform distribution, "
                   "instead of just eyeballing flat bar charts.")
        gof_col = st.selectbox("Variable", all_cat_cols, index=2, key="gof_col")
        obs = df[gof_col].value_counts().sort_index()
        expected = np.full(len(obs), obs.sum() / len(obs))
        chi2_gof, p_gof = stats.chisquare(obs.values, expected)
        c1, c2 = st.columns([2, 1])
        with c1:
            comp = pd.DataFrame({gof_col: obs.index, "Observed": obs.values, "Expected (uniform)": expected})
            fig = go.Figure()
            fig.add_trace(go.Bar(x=comp[gof_col], y=comp["Observed"], name="Observed", marker_color=RUST))
            fig.add_trace(go.Scatter(x=comp[gof_col], y=comp["Expected (uniform)"], mode="lines+markers",
                                      name="Expected (uniform)", line=dict(color=GORGE, dash="dash")))
            fig.update_layout(**LAYOUT_KWARGS, title=f"{gof_col}: Observed vs. Uniform-Expected Counts")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.metric("χ² statistic", f"{chi2_gof:,.2f}")
            st.metric("p-value", f"{p_gof:.4f}")
            verdict = "rejects" if p_gof < 0.05 else "does not reject"
            st.caption(f"The test {verdict} the null hypothesis of a uniform distribution "
                       f"at α = 0.05.")
        max_dev = ((obs - expected) / expected * 100).abs().max()
        st.caption(f"Largest single-category deviation from an even split: {max_dev:.2f}%.")

    # ── 2. Confounding / partial correlation check ─────────────
    with st.expander("② Confounding check — does discount_percentage survive controlling for price?"):
        st.caption("Regresses both discount_percentage and final_price on product_price, "
                   "then correlates the leftover residuals — the correlation that remains "
                   "once product_price's shared influence is removed.")
        x_raw = df["discount_percentage"].values.astype(float)
        y_raw = df["final_price"].values.astype(float)
        z_raw = df["product_price"].values.astype(float)
        raw_r, raw_p = stats.pearsonr(x_raw, y_raw)

        mz, bz = np.polyfit(z_raw, x_raw, 1)
        x_resid = x_raw - (mz * z_raw + bz)
        mz2, bz2 = np.polyfit(z_raw, y_raw, 1)
        y_resid = y_raw - (mz2 * z_raw + bz2)
        partial_r, partial_p = stats.pearsonr(x_resid, y_resid)

        m1, m2 = st.columns(2)
        m1.metric("Raw correlation (discount vs final_price)", f"r = {raw_r:+.4f}", help=f"p = {raw_p:.4g}")
        m2.metric("Partial correlation (controlling for product_price)", f"r = {partial_r:+.4f}",
                   help=f"p = {partial_p:.4g}")
        fig = px.scatter(x=x_resid[np.random.RandomState(0).choice(len(x_resid), min(4000, len(x_resid)), replace=False)],
                          y=y_resid[np.random.RandomState(0).choice(len(y_resid), min(4000, len(y_resid)), replace=False)],
                          opacity=0.3, color_discrete_sequence=[SAGE],
                          title="Residualized discount_percentage vs Residualized final_price (price effect removed)",
                          labels={"x": "discount_percentage residual", "y": "final_price residual"})
        fig.update_layout(**LAYOUT_KWARGS)
        st.plotly_chart(fig, use_container_width=True)
        direction = "strengthens" if abs(partial_r) > abs(raw_r) else "weakens"
        st.caption(f"Controlling for product_price {direction} the discount–revenue relationship "
                   f"(|r| goes from {abs(raw_r):.4f} to {abs(partial_r):.4f}) — a real but modest "
                   "confound/suppression effect, not a spurious one caused entirely by price.")

    # ── 3. Non-parametric robustness suite ──────────────────────
    with st.expander("③ Non-parametric tests — Kruskal-Wallis, Levene, Mann-Whitney U, K-S"):
        st.caption("ANOVA and Pearson correlation assume normal, equal-variance groups. "
                   "These rank-based alternatives don't — run here as a robustness check.")
        np_num = st.selectbox("Numeric variable", NUM_COLS, index=4, key="np_num")
        np_cat = st.selectbox("Group by", all_cat_cols, index=2, key="np_cat")
        groups_np = [g[np_num].values for _, g in df.groupby(np_cat, observed=True)]
        group_names = [name for name, _ in df.groupby(np_cat, observed=True)]

        kw_stat, kw_p = stats.kruskal(*groups_np)
        lev_stat2, lev_p2 = stats.levene(*groups_np)
        c1, c2 = st.columns(2)
        c1.metric("Kruskal-Wallis H", f"{kw_stat:,.2f}", help=f"p = {kw_p:.4g}")
        c1.caption(f"Non-parametric alternative to one-way ANOVA — "
                   f"{'a real' if kw_p < 0.05 else 'no real'} difference in {np_num} across "
                   f"{np_cat} groups (p = {kw_p:.4f}).")
        c2.metric("Levene's test", f"{lev_stat2:,.2f}", help=f"p = {lev_p2:.4g}")
        c2.caption(f"Tests equal variance across groups — variances are "
                   f"{'unequal' if lev_p2 < 0.05 else 'statistically equal'} (p = {lev_p2:.4f}), "
                   f"which tells you whether ANOVA's equal-variance assumption is safe here.")

        st.markdown("**Pairwise comparison (Mann-Whitney U + Kolmogorov-Smirnov)**")
        pc1, pc2 = st.columns(2)
        with pc1:
            grp_a_name = st.selectbox("Group A", group_names, index=0, key="mw_a")
        with pc2:
            grp_b_name = st.selectbox("Group B", group_names, index=min(1, len(group_names) - 1), key="mw_b")
        if grp_a_name == grp_b_name:
            st.info("Choose two different groups to compare.")
        else:
            grp_a = df.loc[df[np_cat] == grp_a_name, np_num].values
            grp_b = df.loc[df[np_cat] == grp_b_name, np_num].values
            mw_stat, mw_p = stats.mannwhitneyu(grp_a, grp_b, alternative="two-sided")
            ks_stat, ks_p = stats.kstest(grp_a, grp_b)
            m1, m2 = st.columns(2)
            m1.metric("Mann-Whitney U p-value", f"{mw_p:.4f}",
                      help="Non-parametric alternative to a two-sample t-test — compares "
                           "medians/rank distributions rather than means.")
            m2.metric("Kolmogorov-Smirnov p-value", f"{ks_p:.4f}",
                      help="Tests whether the two groups' full distributions (not just "
                           "central tendency) differ.")
            mw_verdict = "a real" if mw_p < 0.05 else "no real"
            ks_verdict = "differ" if ks_p < 0.05 else "don't detectably differ"
            st.caption(f"**{grp_a_name}** vs **{grp_b_name}** on `{np_num}`: Mann-Whitney finds "
                       f"{mw_verdict} median difference (p = {mw_p:.4f}); K-S finds the two full "
                       f"distributions {ks_verdict} (p = {ks_p:.4f}).")

    # ── 4. Bootstrap confidence intervals on effect sizes ───────
    with st.expander("④ Bootstrap confidence intervals on effect sizes"):
        st.caption("A single Cramér's V or η² is a point estimate — bootstrap resampling "
                   "shows how precise that estimate actually is.")
        boot_choice = st.radio("Effect size to bootstrap", ["Cramér's V (vs. return_status)",
                                                              "η² (vs. final_price)"],
                                horizontal=True, key="boot_choice")
        boot_col = st.selectbox("Categorical variable", all_cat_cols, index=2, key="boot_col")

        @st.cache_data(show_spinner="Bootstrapping (200 resamples)...")
        def _run_bootstrap(data, col, kind):
            if kind.startswith("Cramér"):
                codes1, _ = pd.factorize(data[col])
                codes2, _ = pd.factorize(data["return_status"])
                return bootstrap_cramers_v(codes1, codes2)
            else:
                codes, _ = pd.factorize(data[col])
                vals = data["final_price"].values.astype(float)
                return bootstrap_eta_squared(codes, vals)

        boot_vals = _run_bootstrap(df, boot_col, boot_choice)
        lo, hi = np.percentile(boot_vals, [2.5, 97.5])
        point_est = np.median(boot_vals)
        m1, m2, m3 = st.columns(3)
        m1.metric("Point estimate (median of resamples)", f"{point_est:.4f}")
        m2.metric("95% CI lower", f"{lo:.4f}")
        m3.metric("95% CI upper", f"{hi:.4f}")
        fig = px.histogram(x=boot_vals, nbins=40, color_discrete_sequence=[SAGE],
                            title=f"Bootstrap distribution — {boot_choice} for {boot_col}")
        fig.add_vline(x=lo, line_dash="dash", line_color=RUST, annotation_text="2.5%")
        fig.add_vline(x=hi, line_dash="dash", line_color=RUST, annotation_text="97.5%")
        fig.update_layout(**LAYOUT_KWARGS, xaxis_title="Effect size", yaxis_title="Resample count")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"A 95% CI of [{lo:.4f}, {hi:.4f}] that sits entirely near 0 means the effect "
                   "isn't just statistically non-significant — it's precisely estimated to be "
                   "small, which is a stronger claim than a single p-value can make.")

    # ── 5. Statistical power / minimum detectable effect ───────
    with st.expander("⑤ Statistical power & minimum detectable effect"):
        st.caption("With n this large, what's the smallest true effect these tests could "
                   "actually detect? Computed via the noncentral F distribution — the same "
                   "logic as the notebook's power analysis, without needing statsmodels.")
        pow_col = st.selectbox("Categorical variable (for group count k)", all_cat_cols,
                                index=2, key="pow_col")
        n_current = len(df)
        k_current = df[pow_col].nunique()
        mde = anova_minimum_detectable_effect(n_current, k_current)
        m1, m2, m3 = st.columns(3)
        m1.metric("Sample size (n)", f"{n_current:,}")
        m2.metric("Groups (k)", f"{k_current}")
        m3.metric("Minimum detectable effect (Cohen's f, 80% power)", f"{mde:.4f}" if not np.isnan(mde) else "—")

        f_range = np.linspace(0.001, max(0.15, mde * 2 if not np.isnan(mde) else 0.15), 100)
        power_curve = [anova_power(f, n_current, k_current) for f in f_range]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=f_range, y=power_curve, mode="lines",
                                  line=dict(color=RUST, width=2.5), name="Power"))
        fig.add_hline(y=0.8, line_dash="dash", line_color=GORGE, annotation_text="80% power")
        if not np.isnan(mde):
            fig.add_vline(x=mde, line_dash="dash", line_color=GORGE,
                          annotation_text=f"MDE ≈ {mde:.3f}")
        fig.update_layout(**LAYOUT_KWARGS, title=f"Power Curve — One-Way ANOVA on `{pow_col}` (n={n_current:,}, k={k_current})",
                           xaxis_title="Effect size (Cohen's f)", yaxis_title="Statistical power")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"At this sample size, an 80%-power test on `{pow_col}` could detect an "
                   f"effect as small as Cohen's f ≈ {mde:.4f} (roughly η² ≈ {mde**2/(1+mde**2):.5f}) "
                   "— far smaller than anything practically meaningful. That's the formal basis "
                   "for treating this dataset's many 'not significant' and 'barely significant' "
                   "results as genuine null findings rather than underpowered tests.")

    # ── 6. Benjamini-Hochberg FDR correction ─────────────────────
    with st.expander("⑥ Benjamini-Hochberg FDR correction across many tests at once"):
        st.caption("Running one test is fine at α = 0.05. Running dozens raises the chance "
                   "some 'significant' results are false positives from multiple testing — "
                   "this consolidates every categorical-association test in this dashboard "
                   "into one table and corrects for that.")

        @st.cache_data(show_spinner="Running the full test battery...")
        def _run_fdr_battery(data):
            rows = []
            for col in all_cat_cols:
                groups = [g["final_price"].values for _, g in data.groupby(col, observed=True)]
                f_stat, p_anova = stats.f_oneway(*groups)
                rows.append({"Test": f"ANOVA: final_price ~ {col}", "Statistic": round(f_stat, 3),
                             "Raw p-value": p_anova})
                ct = pd.crosstab(data[col], data["return_status"])
                chi2, p_chi, _, _ = stats.chi2_contingency(ct)
                rows.append({"Test": f"Chi-square: return_status ~ {col}", "Statistic": round(chi2, 3),
                             "Raw p-value": p_chi})
            return pd.DataFrame(rows)

        fdr_df = _run_fdr_battery(df)
        fdr_df["BH-adjusted p-value"] = benjamini_hochberg(fdr_df["Raw p-value"].values).round(4)
        fdr_df["Raw p-value"] = fdr_df["Raw p-value"].round(4)
        fdr_df["Significant (raw, α=0.05)"] = fdr_df["Raw p-value"] < 0.05
        fdr_df["Significant (BH-adjusted, α=0.05)"] = fdr_df["BH-adjusted p-value"] < 0.05
        fdr_df = fdr_df.sort_values("Raw p-value")
        st.dataframe(fdr_df, hide_index=True, use_container_width=True)

        n_sig_raw = fdr_df["Significant (raw, α=0.05)"].sum()
        n_sig_bh = fdr_df["Significant (BH-adjusted, α=0.05)"].sum()
        st.caption(f"{len(fdr_df)} tests run. {n_sig_raw} significant at raw α = 0.05; "
                   f"{n_sig_bh} remain significant after Benjamini-Hochberg correction. "
                   f"{'Every raw finding survives correction' if n_sig_raw == n_sig_bh else f'{n_sig_raw - n_sig_bh} raw finding(s) were false-positive risks flagged by the correction'} "
                   "— consistent with the notebook's conclusion that its handful of "
                   "significant results are robust, not multiple-testing artifacts.")

# ── TAB 9: Key Insights ──────────────────────────────────────────
with tab9:
    st.markdown("""
    ### 💡 Headline Findings

    - **Product price and quantity are the primary revenue drivers** — discount level,
      demographics, and payment method show negligible influence on order value
      (see the VIF and ANOVA readouts on the Bivariate tab).
    - **Customer segment alone doesn't predict spend** — but Premium customers aged 30–45
      spend meaningfully more, an interaction only visible when segment and age are
      considered together (Multivariate tab).
    - **Returns are hard to predict** from the variables available (price, category,
      demographics, delivery time) — the logistic regression on the Returns Model tab
      typically lands close to AUC ≈ 0.5 (random-guess territory) on this data.
    - **Shipping tier has no measurable effect on delivery speed** in this data (Express vs
      Standard delivery days are statistically identical) — worth flagging as a data-quality
      observation.
    - The near-uniform spread across every categorical variable, the complete absence of
      missing values/duplicates (Data Quality tab), and non-normal numeric distributions
      (Univariate tab) together suggest this dataset is likely **synthetically generated** —
      insights here should be treated as methodologically sound but not yet validated against
      live transaction data.
    - **Multicollinearity is low** among the numeric predictors (VIF well under 5 for all of
      them), so the correlation and regression results aren't being distorted by redundant
      features.

    ### 📌 Recommendations
    1. Prioritize pricing and basket-size (quantity) strategy over segment-based marketing.
    2. Investigate the Premium × 30–45 age segment further for targeted offers.
    3. Don't invest further in a returns-prediction model on this dataset — the AUC on the
       Returns Model tab shows it lacks the signal; richer features (product defect rate,
       review sentiment, complaint reason codes) would be needed first.
    4. If a real `customer_id` becomes available, revisit the Customer/RFM tab for a proper
       per-customer recency/frequency/monetary segmentation — that's currently a gap in
       this dataset.

    *Every claim above has a corresponding chart or test elsewhere in this dashboard —
    use the tabs to verify any of them against whatever filter slice you're currently viewing.*
    """)

    st.markdown("---")
    st.subheader("📌 Business Recommendations (Section 11)")
    reco_df = pd.DataFrame([
        {"Priority": "High", "Insight": "Pricing drives revenue (product_price → final_price, r ≈ +0.73)",
         "Recommendation": "Focus on premium product placement over volume discounting"},
        {"Priority": "High", "Insight": "Discounts reduce revenue as they rise",
         "Recommendation": "Cap discounts near 10%; each % above costs meaningful revenue"},
        {"Priority": "Medium", "Insight": "Customer segments barely differ (≈1% gap on revenue/AOV)",
         "Recommendation": "Redesign segmentation criteria — current segments aren't actionable"},
        {"Priority": "Medium", "Insight": "Express shipping shows no measurable speed advantage over Standard",
         "Recommendation": "Investigate logistics — this looks like an operational anomaly"},
        {"Priority": "Low", "Insight": "Returns are statistically uniform across every dimension tested",
         "Recommendation": "Audit product quality/sizing broadly, not by category"},
        {"Priority": "Low", "Insight": "Category-level revenue is remarkably flat",
         "Recommendation": "Category-level inventory shifts aren't supported by this data"},
    ])
    st.dataframe(reco_df, hide_index=True, use_container_width=True)

    st.subheader("⚠️ Limitations")
    st.markdown("""
    - **Likely synthetic data:** near-perfect symmetry and uniform category distributions
      across the board (see Data Quality tab) strongly suggest this dataset is
      synthetically generated — real retail data is typically far messier.
    - **No causal inference:** all relationships shown are correlational; `product_price`
      correlates with `final_price` because it mathematically determines it, not because
      of market dynamics.
    - **Large-sample inflation, addressed:** with n≈100,000, conventional p<0.05 tests can
      flag trivial effects. This dashboard reports effect size (η², Cramér's V) and VIF
      alongside p-values throughout, rather than p-values alone.
    - **No external context:** no competitor data, marketing spend, or seasonal calendar —
      this dashboard can't explain external drivers of any observed pattern.
    - **No repeat customers:** every `customer_id` in this dataset appears exactly once
      (see the Customer/RFM tab), so true recency/frequency/monetary segmentation isn't
      possible here.
    """)

    st.subheader("🚀 Next Steps")
    next_steps_df = pd.DataFrame([
        {"Model": "Revenue Prediction", "Status": "✅ Completed (Revenue Prediction tab)",
         "Result": "R² ≈ 0.9998 (Random Forest) — essentially solved"},
        {"Model": "Return Prediction", "Status": "✅ Completed (Returns Model tab)",
         "Result": "AUC ≈ 0.50 — not achievable with current fields; needs richer data (reviews, sizing/fit, complaints)"},
        {"Model": "RFM Customer Segmentation", "Status": "🔲 Can't be done on this data",
         "Result": "Requires repeat customers, which this dataset doesn't have"},
        {"Model": "Discount Optimization (A/B Test)", "Status": "🔲 Not yet attempted",
         "Result": "Requires a controlled experiment, not observational data"},
    ])
    st.dataframe(next_steps_df, hide_index=True, use_container_width=True)

st.markdown("---")
st.caption("Built with Streamlit · Data: retail_large_dataset.csv (100,000 transactions, 2023–2025)")
