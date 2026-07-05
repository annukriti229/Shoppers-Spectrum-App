import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import gdown
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity


customer_product_matrix_id = "1ioj5RylWPbRgFDHi-X9WPG9lO0QGQtYq"
product_similarity_id = "1uru84wbnr5PKzGg1AxtlxpEDfCV-NKzN"
def download_file(file_id,output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output, quiet=False)
download_file(customer_product_matrix_id,"customer_product_matrix.pkl")
download_file(product_similarity_id,"product_similarity.pkl")

# Page Config
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)
#Loaders
@st.cache_resource
def load_models():
    with open("kmeans_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("cluster_name_map.pkl", "rb") as f:
        cmap= pickle.load(f)
    with open("product_similarity.pkl", "rb") as f:
        sim = pickle.load(f)
    with open("customer_product_matrix.pkl","rb") as f:
        customer_matrix = pickle.load(f)
    return model, scaler, cmap, sim, customer_matrix


@st.cache_data
def load_rfm():
    return pd.read_csv("rfm_segments.csv", dtype={"CustomerID": str})


@st.cache_data
def load_cluster_profiles():
    return pd.read_csv("cluster_profiles.csv", index_col=0)


@st.cache_data
def load_raw():
    """Load cleaned_online_retail (no .csv extension, as used in notebook)."""
    fname = "cleaned_online_retail"
    if not os.path.exists(fname):
        # fallback: with .csv extension in case user saved it that way
        fname = "cleaned_online_retail.csv"
        if not os.path.exists(fname):
            return None
    df = pd.read_csv(
        fname,
        parse_dates=["InvoiceDate"],
        dtype={
            "InvoiceNo": str, "StockCode": str,
            "Description": str, "CustomerID": str, "Country": str,
        },
    )
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["TotalAmount"] = pd.to_numeric(df["TotalAmount"], errors="coerce")
    return df


# ── recommend function (mirrors notebook logic exactly) ──
def recommend_products(product_name, similarity_df, top_n=5):
    product_name = product_name.strip().upper()
    matched = [p for p in similarity_df.index if p.upper() == product_name]
    if not matched:
        partial = [p for p in similarity_df.index if product_name in p.upper()]
        if partial:
            matched = [partial[0]]
        else:
            return [], None
    selected = matched[0]
    recs = (
        similarity_df[selected]
        .sort_values(ascending=False)
        .drop(selected)
        .head(top_n)
    )
    return recs.index.tolist(), recs.values.tolist()


# ── predict segment (mirrors notebook predict_customer_segment) ──
def predict_customer_segment(recency, frequency, monetary, scaler, model, cmap):
    inp = pd.DataFrame([{
        "Recency": np.log1p(recency),
        "Frequency": np.log1p(frequency),
        "Monetary": np.log1p(monetary),
    }])
    scaled = scaler.transform(inp)
    cluster = int(model.predict(scaled)[0])
    segment = cmap.get(cluster, f"Cluster {cluster}")
    return cluster, segment


# ── bootstrap ──
try:
    model, scaler, cmap, sim_df, customer_product_matrix = load_models()
    rfm = load_rfm()
    profiles = load_cluster_profiles()
    LOADED = True
except FileNotFoundError as e:
    LOADED = False
    MISSING = str(e)

raw_df = load_raw()

if not LOADED:
    st.error(
        f"❌ Missing file: **{MISSING}**\n\n"
        "Run `RFM_analysis.ipynb` first to generate all `.pkl` and `.csv` artifacts, "
        "then place them in the same folder as `app.py`."
    )
    st.stop()

COLORS = {
    "High-Value": "#27ae60", "Regular": "#2980b9",
    "Occasional": "#f39c12", "At-Risk": "#e74c3c",
}
# Sidebars
with st.sidebar:
    st.markdown("## 🛒 Shopper Spectrum")
    st.markdown("---")

    if raw_df is not None:
        countries = ["All"] + sorted(raw_df["Country"].dropna().unique())
        sel_country = st.selectbox("🌍 Select Country", countries)
    else:
        sel_country = "All"

    st.markdown("### Navigation")
    page = st.radio("", [
        "🏠 Executive Dashboard",
        "📈 Sales Analytics",
        "🌍 Country Analysis",
        "📊 RFM Analysis",
        "📐 Elbow Method",
        "🧩 Customer Segmentation",
        "🔗 Similarity Matrix",
        "🔍 Product Recommendation",
        "🎯 Customer Prediction",
        "💡 Business Insights",
    ] , label_visibility="collapsed")

    st.markdown("---")
    st.caption("Built with Streamlit · KMeans · Cosine Similarity")

    filtered = raw_df
    if raw_df is not None and sel_country != "All":
        filtered = raw_df[raw_df["Country"] == sel_country]

# Helpers
def card(label, value):
    return f'<div class="card"><div class="lbl">{label}</div><div class="val">{value}</div></div>'


def hdr(icon, title, sub=""):
    st.markdown(
        f'<div class="page-hdr"><h2>{icon} {title}</h2>'
        + (f"<p>{sub}</p>" if sub else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def insight(txt):
    st.markdown(f'<div class="insight">💡 {txt}</div>', unsafe_allow_html=True)


if page == "🏠 Executive Dashboard":
    hdr("📊", "Executive Dashboard", "Real-time overview of your e-commerce performance")

    if filtered is not None:
        rev = filtered["TotalAmount"].sum()
        custs = filtered["CustomerID"].nunique()
        ords = filtered["InvoiceNo"].nunique()
        prods = filtered["Description"].nunique()
    else:
        rev = rfm["Monetary"].sum()
        custs = rfm["CustomerID"].nunique()
        ords = int(rfm["Frequency"].sum())
        prods = len(sim_df.columns)

    try:
        from sklearn.metrics import silhouette_score

        tmp = rfm[["Recency", "Frequency", "Monetary"]].copy()
        for c in tmp.columns: tmp[c] = np.log1p(tmp[c])
        sc_vals = scaler.transform(tmp)
        sil = silhouette_score(sc_vals, rfm["Cluster"])
        sil_str = f"{sil:.3f}"
    except Exception:
        sil_str = "—"

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, lbl, val in zip(
            [c1, c2, c3, c4, c5],
            ["💰 Revenue", "👥 Customers", "📦 Orders", "🏷️ Products", "🎯 Silhouette"],
            [f"₹{rev:,.0f}", f"{custs:,}", f"{ords:,}", f"{prods:,}", sil_str],
    ):
        col.markdown(card(lbl, val), unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### 🧩 Segment Distribution")
        sc_df = rfm["Segment"].value_counts().reset_index()
        sc_df.columns = ["Segment", "Count"]
        fig = px.pie(sc_df, names="Segment", values="Count", hole=0.45,
                     color="Segment", color_discrete_map=COLORS)
        fig.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("##### 💰 Revenue by Segment")
        sr = rfm.groupby("Segment")["Monetary"].sum().reset_index()
        fig = px.bar(sr, x="Segment", y="Monetary", color="Segment",
                     color_discrete_map=COLORS, text_auto=".2s")
        fig.update_layout(showlegend=False, yaxis_title="Revenue (₹)", margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    if filtered is not None:
        st.markdown("##### 📅 Monthly Revenue Trend")
        mn = filtered.set_index("InvoiceDate").resample("ME")["TotalAmount"].sum().reset_index()
        fig = px.area(mn, x="InvoiceDate", y="TotalAmount",
                      color_discrete_sequence=["#e94560"])
        fig.update_layout(yaxis_title="Revenue (₹)", xaxis_title="", margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("##### ✅ Project Overview")
    for item in [
        "Customer Segmentation (RFM + KMeans)",
        "Product Recommendation (Item-based Collaborative Filtering)",
        "Interactive EDA & Visualisations",
        "Real-time Cluster Prediction",
        "Product Similarity Matrix",
    ]:
        st.markdown(f"✔️ &nbsp; {item}")

# Sales Analytics
elif page == "📈 Sales Analytics":
    hdr("📈", "Sales Analytics", "Explore revenue trends and top-performing products")

    if filtered is None:
        st.info("Place `cleaned_online_retail` next to `app.py` to enable this page.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    avg_ov = filtered.groupby("InvoiceNo")["TotalAmount"].sum().mean()
    for col, lbl, val in zip(
            [c1, c2, c3],
            ["Total Revenue", "Total Orders", "Avg Order Value"],
            [f"₹{filtered['TotalAmount'].sum():,.0f}",
             f"{filtered['InvoiceNo'].nunique():,}",
             f"₹{avg_ov:,.0f}"],
    ):
        col.markdown(card(lbl, val), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 📅 Monthly Revenue Trend")
    mn = filtered.set_index("InvoiceDate").resample("ME")["TotalAmount"].sum().reset_index()
    fig = px.line(mn, x="InvoiceDate", y="TotalAmount", markers=True,
                  color_discrete_sequence=["#0f3460"])
    fig.update_layout(yaxis_title="Revenue (₹)", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        n = st.slider("Top N Products", 5, 30, 10)
        tp = (
            filtered.groupby("Description")["Quantity"].sum()
            .sort_values(ascending=False).head(n).reset_index()
        )
        fig = px.bar(tp, x="Quantity", y="Description", orientation="h",
                     color="Quantity", color_continuous_scale="Tealgrn",
                     title=f"Top {n} Products by Quantity")
        fig.update_layout(yaxis=dict(autorange="reversed"), height=max(400, n * 26))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("##### Order Value Distribution")
        ov = filtered.groupby("InvoiceNo")["TotalAmount"].sum()
        ov = ov[ov < ov.quantile(0.95)]
        fig = px.histogram(ov, nbins=50, color_discrete_sequence=["#e94560"])
        fig.update_layout(xaxis_title="Order Value (₹)", yaxis_title="Count", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### 📆 Sales by Day of Week")
    dow = filtered.copy()
    dow["DayOfWeek"] = dow["InvoiceDate"].dt.day_name()
    dow_s = (
        dow.groupby("DayOfWeek")["TotalAmount"].sum()
        .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        .reset_index()
    )
    fig = px.bar(dow_s, x="DayOfWeek", y="TotalAmount",
                 color="TotalAmount", color_continuous_scale="Blues", text_auto=".2s")
    fig.update_layout(yaxis_title="Revenue (₹)", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Country Analysis
elif page == "🌍 Country Analysis":
    hdr("🌍", "Country Analysis", "Revenue and customer breakdown by geography")

    if raw_df is None:
        st.info("Place `cleaned_online_retail` next to `app.py` to enable this page.")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        cr = (raw_df.groupby("Country")["TotalAmount"].sum()
              .sort_values(ascending=False).head(10).reset_index())
        fig = px.bar(cr, x="TotalAmount", y="Country", orientation="h",
                     color="TotalAmount", color_continuous_scale="Blues",
                     title="Revenue – Top 10 Countries")
        fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Revenue (₹)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        co = (raw_df.groupby("Country")["InvoiceNo"].nunique()
              .sort_values(ascending=False).head(10).reset_index(name="Orders"))
        fig = px.bar(co, x="Orders", y="Country", orientation="h",
                     color="Orders", color_continuous_scale="Oranges",
                     title="Orders – Top 10 Countries")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### 🗺️ Customer Count Treemap")
    cc = (raw_df.groupby("Country")["CustomerID"].nunique()
          .sort_values(ascending=False).head(20).reset_index(name="Customers"))
    fig = px.treemap(cc, path=["Country"], values="Customers",
                     color="Customers", color_continuous_scale="Viridis")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### 📋 Country Summary Table")
    cs = raw_df.groupby("Country").agg(
        Revenue=("TotalAmount", "sum"),
        Orders=("InvoiceNo", "nunique"),
        Customers=("CustomerID", "nunique"),
    ).sort_values("Revenue", ascending=False).round(0)
    st.dataframe(cs, use_container_width=True)

# RFM Analysis
elif page == "📊 RFM Analysis":
    hdr("📊", "RFM Analysis", "Understand Recency, Frequency & Monetary patterns")

    c1, c2, c3 = st.columns(3)
    for col, lbl, val in zip(
            [c1, c2, c3],
            ["Avg Recency", "Avg Frequency", "Avg Monetary"],
            [f"{rfm['Recency'].mean():.0f} days",
             f"{rfm['Frequency'].mean():.1f} orders",
             f"₹{rfm['Monetary'].mean():,.0f}"],
    ):
        col.markdown(card(lbl, val), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Distributions  *(mirrors notebook histograms)*")
    c1, c2, c3 = st.columns(3)

    with c1:
        fig = px.histogram(rfm, x="Recency", nbins=30,
                           color_discrete_sequence=["dodgerblue"])
        fig.update_layout(title="Recency Distribution",
                          xaxis_title="Recency", yaxis_title="Number of Customers")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        freq_plot = rfm[rfm["Frequency"] <= rfm["Frequency"].quantile(0.99)]
        fig = px.histogram(freq_plot, x="Frequency", nbins=30,
                           color_discrete_sequence=["dodgerblue"])
        fig.update_layout(title="Frequency Distribution",
                          xaxis_title="Frequency", yaxis_title="Number of Customers")
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        mon_plot = rfm[rfm["Monetary"] <= rfm["Monetary"].quantile(0.99)]
        fig = px.histogram(mon_plot, x="Monetary", nbins=40,
                           color_discrete_sequence=["dodgerblue"])
        fig.update_layout(title="Monetary Distribution",
                          xaxis_title="Monetary", yaxis_title="Number of Customers")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### RFM Correlation Heatmap")
    corr = rfm[["Recency", "Frequency", "Monetary"]].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### 🔍 Browse RFM Table")
    seg_f = st.selectbox("Filter by Segment", ["All"] + sorted(rfm["Segment"].unique()))
    view = rfm if seg_f == "All" else rfm[rfm["Segment"] == seg_f]
    st.dataframe(
        view[["CustomerID", "Recency", "Frequency", "Monetary", "Cluster", "Segment"]].head(200),
        use_container_width=True,
    )

# Elbow Method
elif page == "📐 Elbow Method":
    hdr("📐", "Elbow Method", "Choosing the optimal number of clusters (k=2…8)")

    st.info("Recomputing Elbow & Silhouette curves from `rfm_segments.csv` — takes a moment.")

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # replicate notebook log-transform + scale
    tmp = rfm[["Recency", "Frequency", "Monetary"]].copy()
    for c in tmp.columns:
        tmp[c] = np.log1p(tmp[c])
    rfm_scaled = scaler.transform(tmp)

    k_values = range(2, 9)
    inertia_vals, sil_vals = [], []

    prog = st.progress(0, text="Running KMeans…")
    for i, k in enumerate(k_values):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(rfm_scaled)
        inertia_vals.append(km.inertia_)
        sil_vals.append(silhouette_score(rfm_scaled, labels))
        prog.progress((i + 1) / len(k_values), text=f"k = {k}")
    prog.empty()

    best_k = list(k_values)[int(np.argmax(sil_vals))]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Elbow Curve")
        fig = px.line(x=list(k_values), y=inertia_vals, markers=True,
                      color_discrete_sequence=["#0f3460"])
        fig.add_vline(x=best_k, line_dash="dash", line_color="red",
                      annotation_text=f"Best k={best_k}")
        fig.update_layout(xaxis_title="Number of Clusters", yaxis_title="Inertia")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("##### Silhouette Scores")
        fig = px.line(x=list(k_values), y=sil_vals, markers=True,
                      color_discrete_sequence=["green"])
        fig.add_vline(x=best_k, line_dash="dash", line_color="red",
                      annotation_text=f"Best k={best_k}")
        fig.update_layout(xaxis_title="Number of Clusters", yaxis_title="Silhouette Score")
        st.plotly_chart(fig, use_container_width=True)

    st.success(f"✅ Best k = **{best_k}**  (silhouette score: {max(sil_vals):.3f})")

# Customer Segmentation
elif page == "🧩 Customer Segmentation":
    hdr("🧩", "Customer Segmentation", "RFM-based cluster overview")

    # segment counts
    sc_df = rfm["Segment"].value_counts().reset_index()
    sc_df.columns = ["Segment", "Count"]
    cols = st.columns(len(sc_df))
    for i, row in sc_df.iterrows():
        cols[i].markdown(card(row["Segment"], f"{row['Count']:,}"), unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Segment Pie Chart")
        fig = px.pie(sc_df, names="Segment", values="Count", hole=0.4,
                     color="Segment", color_discrete_map=COLORS)
        fig.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("##### Average RFM per Segment")
        fig = go.Figure()
        for metric, clr in zip(["Recency", "Frequency", "Monetary"],
                               ["#EF553B", "#636EFA", "#00CC96"]):
            fig.add_trace(go.Bar(
                name=metric,
                x=profiles.index.tolist(),
                y=profiles[metric].tolist(),
                marker_color=clr,
            ))
        fig.update_layout(barmode="group", yaxis_title="Value", legend_title="Metric")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### 2D Scatter — Frequency vs Monetary  *(notebook plot)*")
    fig = px.scatter(
        rfm, x="Frequency", y="Monetary", color="Segment",
        color_discrete_map=COLORS, opacity=0.7, size_max=8,
        title="Customer Segments by Frequency and Monetary",
    )
    fig.update_layout(xaxis_title="Frequency", yaxis_title="Monetary")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### 3D Cluster View  *(notebook plot)*")
    fig = px.scatter_3d(
        rfm, x="Recency", y="Frequency", z="Monetary",
        color="Segment", opacity=0.65, color_discrete_map=COLORS,
        title="3D View of Customer Segments",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### 📋 Cluster Profiles  (from `cluster_profiles.csv`)")
    st.dataframe(profiles, use_container_width=True)

# Similarity Matrix
elif page == "🔗 Similarity Matrix":
    hdr("🔗", "Similarity Matrix",
        "Cosine similarity heatmap between products  (from product_similarity.pkl)")

    all_prods = sorted(sim_df.columns.tolist())

    # default: top-20 by notebook heatmap logic
    if raw_df is not None:
        top20 = raw_df["Description"].value_counts().head(20).index.tolist()
        top20 = [p for p in top20 if p in sim_df.index]
    else:
        top20 = all_prods[:20]

    selected = st.multiselect(
        "Select products to compare (default = top-20 most purchased)",
        options=all_prods,
        default=top20[:min(20, len(top20))],
    )

    if selected:
        sub = sim_df.loc[selected, selected]
        fig = px.imshow(
            sub, color_continuous_scale="Viridis",
            aspect="auto", labels=dict(color="Similarity"),
            title="Product Similarity Heatmap",
        )
        fig.update_layout(height=620)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one product above.")

    st.markdown("##### Top 15 Most Similar Product Pairs")
    sm_copy = sim_df.copy()
    arr = sm_copy.to_numpy(copy=True)
    np.fill_diagonal(arr,0)
    sm_copy = pd.DataFrame(arr,index=sim_df.index,columns=sim_df.columns)
    pairs = (sm_copy.rename_axis(index="Product_A",columns="Product_B").stack().reset_index(name="Similarity")
             )
    pairs.columns = ["Product A", "Product B", "Similarity"]
    pairs = pairs[pairs["Product A"] < pairs["Product B"]]
    pairs = pairs.sort_values(by="Similarity", ascending=False).head(15)
    pairs.reset_index(drop=True,inplace=True)
    pairs.index = pairs.index + 1
    st.dataframe(pairs, use_container_width=True)

# Product Recommendation
elif page == "🔍 Product Recommendation":
    hdr("🔍", "Product Recommendation",
        "Enter a product name to get 5 similar items via collaborative filtering")

    product_list = sorted(sim_df.columns.tolist())

    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown("#### 🏷️ Enter Product Name")
        # ── text input (as required by assignment) ──
        typed = st.text_input(
            "Product Name",
            placeholder="e.g. WHITE HANGING HEART T-LIGHT HOLDER",
        )

        # live auto-complete suggestions
        matched_product = None
        if typed:
            suggestions = [p for p in product_list if typed.upper() in p.upper()]
            if suggestions:
                choice = st.selectbox(
                    f"Found {len(suggestions)} match(es) — select one:",
                    ["— select —"] + suggestions[:25],
                )
                if choice != "— select —":
                    matched_product = choice
            else:
                st.warning("⚠️ No products matched. Try a different keyword.")

        get_btn = st.button("🔍 Get Recommendations")

    with col_right:
        if matched_product:
            st.markdown("##### ✅ Selected Product")
            st.success(f"🏷️ **{matched_product}**")
        else:
            st.markdown("##### ℹ️ How it works")
            st.markdown(
                "- Powered by **item-based collaborative filtering**\n"
                "- Uses cosine similarity on the customer-product purchase matrix\n"
                "- Returns the **5 most similar products** to your input"
            )

    # ── results ──
    if get_btn:
        if not matched_product and typed:
            # try direct lookup via recommend function
            recs, scores = recommend_products(typed, sim_df)
            if recs:
                matched_product = typed.strip().upper()
            else:
                st.error("❌ Product not found. Select from the dropdown after typing.")
        elif not typed:
            st.error("❌ Please enter a product name first.")

        if matched_product:
            recs, scores = recommend_products(matched_product, sim_df)

            if not recs:
                st.error("No recommendations found for this product.")
            else:
                st.markdown("---")
                st.markdown(f"### 🎁 Top 5 Products Similar to  _{matched_product}_")

                # ── styled cards (assignment spec: "card view") ──
                cols = st.columns(5)
                for i, (prod, score) in enumerate(zip(recs, scores)):
                    with cols[i]:
                        st.markdown(f"""
                        <div class="prod-card">
                          <div class="rank">#{i + 1} Recommended</div>
                          <div class="pname">{prod}</div>
                          <div class="score">Similarity: {score:.3f}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("---")
                c1, c2 = st.columns(2)
                top5_df = pd.DataFrame({"Product": recs, "Similarity": scores})

                with c1:
                    st.markdown("##### 📊 Similarity Bar Chart")
                    fig = px.bar(
                        top5_df, x="Similarity", y="Product", orientation="h",
                        color="Similarity", color_continuous_scale="Purples",
                        text_auto=".3f",
                    )
                    fig.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                with c2:
                    st.markdown("##### 🕸️ Similarity Radar Chart")
                    labels = [f"#{i + 1} {p[:18]}…" if len(p) > 18 else f"#{i + 1} {p}"
                              for i, p in enumerate(recs)]
                    fig = go.Figure(go.Scatterpolar(
                        r=scores + [scores[0]],
                        theta=labels + [labels[0]],
                        fill="toself",
                        line_color="#e94560",
                        fillcolor="rgba(233,69,96,0.15)",
                    ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("##### 📋 Recommendations Table")
                top5_df.index = range(1, 6)
                top5_df.index.name = "Rank"
                st.dataframe(top5_df, use_container_width=True)
    else:
        # empty state
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#aaa;">
          <div style="font-size:52px;">🔍</div>
          <div style="font-size:16px;margin-top:14px;">
            Type a product name above and click <b>Get Recommendations</b>
          </div>
        </div>""", unsafe_allow_html=True)

# Customer Segmentation
elif page == "🎯 Customer Prediction":
    hdr("🎯", "Customer Prediction",
        "Enter RFM values to predict the customer segment using the trained KMeans model")

    col_form, col_result = st.columns([1, 1.2])

    with col_form:
        st.markdown("#### 📋 Customer Details")
        recency = st.number_input("📅 Recency  (days since last purchase)",
                                  min_value=0, max_value=1000, value=30, step=1)
        frequency = st.number_input("🔁 Frequency  (number of purchases)",
                                    min_value=1, max_value=1000, value=5, step=1)
        monetary = st.number_input("💰 Monetary  (total spend ₹)",
                                   min_value=0.0, max_value=500000.0,
                                   value=500.0, step=50.0)

        predict_btn = st.button("🎯 Predict Cluster")

    with col_result:
        st.markdown("#### 🏷️ Prediction Result")

        if predict_btn:
            cluster, segment = predict_customer_segment(
                recency, frequency, monetary, scaler, model, cmap
            )

            # coloured badge
            seg_css = segment.replace(" ", "-")
            st.markdown(f"""
            <div style="margin:10px 0 18px;">
              <div style="font-size:13px;color:#666;margin-bottom:6px;">Predicted Segment</div>
              <div class="seg-badge seg-{seg_css}">{segment}</div>
              <div style="font-size:12px;color:#999;margin-top:8px;">Cluster ID: {cluster}</div>
            </div>""", unsafe_allow_html=True)

            # segment descriptions
            descs = {
                "High-Value": "🟢 Recent, frequent, high-spend customer. Offer loyalty rewards and VIP perks.",
                "Regular": "🔵 Steady purchaser with moderate spend. Cross-sell to grow basket size.",
                "Occasional": "🟡 Infrequent buyer with lower spend. Target with time-limited promotions.",
                "At-Risk": "🔴 Hasn't purchased in a long time. Launch a win-back campaign.",
            }
            st.info(descs.get(segment, "Segment identified."))

            # gauges
            st.markdown("##### Input at a Glance")
            gc1, gc2, gc3 = st.columns(3)


            def gauge(val, max_val, label, color):
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=val,
                    gauge=dict(axis=dict(range=[0, max_val]),
                               bar=dict(color=color)),
                    title=dict(text=label, font=dict(size=12)),
                    number=dict(font=dict(size=18)),
                ))
                fig.update_layout(height=175, margin=dict(t=40, b=0, l=10, r=10))
                return fig


            gc1.plotly_chart(gauge(recency, 365, "Recency (days)", "#EF553B"), use_container_width=True)
            gc2.plotly_chart(gauge(frequency, 100, "Frequency", "#636EFA"), use_container_width=True)
            gc3.plotly_chart(gauge(monetary, float(rfm["Monetary"].quantile(0.95)),
                                   "Monetary (₹)", "#00CC96"), use_container_width=True)

            # comparison table vs cluster averages
            st.markdown("##### How This Customer Compares to Segment Averages")
            comp = profiles.copy()
            comp.loc["→ This Customer"] = [recency, frequency, monetary]
            st.dataframe(comp.round(1), use_container_width=True)

        else:
            st.markdown("""
            <div style="text-align:center;padding:60px 10px;color:#aaa;">
              <div style="font-size:52px;">🎯</div>
              <div style="font-size:15px;margin-top:14px;">
                Fill in the values on the left and click <b>Predict Cluster</b>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📖 Segment Reference Guide  *(cluster_name_map from notebook)*")
    ref = pd.DataFrame({
        "Cluster ID": [0, 1, 2, 3],
        "Segment": ["High-Value", "Regular", "Occasional", "At-Risk"],
        "Recency": ["Low", "Low", "High", "Very High"],
        "Frequency": ["High", "Medium", "Low", "Low"],
        "Monetary": ["High", "Medium", "Low", "Low"],
        "Recommended Action": [
            "Loyalty rewards, VIP perks, early access",
            "Cross-sell offers to grow basket size",
            "Time-limited discounts, re-engagement emails",
            "Win-back campaign, surveys on churn reasons",
        ],
    })
    st.table(ref)

# Business Insights
elif page == "💡 Business Insights":
    hdr("💡", "Business Insights", "Data-driven strategy recommendations")

    hv_pct = (rfm["Segment"] == "High-Value").mean() * 100
    ar_pct = (rfm["Segment"] == "At-Risk").mean() * 100
    hv_rev = (rfm.loc[rfm["Segment"] == "High-Value", "Monetary"].sum()
              / rfm["Monetary"].sum() * 100)

    c1, c2, c3 = st.columns(3)
    for col, lbl, val in zip(
            [c1, c2, c3],
            ["High-Value %", "At-Risk %", "HV Revenue Share"],
            [f"{hv_pct:.1f}%", f"{ar_pct:.1f}%", f"{hv_rev:.1f}%"],
    ):
        col.markdown(card(lbl, val), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🔍 Key Findings")
    insight(
        f"High-Value customers ({hv_pct:.1f}%) contribute {hv_rev:.1f}% of revenue — protect with loyalty programmes.")
    insight(f"{ar_pct:.1f}% of customers are At-Risk — targeted win-back campaigns can recover significant revenue.")
    insight(
        "Regular and Occasional segments respond well to personalised cross-sell offers from the recommendation engine.")
    insight("Products with high cosine similarity (Similarity Matrix page) are ideal bundle or cross-sell candidates.")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Revenue by Segment")
        sr = rfm.groupby("Segment")["Monetary"].sum().reset_index()
        fig = px.bar(sr, x="Segment", y="Monetary", color="Segment",
                     color_discrete_map=COLORS, text_auto=".2s")
        fig.update_layout(showlegend=False, yaxis_title="Revenue (₹)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("##### Avg Spend per Segment")
        am = rfm.groupby("Segment")["Monetary"].mean().reset_index()
        fig = px.bar(am, x="Segment", y="Monetary", color="Segment",
                     color_discrete_map=COLORS, text_auto=".2s")
        fig.update_layout(showlegend=False, yaxis_title="Avg Spend (₹)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🗂️ Recommended Actions by Segment")
    actions = pd.DataFrame({
        "Segment": ["High-Value", "Regular", "Occasional", "At-Risk"],
        "Action": [
            "Loyalty rewards, early access, VIP support",
            "Personalised cross-sell to increase order frequency",
            "Time-limited discounts, email re-engagement",
            "Win-back campaigns, surveys to understand churn",
        ],
        "Priority": ["🔴 Critical", "🟠 High", "🟡 Medium", "🔴 Urgent"],
    })
    st.table(actions)