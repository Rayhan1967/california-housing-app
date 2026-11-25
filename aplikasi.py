"""Dashboard Machine Learning Prediksi Harga Rumah California (multi-halaman)."""
import io
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET, CAT = "median_house_value", "ocean_proximity"
NUM_RAW = ["longitude", "latitude", "housing_median_age", "total_rooms",
           "total_bedrooms", "population", "households", "median_income"]
ENG = ["rooms_per_household", "bedrooms_per_room", "population_per_household"]
KAMUS = {
    "longitude": "Garis bujur lokasi blok", "latitude": "Garis lintang lokasi blok",
    "housing_median_age": "Median umur rumah (tahun)", "total_rooms": "Total kamar di blok",
    "total_bedrooms": "Total kamar tidur di blok", "population": "Jumlah penduduk blok",
    "households": "Jumlah rumah tangga blok", "median_income": "Median pendapatan (x $10.000)",
    "median_house_value": "TARGET: median harga rumah (USD)",
    "ocean_proximity": "Kategori jarak ke laut"}
DEFAULT_CFG = {"fe": True, "log": True, "cap": False, "test": 0.2}

st.set_page_config(page_title="Prediksi Harga Rumah", page_icon="🏠", layout="wide")


# ======================= DATA & PIPELINE =======================
@st.cache_data(show_spinner=False)
def _read(src):
    return pd.read_csv(src)


def get_df():
    up = st.session_state.get("upload")
    path = Path(__file__).parent / "housing.csv"
    if up is not None:
        df = _read(up)
    elif path.exists():
        df = _read(path)
    else:
        st.error("housing.csv tidak ditemukan. Upload lewat sidebar.")
        st.stop()
    miss = [c for c in NUM_RAW + [CAT, TARGET] if c not in df.columns]
    if miss:
        st.error(f"Kolom wajib tidak ada: {miss}")
        st.stop()
    return df


def get_cfg():
    return st.session_state.setdefault("cfg", dict(DEFAULT_CFG))


def add_features(df):
    d = df.copy()
    d["rooms_per_household"] = d["total_rooms"] / d["households"].replace(0, np.nan)
    d["bedrooms_per_room"] = d["total_bedrooms"] / d["total_rooms"].replace(0, np.nan)
    d["population_per_household"] = d["population"] / d["households"].replace(0, np.nan)
    return d


def num_cols(cfg):
    return NUM_RAW + (ENG if cfg["fe"] else [])


def split(df, cfg):
    d = add_features(df)
    if cfg["cap"]:
        d = d[d[TARGET] < 500001]
    return train_test_split(d.drop(columns=[TARGET]), d[TARGET],
                            test_size=cfg["test"], random_state=42)


def make_pre(cfg):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols(cfg)),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [CAT])])


def make_pipe(est, cfg, k="all"):
    pipe = Pipeline([("pre", make_pre(cfg)),
                     ("select", SelectKBest(f_regression, k=k)), ("model", est)])
    if cfg["log"]:
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    return pipe


def pfx(cfg):
    return "regressor__" if cfg["log"] else ""


def zoo():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=18, n_jobs=-1, random_state=42),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=42)}


def evaluate(y, p):
    return {"R²": r2_score(y, p), "MAE": mean_absolute_error(y, p),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAPE (%)": mean_absolute_percentage_error(y, p) * 100}


@st.cache_resource(show_spinner=False)
def run_benchmark(df, cfg, cv):
    X_tr, X_te, y_tr, y_te = split(df, cfg)
    rows, fitted = [], {}
    for name, est in zoo().items():
        pipe = make_pipe(est, cfg)
        t0 = time.time()
        cvs = cross_val_score(pipe, X_tr, y_tr, scoring="r2",
                              cv=KFold(cv, shuffle=True, random_state=42))
        pipe.fit(X_tr, y_tr)
        dt = time.time() - t0
        pred = pipe.predict(X_te)
        rows.append({"Model": name, "R² CV (mean)": cvs.mean(), "R² CV (std)": cvs.std(),
                     **{f"{k} Test": v for k, v in evaluate(y_te, pred).items()},
                     "Waktu (dtk)": dt})
        fitted[name] = (pipe, pred)
    res = pd.DataFrame(rows).sort_values("R² Test", ascending=False).reset_index(drop=True)
    return res, fitted


@st.cache_resource(show_spinner=False)
def default_model(df, cfg):
    X_tr, X_te, y_tr, y_te = split(df, cfg)
    pipe = make_pipe(HistGradientBoostingRegressor(random_state=42), cfg).fit(X_tr, y_tr)
    return pipe, evaluate(y_te, pipe.predict(X_te))["RMSE"]


def active_model(df, cfg):
    """Model untuk prediksi: pilihan pengguna, atau default jika belum ada."""
    a = st.session_state.get("active")
    if a:
        return a["model"], a["name"], a["rmse"]
    m, rmse = default_model(df, cfg)
    return m, "Hist Gradient Boosting (default)", rmse


def card(title, body, color="#2ba39a"):
    st.markdown(f"""<div style="border:1px solid #3a3a3a;border-left:5px solid {color};
    border-radius:10px;padding:14px 18px;margin-bottom:12px">
    <b>{title}</b><br><span style="opacity:.85">{body}</span></div>""", unsafe_allow_html=True)


def vif_table(df):
    X = df[NUM_RAW].fillna(df[NUM_RAW].median())
    out = []
    for c in NUM_RAW:
        r2 = LinearRegression().fit(X.drop(columns=c), X[c]).score(X.drop(columns=c), X[c])
        out.append({"Fitur": c, "VIF": 1 / max(1 - r2, 1e-9)})
    return pd.DataFrame(out).sort_values("VIF", ascending=False)


# ======================= HALAMAN: PENGANTAR =======================
def page_beranda():
    df = get_df()
    st.title("🏠 Prediksi Harga Rumah California")
    st.caption("Tugas Machine Learning · Regresi pada dataset California Housing")
    c = st.columns(4)
    c[0].metric("Jumlah blok", f"{len(df):,}")
    c[1].metric("Jumlah fitur", df.shape[1] - 1)
    c[2].metric("Rata-rata harga", f"${df[TARGET].mean():,.0f}")
    c[3].metric("Missing value", int(df.isna().sum().sum()))
    st.subheader("Alur aplikasi")
    a, b, c3 = st.columns(3)
    with a:
        card("1. Pengantar", "Kenali dataset dan lakukan EDA: distribusi, peta, korelasi, VIF, outlier.")
    with b:
        card("2. Pengolahan & Model", "Preprocessing, PCA, optimasi hyperparameter, lalu benchmark beberapa model.", "#e0a030")
    with c3:
        card("3. Penerapan", "Prediksi satu rumah, prediksi massal via CSV, dan kesimpulan otomatis.", "#d0476b")
    st.info("Mulailah dari **Eksplorasi Data**, atur opsi di **Preprocessing**, "
            "lalu jalankan **Modelling & Evaluasi** agar halaman prediksi memakai model terbaik.")


def page_dataset():
    df = get_df()
    st.title("Tentang Dataset")
    st.write("Setiap baris mewakili satu *block group* sensus California (1990).")
    kamus = pd.DataFrame({"Kolom": df.columns, "Tipe": df.dtypes.astype(str).values,
                          "Missing": df.isna().sum().values,
                          "Unik": df.nunique().values,
                          "Keterangan": [KAMUS.get(c, "-") for c in df.columns]})
    st.dataframe(kamus, hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        kat = st.multiselect("Filter ocean_proximity", sorted(df[CAT].unique()),
                             default=sorted(df[CAT].unique()))
    with c2:
        lo, hi = float(df[TARGET].min()), float(df[TARGET].max())
        rng = st.slider("Rentang harga (USD)", lo, hi, (lo, hi))
    f = df[df[CAT].isin(kat) & df[TARGET].between(*rng)]
    st.caption(f"{len(f):,} baris cocok dengan filter")
    st.dataframe(f.head(500))
    st.download_button("Unduh hasil filter (CSV)", f.to_csv(index=False), "housing_filter.csv")
    with st.expander("Statistik deskriptif"):
        st.dataframe(df.describe().T)


def page_eda():
    df = get_df()
    st.title("Eksplorasi Data (EDA)")
    t1, t2, t3, t4 = st.tabs(["Distribusi", "Geospasial", "Korelasi & VIF", "Outlier & Anomali"])
    with t1:
        col = st.selectbox("Kolom", NUM_RAW + [TARGET], index=7)
        st.plotly_chart(px.histogram(df, x=col, nbins=60, marginal="box", color=CAT))
        st.plotly_chart(px.box(df, x=CAT, y=TARGET, color=CAT, title="Harga per kategori jarak laut"))
    with t2:
        smp = df.sample(min(8000, len(df)), random_state=1)
        warna = st.selectbox("Warnai berdasarkan", [TARGET, "median_income", "population", "housing_median_age"])
        fig = px.scatter(smp, x="longitude", y="latitude", color=warna, hover_data=[CAT],
                         color_continuous_scale="Turbo", opacity=0.7, height=600)
        fig.update_yaxes(scaleanchor="x")
        st.plotly_chart(fig)
    with t3:
        cor = df[NUM_RAW + [TARGET]].corr()
        st.plotly_chart(px.imshow(cor, text_auto=".2f", color_continuous_scale="RdBu_r",
                                  zmin=-1, zmax=1, height=600))
        st.subheader("Variance Inflation Factor (multikolinearitas)")
        v = vif_table(df)
        st.dataframe(v, hide_index=True)
        st.caption("VIF > 10 menandakan multikolinearitas kuat (umumnya total_rooms, total_bedrooms, "
                   "population, households saling berkaitan). Ini alasan dibuat fitur rasio dan dicoba PCA.")
    with t4:
        rows = []
        for c in NUM_RAW + [TARGET]:
            q1, q3 = df[c].quantile([.25, .75])
            iqr = q3 - q1
            n = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
            rows.append({"Kolom": c, "Outlier IQR": n, "% data": 100 * n / len(df)})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        cap_t = int((df[TARGET] >= 500001).sum())
        cap_a = int((df["housing_median_age"] >= 52).sum())
        card("Sensor pada target", f"{cap_t:,} baris punya harga = $500.001 (batas atas data). "
             "Nilai sebenarnya mungkin lebih tinggi sehingga model cenderung menaksir terlalu rendah.", "#d0476b")
        card("Sensor pada umur rumah", f"{cap_a:,} baris memiliki umur 52 tahun (nilai maksimum).", "#e0a030")
        card("Missing value", f"total_bedrooms kosong {int(df['total_bedrooms'].isna().sum())} baris; diimputasi median di dalam pipeline.")


# ======================= HALAMAN: PENGOLAHAN & MODEL =======================
def page_preprocessing():
    df, cfg = get_df(), get_cfg()
    st.title("Preprocessing")
    st.write("Pengaturan di sini dipakai oleh seluruh halaman model.")
    c1, c2 = st.columns(2)
    with c1:
        cfg["fe"] = st.toggle("Feature engineering (rasio kamar, kamar tidur, populasi)", cfg["fe"])
        cfg["log"] = st.toggle("Transformasi log pada target (log1p)", cfg["log"])
    with c2:
        cfg["cap"] = st.toggle("Buang baris harga tersensor ($500.001)", cfg["cap"])
        cfg["test"] = st.slider("Proporsi data uji", 0.1, 0.4, cfg["test"], 0.05)
    X_tr, X_te, y_tr, y_te = split(df, cfg)
    m = st.columns(4)
    m[0].metric("Data latih", f"{len(X_tr):,}")
    m[1].metric("Data uji", f"{len(X_te):,}")
    m[2].metric("Fitur numerik", len(num_cols(cfg)))
    m[3].metric("Missing (latih)", int(X_tr[num_cols(cfg)].isna().sum().sum()))
    st.subheader("Rancangan pipeline")
    st.code("Imputer(median) → StandardScaler (numerik)\nOneHotEncoder (ocean_proximity)\n"
            "→ SelectKBest (opsional) → Model"
            + ("\nTarget: log1p → expm1 saat prediksi" if cfg["log"] else ""), language="text")
    pre = make_pre(cfg).fit(X_tr)
    Z = pd.DataFrame(pre.transform(X_tr.head(8)),
                     columns=[n.split("__")[-1] for n in pre.get_feature_names_out()])
    st.subheader("Contoh hasil transformasi")
    st.dataframe(Z.round(3))
    if cfg["fe"]:
        st.subheader("Fitur baru")
        st.dataframe(add_features(df)[ENG].describe().T)


def page_pca():
    df, cfg = get_df(), get_cfg()
    st.title("Reduksi Dimensi (PCA)")
    X_tr, X_te, y_tr, y_te = split(df, cfg)
    pre = make_pre(cfg)
    Z = pre.fit_transform(X_tr)
    names = [n.split("__")[-1] for n in pre.get_feature_names_out()]
    pca = PCA().fit(Z)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n = st.slider("Jumlah komponen", 1, len(names), min(8, len(names)))
    st.metric(f"Varians terjelaskan oleh {n} komponen", f"{cum[n - 1] * 100:.2f}%")
    fig = go.Figure()
    fig.add_bar(x=list(range(1, len(cum) + 1)), y=pca.explained_variance_ratio_, name="Per komponen")
    fig.add_scatter(x=list(range(1, len(cum) + 1)), y=cum, name="Kumulatif")
    fig.update_layout(xaxis_title="Komponen", yaxis_tickformat=".0%")
    st.plotly_chart(fig)
    c1, c2 = st.columns(2)
    with c1:
        idx = np.random.RandomState(0).choice(len(Z), min(4000, len(Z)), replace=False)
        P = pca.transform(Z[idx])
        st.plotly_chart(px.scatter(x=P[:, 0], y=P[:, 1], color=y_tr.values[idx],
                                   labels={"x": "PC1", "y": "PC2", "color": "Harga"},
                                   color_continuous_scale="Turbo", opacity=.6, title="Proyeksi 2D"))
    with c2:
        load = pd.DataFrame(pca.components_[:3].T, index=names, columns=["PC1", "PC2", "PC3"])
        st.plotly_chart(px.imshow(load, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                  aspect="auto", title="Loading 3 komponen pertama"))
    if st.button("Bandingkan model dengan vs tanpa PCA"):
        rows = []
        for nm, est in {"Ridge": Ridge(), "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=42)}.items():
            for label, steps in [("Tanpa PCA", [("pre", make_pre(cfg))]),
                                 (f"PCA {n} komponen", [("pre", make_pre(cfg)), ("pca", PCA(n))])]:
                t0 = time.time()
                p = Pipeline(steps + [("m", est)]).fit(X_tr, y_tr)
                rows.append({"Model": nm, "Skenario": label, **{k: v for k, v in evaluate(y_te, p.predict(X_te)).items() if k in ("R²", "RMSE")},
                             "Waktu (dtk)": time.time() - t0})
        st.dataframe(pd.DataFrame(rows), hide_index=True)


def page_optimasi():
    df, cfg = get_df(), get_cfg()
    st.title("Teknik Optimasi")
    st.write("Pencarian hyperparameter dengan **RandomizedSearchCV**, termasuk pemilihan jumlah fitur (**SelectKBest**).")
    c1, c2, c3, c4 = st.columns(4)
    nama = c1.selectbox("Model", ["Hist Gradient Boosting", "Random Forest"])
    n_iter = c2.slider("Jumlah kombinasi", 5, 40, 12)
    cv = c3.slider("Fold CV", 3, 5, 3)
    smp = c4.slider("Sampel untuk pencarian", 2000, 16000, 6000, 1000)
    p = pfx(cfg) + "model__"
    if nama == "Random Forest":
        base = RandomForestRegressor(n_jobs=-1, random_state=42)
        space = {p + "n_estimators": [100, 200, 300], p + "max_depth": [10, 15, 20, None],
                 p + "min_samples_leaf": [1, 2, 4], p + "max_features": [0.5, 0.7, 1.0]}
    else:
        base = HistGradientBoostingRegressor(random_state=42)
        space = {p + "learning_rate": [0.03, 0.05, 0.1, 0.2], p + "max_iter": [150, 300, 500],
                 p + "max_leaf_nodes": [15, 31, 63], p + "min_samples_leaf": [10, 20, 40],
                 p + "l2_regularization": [0.0, 0.1, 1.0]}
    space[pfx(cfg) + "select__k"] = [6, 8, 10, 12, "all"]
    if st.button("Jalankan optimasi", type="primary"):
        X_tr, X_te, y_tr, y_te = split(df, cfg)
        sub = X_tr.sample(min(smp, len(X_tr)), random_state=0)
        with st.spinner("Mencari hyperparameter terbaik..."):
            t0 = time.time()
            rs = RandomizedSearchCV(make_pipe(base, cfg), space, n_iter=n_iter, cv=cv, scoring="r2",
                                    random_state=42, refit=False)
            rs.fit(sub, y_tr.loc[sub.index])
            tuned = make_pipe(base, cfg).set_params(**rs.best_params_).fit(X_tr, y_tr)
            basel = make_pipe(base, cfg).fit(X_tr, y_tr)
        st.session_state["tune"] = {
            "name": nama, "rs": pd.DataFrame(rs.cv_results_), "best": rs.best_params_, "model": tuned,
            "base": evaluate(y_te, basel.predict(X_te)), "tuned": evaluate(y_te, tuned.predict(X_te)),
            "sec": time.time() - t0}
    t = st.session_state.get("tune")
    if not t:
        return
    st.success(f"Selesai dalam {t['sec']:.0f} detik ({t['name']}).")
    cmp = pd.DataFrame({"Baseline": t["base"], "Setelah optimasi": t["tuned"]}).T
    st.dataframe(cmp.style.format("{:,.3f}"))
    st.write("**Hyperparameter terbaik:**")
    st.json({k.split("__", 1)[-1] if "select" not in k else "select__k": (v if v is None or isinstance(v, str) else float(v) if isinstance(v, float) else int(v))
             for k, v in t["best"].items()})
    top = t["rs"].sort_values("rank_test_score").head(10)
    cols = ["rank_test_score", "mean_test_score", "std_test_score"] + [c for c in top.columns if c.startswith("param_")]
    st.dataframe(top[cols].rename(columns=lambda c: c.replace("param_", "").replace("regressor__", "")), hide_index=True)
    if st.button("Gunakan model hasil optimasi untuk prediksi"):
        st.session_state["active"] = {"model": t["model"], "name": f"{t['name']} (tuned)", "rmse": t["tuned"]["RMSE"]}
        st.toast("Model aktif diperbarui", icon="✅")


def page_model():
    df, cfg = get_df(), get_cfg()
    st.title("Modelling & Evaluasi")
    cv = st.slider("Fold cross-validation", 3, 5, 3)
    if st.button("Jalankan benchmark semua model", type="primary"):
        with st.spinner("Melatih 4 model dengan cross-validation..."):
            res, fitted = run_benchmark(df, cfg, cv)
        st.session_state["bench"] = (res, fitted)
        best = res.loc[0, "Model"]
        st.session_state["active"] = {"model": fitted[best][0], "name": best, "rmse": float(res.loc[0, "RMSE Test"])}
    if "bench" not in st.session_state:
        st.info("Tekan tombol di atas untuk memulai.")
        return
    res, fitted = st.session_state["bench"]
    st.dataframe(res.style.format({c: "{:,.3f}" for c in res.columns if c != "Model"}), hide_index=True)
    st.plotly_chart(px.bar(res, x="Model", y="R² Test", color="Model", text_auto=".3f", title="R² data uji"))
    pilih = st.selectbox("Analisis model", res["Model"], index=0)
    pipe, pred = fitted[pilih]
    _, X_te, _, y_te = split(df, cfg)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(x=y_te, y=pred, opacity=.4, labels={"x": "Aktual", "y": "Prediksi"}, title="Aktual vs prediksi")
        fig.add_shape(type="line", x0=y_te.min(), y0=y_te.min(), x1=y_te.max(), y1=y_te.max(), line=dict(color="red", dash="dash"))
        st.plotly_chart(fig)
    with c2:
        st.plotly_chart(px.histogram(x=y_te.values - pred, nbins=60, labels={"x": "Residual (aktual − prediksi)"}, title="Distribusi residual"))
    if st.toggle("Hitung permutation importance (butuh beberapa detik)"):
        cols = num_cols(cfg) + [CAT]
        idx = X_te.sample(min(1500, len(X_te)), random_state=0).index
        pi = permutation_importance(pipe, X_te.loc[idx, cols], y_te.loc[idx], n_repeats=3, random_state=0, n_jobs=-1)
        imp = pd.DataFrame({"Fitur": cols, "Importance": pi.importances_mean}).sort_values("Importance")
        st.plotly_chart(px.bar(imp, x="Importance", y="Fitur", orientation="h", title="Permutation importance"))
    buf = io.BytesIO()
    joblib.dump(pipe, buf)
    st.download_button(f"Unduh model {pilih} (.joblib)", buf.getvalue(), "model_harga_rumah.joblib")
    st.caption(f"Model aktif untuk prediksi: **{st.session_state['active']['name']}**")


# ======================= HALAMAN: PENERAPAN =======================
def page_single():
    df, cfg = get_df(), get_cfg()
    model, nama, rmse = active_model(df, cfg)
    st.title("Prediksi Satu Rumah")
    st.caption(f"Model aktif: {nama}")
    med = df[NUM_RAW].median()
    with st.form("single"):
        a, b, c = st.columns(3)
        lon = a.number_input("Longitude", value=float(med.longitude), format="%.2f")
        lat = a.number_input("Latitude", value=float(med.latitude), format="%.2f")
        age = a.slider("Umur median rumah", 1, 52, int(med.housing_median_age))
        inc = b.slider("Median pendapatan (x $10.000)", 0.5, 15.0, float(round(med.median_income, 1)), 0.1)
        rooms = b.number_input("Total kamar", 1, 40000, int(med.total_rooms))
        bed = b.number_input("Total kamar tidur", 1, 7000, int(med.total_bedrooms))
        pop = c.number_input("Populasi", 1, 36000, int(med.population))
        hh = c.number_input("Rumah tangga", 1, 6100, int(med.households))
        loc = c.selectbox("Jarak ke laut", sorted(df[CAT].unique()))
        ok = st.form_submit_button("Prediksi", type="primary")
    if not ok:
        return
    row = add_features(pd.DataFrame([{
        "longitude": lon, "latitude": lat, "housing_median_age": age, "total_rooms": rooms,
        "total_bedrooms": bed, "population": pop, "households": hh, "median_income": inc, CAT: loc,
        TARGET: 0}]))
    hasil = float(model.predict(row)[0])
    rata = df.loc[df[CAT] == loc, TARGET].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Perkiraan harga", f"${hasil:,.0f}")
    c2.metric("Rentang ±1 RMSE", f"${max(hasil - rmse, 0):,.0f} – ${hasil + rmse:,.0f}")
    c3.metric(f"vs rata-rata '{loc}'", f"${rata:,.0f}", f"{(hasil - rata) / rata * 100:+.1f}%")
    smp = df.sample(min(5000, len(df)), random_state=1)
    fig = px.scatter(smp, x="longitude", y="latitude", color=TARGET, color_continuous_scale="Turbo", opacity=.5, height=500)
    fig.add_scatter(x=[lon], y=[lat], mode="markers", marker=dict(size=18, color="white", symbol="star", line=dict(width=2, color="black")), name="Lokasi input")
    st.plotly_chart(fig)


def page_bulk():
    df, cfg = get_df(), get_cfg()
    model, nama, _ = active_model(df, cfg)
    st.title("Prediksi Massal (CSV)")
    st.caption(f"Model aktif: {nama}")
    st.download_button("Unduh template CSV", df[NUM_RAW + [CAT]].head(5).to_csv(index=False), "template.csv")
    up = st.file_uploader("Upload CSV berisi kolom: " + ", ".join(NUM_RAW + [CAT]), type="csv", key="bulk")
    if up is None:
        return
    new = pd.read_csv(up)
    miss = [c for c in NUM_RAW + [CAT] if c not in new.columns]
    if miss:
        st.error(f"Kolom belum lengkap: {miss}")
        return
    X = add_features(new)
    out = new.copy()
    out["prediksi_harga"] = model.predict(X).round(0)
    st.dataframe(out)
    c = st.columns(3)
    c[0].metric("Jumlah baris", f"{len(out):,}")
    c[1].metric("Rata-rata prediksi", f"${out.prediksi_harga.mean():,.0f}")
    c[2].metric("Median prediksi", f"${out.prediksi_harga.median():,.0f}")
    if TARGET in new.columns:
        st.subheader("Evaluasi (kolom target tersedia)")
        st.dataframe(pd.DataFrame([evaluate(new[TARGET], out.prediksi_harga)]).style.format("{:,.3f}"), hide_index=True)
    st.plotly_chart(px.histogram(out, x="prediksi_harga", nbins=40, title="Distribusi prediksi"))
    st.download_button("Unduh hasil prediksi", out.to_csv(index=False), "hasil_prediksi.csv", type="primary")


def page_kesimpulan():
    df, cfg = get_df(), get_cfg()
    st.title("Kesimpulan & Saran")
    st.header("Kesimpulan")
    cap_t = int((df[TARGET] >= 500001).sum())
    a, b = st.columns(2)
    with a:
        if "bench" in st.session_state:
            res = st.session_state["bench"][0]
            card("1. Model terbaik", f"{res.loc[0, 'Model']} memperoleh R² uji {res.loc[0, 'R² Test']:.3f} "
                 f"dan RMSE ${res.loc[0, 'RMSE Test']:,.0f}; model terlemah R² {res['R² Test'].min():.3f}.")
        else:
            card("1. Model terbaik", "Belum ada hasil. Jalankan benchmark di halaman Modelling & Evaluasi.", "#888")
        v = vif_table(df)
        card("2. Multikolinearitas", f"{int((v.VIF > 10).sum())} dari {len(v)} fitur memiliki VIF > 10 "
             f"(tertinggi: {v.iloc[0].Fitur}, {v.iloc[0].VIF:,.0f}).", "#e0a030")
    with b:
        t = st.session_state.get("tune")
        if t:
            card("3. Efek optimasi", f"R² {t['name']} berubah dari {t['base']['R²']:.3f} menjadi {t['tuned']['R²']:.3f}; "
                 f"RMSE dari ${t['base']['RMSE']:,.0f} menjadi ${t['tuned']['RMSE']:,.0f}.")
        else:
            card("3. Efek optimasi", "Belum dijalankan di halaman Teknik Optimasi.", "#888")
        card("4. Sensor data", f"{cap_t:,} baris ({cap_t / len(df) * 100:.1f}%) memiliki harga tersensor $500.001.", "#d0476b")
    card("Kualitas data", f"{len(df):,} baris; hanya total_bedrooms yang memiliki missing value "
         f"({int(df.total_bedrooms.isna().sum())} baris), ditangani imputasi median dalam pipeline sehingga tidak ada kebocoran data.")
    l, s = st.columns(2)
    with l:
        st.header("Keterbatasan")
        st.markdown("- Data tahun 1990 dan agregat per blok, bukan per rumah.\n"
                    "- Evaluasi memakai satu kali split ditambah CV; hasil dapat berubah pada pembagian lain.\n"
                    "- Harga tersensor membuat prediksi rumah mahal cenderung terlalu rendah.")
    with s:
        st.header("Saran")
        st.markdown("- Gunakan nested cross-validation untuk estimasi yang lebih stabil.\n"
                    "- Tambahkan fitur geospasial (jarak ke kota besar, clustering lokasi).\n"
                    "- Coba model lain (XGBoost/LightGBM) dan metode ensembel stacking.")


# ======================= NAVIGASI =======================
def main():
    st.session_state.setdefault("cfg", dict(DEFAULT_CFG))
    P = st.Page
    pg = st.navigation({
        "Pengantar": [P(page_beranda, title="Beranda", icon=":material/home:", default=True),
                      P(page_dataset, title="Tentang Dataset", icon=":material/database:"),
                      P(page_eda, title="Eksplorasi Data (EDA)", icon=":material/bar_chart:")],
        "Pengolahan & Model": [P(page_preprocessing, title="Preprocessing", icon=":material/tune:"),
                               P(page_pca, title="Reduksi Dimensi (PCA)", icon=":material/scatter_plot:"),
                               P(page_optimasi, title="Teknik Optimasi", icon=":material/auto_fix_high:"),
                               P(page_model, title="Modelling & Evaluasi", icon=":material/model_training:")],
        "Penerapan": [P(page_single, title="Prediksi Satu Rumah", icon=":material/home_work:"),
                      P(page_bulk, title="Prediksi Massal (CSV)", icon=":material/upload_file:"),
                      P(page_kesimpulan, title="Kesimpulan & Saran", icon=":material/task_alt:")]})
    with st.sidebar:
        st.divider()
    with st.sidebar.container(border=True):
        st.markdown("**Tugas Machine Learning**")
        st.caption("Regresi Harga Rumah – California Housing")
        st.markdown(
            """
            <small>
        • Muhamad Habib Raihan (241011400625)<br>
        • Ryu Barena (241011400648)<br>
        • Muhammad Bintang Rivaldi (241011401764)
        </small>
        """,
        unsafe_allow_html=True
    )
    pg.run()

if __name__ == "__main__":
    main()