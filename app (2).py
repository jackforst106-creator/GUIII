"""
GUI Streamlit: Pemodelan Harga Saham INDF & ICBP dengan MGBM
dan Pengukuran Risiko VaR Variance-Covariance + Backtesting Kupiec.

Jalankan:  streamlit run app.py
Input   :  CSV Investing.com (Date, Price, Open, High, Low, Vol., Change %)
           Upload lewat sidebar, atau taruh INDF.csv dan ICBP.csv di folder yang sama.
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import stats
from scipy.special import xlogy

st.set_page_config(page_title="MGBM & VaR INDF-ICBP", layout="wide")

Z_95 = 1.645
P_EXP = 0.05
CV_CHI2 = 3.841
STOCKS = ["INDF", "ICBP"]


# ---------------------------------------------------------------- data
def load_csv(src):
    df = pd.read_csv(src)
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    for c in ["Price", "Open", "High", "Low"]:
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(str).str.replace(",", "", regex=False).astype(float)
    return df.sort_values("Date").reset_index(drop=True)


def get_source(name, uploaded):
    if uploaded is not None:
        return uploaded
    if os.path.exists(f"{name}.csv"):
        return f"{name}.csv"
    return None


def prepare(dfs):
    """Pra-pemrosesan: urutkan, samakan tanggal, hitung log return."""
    px = pd.concat(
        {k: v.set_index("Date")["Price"] for k, v in dfs.items()}, axis=1
    ).dropna()
    ret = np.log(px / px.shift(1)).dropna()
    return px, ret


# ---------------------------------------------------------------- analisis
def describe(ret):
    rows = {}
    for s in ret.columns:
        r = ret[s]
        xr = np.asarray(r, dtype=float)
        rows[s] = {
            "Jumlah return": len(r),
            "Minimum": r.min(),
            "Maksimum": r.max(),
            "Rata-rata": r.mean(),
            "Varians": r.var(ddof=1),
            "Std. deviasi": r.std(ddof=1),
            "Skewness": float(stats.skew(xr)),
            "Excess kurtosis": float(stats.kurtosis(xr)),
        }
    return pd.DataFrame(rows)


def ks_test(r):
    x = np.asarray(r, dtype=float)
    mu, sd = float(x.mean()), float(x.std(ddof=1))
    res = stats.kstest(x, stats.norm(loc=mu, scale=sd).cdf)
    return float(res.statistic), float(res.pvalue)


def mgbm_params(ret):
    rbar = ret.mean()
    sig = ret.std(ddof=1)
    mu_hat = rbar + sig**2 / 2  # Pers. 2.11
    cov = ret.cov()
    corr = ret.corr()
    chol = np.linalg.cholesky(cov.values)  # Sigma = L L^T
    return rbar, sig, mu_hat, cov, corr, chol


def simulate_mgbm(s0, mu, chol, horizon, n_paths, seed):
    """S_i(t)=S_i(t-1)exp[(mu_i - 1/2 sum_j s_ij^2)dt + sum_j s_ij sqrt(dt) Z_j]"""
    rng = np.random.default_rng(seed)
    n = len(s0)
    drift = mu - 0.5 * (chol**2).sum(axis=1)
    paths = np.zeros((n_paths, horizon + 1, n))
    paths[:, 0, :] = s0
    for t in range(1, horizon + 1):
        z = rng.standard_normal((n_paths, n))
        paths[:, t, :] = paths[:, t - 1, :] * np.exp(drift + z @ chol.T)
    return paths


def var_value(rbar, sig, w0):
    return w0 * (Z_95 * sig - rbar)  # VaR = W0 (1,645 sigma - rbar)


def kupiec(r, var_pct):
    T = len(r)
    N = int((r < -var_pct).sum())
    p = P_EXP
    p_hat = N / T
    ll0 = xlogy(T - N, 1 - p) + xlogy(N, p)
    ll1 = xlogy(T - N, 1 - p_hat) + xlogy(N, p_hat)
    lr = -2 * ll0 + 2 * ll1
    return T, N, p_hat, lr, lr < CV_CHI2


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Pengaturan")
up_indf = st.sidebar.file_uploader("CSV INDF", type="csv")
up_icbp = st.sidebar.file_uploader("CSV ICBP", type="csv")
selected = st.sidebar.multiselect("Pemilihan saham (tampilan)", STOCKS, default=STOCKS)
w0 = st.sidebar.number_input(
    "Nilai investasi awal W₀ (Rp)", min_value=1_000_000, value=100_000_000, step=1_000_000
)
st.sidebar.markdown("**Tingkat kepercayaan:** 95%  \n**Horizon:** 1 hari perdagangan")
st.sidebar.subheader("Simulasi MGBM")
horizon = st.sidebar.slider("Horizon simulasi (hari)", 5, 120, 30)
n_paths = st.sidebar.slider("Jumlah lintasan", 50, 2000, 500, step=50)
seed = st.sidebar.number_input("Seed", value=42, step=1)

st.title("Pemodelan Harga Saham INDF dan ICBP dengan MGBM")
st.caption("Pengukuran risiko: Value at Risk Variance-Covariance • Validasi: Kupiec Test")

srcs = {"INDF": get_source("INDF", up_indf), "ICBP": get_source("ICBP", up_icbp)}
if any(v is None for v in srcs.values()):
    st.info("Upload CSV INDF dan ICBP (format Investing.com) di sidebar untuk memulai.")
    st.stop()

try:
    dfs = {k: load_csv(v) for k, v in srcs.items()}
    px, ret = prepare(dfs)
except Exception as e:
    st.error(f"Gagal membaca data: {e}")
    st.stop()

if not selected:
    st.warning("Pilih minimal satu saham.")
    st.stop()

# ---------------------------------------------------------------- a. periode
st.header("1. Informasi Periode Data")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tanggal awal", px.index.min().strftime("%d %b %Y"))
c2.metric("Tanggal akhir", px.index.max().strftime("%d %b %Y"))
c3.metric("Observasi harga", len(px))
c4.metric("Observasi return", len(ret))

# ---------------------------------------------------------------- c. harga
st.header("2. Data Harga Saham")
fig = go.Figure()
for s in selected:
    fig.add_scatter(x=px.index, y=px[s], name=s, mode="lines")
fig.update_layout(height=380, yaxis_title="Harga penutupan (Rp)", margin=dict(t=20))
st.plotly_chart(fig, use_container_width=True)
with st.expander("Lihat tabel data"):
    st.dataframe(px[selected].sort_index(ascending=False), use_container_width=True)

# ---------------------------------------------------------------- d. statistik
st.header("3. Statistik dan Normalitas")
desc = describe(ret)
st.dataframe(desc[selected].style.format("{:.6f}"), use_container_width=True)

cols = st.columns(len(selected))
for col, s in zip(cols, selected):
    d, p = ks_test(ret[s])
    with col:
        st.subheader(s)
        st.write(f"Statistik D = **{d:.4f}**, p-value = **{p:.4f}**")
        if p < 0.05:
            st.error("H₀ ditolak: return tidak berdistribusi normal (α = 0,05).")
        else:
            st.success("H₀ diterima: return berdistribusi normal (α = 0,05).")
        h = go.Figure()
        h.add_histogram(x=ret[s], histnorm="probability density", nbinsx=30, name="Return")
        xs = np.linspace(ret[s].min(), ret[s].max(), 200)
        h.add_scatter(
            x=xs, y=stats.norm.pdf(xs, float(ret[s].mean()), float(ret[s].std(ddof=1))), name="Normal"
        )
        h.update_layout(height=280, margin=dict(t=10), showlegend=False)
        st.plotly_chart(h, use_container_width=True)
st.caption(
    "Catatan: K-S dengan parameter hasil estimasi cenderung konservatif; "
    "uji Lilliefors dapat dipakai sebagai pembanding."
)

# ---------------------------------------------------------------- e. MGBM
st.header("4. Korelasi dan Parameter MGBM")
rbar, sig, mu_hat, cov, corr, chol = mgbm_params(ret)
params = pd.DataFrame(
    {"r̄ (rata-rata log return)": rbar, "σ (volatilitas)": sig, "μ̂ = r̄ + σ²/2 (drift)": mu_hat}
)
a, b = st.columns(2)
with a:
    st.markdown("**Parameter**")
    st.dataframe(params.style.format("{:.6f}"), use_container_width=True)
    st.markdown("**Matriks varians-kovarians Σ**")
    st.dataframe(cov.style.format("{:.3e}"), use_container_width=True)
with b:
    st.metric("Korelasi INDF–ICBP", f"{corr.loc['INDF', 'ICBP']:.4f}")
    st.markdown("**Matriks volatilitas σ (Cholesky, Σ = σσᵀ)**")
    st.dataframe(
        pd.DataFrame(chol, index=STOCKS, columns=["Z₁", "Z₂"]).style.format("{:.6f}"),
        use_container_width=True,
    )

st.subheader("Simulasi harga MGBM dua dimensi")
s0 = px.iloc[-1].values
paths = simulate_mgbm(s0, mu_hat.values, chol, horizon, n_paths, int(seed))
sim_cols = st.columns(len(selected))
for col, s in zip(sim_cols, selected):
    i = STOCKS.index(s)
    f = go.Figure()
    for k in range(min(n_paths, 60)):
        f.add_scatter(
            y=paths[k, :, i], mode="lines", line=dict(width=0.6, color="rgba(120,120,120,.35)"),
            showlegend=False,
        )
    f.add_scatter(y=paths[:, :, i].mean(axis=0), name="Rata-rata", line=dict(color="crimson", width=2.5))
    f.update_layout(
        title=f"{s}: harga awal Rp{s0[i]:,.0f}", height=330, xaxis_title="Hari ke-",
        yaxis_title="Harga (Rp)", margin=dict(t=40),
    )
    with col:
        st.plotly_chart(f, use_container_width=True)
        st.write(
            f"Rata-rata hari ke-{horizon}: **Rp{paths[:, -1, i].mean():,.0f}** "
            f"(persentil 5–95%: Rp{np.percentile(paths[:, -1, i], 5):,.0f} – "
            f"Rp{np.percentile(paths[:, -1, i], 95):,.0f})"
        )

# ---------------------------------------------------------------- f. VaR
st.header("5. Hasil Value at Risk (95%, 1 hari)")
st.latex(r"VaR = W_0\,(1{,}645\,\sigma - \bar r)")
var_rp = {s: var_value(rbar[s], sig[s], w0) for s in STOCKS}
cols = st.columns(len(selected))
for col, s in zip(cols, selected):
    col.metric(f"VaR {s}", f"Rp{var_rp[s]:,.0f}", f"{var_rp[s] / w0:.2%} dari W₀", delta_color="off")
    col.caption(
        f"Dengan keyakinan 95%, kerugian harian {s} tidak melebihi Rp{var_rp[s]:,.0f}."
    )

# ---------------------------------------------------------------- g. Kupiec
st.header("6. Kupiec (Backtesting)")
st.latex(
    r"LR_{POF} = -2\ln\!\left[(1-p)^{T-N}p^{N}\right] + 2\ln\!\left[(1-\hat p)^{T-N}\hat p^{N}\right]"
)
rows = []
figs = {}
for s in selected:
    var_pct = var_rp[s] / w0
    T, N, p_hat, lr, valid = kupiec(ret[s], var_pct)
    rows.append(
        {
            "Saham": s,
            "T": T,
            "N (exception)": N,
            "Harapan N (5%)": round(T * P_EXP, 1),
            "p̂ = N/T": round(p_hat, 4),
            "LR_POF": round(lr, 4),
            "CV χ²(0,05;1)": CV_CHI2,
            "Keputusan": "H₀ diterima: VaR valid" if valid else "H₀ ditolak: VaR tidak valid",
        }
    )
    f = go.Figure()
    f.add_scatter(x=ret.index, y=ret[s], mode="lines", name="Return aktual")
    f.add_hline(y=-var_pct, line_color="crimson", annotation_text="-VaR")
    exc = ret[s][ret[s] < -var_pct]
    f.add_scatter(x=exc.index, y=exc, mode="markers", marker=dict(color="crimson", size=8), name="Exception")
    f.update_layout(title=f"{s}: return vs -VaR", height=320, margin=dict(t=40))
    figs[s] = f
st.dataframe(pd.DataFrame(rows).set_index("Saham"), use_container_width=True)
cols = st.columns(len(selected))
for col, s in zip(cols, selected):
    col.plotly_chart(figs[s], use_container_width=True)
