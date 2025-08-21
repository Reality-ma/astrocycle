import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astrocycle.pipeline import run_pipeline
from astrocycle.spectrum import bandpass_hilbert

# Ensure 'src' is on path so we can import astrocycle without installation
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURR_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from astrocycle import run_pipeline, bandpass_hilbert

st.set_page_config(page_title="AstroCycle-MVP", layout="wide")
st.title("AstroCycle-MVP : 深度→时间 与 天文旋回初筛 (v2)")

with st.sidebar:
    st.header("输入数据")
    data_file = st.file_uploader("主数据CSV（列名：depth,value[,sigma]）", type=["csv"])
    anchors_file = st.file_uploader("锚点CSV（列名：depth,age_kyr[,age_sigma_kyr]）", type=["csv"])

    st.header("年龄–深度")
    model = st.selectbox("模型", ["pchip","linear"], index=0)
    mc = st.number_input("蒙特卡罗次数(anchors不确定度)", min_value=0, max_value=2000, value=0, step=50)

    st.header("频谱参数（cpk=每千年循环数）")
    fmin = st.number_input("最小频率 cpk", min_value=0.0005, max_value=1.0, value=0.002, step=0.001, format="%.4f")
    fmax = st.number_input("最大频率 cpk", min_value=0.005, max_value=2.0, value=0.1, step=0.005)
    nfreq = st.slider("频率采样点数", min_value=300, max_value=5000, value=1000, step=100)

    st.header("带通与包络（检查调幅）")
    bands = {
        "Precession ~ 19-23 kyr": (1/23.0, 1/19.0),
        "Obliquity ~ 41 kyr": (1/48.0, 1/35.0),
        "Eccentricity ~ 100 kyr": (1/140.0, 1/80.0),
        "Long Ecc ~ 405 kyr": (1/520.0, 1/320.0),
    }
    band_key = st.selectbox("目标频带", list(bands.keys()), index=0)

st.markdown("### 1) 加载与预览")
if data_file is not None:
    df = pd.read_csv(data_file)
    st.write(df.head())
else:
    st.info("请上传主数据 CSV。也可用 `data/example_series.csv` 与 `data/example_anchors.csv` 进行试跑。")

anchors = None
if anchors_file is not None:
    anchors = pd.read_csv(anchors_file)
    st.write("Anchors:", anchors)

if data_file is not None:
    depth = df["depth"].to_numpy()
    value = df["value"].to_numpy()
    if anchors is not None:
        ad = anchors["depth"].to_numpy()
        aa = anchors["age_kyr"].to_numpy()
        asa = anchors["age_sigma_kyr"].to_numpy() if "age_sigma_kyr" in anchors.columns else None
    else:
        ad = aa = asa = None

    st.markdown("### 2) 运行管线")
    res = run_pipeline(depth, value, anchors_depth=ad, anchors_age_kyr=aa, anchors_age_sigma_kyr=asa,
                       model=model, mc=mc, freq_min=fmin, freq_max=fmax, nfreq=nfreq)
    t_kyr = res["t_kyr"]; x = res["x"]
    freq = res["freq_cpk"]; pwr = res["power"]; thr95 = res["ar1_95"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("年龄–深度预览 & 时间域序列")
        fig1 = plt.figure()
        plt.plot(t_kyr, x, lw=1)
        plt.xlabel("Age (kyr)")
        plt.ylabel("Value")
        plt.title("Time-domain series (sorted by age)")
        st.pyplot(fig1)

    with col2:
        st.subheader("Lomb–Scargle 频谱 (cpk)")
        fig2 = plt.figure()
        plt.plot(freq, pwr, label="LS Power")
        if np.all(np.isfinite(thr95)):
            plt.plot(freq, thr95, ls="--", label="AR1 95% approx")
        plt.xlabel("Frequency (cycles per kyr)")
        plt.ylabel("Power")
        plt.legend()
        st.pyplot(fig2)

    st.markdown("### 3) 目标频带带通 + Hilbert 包络")
    fl, fh = bands[band_key]
    try:
        x_f, env = bandpass_hilbert(t_kyr, x, fl, fh)
        fig3 = plt.figure()
        plt.plot(t_kyr, x_f, lw=1, label="bandpass")
        plt.plot(t_kyr, env, lw=1, label="envelope")
        plt.xlabel("Age (kyr)"); plt.legend()
        plt.title(f"Band: {band_key}")
        st.pyplot(fig3)
    except Exception as e:
        st.error(f"带通失败: {e}")

    if "power_mc_mean" in res:
        st.markdown("### 4) MC 不确定度（功率均值±std）")
        fig4 = plt.figure()
        mu = res["power_mc_mean"]; sd = res["power_mc_std"]
        plt.plot(freq, mu, label="MC mean")
        plt.fill_between(freq, mu-sd, mu+sd, alpha=0.3, label="±1σ")
        plt.xlabel("Frequency (cpk)"); plt.ylabel("Power")
        plt.legend()
        st.pyplot(fig4)

    st.success("完成。可在侧栏调整参数重跑。")
