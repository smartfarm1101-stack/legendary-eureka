import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="Streamlit Test",
    page_icon="✅"
)

st.title("✅ Streamlit 연결 테스트")

st.write("이 화면이 보이면 GitHub와 Streamlit이 정상적으로 연결되었습니다.")

st.divider()

st.write("⏰ 현재 시간:")
st.write(datetime.now())

st.caption("페이지를 새로고침하면 시간이 바뀌면 정상입니다.")

st.success("연결 성공!")


import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import unicodedata
import io
import numpy as np

# ===============================
# 기본 설정
# ===============================
st.set_page_config(
    page_title="pH/EC와 생장의 상관관계",
    layout="wide"
)

# 한글 폰트 (Streamlit + Plotly)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# 파일 유틸 (NFC/NFD 완전 대응)
# ===============================
def normalize_all(name: str):
    return {
        unicodedata.normalize("NFC", name),
        unicodedata.normalize("NFD", name)
    }

def find_file(data_dir: Path, target_name: str):
    targets = normalize_all(target_name)
    for f in data_dir.iterdir():
        if f.is_file():
            if unicodedata.normalize("NFC", f.name) in targets or \
               unicodedata.normalize("NFD", f.name) in targets:
                return f
    return None

# ===============================
# 데이터 로딩
# ===============================
@st.cache_data
def load_environment_data():
    data_dir = Path("data")
    env_data = {}

    school_files = {
        "송도고": "송도고_환경데이터.csv",
        "하늘고": "하늘고_환경데이터.csv",
        "아라고": "아라고_환경데이터.csv",
        "동산고": "동산고_환경데이터.csv",
    }

    for school, fname in school_files.items():
        file_path = find_file(data_dir, fname)
        if file_path is None:
            st.error(f"환경 데이터 파일을 찾을 수 없습니다: {fname}")
            return None
        df = pd.read_csv(file_path)
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")

        # 결측치 보정
        df.interpolate(method="linear", inplace=True)

        # IQR 이상치 제거
        for col in ["temperature", "humidity", "ph", "ec"]:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]

        env_data[school] = df.reset_index(drop=True)

    return env_data


@st.cache_data
def load_growth_data():
    data_dir = Path("data")
    xlsx_path = find_file(data_dir, "4개교_생육결과데이터.xlsx")
    if xlsx_path is None:
        st.error("생육 결과 데이터 파일을 찾을 수 없습니다.")
        return None

    xls = pd.ExcelFile(xlsx_path)
    growth_data = {}

    for sheet in xls.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        growth_data[sheet] = df

    return growth_data


# ===============================
# 데이터 로딩 실행
# ===============================
with st.spinner("데이터 로딩 중..."):
    env_data = load_environment_data()
    growth_data = load_growth_data()

if env_data is None or growth_data is None:
    st.stop()

# ===============================
# 사이드바
# ===============================
school_option = st.sidebar.selectbox(
    "학교 선택",
    ["전체", "송도고", "하늘고", "아라고", "동산고"]
)

# ===============================
# 제목
# ===============================
st.title("🌱 pH/EC와 생장의 상관관계")

tab1, tab2, tab3 = st.tabs([
    "📈 pH/EC 변화",
    "📉 선형 회귀 분석",
    "🔁 pH-EC 쌍곡선 모델"
])

# ===============================
# Tab 1: 시간 변화
# ===============================
with tab1:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=["pH 변화", "EC 변화"]
    )

    for school, df in env_data.items():
        if school_option != "전체" and school != school_option:
            continue

        fig.add_trace(
            go.Scatter(x=df["time"], y=df["ph"], mode="lines", name=f"{school} pH"),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df["time"], y=df["ec"], mode="lines", name=f"{school} EC"),
            row=2, col=1
        )

    fig.update_layout(
        height=700,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        legend_title="학교"
    )

    st.plotly_chart(fig, use_container_width=True)

# ===============================
# Tab 2: 선형 회귀
# ===============================
with tab2:
    all_df = pd.concat(env_data.values(), ignore_index=True)
    t = np.arange(len(all_df))

    ph_coef = np.polyfit(t, all_df["ph"], 1)
    ec_coef = np.polyfit(t, all_df["ec"], 1)

    fig = make_subplots(rows=2, cols=1, subplot_titles=[
        f"pH 선형 회귀 (기울기 = {ph_coef[0]:.6f})",
        f"EC 선형 회귀 (기울기 = {ec_coef[0]:.6f})"
    ])

    fig.add_trace(
        go.Scatter(y=all_df["ph"], mode="markers", name="pH"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=np.polyval(ph_coef, t), mode="lines", name="pH Regression"),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(y=all_df["ec"], mode="markers", name="EC"),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(y=np.polyval(ec_coef, t), mode="lines", name="EC Regression"),
        row=2, col=1
    )

    fig.update_layout(
        height=700,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
    )

    st.plotly_chart(fig, use_container_width=True)

# ===============================
# Tab 3: 쌍곡선 함수
# ===============================
with tab3:
    ph = all_df["ph"]
    ec = all_df["ec"]

    x = 1 / ph
    coef = np.polyfit(x, ec, 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ph, y=ec, mode="markers", name="실측값"))
    fig.add_trace(go.Scatter(
        x=ph,
        y=coef[0] * (1 / ph) + coef[1],
        mode="lines",
        name=f"EC = {coef[1]:.3f} + {coef[0]:.3f} × (1/pH)"
    ))

    corr = np.corrcoef(ph, ec)[0, 1]

    fig.update_layout(
        title=f"pH-EC 쌍곡선 관계 (상관계수 = {corr:.2f})",
        xaxis_title="pH",
        yaxis_title="EC",
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
    )

    st.plotly_chart(fig, use_container_width=True)

# ===============================
# XLSX 다운로드
# ===============================
st.subheader("📥 데이터 다운로드")

buffer = io.BytesIO()
all_df.to_excel(buffer, index=False, engine="openpyxl")
buffer.seek(0)

st.download_button(
    label="통합 환경 데이터 다운로드 (XLSX)",
    data=buffer,
    file_name="통합_환경데이터.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
