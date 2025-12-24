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
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import unicodedata
import io

# ===============================
# 기본 설정
# ===============================
st.set_page_config(
    page_title="남곤 물고기 남덩이",
    layout="wide"
)

# ===============================
# 한글 폰트 (Streamlit)
# ===============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.title("🐟 남곤 물고기 남덩이")
st.caption("극지식물 최적 EC 농도 연구 대시보드")

# ===============================
# 유틸: NFC/NFD 파일 탐색
# ===============================
def normalize_name(name: str, form: str):
    return unicodedata.normalize(form, name)

def find_file_by_name(directory: Path, target_name: str):
    for file in directory.iterdir():
        if normalize_name(file.name, "NFC") == normalize_name(target_name, "NFC"):
            return file
        if normalize_name(file.name, "NFD") == normalize_name(target_name, "NFD"):
            return file
    return None

# ===============================
# 데이터 로딩
# ===============================
@st.cache_data
def load_env_data():
    data_dir = Path("data")
    school_files = {
        "송도고": "송도고_환경데이터.csv",
        "하늘고": "하늘고_환경데이터.csv",
        "아라고": "아라고_환경데이터.csv",
        "동산고": "동산고_환경데이터.csv",
    }

    env_data = {}
    for school, fname in school_files.items():
        file_path = find_file_by_name(data_dir, fname)
        if file_path is None:
            st.error(f"❌ {school} 환경 데이터 파일을 찾을 수 없습니다.")
            return None

        df = pd.read_csv(file_path)
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")

        # 결측치 보정
        df = df.interpolate(method="linear")

        # IQR 이상치 제거
        for col in ["temperature", "humidity", "ph", "ec"]:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]

        env_data[school] = df

    return env_data

@st.cache_data
def load_growth_data():
    data_dir = Path("data")
    xlsx_path = find_file_by_name(data_dir, "4개교_생육결과데이터.xlsx")
    if xlsx_path is None:
        st.error("❌ 생육 결과 XLSX 파일을 찾을 수 없습니다.")
        return None

    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    growth = {}
    for sheet in xls.sheet_names:
        growth[sheet] = pd.read_excel(xls, sheet_name=sheet)
    return growth

with st.spinner("📊 데이터 로딩 중..."):
    env_data = load_env_data()
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

selected_schools = list(env_data.keys()) if school_option == "전체" else [school_option]

# ===============================
# 탭 구성
# ===============================
tab1, tab2, tab3 = st.tabs([
    "📈 pH/EC 변화",
    "📉 선형회귀 분석",
    "🔁 pH-EC 쌍곡선 관계"
])

# ===============================
# Tab 1: 시간에 따른 변화
# ===============================
with tab1:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("시간에 따른 pH 변화", "시간에 따른 EC 변화")
    )

    for school in selected_schools:
        df = env_data[school]
        fig.add_trace(
            go.Scatter(x=df["time"], y=df["ph"], name=f"{school} pH"),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df["time"], y=df["ec"], name=f"{school} EC"),
            row=2, col=1
        )

    fig.update_layout(
        height=700,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
    )
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# Tab 2: 선형회귀
# ===============================
with tab2:
    all_df = pd.concat(env_data.values(), ignore_index=True)

    t = np.arange(len(all_df))
    ph_coef = np.polyfit(t, all_df["ph"], 1)
    ec_coef = np.polyfit(t, all_df["ec"], 1)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("pH 선형회귀", "EC 선형회귀")
    )

    fig.add_trace(go.Scatter(y=all_df["ph"], mode="markers", name="pH"), row=1, col=1)
    fig.add_trace(go.Scatter(y=ph_coef[0]*t + ph_coef[1], name="Regression"), row=1, col=1)

    fig.add_trace(go.Scatter(y=all_df["ec"], mode="markers", name="EC"), row=2, col=1)
    fig.add_trace(go.Scatter(y=ec_coef[0]*t + ec_coef[1], name="Regression"), row=2, col=1)

    fig.update_layout(
        height=700,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"""
- pH 기울기: **{ph_coef[0]:.5f}**
- EC 기울기: **{ec_coef[0]:.5f}**
    """)

# ===============================
# Tab 3: 쌍곡선 함수
# ===============================
with tab3:
    ph = all_df["ph"]
    ec = all_df["ec"]

    coef = np.polyfit(1/ph, ec, 1)
    ec_pred = coef[0] * (1/ph) + coef[1]

    corr = np.corrcoef(ph, ec)[0, 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ph, y=ec, mode="markers", name="Observed"))
    fig.add_trace(go.Scatter(x=ph, y=ec_pred, mode="lines", name="Hyperbolic Fit"))

    fig.update_layout(
        title="pH-EC 쌍곡선 관계",
        xaxis_title="pH",
        yaxis_title="EC",
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.success(f"""
회귀식:
EC = {coef[1]:.3f} + {coef[0]:.3f} × (1/pH)

상관계수: **{corr:.2f}**
    """)

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
