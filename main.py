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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import unicodedata
import io

# =========================
# Streamlit 기본 설정
# =========================
st.set_page_config(
    page_title="극지식물 최적 EC 농도 연구",
    layout="wide"
)

# =========================
# 한글 폰트 깨짐 방지 (CSS)
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 공통 설정
# =========================
DATA_DIR = Path("data")

SCHOOL_EC = {
    "송도고": 1.0,
    "하늘고": 2.0,
    "아라고": 4.0,
    "동산고": 8.0
}

SCHOOL_COLOR = {
    "송도고": "#1f77b4",
    "하늘고": "#2ca02c",
    "아라고": "#ff7f0e",
    "동산고": "#d62728"
}

# =========================
# 파일 탐색 (NFC/NFD 안전)
# =========================
def find_file(target_name: str):
    target_nfc = unicodedata.normalize("NFC", target_name)
    target_nfd = unicodedata.normalize("NFD", target_name)

    for f in DATA_DIR.iterdir():
        fname_nfc = unicodedata.normalize("NFC", f.name)
        fname_nfd = unicodedata.normalize("NFD", f.name)
        if fname_nfc == target_nfc or fname_nfd == target_nfd:
            return f
    return None

# =========================
# 데이터 로딩
# =========================
@st.cache_data
def load_environment_data():
    data = {}
    for school in SCHOOL_EC.keys():
        file_path = find_file(f"{school}_환경데이터.csv")
        if file_path is None:
            st.error(f"❌ {school} 환경 데이터 파일을 찾을 수 없습니다.")
            continue
        df = pd.read_csv(file_path)
        df["학교"] = school
        data[school] = df
    return data

@st.cache_data
def load_growth_data():
    file_path = find_file("4개교_생육결과데이터.xlsx")
    if file_path is None:
        st.error("❌ 생육 결과 XLSX 파일을 찾을 수 없습니다.")
        return {}

    xls = pd.ExcelFile(file_path, engine="openpyxl")
    data = {}
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        df["학교"] = sheet
        df["EC"] = SCHOOL_EC.get(sheet, None)
        data[sheet] = df
    return data

# =========================
# 데이터 로딩
# =========================
with st.spinner("📂 데이터 로딩 중..."):
    env_data = load_environment_data()
    growth_data = load_growth_data()

if not env_data or not growth_data:
    st.stop()

# =========================
# 제목
# =========================
st.title("🌱 극지식물 최적 EC 농도 연구")

# =========================
# 사이드바
# =========================
selected_school = st.sidebar.selectbox(
    "🏫 학교 선택",
    ["전체"] + list(SCHOOL_EC.keys())
)

# =========================
# 탭 구성
# =========================
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

# ======================================================
# Tab 1 : 실험 개요
# ======================================================
with tab1:
    st.subheader("연구 배경 및 목적")
    st.write(
        """
        본 연구는 **극지식물의 최적 EC(전기전도도) 농도**를 규명하기 위해  
        4개 학교에서 서로 다른 EC 조건 하에 환경 데이터와 생육 결과를 비교·분석하였다.
        """
    )

    info_df = pd.DataFrame([
        {
            "학교명": s,
            "EC 목표": SCHOOL_EC[s],
            "개체수": len(growth_data[s]),
            "색상": SCHOOL_COLOR[s]
        } for s in SCHOOL_EC
    ])

    st.dataframe(info_df, use_container_width=True)

    total_count = sum(len(df) for df in growth_data.values())
    avg_temp = pd.concat(env_data.values())["temperature"].mean()
    avg_hum = pd.concat(env_data.values())["humidity"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 개체수", f"{total_count}개")
    col2.metric("평균 온도", f"{avg_temp:.1f} ℃")
    col3.metric("평균 습도", f"{avg_hum:.1f} %")
    col4.metric("최적 EC", "2.0 (하늘고) ⭐")

# ======================================================
# Tab 2 : 환경 데이터
# ======================================================
with tab2:
    st.subheader("학교별 환경 데이터 비교")

    env_all = pd.concat(env_data.values())

    avg_env = env_all.groupby("학교").mean(numeric_only=True).reset_index()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("평균 온도", "평균 습도", "평균 pH", "목표 EC vs 실측 EC")
    )

    fig.add_bar(x=avg_env["학교"], y=avg_env["temperature"], row=1, col=1)
    fig.add_bar(x=avg_env["학교"], y=avg_env["humidity"], row=1, col=2)
    fig.add_bar(x=avg_env["학교"], y=avg_env["ph"], row=2, col=1)

    fig.add_bar(x=avg_env["학교"], y=avg_env["ec"], name="실측 EC", row=2, col=2)
    fig.add_bar(
        x=list(SCHOOL_EC.keys()),
        y=list(SCHOOL_EC.values()),
        name="목표 EC",
        row=2, col=2
    )

    fig.update_layout(
        height=700,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    if selected_school != "전체":
        df = env_data[selected_school]

        fig_ts = go.Figure()
        fig_ts.add_scatter(x=df["time"], y=df["temperature"], name="온도")
        fig_ts.add_scatter(x=df["time"], y=df["humidity"], name="습도")
        fig_ts.add_scatter(x=df["time"], y=df["ec"], name="EC")
        fig_ts.add_hline(
            y=SCHOOL_EC[selected_school],
            line_dash="dash",
            annotation_text="목표 EC"
        )

        fig_ts.update_layout(
            title=f"{selected_school} 시계열 변화",
            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
        )

        st.plotly_chart(fig_ts, use_container_width=True)

    with st.expander("📄 환경 데이터 원본"):
        st.dataframe(env_all, use_container_width=True)

        csv_buffer = io.BytesIO()
        env_all.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        st.download_button(
            "CSV 다운로드",
            data=csv_buffer,
            file_name="환경데이터_전체.csv",
            mime="text/csv"
        )

# ======================================================
# Tab 3 : 생육 결과
# ======================================================
with tab3:
    growth_all = pd.concat(growth_data.values())

    avg_weight = growth_all.groupby("EC")["생중량(g)"].mean().reset_index()
    best_ec = avg_weight.loc[avg_weight["생중량(g)"].idxmax(), "EC"]

    st.metric("🥇 평균 생중량 최고 EC", f"{best_ec}")

    fig_bar = px.bar(
        avg_weight,
        x="EC",
        y="생중량(g)",
        title="EC별 평균 생중량",
        text_auto=".2f"
    )

    fig_bar.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    fig_box = px.box(
        growth_all,
        x="학교",
        y="생중량(g)",
        color="학교",
        title="학교별 생중량 분포"
    )

    fig_box.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
    )

    st.plotly_chart(fig_box, use_container_width=True)

    fig_scatter1 = px.scatter(
        growth_all,
        x="잎 수(장)",
        y="생중량(g)",
        color="학교",
        title="잎 수 vs 생중량"
    )

    fig_scatter2 = px.scatter(
        growth_all,
        x="지상부 길이(mm)",
        y="생중량(g)",
        color="학교",
        title="지상부 길이 vs 생중량"
    )

    st.plotly_chart(fig_scatter1, use_container_width=True)
    st.plotly_chart(fig_scatter2, use_container_width=True)

    with st.expander("📄 생육 데이터 원본"):
        st.dataframe(growth_all, use_container_width=True)

        buffer = io.BytesIO()
        growth_all.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)

        st.download_button(
            "XLSX 다운로드",
            data=buffer,
            file_name="생육결과_전체.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
