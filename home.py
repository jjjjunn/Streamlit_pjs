import streamlit as st

st.set_page_config(
    page_title="재활용 이벤트 성과 지표",
    page_icon= "📋"
)

st.title(":recycle: :rainbow[재활용 이벤트 성과 지표] :recycle:")
st.sidebar.success("페이지를 선택해 주세요.")

st.markdown(
    """
    재활용 관련 오프라인 캠페인과 온라인 마케팅을
    2023년 1월 1일부터 2024년 12월 31일까지 2년간 진행한 결과를
    집계 및 분석한 페이지 입니다.
    필요한 데이터를 해당 사이트로 이동하셔서 조회하시면 됩니다.
    **👈 이동하려는 페이지 선택하기**
    ### 어떤 데이터를 보고싶으세요?
    - (전체 기간 합산) 요약 데이터
    - 오프라인 캠페인 진행 관련 데이터
    - 온라인 마케팅 진행 관련 데이터 
"""
)

