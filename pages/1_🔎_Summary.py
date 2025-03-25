import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import time  
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import folium
import plotly.graph_objs as go


# 메인 페이지 너비 넓게 (가장 처음에 설정해야 함)
st.set_page_config(layout="wide") 

with st.spinner("잠시만 기다려 주세요..."):
    time.sleep(1)  # 대기 시간 시뮬레이션
st.success("Data Loaded!")

# 한글 및 마이너스 깨짐
from matplotlib import rc
plt.rcParams['font.family'] = "Malgun Gothic"
plt.rcParams['axes.unicode_minus'] = False

# CSV 파일 경로 설정
CSV_FILE_PATH = 'https://raw.githubusercontent.com/jjjjunn/YH_project/refs/heads/main/'

off_csv = 'recycling_off.csv'
on_csv = 'recycling_online.csv'

off_df = pd.read_csv(CSV_FILE_PATH + off_csv, encoding="UTF8")
on_df = pd.read_csv(CSV_FILE_PATH + on_csv, encoding="UTF8")

# 지역 리스트 출력
# cities = off_df['지역'].unique().tolist()
# st.write(cities)

# 오프라인 전체 데이터
# st.dataframe(off_df, use_container_width=True)

# 각 지역에 대한 위도와 경도 정보를 포함한 사전 생성
coordinates = {
    "인천": (37.4563, 126.7052),
    "강원": (37.8228, 128.1555),
    "충북": (36.6351, 127.4915),
    "경기": (37.4138, 127.5183),
    "울산": (35.5373, 129.3167),
    "제주": (33.4997, 126.5318),
    "전북": (35.7210, 127.1454),
    "대전": (36.3504, 127.3845),
    "대구": (35.8714, 128.6014),
    "서울": (37.5665, 126.9780),
    "충남": (36.6887, 126.7732),
    "경남": (35.2345, 128.6880),
    "세종": (36.4805, 127.2898),
    "경북": (36.1002, 128.6295),
    "부산": (35.1796, 129.0756),
    "광주": (35.1595, 126.8526),
    "전남": (34.7802, 126.1322)
}

off_data_by_city = off_df.groupby('지역').agg({'방문자수':'sum', '참여자수':'sum'}).reset_index()
off_data_by_city = off_data_by_city.dropna(subset=['방문자수', '참여자수'])  # NaN 제거
# 참여율 계산
off_data_by_city['참여율'] = off_data_by_city.apply(
    lambda row: (row['참여자수'] / row['방문자수'] * 100) if row['방문자수'] > 0 else 0,
    axis=1
)

# 위도와 경도 열 추가
off_data_by_city['위도'] = off_data_by_city['지역'].map(lambda x: coordinates[x][0] if x in coordinates else None)
off_data_by_city['경도'] = off_data_by_city['지역'].map(lambda x: coordinates[x][1] if x in coordinates else None)

tab1, tab2 = st.tabs(['오프라인', '온라인'])
with tab1:
    st.markdown('**:rainbow[지역별 방문자수 데이터]**')
    st.dataframe(off_data_by_city, use_container_width=True) # 오프라인 지역별 데이터

    # 비어있는 데이터가 없는지를 확인
    valid_data = off_data_by_city.dropna(subset=['위도', '경도'])

    # Map 추가
    my_map = folium.Map(
        location=[valid_data['위도'].mean(), valid_data['경도'].mean()],
        zoom_start=7
    )

    # 지도 커스텀 - 원형 마커와 값 추가
    for index, row in valid_data.iterrows():
        folium.CircleMarker(
            location=[row['위도'], row['경도']],
            radius=row['참여율'],  # 반지름은 참여율에서 가져옴
            color='#3186cc',
            fill_color='#3186cc',
            fill_opacity=0.6  # 적절한 투명도로 설정
        ).add_to(my_map)
        
        # 마커 추가
        folium.Marker(
            location=[row['위도'], row['경도']],
            icon=folium.DivIcon(
                html=f"<div style='font-size: 11pt'><b>{row['지역']} {(row['참여율']):.1f}%</b></div>"  # 값 표시 방식
            )
        ).add_to(my_map) 

    # Streamlit으로 지도 표시
    st.caption("### 지역별 참여율 데이터")
    st.components.v1.html(my_map._repr_html_(), height=600)  # Streamlit에서 지도를 표시


with tab2:
    st.title('**:rainbow[온라인 마케팅 데이터]**')
    on_by_route = on_df.groupby('유입경로').agg({'노출수':'sum', '유입수':'sum', '체류시간(min)':'sum', '페이지뷰':'sum', '이탈수':'sum',
                                             '회원가입':'sum', '앱 다운':'sum', '구독':'sum'}).reset_index()
    on_by_route = on_by_route.dropna(subset=['노출수', '유입수', '체류시간(min)', '페이지뷰', '이탈수', '회원가입', '앱 다운', '구독'])  # NaN 제거
    on_by_route_ex = on_by_route[on_by_route['유입경로']!='키워드 검색'] # 키워드 검색 제외
    st.dataframe(on_by_route, use_container_width=True)

    # 산점도 생성을 위한 Plotly 시각화
    fig = go.Figure()

    # 산점도 추가
    fig.add_trace(go.Scatter(
        x=on_by_route_ex['유입수'],
        y=on_by_route_ex['유입경로'],
        mode='markers+text',
        name='유입수 데이터',
        text=on_by_route_ex['유입수'],
        textposition='top center',  # 텍스트 표시 위치
        marker=dict(color='#d00000', size=10)
    ))

    # 레이아웃 설정
    fig.update_layout(
        title='유입경로별 유입수 Scatter Plot',
        xaxis_title='유입수',
        yaxis_title='유입경로',
        boxmode='group',  # 그룹화된 박스 플롯
        height=600,
        showlegend=True
    )

    # 결과 출력
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

    # 키워드별 전환수
    act_by_keyword = on_df[on_df['유입경로'] =='키워드 검색']
    act_by_keyword = act_by_keyword.groupby('키워드').agg({'노출수':'sum', '유입수':'sum', '체류시간(min)':'sum', '페이지뷰':'sum', '이탈수':'sum',
                                             '회원가입':'sum', '앱 다운':'sum', '구독':'sum', '전환수':'sum'}).reset_index()
    act_by_keyword = act_by_keyword.dropna(subset=['노출수', '유입수', '체류시간(min)', '페이지뷰', '이탈수', '회원가입', '앱 다운', '구독', '전환수'])  # NaN 제거

    # 바 레이스 차트 생성
    fig = px.bar(act_by_keyword,
                x='전환수',
                y='키워드',
                range_x=[0, act_by_keyword['전환수'].max() + 100],  # x축 범위 설정
                title='전환수 바 레이스 차트',
                orientation='h',  # 수평 바 차트
                labels={'키워드': '키워드', '전환수': '전환수'},
                color='전환수',  # 색상 기준 설정
                template='plotly_white'  # 템플릿 설정
)

    # 결과 출력
    st.plotly_chart(fig, use_container_width=True)

