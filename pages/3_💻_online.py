import streamlit as st
import pandas as pd
import mysql.connector
from mysql.connector import Error
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import time  
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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
CSV_FILE_PATH = 'https://raw.githubusercontent.com/jjjjunn/YH_project/refs/heads/main/recycling_online.csv'

# Streamlit emoji: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/

# 테이블 이름 설정
DB_TABLE = 'recycle_on'

# MySQL 연결 정보 (secrets.toml에서 가져옴)
def get_db_connection():
    try:
        secrets = st.secrets["mysql"]
        connection = mysql.connector.connect(
            host=secrets["host"],
            database=secrets["database"],
            user=secrets["user"],
            password=secrets["password"]
        )
        if connection.is_connected():
            st.write()
        return connection
    except Error as e:
        st.error(f"데이터베이스 연결 실패: {e}")
        return None

# 프로그램에서 테이블 생성
@st.cache_data
def create_table(_connection):  # 인자 이름에 "_" 추가
    try:
        cursor = _connection.cursor()
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {DB_TABLE} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            날짜 DATE NULL,
            디바이스 VARCHAR(20) NULL,
            유입경로 VARCHAR(20) NULL,
            키워드 VARCHAR(20) NULL,
            노출수 INT NULL,
            유입수 INT NULL,
            `유입률(%)` FLOAT NULL,
            `체류시간(min)` INT NULL,
            `평균체류시간(min)` FLOAT NULL,
            페이지뷰 INT NULL,
            평균페이지뷰 FLOAT NULL,
            이탈수 INT NULL,
            `이탈률(%)` FLOAT NULL,
            회원가입 INT NULL,
            `전환율(가입)` FLOAT NULL,
            `앱 다운` INT NULL,
            `전환율(앱)` FLOAT NULL,
            구독 INT NULL,
            `전환율(구독)` FLOAT NULL,
            전환수 INT NULL,
            `전환율(%)` FLOAT NULL
        );
        """
        cursor.execute(create_table_query)
        _connection.commit()
        print(f"'{DB_TABLE}' 테이블이 성공적으로 생성되었습니다.")
    except Error as e:
        print(f"테이블 생성 중 오류 발생: {e}")
    finally:
        cursor.close()

# 데이터 삽입 함수
@st.cache_data
def insert_data(_connection, df):  # 인자 이름에 "_" 추가
    try:
        cursor = _connection.cursor()
        insert_query = f"""
        INSERT INTO `{DB_TABLE}` (
            날짜, 디바이스, 유입경로, 키워드, 노출수, 유입수, `유입률(%)`, 
            `체류시간(min)`, `평균체류시간(min)`, 페이지뷰, 평균페이지뷰, 이탈수, `이탈률(%)`, 회원가입, `전환율(가입)`,
            `앱 다운`, `전환율(앱)`, 구독, `전환율(구독)`, 전환수, `전환율(%)`
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        data = [tuple(row) for row in df.values]
        cursor.executemany(insert_query, data)
        _connection.commit()
        print(f"{cursor.rowcount}개의 레코드가 '{DB_TABLE}' 테이블에 삽입되었습니다.")
    except Error as e:
        print(f"데이터 삽입 중 오류 발생: {e}")
    finally:
        cursor.close()

# 데이터 가져오기 (캐싱 적용)
@st.cache_data
def get_data():
    connection = get_db_connection()
    if connection is None:
        return pd.DataFrame()
    
    try:
        query = f"SELECT * FROM {DB_TABLE};"
        df = pd.read_sql(query, connection)
        # 열 이름 조정
        df = df.rename(columns={
            "날짜": "DATE",
            "디바이스": "Device",
            "유입경로": "Route",
            "키워드": "KeyWord",
            "노출수": "Exposure",
            "유입수": "Inflow",
            "유입률(%)": "In_rate",
            "체류시간(min)": "Stay_min",
            "평균체류시간(min)": "M_Stay_min",
            "페이지뷰": "P_view",
            "평균페이지뷰": "M_P_view",
            "이탈수": "Exit",
            "이탈률(%)": "Exit_R",
            "회원가입": "Join",
            "전환율(가입)": "Join_R",
            "앱 다운": "Down",
            "전환율(앱)" : "Down_R",
            "구독": "Scribe",
            "전환율(구독)": "Scr_R",
            "전환수": "Action",
            "전환율(%)": "Act_R",
       })
        df['DATE'] = pd.to_datetime(df['DATE'])  # DATE 컬럼을 datetime 형식으로 변환
        df['WEEKDAY'] = df['DATE'].dt.day_of_week # 요일 정보 추가하여 데이터프레임에 추가
        # 요일의 인덱스와 매칭하기 위한 딕셔너리
        week_mapping = {
            0:'월',
            1:'화',
            2:'수',
            3:'목',
            4:'금',
            5:'토',
            6:'일'
        }
        # WEEKDAY 컬럼의 숫자를 한글 요일로 변환
        df['WEEKDAY'] = df['WEEKDAY'].map(week_mapping)
        return df
    except Error as e:
        print(f"데이터 조회 중 오류 발생: {e}")
        return pd.DataFrame()
    finally:
        if connection.is_connected():
            connection.close()

# 메인 함수
def main():
    
    # CSV 파일 읽기
    try:
        df = pd.read_csv(CSV_FILE_PATH, encoding="UTF8").fillna(0) # 결측값 0으로 대체
        df.replace([np.inf, -np.inf], np.nan, inplace=True) # 무한대는 Nan으로 대체
        df.fillna(0, inplace=True) # Nan 값 0으로 대체
        print("CSV 파일을 성공적으로 읽었습니다.")
    except FileNotFoundError:
        st.error(f"파일 '{CSV_FILE_PATH}'을 찾을 수 없습니다.")
        return
    except Exception as e:
        st.error(f"CSV 파일 읽기 중 오류 발생: {e}")
        return

    # 데이터베이스 연결
    connection = get_db_connection()
    if connection is None:
        return

    # 테이블 생성
    create_table(connection)  # 호출 시 변경된 인자 사용

    # 데이터 삽입
    insert_data(connection, df)  # 호출 시 변경된 인자 사용

    # 연결 종료
    if connection.is_connected():
        connection.close()

# 메인 함수 실행
if __name__ == "__main__":
    main()


# 필터링된 데이터 캐싱
@st.cache_data
def filter_data(df, start_date, end_date, selected_day, selected_dv):
    # 날짜 필터링
    df_filtered = df[(df['DATE'] >= start_date) & (df['DATE'] <= end_date)]
    # 필터링 후 NaT 값 확인
    if df_filtered['DATE'].isnull().any():
        st.warning("필터링된 후 데이터에 NaT 값이 포함되어 있습니다.")
        df_filtered = df_filtered.dropna(subset=['DATE'])   

    # 요일 필터링: All이 선택되었거나 selected_day가 비어 있을 시 필터링 하지 않음
    if selected_day and 'All' not in selected_day:
        df_filtered = df_filtered[df_filtered['WEEKDAY'].isin(selected_day)]
    
    # 디바이스 필터링: 'All_DV'가 선택되면 모든 디바이스를 반환
    if 'All_DV' not in selected_dv:
        df_filtered = df_filtered[df_filtered['Device'].isin(selected_dv)]
    
    return df_filtered

# Streamlit UI
st.markdown("# :recycle: :rainbow[재활용 온라인 마케팅 성과 지표] :recycle:")

# 데이터에서 날짜 정보 추출
df = get_data() # 데이터프레임 가져오기

df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')  # FORMAT 변경
period_q1 = df['DATE'].quantile(0.25) # 전체 기간의 1사분위 지점
period_q3 = df['DATE'].quantile(0.75) # 전체 기간의 3사분위 지점
start_date = df['DATE'].min() # 데이터의 시작 일자
end_date = df['DATE'].max() # 데이터의 종료 일자

# 사이드바: 화면 왼쪽의 영역을 나누어 사용
st.sidebar.header('데이터 조회 옵션 선택')

# 1. 날짜 조회 슬라이더
st.sidebar.write(":calendar: **기간 조회** :calendar:")
start_date_input = st.sidebar.date_input(
    "시작날짜",
    value=period_q1,
    min_value=start_date,  # 데이터의 최소 날짜
    max_value=end_date   # 데이터의 최대 날짜
)
end_date_input = st.sidebar.date_input(
    "종료날짜",
    value=period_q3,
    min_value=start_date,  # 데이터의 최소 날짜
    max_value=end_date   # 데이터의 최대 날짜
)

# 입력된 날짜를 datetime 타입으로 변환
start_date_input = pd.to_datetime(start_date_input)
end_date_input = pd.to_datetime(end_date_input)

# 요일 조회 레이아웃 추가
wdays_options = ['All'] + df['WEEKDAY'].unique().tolist()  # 전체 선택 및 요일의 유니크 값

selected_day_w = st.sidebar.multiselect(
    ('**:blue[요일을 선택해 주세요]**'),
    options=wdays_options,
    default=['All'],  # 기본값: 전체
    placeholder='요일 선택'
)

# 디아비스 조회 레이아웃 추가
dv_options = ['All_DV'] + df['Device'].unique().tolist() # 전체 선택 및 디바이스 유니크 값
selected_dv = st.sidebar.multiselect(
    ('**:blue[디바이스 선택]**'),
    options=dv_options,
    default=['All_DV'], # 기본값: 전체
    placeholder='디바이스 선택'
)

# 데이터 필터링
df_select = filter_data(df, start_date_input, end_date_input, selected_day_w, selected_dv)
# 특정열만 출력
columns_to_display = ['DATE', 'WEEKDAY', 'Device', 'Route', 'KeyWord', 'Exposure', 'Inflow', 'In_rate', 'Stay_min', 'M_Stay_min', 'P_view', 'M_P_view', 'Exit', 'Exit_R', 'Join', 'Join_R', 
                    'Down', 'Down_R', 'Scribe', 'Scr_R', 'Action', 'Act_R']
# 선택열만 출력
filtered_selected_df = df_select[columns_to_display]
        
# 탭 생성하기
tab1, tab2, tab3 = st.tabs(['데이터(지표)', '분석', '예측'])

with tab1: # 데이터프레임, 지표
    # 데이터 조회 버튼
    if st.sidebar.button("데이터 조회"):
        if not filtered_selected_df.empty:
            st.dataframe(filtered_selected_df, use_container_width=True)  # 필터링된 데이터프레임 출력

            # (핵심 지표) 계산
            total_exposure = int(filtered_selected_df['Exposure'].sum())
            total_in = int(filtered_selected_df['Inflow'].sum())
            in_ratio = (total_in / total_exposure * 100) if total_exposure > 0 else 0
            total_stay = int(filtered_selected_df['Stay_min'].sum())
            total_exit = int(filtered_selected_df['Exit'].sum())
            exit_ratio = (total_exit / total_in * 100) if total_in > 0 else 0
            total_act = int(filtered_selected_df['Action'].sum())
            act_ratio = (total_act / total_in * 100) if total_in > 0 else 0

            st.markdown('#####')

            # KPI 표시
            first_column, second_column, third_column, fourth_column, fifth_column, sixth_column, seventh_column, eighth_column = st.columns(8)
            with first_column:
                st.subheader("노출수:")
                st.subheader(f"{(total_exposure):,.0f}")
            with second_column:
                st.subheader("유입수:")
                st.subheader(f"{(total_in):,.0f}")
            with third_column:
                st.subheader("유입비율:")
                st.subheader(f"{(in_ratio):.1f}%")
            with fourth_column:
                st.subheader("체류시간(min):")
                st.subheader(f"{(total_stay):,.0f}")
            with fifth_column:
                st.subheader("이탈수:")
                st.subheader(f"{(total_exit):,.0f}")      
            with sixth_column:
                st.subheader("이탈률(%):")
                st.subheader(f"{(exit_ratio):.1f}%")      
            with seventh_column:
                st.subheader("전환수:")
                st.subheader(f"{(total_act):,.0f}")      
            with eighth_column:
                st.subheader("전환율(%):")
                st.subheader(f"{(act_ratio):.1f}%")     

            st.divider()

with tab2: # 데이터 시각화
    # 'DATE'를 월 단위로 그룹화하고 각 디바이스별 유입수, 전환수 합산
    # 차트를 생성하는 함수
    def create_monthly_bar_chart(data, value_col, title):
        # 월별로 그룹화하여 합계 계산
        monthly_data = data.groupby(['Device', pd.Grouper(key='DATE', freq='M')])[value_col].sum().reset_index()
        monthly_data = monthly_data.dropna(subset=[value_col])  # NaN 제거

        # Plotly로 라인차트 생성
        fig = px.line(
            monthly_data,
            x="DATE",
            y=value_col,
            orientation="v",
            title=f"<b>{title}</b>",
            color = "Device",
            template="plotly_white",
        )
        
        # 레이아웃 업데이트
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False)
        )
        return fig

    # 차트 생성
    left_column, right_column = st.columns(2)

    with left_column:
        fig_month_in = create_monthly_bar_chart(filtered_selected_df, 'Inflow', '디바이스별 월간 유입수')
        st.plotly_chart(fig_month_in, use_container_width=True)

    with right_column:
        fig_month_act = create_monthly_bar_chart(filtered_selected_df, 'Action', '디바이스별 월간 전환수')
        st.plotly_chart(fig_month_act, use_container_width=True)

    st.divider()

    # 'DATE'를 월 단위로 그룹화하고 각 유입경로별 유입수, 전환수 합산
    # 차트를 생성하는 함수
    def create_monthly_bar_chart(data, value_col, title):
        # 월별로 그룹화하여 합계 계산
        monthly_data = data.groupby(['Route', pd.Grouper(key='DATE', freq='M')])[value_col].sum().reset_index()
        monthly_data = monthly_data.dropna(subset=[value_col])  # NaN 제거

        # Plotly로 라인차트 생성
        fig = px.line(
            monthly_data,
            x="DATE",
            y=value_col,
            orientation="v",
            title=f"<b>{title}</b>",
            color = "Route",
            template="plotly_white",
        )
        
        # 레이아웃 업데이트
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False)
        )
        return fig

    # '키워드 검색' 경로 제외 (비중 많음)
    filtered_selected_df_excluded = filtered_selected_df[filtered_selected_df['Route'] != '키워드 검색']

    # 차트 생성
    left_column, right_column = st.columns(2)

    with left_column:
        fig_month_in = create_monthly_bar_chart(filtered_selected_df_excluded, 'Inflow', '유입경로별 월간 유입수(키워드검색 제외)')
        st.plotly_chart(fig_month_in, use_container_width=True)

    with right_column:
        fig_month_act = create_monthly_bar_chart(filtered_selected_df_excluded, 'Action', '유입경로별 월간 전환수(키워드검색 제외)')
        st.plotly_chart(fig_month_act, use_container_width=True)

    st.divider()

    # 캠페인별 비율 파이차트: 파이차트 생성 함수
    def create_pie_chart(data, value_col, label_col, title):
        summary = data.groupby([label_col])[value_col].sum().reset_index()
        labels = summary[label_col].unique()
        explode = [0.1 for _ in range(len(labels))]  # 모든 섹터를 약간 부풀리기

        plt.figure(figsize=(4, 4))
        plt.pie(
            summary[value_col],
            labels=labels,
            autopct='%.1f%%',
            startangle=260,
            colors = sns.color_palette("Set2", len(labels)),
            explode=explode,
            shadow=True
        )
        plt.title(title)
        plt.axis('equal')  # 원형 비율을 유지하기 위해 설정

    # '키워드 검색' 경로만 포함
    filtered_selected_df_keyword = filtered_selected_df[filtered_selected_df['Route'] == '키워드 검색']

    # 차트 생성
    left_column, right_column = st.columns(2)

    with left_column:
        create_pie_chart(filtered_selected_df_keyword, 'Inflow', 'KeyWord', '키워드별 유입비중')
        st.pyplot(plt.gcf())  # 현재의 plt Figure를 Streamlit에 표시

    with right_column:
        create_pie_chart(filtered_selected_df_keyword, 'Action', 'KeyWord', '키워드별 전환비중')
        st.pyplot(plt.gcf())  # 현재의 plt Figure를 Streamlit에 표시

    st.divider()

    # 월별 각 전환수 누적 막대그래프
    def create_cumulative_bar_chart(data, value_col, title):
        # Join, Down, Scribe에 대해 각각 월별 합계 계산 후 병합
        join_data = data.groupby(['Join', pd.Grouper(key='DATE', freq='M')])[value_col].sum().reset_index()
        join_data['Type'] = '회원가입'  # 회원가입 타입 레이블 추가

        down_data = data.groupby(['Down', pd.Grouper(key='DATE', freq='M')])[value_col].sum().reset_index()
        down_data['Type'] = '앱 다운'  # 앱 다운 타입 레이블 추가

        scribe_data = data.groupby(['Scribe', pd.Grouper(key='DATE', freq='M')])[value_col].sum().reset_index()
        scribe_data['Type'] = '구독'  # 구독 타입 레이블 추가

        # 데이터 결합
        combined_data = pd.concat([join_data, down_data, scribe_data], ignore_index=True)

        # 결합된 데이터에서 NaN 제거
        combined_data = combined_data.dropna(subset=[value_col])  

            # 누적 막대그래프 생성
        fig = px.bar(
            combined_data,
            x="DATE",
            y=value_col,
            color="Type",
            title=f"<b>{title}</b>",
            text=value_col,
            template="plotly_white",
            orientation='v'
        )
        
        # 누적 설정
        fig.update_traces(texttemplate='%{text:.2f}', textposition='inside')
        fig.update_layout(barmode='stack')  # 누적 모드 설정

        # 레이아웃 업데이트
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False)
        )
        
        return fig

    # 차트 생성
    fig_cumulative_bar = create_cumulative_bar_chart(filtered_selected_df, 'Action', '월간 회원가입, 앱 다운, 구독 수 누적 막대그래프')
    st.plotly_chart(fig_cumulative_bar, use_container_width=True)

    st.divider()

    # 수치형 데이터의 상관관계 히트맵
    numeric_df = filtered_selected_df.select_dtypes(include=[float, int]) # 숫자형 열만 선택
    corr_df = numeric_df.corr()

    # 히트맵 생성
    plt.figure(figsize=(8,6))
    sns.heatmap(corr_df, cmap='flare', annot=True, fmt='.1f', cbar=True)
    plt.title("상관관계 히트맵")
    # Streamlit 표시
    st.pyplot(plt)
    plt.clf()

with tab3: # 머신러닝 모델 구현
    # 1. 일별 데이터 집계
    daily_data = filtered_selected_df.groupby('DATE').agg({'Inflow':'sum', 'Action':'sum'}).reset_index()

    if daily_data.empty:
        st.warning("조회된 데이터가 없습니다.")
    else:
        # 피처와 변수 정의
        X = daily_data[['Inflow']]
        y = daily_data[['Action']]

        # 데이터 분리
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 회귀 모델 훈련
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 예측 수행
        daily_data['PREDICTED_Action'] = model.predict(X)  # 전체 데이터에 대한 예측

        # 성능 평가
        mse = mean_squared_error(y_test, model.predict(X_test))
        # MSE 계산 후 RMSE 계산
        rmse = np.sqrt(mse)
        st.write(f"**:red[회귀 모델 RMSE: {rmse:.2f}]**")
        st.write(f"모델이 예측한 전환수와 실제 수치 사이의 평균차가 **:blue[{rmse:.2f}]** 입니다.")


        # 기간별 실제 및 예측 전환수 시각화
        fig = px.line(daily_data, x='DATE', y=['Action', 'PREDICTED_Action'], 
                    labels={'value': '참여자 수'},
                    title='기간별 실제 전환수와 예측 전환수 비교',
                    markers=True)
        st.plotly_chart(fig)

