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
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


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

memeber_df = pd.read_csv(CSV_FILE_PATH + 'members_data.csv')

# Streamlit emoji: https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/

print_df = memeber_df.rename(columns={
     "age": "나이",
     "gender": "성별",
     "marriage": "혼인여부",
     "city": "도시",
     "channel": "가입경로",
     "before_ev": "참여_전",
     "part_ev": "참여이벤트",
     "after_ev": "참여_후"
})

# 데이터값 변경
print_df['성별'] = print_df['성별'].map({0:'남자', 1:'여자'})
print_df['혼인여부'] = print_df['혼인여부'].map({0:'미혼', 1:'기혼'})
print_df['도시'] = print_df['도시'].map({0:'부산', 1:'대구', 2:'인천', 3:'대전', 4:'울산', 5:'광주', 6:'서울', 
    7:'경기', 8:'강원', 9:'충북', 10:'충남', 11:'전북', 12:'전남', 13:'경북', 14:'경남', 15:'세종', 16:'제주'})
print_df['가입경로'] = print_df['가입경로'].map({0:"직접 유입", 1:"키워드 검색", 2:"블로그", 3:"카페", 4:"이메일", 
        5:"카카오톡", 6:"메타", 7:"인스타그램", 8:"유튜브", 9:"배너 광고", 10:"트위터 X", 11:"기타 SNS"})
print_df['참여_전'] = print_df['참여_전'].map({0:'가입', 1:'미가입'})
print_df['참여이벤트'] = print_df['참여이벤트'].map({0:"워크숍 개최", 1:"재활용 품목 수집 이벤트", 2:"재활용 아트 전시",
          3:"게임 및 퀴즈", 4:"커뮤니티 청소 활동", 5:"업사이클링 마켓", 6:"홍보 부스 운영"})
print_df['참여_후'] = print_df['참여_후'].map({0:'가입', 1:'미가입'})

with st.expander('회원 데이터'):
    st.dataframe(print_df, use_container_width=True)

data = memeber_df[['age', 'gender', 'marriage', 'after_ev']]

tab1, tab2, tab3 = st.tabs(['서비스가입 예측', '추천 캠페인', '추천 채널'])

with tab1: # 서비스 가입 예측 모델
    first_column, second_column, thrid_columns = st.columns([6, 2, 2])
    with first_column:
        st.write("서비스가입 예측 모델입니다. 아래의 조건을 선택해 주세요.")
        ages_1 = st.slider(
            "연령대를 선택해 주세요.",
            25, 65, (35, 45)
        )
        st.write(f"**선택 연령대: :red[{ages_1}]세**")
    
    with second_column:
        gender_1 = st.radio(
            "성별을 선택해 주세요.",
            ["남자", "여자"],
            index=1
        )
    
    with thrid_columns:
        marriage_1 = st.radio(
            "혼인여부를 선택해 주세요.",
            ["미혼", "기혼"],
            index=1
        )
    
    # 예측 모델 학습 및 평가 함수
    def service_predict(data):
        # 데이터 전처리 및 파이프라인 설정
        numeric_features = ['age']
        categorical_features = ['gender', 'marriage']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(categories='auto'), categorical_features)
            ]
        )

        # 랜덤 포레스트 모델
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))  # n_jobs=-1로 모든 코어 사용
        ])

        # 데이터 분할
        X = data.drop(columns=['after_ev'])
        y = data['after_ev']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 하이퍼파라미터 튜닝을 위한 그리드 서치
        param_grid = {
            'classifier__n_estimators': [100, 200],  # 늘리면 성능 향상 가능
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5]
        }

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        # 최적의 모델로 예측 수행
        y_pred = grid_search.predict(X_test)

        # 성능 평가
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"이 모델의 테스트 정확도는 {accuracy * 100:.1f}% 입니다.")

        return grid_search.best_estimator_, grid_search.best_estimator_.named_steps['classifier'].feature_importances_

    # 사용자가 입력한 값을 새로운 데이터로 변환
    def pre_result(model, new_data):
        prediction = model.predict(new_data)
        st.write(f"**모델 예측 결과: :rainbow[{'가입' if prediction[0] == 0 else '미가입'}]**")

    # 특성 중요도 시각화
    def plot_feature_importance(importances, feature_names):
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(2, 1))
        plt.title("특성 중요도")
        plt.barh(range(len(importances)), importances[indices], align="center")
        plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
        plt.xlabel("중요도")
        st.pyplot(plt)

    # 예측하기 버튼 클릭에 따른 동작
    if st.button("예측하기"):
        # 기존 데이터로 모델 학습
        model, feature_importances = service_predict(data)

        # 입력된 값을 새로운 데이터 형식으로 변환
        new_data = pd.DataFrame({
            'age': [(ages_1[0] + ages_1[1]) / 2],  # 나이의 중앙값
            'gender': [1 if gender_1 == '여자' else 0],  # 성별 인코딩
            'marriage': [1 if marriage_1 == '기혼' else 0]  # 혼인 여부 인코딩
        })

        # 예측 수행
        pre_result(model, new_data)

        # 특성 중요도 시각화
        feature_names = ['age'] + list(model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out())
        plot_feature_importance(feature_importances, feature_names)

data_2 = memeber_df[['age', 'gender', 'marriage', 'part_ev', 'after_ev']]

# 참여 이벤트 매핑
event_mapping = {
    0: "워크숍 개최",
    1: "재활용 품목 수집 이벤트",
    2: "재활용 아트 전시",
    3: "게임 및 퀴즈",
    4: "커뮤니티 청소 활동",
    5: "업사이클링 마켓",
    6: "홍보 부스 운영"
}

with tab2: # 캠페인 추천 모델
    first_column, second_column, thrid_columns = st.columns([6, 2, 2])
    with first_column:
        st.write("캠페인 추천 모델입니다. 아래의 조건을 선택해 주세요.")
        ages_2 = st.slider(
            "연령대를 선택해 주세요.",
            25, 65, (35, 45),
            key='slider_2'
        )
        st.write(f"**선택 연령대: :red[{ages_2}]세**")
    
    with second_column:
        gender_2 = st.radio(
            "성별을 선택해 주세요.",
            ["남자", "여자"],
            index=1,
            key='radio2_1'
        )
    
    with thrid_columns:
        marriage_2 = st.radio(
            "혼인여부를 선택해 주세요.",
            ["미혼", "기혼"],
            index=1,
            key='radio2_2'
        )

    # 추천 모델 함수
    def recommend_event(data_2):
        # X, y 설정
        X = data_2[['age', 'gender', 'marriage', 'part_ev']]
        y = data_2['after_ev']

        # 더미 변수 생성하여 참여 이벤트 인코딩
        X = pd.get_dummies(X, columns=['part_ev'], drop_first=True)

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)

        # 렌덤 포레스트 모델 정의 및 학습
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        return model, X_train.columns  # 모델과 피쳐 이름을 반환

    # 사용자 정보 입력을 통한 추천 이벤트 평가
    if st.button("효과적인 이벤트 추천받기"):
        # 추천 모델 훈련
        model, feature_names = recommend_event(data_2)

        event_results = {}

        # 각 이벤트에 대한 추천 가능성 평가
        for event in range(7):  # part_ev가 0에서 6까지의 숫자이므로, 7개의 이벤트에 대해 반복
            # 새로운 사용자 정보 세팅
            new_user_data = pd.DataFrame({
                'age': [(ages_2[0] + ages_2[1]) / 2],  # 연령대의 중앙값
                'gender': [1 if gender_2 == '여자' else 0],  # 성별 인코딩
                'marriage': [1 if marriage_2 == '기혼' else 0],  # 혼인 여부 인코딩
                'part_ev': [event]  # 번호로 매핑된 이벤트
            })

            # 더미 변수 생성
            new_user_data = pd.get_dummies(new_user_data, columns=['part_ev'], drop_first=True)

            # 피쳐 정렬
            new_user_data = new_user_data.reindex(columns=feature_names, fill_value=0)

            # 예측 수행
            prediction = model.predict(new_user_data)
            event_results[event] = prediction[0]  # 가입 여부 저장 (0: 가입, 1: 미가입)

        # 가입(0) 가능성이 높은 이벤트 중 가장 높은 것
        possible_events = {event: result for event, result in event_results.items() if result == 0} 

        if possible_events:
            best_event = max(possible_events, key=possible_events.get)
            st.write(f"**추천 이벤트: :violet[{event_mapping[best_event]}] 👈 이벤트가 가장 효과적입니다!**")
        else:
            st.write("추천 이벤트: 가입 확률이 높지 않으므로 다른 캠페인을 고려해보세요.")

data_3 = memeber_df[['age', 'gender', 'marriage', 'channel', 'after_ev']]

# 가입 시 유입경로 매핑
register_channel = {
    0:"직접 유입",
    1:"키워드 검색",
    2:"블로그",
    3:"카페",
    4:"이메일",
    5:"카카오톡",
    6:"메타",
    7:"인스타그램",
    8:"유튜브", 
    9:"배너 광고", 
    10:"트위터 X", 
    11:"기타 SNS"
}

with tab3: # 마케팅 채널 추천 모델
    col1, col2, col3 = st.columns([6, 2, 2])
    with col1:
        st.write("마케팅 채널 추천 모델입니다. 아래의 조건을 선택해 주세요")
        ages_3 = st.slider(
            "연령대를 선택해 주세요.",
            25, 65, (35, 45),
            key='slider_3'
        )
        st.write(f"**선택 연령대: :red[{ages_3}]세**")
    
    with col2:
        gender_3 = st.radio(
            "성별을 선택해 주세요.",
            ["남자", "여자"],
            index=1,
            key='radio3_1'
        )
    
    with col3:
        marriage_3 = st.radio(
            "혼인여부를 선택해 주세요.",
            ["미혼", "기혼"],
            index=1,
            key='radio3_2'
        )

        # 추천 모델 함수
    def recommend_channel(data_3):
        # X, y 설정
        X = data_3[['age', 'gender', 'marriage', 'channel']]
        y = data_3['after_ev']

        # 더미 변수 생성하여 유입 채널 인코딩
        X = pd.get_dummies(X, columns=['channel'], drop_first=True)

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=42)

        # 렌덤 포레스트 모델 정의 및 학습
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        return model, X_train.columns  # 모델과 피쳐 이름을 반환

    # 사용자 정보 입력을 통한 추천 이벤트 평가
    if st.button("효과적인 마케팅 채널 추천받기"):
        # 추천 모델 훈련
        model, feature_names = recommend_channel(data_3)

        channel_results = {}

        # 각 이벤트에 대한 추천 가능성 평가
        for channel in range(12):  # part_ev가 0에서 12까지의 숫자이므로, 12개의 이벤트에 대해 반복
            if channel in (0,1): # 직접 유입과 키워드 검색 채널 제외
                continue

            # 새로운 사용자 정보 세팅
            new_user_data = pd.DataFrame({
                'age': [(ages_2[0] + ages_2[1]) / 2],  # 연령대의 중앙값
                'gender': [1 if gender_2 == '여자' else 0],  # 성별 인코딩
                'marriage': [1 if marriage_2 == '기혼' else 0],  # 혼인 여부 인코딩
                'channel': [channel]  # 번호로 매핑된 채널
            })

            # 더미 변수 생성
            new_user_data = pd.get_dummies(new_user_data, columns=['channel'], drop_first=True)

            # 피쳐 정렬
            new_user_data = new_user_data.reindex(columns=feature_names, fill_value=0)

            # 예측 수행
            prediction = model.predict(new_user_data)
            channel_results[channel] = prediction[0]  # 가입 여부 저장 (0: 가입, 1: 미가입)

        # 가입(0) 가능성이 높은 채널 중 가장 높은 것 3개
        possible_channels = {channel: result for channel, result in channel_results.items() if result == 0} 

        if possible_channels:
            best_channels = sorted(possible_channels.keys(), key=lambda x: possible_channels[x])[:3]  # 가장 좋은 3개 채널
            recommended_channels = [register_channel[ch] for ch in best_channels]
            st.write(f"**추천 마케팅 채널:** :violet[{', '.join(recommended_channels)}] 👈 이 채널들이 가장 효과적입니다!")
        else:
            st.write("추천 마케팅 채널: 가입 확률이 높지 않으므로 다른 채널을 고려해보세요.")