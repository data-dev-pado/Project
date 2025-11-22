## 프로젝트 목표
- Netflix 콘텐츠 추가 전략 및 트렌드 분석
- 데이터 기반 콘텐츠 추천 시스템 구축
- 인터랙티브 대시보드 및 자동화된 분석 보고서 생성

## 데이터셋
- **출처**: [Kaggle - Netflix Shows Dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows)
- **규모**: 8,807개 콘텐츠 (전처리 후 8,709개)
- **기간**: 2008년 ~ 2021년
- **변수**: 제목, 타입, 감독, 출연진, 국가, 등급, 설명 등 12개

## 주요 알고리즘
- **TF-IDF Vectorizer**: 텍스트 벡터화 (max_features=1000)
- **Cosine Similarity**: 콘텐츠 간 유사도 계산
- **Linear Regression**: TV 프로그램 비율 예측
- **NetworkX Centrality**: Degree/Betweenness 중심성
- **Moving Average**: 3/6개월 이동평균

## 주요 기능

### 핵심 분석 (5가지)
- **콘텐츠 전략 분석**: 연도별 영화/TV 프로그램 추가 추세
- **글로벌 분포 분석**: 국가별 콘텐츠 수 및 다양성
- **네트워크 분석**: 배우-감독 협업 네트워크 (Degree/Betweenness Centrality)
- **유사도 분석**: TF-IDF 기반 콘텐츠 추천 시스템
- **성장 모델링**: 시계열 분석 및 미래 예측 (LinearRegression)

### 고급 분석 (3가지)
- **퍼널 분석**: 데이터 품질 단계별 전환율
- **코호트 분석**: 장르별 연도별 트렌드 변화
- **시계열 분석**: 월별 패턴 및 계절성 분석

## 프로젝트 구조
```
NETFLIX/ 
|── KPI_Netflix_Dashboard          # 종합 파일 
│
├── data_preprocessing.py          # 데이터 전처리 모듈
│   ├── load_and_preprocess_netflix_data()
│   ├── get_data_summary()
│   └── validate_data()
│
├── analysis_functions.py          # 핵심 분석 
│   ├── analyze_content_strategy_shift()
│   ├── analyze_global_content_distribution()
│   ├── create_actor_director_network()
│   ├── content_similarity_analysis()
│   └── growth_modeling_analysis()
│
├── advanced_analysis.py           # 고급 분석 
│   ├── funnel_analysis()
│   ├── cohort_analysis()
│   └── time_series_analysis()
│
├── dashboard_manager.py           # 대시보드 생성
│   ├── create_comprehensive_dashboard()
│   ├── save_charts_as_html()
│   ├── export_analysis_report()
│   └── interactive_recommendation_system()
│
├── main.py                        # 메인 실행 파일
│
├── netflix_titles.csv             # 원본 데이터
│
├── requirements.txt               # 패키지 의존성
| 
└── README.md                      # 프로젝트 문서
```


## 주요 인사이트

### 1. 콘텐츠 전략 변화
- **TV 프로그램 비율 증가**: 2015년 이후 지속적 증가 (2021년 33.7%)
- **향후 예측**: 2024년 38.8%까지 증가 예상
- **월별 패턴**: 7월 최고(257개), 2월 최저 콘텐츠 추가

### 2. 글로벌 확장
- **상위 3개국**: 미국(3,642), 인도(1,045), 영국(785)
- **국가 다양성**: 2015년 이후 연평균 20% 증가
- **지역별 특징**: 미국 영화 중심, 인도/영국 TV 프로그램 확대

### 3. 장르 트렌드
- **하락 장르**: 드라마(-2.41%/년), 독립영화(-1.11%/년)
- **상승 장르**: 국제 TV 프로그램(+1.10%/년), 액션(+0.85%/년)
- **최다 장르**: 기타(3,962개), 드라마, 코미디 순

### 4. 네트워크 분석
- **협업 허브**: 상위 10명의 감독/배우가 전체 네트워크의 40% 연결
- **중심 인물**: Degree Centrality 기준 상위 10명 도출

### 5. 추천 시스템
- **유사도 정확도**: 평균 코사인 유사도 0.42
- **추천 성능**: 동일 장르 매칭률 80%+