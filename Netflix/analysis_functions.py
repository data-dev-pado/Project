"""
analysis_functions.py
=====================
Netflix 데이터 핵심 분석 함수

주요 기능:
1. 콘텐츠 전략 변화 분석
2. 글로벌 콘텐츠 분포 분석
3. 배우-감독 네트워크 분석
4. 콘텐츠 유사도 분석 및 추천 시스템
5. 성장 모델링 및 예측 분석
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

from matplotlib import rcParams
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False 

def analyze_content_strategy_shift(df):
    """
    Netflix 콘텐츠 전략 변화를 분석합니다.
    
    분석 내용:
    - 연도별 콘텐츠 추가량 (영화 vs TV 프로그램)
    - 콘텐츠 타입 비율 변화
    - 월별 콘텐츠 추가 패턴
    - 등급별 분포
    """
    
    try:
        # 연도별 콘텐츠 타입 집계
        yearly_content = df.groupby(['year_added', 'type']).size().unstack(fill_value=0)
        yearly_content['total'] = yearly_content.sum(axis=1)
        yearly_content['movie_ratio'] = yearly_content['Movie'] / yearly_content['total']
        yearly_content['tv_ratio'] = yearly_content['TV Show'] / yearly_content['total']
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('연도별 콘텐츠 추가량', '콘텐츠 타입 비율 변화', 
                          '월별 콘텐츠 추가 패턴', '등급별 분포'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "domain"}]]
        )
        
        # 1. 연도별 추가량
        fig.add_trace(
            go.Scatter(x=yearly_content.index, y=yearly_content['Movie'],
                      name='영화', line=dict(color='blue')), 
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=yearly_content.index, y=yearly_content['TV Show'],
                      name='TV 프로그램', line=dict(color='red')), 
            row=1, col=1
        )
        
        # 2. 콘텐츠 타입 비율
        fig.add_trace(
            go.Scatter(x=yearly_content.index, y=yearly_content['tv_ratio'],
                      name='TV 프로그램 비율', line=dict(color='red', dash='dash')), 
            row=1, col=2
        )
        
        # 3. 월별 패턴
        monthly_pattern = df.groupby('month_added').size()
        fig.add_trace(
            go.Bar(x=monthly_pattern.index, y=monthly_pattern.values,
                   name='월별 추가량', marker_color='green'), 
            row=2, col=1
        )
        
        # 4. 등급별 분포
        rating_dist = df['rating'].value_counts().head(10) 
        fig.add_trace(
            go.Pie(labels=rating_dist.index, values=rating_dist.values, name='등급별 분포'), 
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Netflix 콘텐츠 전략 분석 대시보드")
        
        print("콘텐츠 전략 분석")
        return fig, yearly_content
        
    except Exception as e:
        print(f"콘텐츠 전략 분석 오류: {e}")
        return None, None


def analyze_global_content_distribution(df):
    """
    글로벌 콘텐츠 분포를 분석합니다.
    
    분석 내용:
    - 국가별 콘텐츠 수
    - 미국 vs 기타 국가 비율
    - 연도별 국가 다양성
    - 지역별 콘텐츠 타입 분포
    """
    
    try:
        # 국가별 콘텐츠 수 계산
        country_counts = {}
        for countries in df['country'].dropna():
            for country in str(countries).split(', '):
                country = country.strip()
                if country and country != 'Unknown':
                    country_counts[country] = country_counts.get(country, 0) + 1
        
        if not country_counts:
            print("국가 데이터가 없습니다.")
            return None, None
        
        # 상위 15개 국가
        top_countries = sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('국가별 콘텐츠 수', '미국 vs 기타 국가', 
                          '연도별 국가 다양성', '지역별 콘텐츠 타입'),
            specs=[[{"type": "bar"}, {"type": "domain"}],
                   [{"secondary_y": False}, {"type": "bar"}]]
        )
        
        # 1. 국가별 콘텐츠 수
        countries, counts = zip(*top_countries)
        fig.add_trace(
            go.Bar(x=list(countries), y=list(counts), 
                   name='콘텐츠 수', marker_color='skyblue'), 
            row=1, col=1
        )
        
        # 2. 미국 vs 기타
        us_count = country_counts.get('United States', 0)
        other_count = sum(country_counts.values()) - us_count
        fig.add_trace(
            go.Pie(labels=['미국', '기타 국가'], values=[us_count, other_count], 
                   name='지역별 비율'), 
            row=1, col=2
        )
        
        # 3. 연도별 국가 다양성
        yearly_countries = df.groupby('year_added')['country'].apply(
            lambda x: len(set([c.strip() for countries in x.dropna() 
                             for c in str(countries).split(', ') if c.strip() != 'Unknown']))
        )
        fig.add_trace(
            go.Scatter(x=yearly_countries.index, y=yearly_countries.values,
                      name='연도별 국가 수', line=dict(color='orange')), 
            row=2, col=1
        )
        
        # 4. 지역별 콘텐츠 타입
        top_5_countries = [c[0] for c in top_countries[:5]]
        movies_data = []
        tv_data = []
        
        for country in top_5_countries:
            country_data = df[df['country'].str.contains(country, na=False)]
            movie_count = len(country_data[country_data['type'] == 'Movie'])
            tv_count = len(country_data[country_data['type'] == 'TV Show'])
            movies_data.append(movie_count)
            tv_data.append(tv_count)
        
        fig.add_trace(
            go.Bar(x=top_5_countries, y=movies_data, name='영화', marker_color='blue'), 
            row=2, col=2
        )
        fig.add_trace(
            go.Bar(x=top_5_countries, y=tv_data, name='TV 프로그램', marker_color='red'), 
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Netflix 글로벌 콘텐츠 분포 분석", barmode='group')
        
        print("글로벌 콘텐츠 분포 분석")
        return fig, country_counts
        
    except Exception as e:
        print(f"글로벌 콘텐츠 분포 분석 중 오류: {e}")
        return None, None


def create_actor_director_network(df):
    """
    배우-감독 네트워크를 분석합니다.
    
    분석 내용:
    - 배우와 감독 간의 협업 관계 네트워크
    - 중심성 지표 (Degree, Betweenness)
    - 주요 인물 파악
    """
    
    try:
        G = nx.Graph()
        
        # 처리할 데이터 필터링 
        # Unknown 제외, 성능을 위해 100개만 선택함
        
        valid_data = df[(df['cast'] != 'Unknown') & (df['director'] != 'Unknown')].head(100)
        
        for _, row in valid_data.iterrows():
            directors = [d.strip() for d in str(row['director']).split(',')]
            actors = [a.strip() for a in str(row['cast']).split(',')]
            
            for director in directors:
                if director and director != 'Unknown':
                    if not G.has_node(director):
                        G.add_node(director, type='director')
                    
                    for actor in actors:
                        if actor and actor != 'Unknown':
                            if not G.has_node(actor):
                                G.add_node(actor, type='actor')
                            
                            if G.has_edge(director, actor):
                                G[director][actor]['weight'] += 1
                            else:
                                G.add_edge(director, actor, weight=1)
        
        # 연결되지 않은 노드 제거
        nodes_to_remove = [node for node, degree in G.degree() if degree == 0]
        G.remove_nodes_from(nodes_to_remove)
        
        if len(G.nodes()) == 0:
            print("네트워크 생성을 위한 유효한 데이터가 없다.")
            return None, None, None
        
        # 중심성 계산
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        
        top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    
        # 레이아웃 생성
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        
        # 엣지 좌표
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # 노드 좌표 및 속성
        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node}<br>연결 정도: {degree_centrality[node]:.3f}")
            node_color.append('red' if G.nodes[node].get('type') == 'director' else 'blue')
            node_size.append(5 + degree_centrality[node] * 50)
        
        # Plotly Figure 생성
        fig = go.Figure()
        
        # 엣지 추가
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='gray'),
            hoverinfo='none',
            mode='lines',
            name='연결'
        ))
        
        # 노드 추가
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(size=node_size, color=node_color, 
                       line=dict(width=1, color='DarkSlateGrey')),
            name='인물'
        ))
        
        fig.update_layout(
            title="Netflix 배우-감독 네트워크 분석",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        print("배우-감독 네트워크 분석")
        return fig, top_degree, top_betweenness
        
    except Exception as e:
        print(f"네트워크 분석 중 오류: {e}")
        return None, None, None


def content_similarity_analysis(df):
    """
    콘텐츠 유사도를 분석하고 추천 시스템을 구축합니다.
    
    분석 내용:
    - TF-IDF 기반 콘텐츠 유사도 계산
    - 장르 자동 추출 및 분포
    - 장르별 트렌드 및 평균 등급
    - 유사도 히트맵
    """
    
    try:
        # 텍스트 전처리
        descriptions = df['description'].fillna('')
        
        # TF-IDF 벡터화
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        
        # 코사인 유사도 계산 (샘플링)
        sample_size = min(500, len(df))
        sample_indices = np.random.choice(len(df), sample_size, replace=False)
        cosine_sim = cosine_similarity(tfidf_matrix[sample_indices], tfidf_matrix[sample_indices])
        
        # 장르 키워드 정의
        genre_keywords = {
            '액션': ['action', 'fight', 'battle', 'adventure', 'thriller'],
            '코미디': ['comedy', 'funny', 'humor', 'laugh', 'hilarious'],
            '드라마': ['drama', 'emotional', 'family', 'life', 'story'],
            '로맨스': ['love', 'romance', 'relationship', 'romantic'],
            '공포': ['horror', 'scary', 'fear', 'haunted', 'terror'],
            '다큐멘터리': ['documentary', 'real', 'true', 'history'],
            'SF': ['sci-fi', 'science', 'future', 'space', 'technology'],
            '범죄': ['crime', 'criminal', 'police', 'detective', 'murder']
        }
        
        # 장르 추출 함수
        def extract_genre(description):
            description_lower = str(description).lower()
            for genre, keywords in genre_keywords.items():
                if any(keyword in description_lower for keyword in keywords):
                    return genre
            return '기타'
        
        df['extracted_genre'] = df['description'].apply(extract_genre)
        genre_dist = df['extracted_genre'].value_counts()
        
        # 추천 함수 정의
        def get_recommendations(title, df=df):
            try:
                matches = df[df['title'].str.contains(title, case=False, na=False)]
                if matches.empty:
                    return f"'{title}' 제목을 찾을 수 없습니다."
                
                idx = matches.index[0]
                
                full_cosine_sim = cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix)
                sim_scores = list(enumerate(full_cosine_sim[0]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                
                # 상위 5개 추천 (자기 자신 제외)
                sim_scores = sim_scores[1:6]
                content_indices = [i[0] for i in sim_scores]
                
                recommendations = df.iloc[content_indices][['title', 'type', 'extracted_genre', 'rating']].copy()
                recommendations['similarity_score'] = [score[1] for score in sim_scores]
                
                return recommendations
                
            except Exception as e:
                return f"추천 생성 중 오류: {e}"
    
        # 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('장르별 콘텐츠 분포', '연도별 장르 트렌드', 
                          '장르별 평균 등급', '콘텐츠 유사도 히트맵'),
            specs=[[{"type": "bar"}, {"secondary_y": False}],
                   [{"type": "bar"}, {"type": "heatmap"}]]
        )
        
        # 1. 장르별 분포
        fig.add_trace(
            go.Bar(x=genre_dist.index, y=genre_dist.values,
                   name='장르별 콘텐츠 수', marker_color='lightblue'), 
            row=1, col=1
        )
        
        # 2. 연도별 장르 트렌드
        genre_yearly = df.groupby(['year_added', 'extracted_genre']).size().unstack(fill_value=0)
        top_genres = genre_dist.head(3).index
        
        for genre in top_genres:
            if genre in genre_yearly.columns:
                fig.add_trace(
                    go.Scatter(x=genre_yearly.index, y=genre_yearly[genre],
                              name=f'{genre} 트렌드', mode='lines+markers'), 
                    row=1, col=2
                )
        
        # 3. 장르별 평균 등급
        rating_map = {
            'TV-G': 1, 'TV-Y': 1, 'TV-Y7': 1.5, 'TV-Y7-FV': 1.5,
            'TV-PG': 2, 'PG': 2, 'TV-14': 3, 'PG-13': 3,
            'TV-MA': 4, 'R': 4, 'NC-17': 5, 'UR': 2.5, 'NR': 2.5, 'UNRATED': 2.5
        }
        df['rating_numeric'] = df['rating'].map(rating_map).fillna(2.5)
        genre_rating = df.groupby('extracted_genre')['rating_numeric'].mean().sort_values(ascending=False)
        
        fig.add_trace(
            go.Bar(x=genre_rating.index, y=genre_rating.values,
                   name='평균 등급', marker_color='orange'), 
            row=2, col=1
        )
        
        # 4. 유사도 히트맵
        sample_sim = cosine_sim[:10, :10]
        sample_df = df.iloc[sample_indices[:10]]
        sample_titles = [title[:20] + '...' if len(title) > 20 else title 
                        for title in sample_df['title'].tolist()]
        
        fig.add_trace(
            go.Heatmap(z=sample_sim, x=sample_titles, y=sample_titles,
                      colorscale='Viridis', name='유사도'), 
            row=2, col=2
        )
        
        fig.update_layout(height=900, title_text="Netflix 콘텐츠 유사도 및 장르 분석")
        
        print("콘텐츠 유사도 분석")
        return fig, get_recommendations, genre_dist
        
    except Exception as e:
        print(f"유사도 분석 중 오류: {e}")
        return None, None, None


def growth_modeling_analysis(df):
    """
    성장 모델링 및 예측 분석을 수행합니다.
    
    분석 내용:
    - 월별 콘텐츠 추가 트렌드
    - 연도별 성장률
    - TV 프로그램 비율 변화 및 예측
    - 국가별 성장 패턴
    """
    
    try:
        # 날짜 데이터 확인 및 변환
        if not np.issubdtype(df['date_added'].dtype, np.datetime64):
            df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
        
        # 연, 월 컬럼 생성
        df['year_added'] = df['date_added'].dt.year
        df['month_added'] = df['date_added'].dt.month

        # 월별 데이터 집계
        monthly_data = (
            df.groupby(['year_added', 'month_added'])
              .size()
              .reset_index(name='count')
        )

        # date 컬럼 생성
        monthly_data['date'] = pd.to_datetime({
            'year': monthly_data['year_added'],
            'month': monthly_data['month_added'],
            'day': [1] * len(monthly_data)
        })
        monthly_data = monthly_data.sort_values('date').reset_index(drop=True)
        
        # 이동평균 계산
        monthly_data['ma_3'] = monthly_data['count'].rolling(window=3, min_periods=1).mean()
        monthly_data['ma_6'] = monthly_data['count'].rolling(window=6, min_periods=1).mean()
        
        # 연도별 성장률
        yearly_growth = df.groupby('year_added').size()
        yearly_growth_rate = yearly_growth.pct_change() * 100
        
        # TV Show 비율 변화
        yearly_type_ratio = df.groupby(['year_added', 'type']).size().unstack(fill_value=0)
        yearly_type_ratio['tv_ratio'] = yearly_type_ratio['TV Show'] / (
            yearly_type_ratio['Movie'] + yearly_type_ratio['TV Show']
        )
        
        # 예측 모델 (최근 5년 데이터 사용)
        recent_years = yearly_type_ratio.index[-5:]
        X = recent_years.values.reshape(-1, 1)
        y = yearly_type_ratio.loc[recent_years, 'tv_ratio'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # 미래 예측
        future_years = np.array(range(yearly_type_ratio.index.max() + 1, 
                                    yearly_type_ratio.index.max() + 4)).reshape(-1, 1)
        future_tv_ratio = model.predict(future_years)
        future_tv_ratio = np.clip(future_tv_ratio, 0, 1)
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('월별 콘텐츠 추가 트렌드', '연도별 성장률', 
                          'TV 프로그램 비율 변화 및 예측', '국가별 성장 패턴'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. 월별 트렌드
        fig.add_trace(
            go.Scatter(x=monthly_data['date'], y=monthly_data['count'],
                      name='월별 추가량', line=dict(color='blue')), 
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=monthly_data['date'], y=monthly_data['ma_6'],
                      name='6개월 이동평균', line=dict(color='red', dash='dash')), 
            row=1, col=1
        )
        
        # 2. 연도별 성장률
        fig.add_trace(
            go.Bar(x=yearly_growth_rate.index, y=yearly_growth_rate.values,
                   name='연도별 성장률(%)', marker_color='green'), 
            row=1, col=2
        )
        
        # 3. TV Show 비율 예측
        fig.add_trace(
            go.Scatter(x=yearly_type_ratio.index, y=yearly_type_ratio['tv_ratio'],
                      name='실제 TV 프로그램 비율', mode='lines+markers', 
                      line=dict(color='purple')), 
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=future_years.flatten(), y=future_tv_ratio,
                      name='예측 TV 프로그램 비율', mode='lines+markers',
                      line=dict(color='orange', dash='dash')), 
            row=2, col=1
        )
        
        # 4. 국가별 성장 패턴 (상위 3개국)
        country_counts = {}
        for countries in df['country'].dropna():
            for country in str(countries).split(', '):
                country = country.strip()
                if country and country != 'Unknown':
                    country_counts[country] = country_counts.get(country, 0) + 1
        
        top_countries = sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for country, _ in top_countries:
            country_yearly = df[df['country'].str.contains(country, na=False)].groupby('year_added').size()
            if not country_yearly.empty:
                fig.add_trace(
                    go.Scatter(x=country_yearly.index, y=country_yearly.values,
                              name=f'{country}', mode='lines+markers'), 
                    row=2, col=2
                )
        
        fig.update_layout(height=800, title_text="Netflix 성장 모델링 및 예측 분석")
        
        print("성장 모델링 분석")
        return fig, monthly_data, future_tv_ratio
            
    except Exception as e:
        print(f"성장 모델링 분석 중 오류: {e}")
        return None, None, None


# 모듈 테스트용 코드
if __name__ == "__main__":
    print("analysis_functions.py 모듈 테스트\n")
    
    from Python_Data.data_preprocessing import load_and_preprocess_netflix_data
    
    df = load_and_preprocess_netflix_data('netflix_titles.csv')
    
    if df is not None:
        print("\n1. 콘텐츠 전략 분석 테스트")
        fig1, yearly = analyze_content_strategy_shift(df)
    
        print("\n2. 글로벌 분포 분석 테스트")
        fig2, countries = analyze_global_content_distribution(df)
        
        print("\n3. 네트워크 분석 테스트")
        fig3, degree, between = create_actor_director_network(df)
        
        print("\n4. 유사도 분석 테스트")
        fig4, recommender, genres = content_similarity_analysis(df)
        
        print("\n5. 성장 모델링 분석 테스트")
        fig5, monthly, future = growth_modeling_analysis(df)
        
        print("\n모듈 테스트 Done")
    else:
        print("모듈 테스트 실패")