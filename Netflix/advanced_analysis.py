"""
advanced_analysis.py
====================
Netflix 데이터 고급 분석 모듈

주요 기능:
1. 퍼널 분석 (Funnel Analysis) - 데이터 품질 단계별 분석
2. 코호트 분석 (Cohort Analysis) - 장르별 연도별 트렌드
3. 시계열 분석 (Time Series Analysis) - 월별 패턴 및 계절성
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def funnel_analysis(df):
    """
    분석 단계:
    1. 전체 콘텐츠
    2. 등급 정보 보유
    3. 설명 정보 보유
    4. 출연진 정보 보유
    """
    
    try:
        df_clean = df.copy()
        total_content = len(df_clean)
        
        # 퍼널 데이터 구조 생성
        funnel_data = {
            'stage': [], 
            'count': [], 
            'step_conversion_rate': [],   
            'cumulative_conversion_rate': [],  
            'drop_off_rate': []      
        }
        
        # 1단계: 전체 콘텐츠
        funnel_data['stage'].append('1. 전체 콘텐츠')
        funnel_data['count'].append(total_content)
        funnel_data['step_conversion_rate'].append(100.0)
        funnel_data['cumulative_conversion_rate'].append(100.0)
        funnel_data['drop_off_rate'].append(0.0)
        
        # 2단계: 등급 정보 보유
        rated_content = len(df_clean[df_clean['rating'].notna() & (df_clean['rating'] != '')])
        step_conv2 = rated_content / total_content * 100
        cum_conv2 = rated_content / total_content * 100
        funnel_data['stage'].append('2. 등급 정보 보유')
        funnel_data['count'].append(rated_content)
        funnel_data['step_conversion_rate'].append(step_conv2)
        funnel_data['cumulative_conversion_rate'].append(cum_conv2)
        funnel_data['drop_off_rate'].append(100 - step_conv2)
        
        # 3단계: 설명 정보 보유
        described_content = len(df_clean[df_clean['description'].notna() & (df_clean['description'] != '')])
        step_conv3 = described_content / rated_content * 100 if rated_content > 0 else 0
        cum_conv3 = described_content / total_content * 100
        funnel_data['stage'].append('3. 설명 정보 보유')
        funnel_data['count'].append(described_content)
        funnel_data['step_conversion_rate'].append(step_conv3)
        funnel_data['cumulative_conversion_rate'].append(cum_conv3)
        funnel_data['drop_off_rate'].append(100 - step_conv3)
        
        # 4단계: 출연진 정보 보유
        cast_content = len(df_clean[df_clean['cast'].notna() & (df_clean['cast'] != '')])
        step_conv4 = cast_content / described_content * 100 if described_content > 0 else 0
        cum_conv4 = cast_content / total_content * 100
        funnel_data['stage'].append('4. 출연진 정보 보유')
        funnel_data['count'].append(cast_content)
        funnel_data['step_conversion_rate'].append(step_conv4)
        funnel_data['cumulative_conversion_rate'].append(cum_conv4)
        funnel_data['drop_off_rate'].append(100 - step_conv4)
        
        funnel_df = pd.DataFrame(funnel_data)
        
        # 시각화
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        stages = funnel_df['stage'].str.replace(r'^\d+\.\s*', '', regex=True)
        counts = funnel_df['count']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        # 1. 단계별 콘텐츠 수 + 누적 전환율
        bars = ax1.barh(stages, counts, color=colors)
        ax1.set_xlabel('콘텐츠 수')
        ax1.set_title('Netflix 콘텐츠 품질 퍼널 분석', fontsize=14, fontweight='bold')
        
        for i, (bar, count, step_rate, cum_rate) in enumerate(zip(
                bars, counts, funnel_df['step_conversion_rate'], funnel_df['cumulative_conversion_rate'])):
            ax1.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2, 
                     f'{count:,}\n(단계: {step_rate:.1f}%, 누적: {cum_rate:.1f}%)', 
                     ha='left', va='center', fontweight='bold')
        
        # 2. 단계별 전환율 & 이탈율
        x = np.arange(len(stages))
        ax2.bar(x, funnel_df['step_conversion_rate'], color=colors, alpha=0.7, label='단계별 전환율')
        ax2.bar(x, funnel_df['drop_off_rate'], bottom=funnel_df['step_conversion_rate'], 
                color='lightgray', alpha=0.7, label='이탈율')
        
        ax2.set_xlabel('퍼널 단계')
        ax2.set_ylabel('비율 (%)')
        ax2.set_title('단계별 전환율 및 이탈율', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(stages, rotation=45, ha='right')
        ax2.legend()
        ax2.set_ylim(0, 100)
        
        for i, rate in enumerate(funnel_df['step_conversion_rate']):
            ax2.text(i, rate + 2, f'{rate:.1f}%', ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        print(f"퍼널 분석 - 최종 단계 누적 전환율: {cum_conv4:.1f}%")
        
        return fig, funnel_df
        
    except Exception as e:
        print(f"퍼널 분석 오류: {e}")
        return None, None


def cohort_analysis(df):
    """
    분석 내용:
    - 연도별 장르 콘텐츠 수 변화
    - 장르별 비율 변화 (정규화)
    - 장르 트렌드 (상승/하락)
    
    Returns:
    --------
    tuple : (matplotlib.Figure, pd.DataFrame, dict)
        코호트 차트, 코호트 테이블, 트렌드 분석 결과
    """
    
    try:
        df_clean = df.copy()
        df_clean['year_added'] = df_clean['date_added'].dt.year
        
        # 장르 데이터 추출
        genre_data = []
        for idx, row in df_clean.iterrows():
            if pd.notna(row['listed_in']):
                genres = [genre.strip() for genre in str(row['listed_in']).split(',')]
                for genre in genres[:2]:  # 상위 2개 장르만
                    genre_data.append({
                        'year_added': row['year_added'], 
                        'genre': genre, 
                        'type': row['type']
                    })
        
        genre_df = pd.DataFrame(genre_data)
        
        # 코호트 테이블 생성
        cohort_table = genre_df.pivot_table(
            index='genre', columns='year_added', values='type', aggfunc='count', fill_value=0
        )
        
        # 상위 10개 장르
        top_genres = cohort_table.sum(axis=1).nlargest(10).index
        cohort_table_top = cohort_table.loc[top_genres]
        
        # 비율 정규화
        cohort_percentages = cohort_table_top.div(cohort_table_top.sum(axis=0), axis=1) * 100
        
        # 시각화
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # 1. 절대 수치 히트맵
        sns.heatmap(cohort_table_top, annot=True, fmt='d', cmap='YlOrRd', ax=ax1, 
                    cbar_kws={'label': '콘텐츠 수'})
        ax1.set_title('연도별 장르 콘텐츠 수 (코호트 분석)', fontsize=14, fontweight='bold')
        
        # 2. 비율 히트맵
        sns.heatmap(cohort_percentages, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax2, 
                    cbar_kws={'label': '비율 (%)'})
        ax2.set_title('연도별 장르 비율 변화 (정규화)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 장르 트렌드 계산
        trend_analysis = {}
        for genre in top_genres:
            percentages = cohort_percentages.loc[genre].values
            valid_mask = ~np.isnan(percentages)
            if np.sum(valid_mask) > 2:
                slope = np.polyfit(cohort_percentages.columns.values[valid_mask], 
                                 percentages[valid_mask], 1)[0]
                trend_analysis[genre] = slope
        
        print("코호트 분석")
        print("장르 트렌드 (상승: +, 하락: -):")
        for genre, slope in trend_analysis.items():
            direction = "상승" if slope > 0 else "하락"
            print(f"  {genre}: {slope:+.3f} ({direction})")
        
        return fig, cohort_table_top, trend_analysis
        
    except Exception as e:
        print(f"코호트 분석 중 오류: {e}")
        return None, None, None


def cohort_analysis_with_trend(df):
     
    try:
        df_clean = df.copy()
        df_clean['year_added'] = df_clean['date_added'].dt.year
        
        # 장르 데이터 추출
        genre_data = []
        for idx, row in df_clean.iterrows():
            if pd.notna(row['listed_in']):
                genres = [genre.strip() for genre in str(row['listed_in']).split(',')]
                for genre in genres[:2]:
                    genre_data.append({
                        'year_added': row['year_added'], 
                        'genre': genre, 
                        'type': row['type']
                    })
        
        genre_df = pd.DataFrame(genre_data)
        
        cohort_table = genre_df.pivot_table(
            index='genre', columns='year_added', values='type', aggfunc='count', fill_value=0
        )
        
        top_genres = cohort_table.sum(axis=1).nlargest(10).index
        cohort_table_top = cohort_table.loc[top_genres]
        cohort_percentages = cohort_table_top.div(cohort_table_top.sum(axis=0), axis=1) * 100
        
        # 트렌드 계산
        trend_analysis = {}
        for genre in top_genres:
            percentages = cohort_percentages.loc[genre].values
            valid_mask = ~np.isnan(percentages)
            if np.sum(valid_mask) > 2:
                slope = np.polyfit(cohort_percentages.columns.values[valid_mask], 
                                 percentages[valid_mask], 1)[0]
                trend_analysis[genre] = slope
        
        # 시각화
        plt.figure(figsize=(16, 10))
        sns.heatmap(cohort_percentages, annot=True, fmt='.1f', cmap='RdYlBu_r', 
                    cbar_kws={'label': '비율 (%)'})
        
        # 트렌드 화살표 추가
        for i, genre in enumerate(top_genres):
            slope = trend_analysis.get(genre, 0)
            color = 'green' if slope > 0 else 'red' if slope < 0 else 'gray'
            
            plt.arrow(
                x=len(cohort_percentages.columns) + 0.5,
                y=i + 0.5,
                dx=0.5, dy=0,
                head_width=0.3, head_length=0.2, fc=color, ec=color
            )
            plt.text(
                x=len(cohort_percentages.columns) + 1.1, 
                y=i + 0.5, 
                s=f"{slope:+.2f}", 
                va='center', color=color, fontweight='bold'
            )
        
        plt.title('연도별 장르 비율 변화 & 트렌드', fontsize=16, fontweight='bold')
        plt.xlabel('연도')
        plt.ylabel('장르')
        plt.tight_layout()
        
        print("트렌드 포함 코호트 분석")
        
        return plt.gcf()
        
    except Exception as e:
        print(f"트렌드 코호트 분석 오류: {e}")
        return None


def time_series_analysis(df):
    """    
    분석 내용:
    - 월별 전체 콘텐츠 추가량
    - 영화 vs TV 프로그램 추세
    - TV 프로그램 비율 변화
    - 계절성 분석 (월별 패턴)
    
    Returns:
    --------
    tuple : (matplotlib.Figure, dict)
        시계열 차트와 계절성 분석 결과
    """
    
    try:
        df_time = df[df['date_added'].notna()].copy()
        df_time['year_month'] = df_time['date_added'].dt.to_period('M')
        
        # 월별 통계
        monthly_stats = df_time.groupby(['year_month', 'type']).size().unstack(fill_value=0)
        monthly_total = monthly_stats.sum(axis=1)
        
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # 1. 월별 전체 콘텐츠 추가량
        ax1 = axes[0, 0]
        monthly_total.plot(ax=ax1, color='steelblue', linewidth=2)
        ax1.fill_between(monthly_total.index.astype(str), monthly_total.values, 
                        alpha=0.3, color='steelblue')
        ax1.set_title('월별 콘텐츠 추가량 시계열', fontsize=12, fontweight='bold')
        ax1.set_xlabel('기간')
        ax1.set_ylabel('콘텐츠 수')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 영화 vs TV 프로그램
        ax2 = axes[0, 1]
        if 'Movie' in monthly_stats.columns and 'TV Show' in monthly_stats.columns:
            monthly_stats.plot(kind='area', stacked=True, ax=ax2, 
                             color=['lightcoral', 'lightblue'], alpha=0.7)
        ax2.set_title('영화 vs TV 프로그램 월별 추가량', fontsize=12, fontweight='bold')
        ax2.set_xlabel('기간')
        ax2.set_ylabel('콘텐츠 수')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. TV 프로그램 비율 변화
        ax3 = axes[1, 0]
        tv_ratio = monthly_stats.get('TV Show', pd.Series(0, index=monthly_stats.index)) / monthly_total * 100
        tv_ratio.plot(ax=ax3, color='green', linewidth=2, marker='o', markersize=3)
        ax3.set_title('TV 프로그램 비율 변화', fontsize=12, fontweight='bold')
        ax3.set_xlabel('기간')
        ax3.set_ylabel('TV 프로그램 비율 (%)')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 계절성 분석
        ax4 = axes[1, 1]
        df_time['month'] = df_time['date_added'].dt.month
        seasonal_pattern = df_time.groupby('month').size()
        bars = ax4.bar(seasonal_pattern.index, seasonal_pattern.values, 
                      color=plt.cm.Set3(np.linspace(0, 1, 12)))
        ax4.set_title('월별 콘텐츠 추가 패턴 (계절성)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('월')
        ax4.set_ylabel('평균 콘텐츠 수')
        ax4.set_xticks(range(1, 13))
        ax4.set_xticklabels([f'{i}월' for i in range(1, 13)], rotation=45)
        
        # 최고/최저 월 표시
        max_month = seasonal_pattern.idxmax()
        min_month = seasonal_pattern.idxmin()
        ax4.text(max_month, seasonal_pattern[max_month] + seasonal_pattern.max() * 0.05, 
                f'최고\n{seasonal_pattern[max_month]}개', 
                ha='center', fontweight='bold', color='red')
        ax4.text(min_month, seasonal_pattern[min_month] + seasonal_pattern.max() * 0.05, 
                f'최저\n{seasonal_pattern[min_month]}개', 
                ha='center', fontweight='bold', color='blue')
        
        plt.tight_layout()
        
        # 계절성 분석 결과
        seasonality_results = {
            'max_month': max_month,
            'max_count': seasonal_pattern[max_month],
            'min_month': min_month,
            'min_count': seasonal_pattern[min_month],
            'monthly_pattern': seasonal_pattern.to_dict()
        }
        
        print(f"시계열 분석")
        print(f"  최고 활동 월: {max_month}월 ({seasonal_pattern[max_month]}개)")
        print(f"  최저 활동 월: {min_month}월 ({seasonal_pattern[min_month]}개)")
        
        return fig, seasonality_results
        
    except Exception as e:
        print(f"시계열 분석 오류: {e}")
        return None, None


def run_all_advanced_analysis(df):
    
    results = {}
    
    # 1. 퍼널 분석
    funnel_fig, funnel_df = funnel_analysis(df)
    results['funnel'] = {'figure': funnel_fig, 'data': funnel_df}
    
    # 2. 코호트 분석
    cohort_fig, cohort_table, trends = cohort_analysis(df)
    results['cohort'] = {'figure': cohort_fig, 'table': cohort_table, 'trends': trends}
    
    # 3. 트렌드 포함 코호트
    trend_fig = cohort_analysis_with_trend(df)
    results['cohort_trend'] = {'figure': trend_fig}
    
    # 4. 시계열 분석
    ts_fig, seasonality = time_series_analysis(df)
    results['time_series'] = {'figure': ts_fig, 'seasonality': seasonality}
    
    return results


# 모듈 테스트용 코드
if __name__ == "__main__":
    print("advanced_analysis.py 모듈 테스트\n")
    
    from Python_Data.data_preprocessing import load_and_preprocess_netflix_data
    
    df = load_and_preprocess_netflix_data('netflix_titles.csv')
    
    if df is not None:
        print("\n1. 퍼널 분석 테스트")
        funnel_fig, funnel_df = funnel_analysis(df)
        if funnel_fig:
            plt.show()
            print(funnel_df)
        
        print("\n2. 코호트 분석 테스트")
        cohort_fig, cohort_table, trends = cohort_analysis(df)
        if cohort_fig:
            plt.show()
        
        print("\n3. 시계열 분석 테스트")
        ts_fig, seasonality = time_series_analysis(df)
        if ts_fig:
            plt.show()
        
        print("\nModule Done")
    else:
        print("모듈 테스트 실패: 데이터를 로드할 수 없습니다.")