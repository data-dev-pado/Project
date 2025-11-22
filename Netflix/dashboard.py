"""
dashboard_manager.py
====================
Netflix 대시보드 생성 및 관리 모듈

주요 기능:
1. 종합 대시보드 생성
2. 개별 차트 순차 표시
3. HTML 파일로 차트 저장
4. 통합 HTML 대시보드 생성
5. 분석 보고서 텍스트 내보내기
6. 인터랙티브 추천 시스템
"""

import os
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from matplotlib import rcParams
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False 


def create_comprehensive_dashboard(df):
    
    if df is None or df.empty:
        print("유효한 데이터가 없다.")
        return None
    
    print("\n" + "="*60)
    print("Netflix 데이터 분석 대시보드")
    print("="*60)
    
    # 기본 통계 출력
    print(f"   총 콘텐츠: {len(df):,}개")
    movie_count = len(df[df['type'] == 'Movie'])
    tv_count = len(df[df['type'] == 'TV Show'])
    print(f"   영화: {movie_count:,}개 ({movie_count/len(df)*100:.1f}%)")
    print(f"   TV 프로그램: {tv_count:,}개 ({tv_count/len(df)*100:.1f}%)")
    print(f"   데이터 기간: {df['date_added'].min().strftime('%Y-%m-%d')} ~ {df['date_added'].max().strftime('%Y-%m-%d')}")
    
    dashboard_results = {}
    
    # 분석 함수 import
    from Python_Data.analysis_functions import (
        analyze_content_strategy_shift,
        analyze_global_content_distribution,
        create_actor_director_network,
        content_similarity_analysis,
        growth_modeling_analysis
    )
    
    # 1. 콘텐츠 전략 분석
    strategy_fig, yearly_content = analyze_content_strategy_shift(df)
    dashboard_results['strategy_analysis'] = strategy_fig
    print("Content Strategy Analysis")
    
    # 2. 글로벌 콘텐츠 분포
    global_fig, country_counts = analyze_global_content_distribution(df)
    dashboard_results['global_distribution'] = global_fig
    print("Global Content Distribution Analysis")
    
    # 3. 배우-감독 네트워크
    network_fig, top_degree, top_betweenness = create_actor_director_network(df)
    dashboard_results['network_analysis'] = network_fig
    print("Actor-Director Network Analysis")
    
    # 4. 콘텐츠 유사도 분석
    similarity_fig, recommender, genre_dist = content_similarity_analysis(df)
    dashboard_results['similarity_analysis'] = similarity_fig
    dashboard_results['recommender'] = recommender
    print("Content Similarity Analysis")
    
    # 5. 성장 모델링
    growth_fig, monthly_data, future_predictions = growth_modeling_analysis(df)
    dashboard_results['growth_modeling'] = growth_fig
    print("Growth Modeling Analysis")
    
    # 인사이트 저장
    insights = {
        'yearly_content': yearly_content,
        'country_counts': country_counts,
        'genre_distribution': genre_dist,
        'monthly_trends': monthly_data,
        'future_predictions': future_predictions
    }
    dashboard_results['insights'] = insights
    
    # 주요 인사이트 출력
    
    if yearly_content is not None:
        print(f"TV 프로그램 비율: {yearly_content['tv_ratio'].iloc[-1]:.1%}")
    
    if country_counts:
        top_3 = sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"상위 3개국: {', '.join([f'{c}({n})' for c, n in top_3])}")
    
    if genre_dist is not None:
        print(f"인기 장르: {genre_dist.index[0]} ({genre_dist.iloc[0]}개)")
    
    if future_predictions is not None:
        print(f"향후 TV 비율 예측: {future_predictions[-1]:.1%}")
    
    print("="*60 + "\n")
    
    return dashboard_results


def display_individual_charts(dashboard):
    
    if dashboard is None:
        print("표시할 대시보드가 없습니다.")
        return
    
    print("\n개별 차트")
    print("="*30)
    
    charts = [
        ('strategy_analysis', '콘텐츠 전략 변화 분석'),
        ('global_distribution', '글로벌 콘텐츠 분포 분석'),
        ('network_analysis', '배우-감독 네트워크 분석'),
        ('similarity_analysis', '콘텐츠 유사도 분석'),
        ('growth_modeling', '성장 모델링 분석')
    ]
    
    for key, name in charts:
        if dashboard.get(key):
            print(f"\n{name}")
            dashboard[key].show()
            input("다음 차트를 보려면 Enter")
    
    print("\n모든 차트 표시")


def create_combined_html(chart_files, output_dir):
    """
    통합 HTML 대시보드를 생성합니다.
    
    Parameters:
    -----------
    chart_files : list
        차트 파일 경로 리스트
    output_dir : str
        출력 디렉토리 경로
    
    Returns:
    --------
    str
        생성된 HTML 파일 경로
    """
    
    try:
        combined_html = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Netflix 콘텐츠 분석 대시보드</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            text-align: center;
            background-color: #e50914;
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .nav {
            text-align: center;
            margin-bottom: 30px;
        }
        .nav a {
            display: inline-block;
            padding: 10px 20px;
            margin: 5px;
            background-color: #333;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            transition: background-color 0.3s;
        }
        .nav a:hover {
            background-color: #e50914;
        }
        .chart-section {
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .chart-title {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #333;
            border-bottom: 2px solid #e50914;
            padding-bottom: 10px;
        }
        iframe {
            width: 100%;
            height: 800px;
            border: none;
            border-radius: 5px;
        }
        .footer {
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            background-color: #333;
            color: white;
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Netflix 콘텐츠 분석 대시보드</h1>
        <p>종합적인 Netflix 데이터 분석 및 시각화</p>
    </div>
    
    <div class="nav">
"""
        
        # 네비게이션 생성
        for i, filepath in enumerate(chart_files):
            filename = os.path.basename(filepath).replace('.html', '').replace('_', ' ')
            combined_html += f'        <a href="#chart{i+1}">{filename}</a>\n'
        
        combined_html += """    </div>
    
"""
        
        # 차트 섹션 생성
        for i, filepath in enumerate(chart_files):
            chart_filename = os.path.basename(filepath)
            display_name = chart_filename.replace('.html', '').replace('_', ' ')
            combined_html += f"""    <div class="chart-section" id="chart{i+1}">
        <div class="chart-title">{i+1}. {display_name}</div>
        <iframe src="{chart_filename}"></iframe>
    </div>
    
"""
        
        combined_html += """    <div class="footer">
        <p>Generated by Netflix Analysis Dashboard</p>
        <p>2024 Netflix 콘텐츠 분석 프로젝트</p>
    </div>
</body>
</html>"""
        
        combined_filepath = os.path.join(output_dir, "netflix_dashboard.html")
        with open(combined_filepath, 'w', encoding='utf-8') as f:
            f.write(combined_html)
        
        print(f"통합 대시보드 생성: {combined_filepath}")
        return combined_filepath
        
    except Exception as e:
        print(f"HTML 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_charts_as_html(dashboard, output_dir='netflix_charts'):
    
    if dashboard is None:
        print("저장할 대시보드가 없다.")
        return None
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    chart_files = []
    
    charts = [
        ('strategy_analysis', '콘텐츠_전략_분석'),
        ('global_distribution', '글로벌_콘텐츠_분포'),
        ('network_analysis', '배우_감독_네트워크'),
        ('similarity_analysis', '콘텐츠_유사도_분석'),
        ('growth_modeling', '성장_모델링')
    ]
    
    for key, filename in charts:
        if dashboard.get(key):
            filepath = os.path.join(output_dir, f"{filename}.html")
            dashboard[key].write_html(filepath)
            chart_files.append(filepath)
            print(f"{filename} 저장")
    
    # 통합 HTML 생성
    combined_path = create_combined_html(chart_files, output_dir)
    
    if combined_path:
        print(f"\n모든 차트가 '{output_dir}' 폴더에 저장되었습니다!")
        print(f"통합 대시보드: {combined_path}")
    
    return chart_files


def export_analysis_report(dashboard, filename='netflix_analysis_report.txt'):
    """
    Parameters:
    -----------
    dashboard : dict
        대시보드 결과 딕셔너리
    filename : str
        출력 파일명
    """
    
    if dashboard is None:
        return
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("Netflix 콘텐츠 분석 보고서\n")
            f.write("="*60 + "\n\n")
            
            insights = dashboard.get('insights', {})
            
            # 1. 연도별 콘텐츠
            if 'yearly_content' in insights:
                f.write("1. 연도별 콘텐츠 추가 현황\n")
                f.write("-"*40 + "\n")
                yearly_content = insights['yearly_content']
                f.write(yearly_content.to_string())
                f.write("\n\n")
            
            # 2. 국가별 분포
            if 'country_counts' in insights:
                f.write("2. 국가별 콘텐츠 분포 (Top 10)\n")
                f.write("-"*40 + "\n")
                country_counts = insights['country_counts']
                top_10 = sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                for i, (country, count) in enumerate(top_10, 1):
                    f.write(f"{i}. {country}: {count:,}개\n")
                f.write("\n")
            
            # 3. 장르별 분포
            if 'genre_distribution' in insights:
                f.write("3. 장르별 콘텐츠 분포\n")
                f.write("-"*40 + "\n")
                genre_dist = insights['genre_distribution']
                f.write(genre_dist.to_string())
                f.write("\n\n")
            
            # 4. 미래 예측
            if 'future_predictions' in insights:
                f.write("4. 미래 예측\n")
                f.write("-"*40 + "\n")
                future_predictions = insights['future_predictions']
                for i, pred in enumerate(future_predictions, 1):
                    f.write(f"예측 {i}: TV 프로그램 비율 {pred:.1%}\n")
                f.write("\n")
            
            f.write(f"보고서 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
        
        print(f"분석 보고서가 '{filename}'에 저장되었습니다.")
        
    except Exception as e:
        print(f"보고서 저장 오류: {e}")


def interactive_recommendation_system(dashboard, df):
    """
    인터랙티브 추천 시스템을 실행합니다.
    
    Parameters:
    -----------
    dashboard : dict
        대시보드 결과 딕셔너리
    df : pd.DataFrame
        Netflix 데이터
    """
    
    if dashboard is None or 'recommender' not in dashboard:
        print("추천 시스템을 사용할 수 없습니다.")
        return
    
    print("\nNetflix 콘텐츠 추천 시스템")
    print("="*50)
    print("Tip: 제목의 일부만 입력해도 검색됩니다.")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.\n")
    
    while True:
        try:
            user_input = input("추천받고 싶은 콘텐츠 제목: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '종료', 'q']:
                print("\n추천 시스템을 종료합니다.")
                break
            
            if not user_input:
                print("제목을 입력해주세요.\n")
                continue
            
            print(f"\n'{user_input}'와 유사한 콘텐츠를 찾는 중...\n")
            recommendations = dashboard['recommender'](user_input)
            
            if isinstance(recommendations, pd.DataFrame):
                print("추천 결과:")
                print("-"*50)
                for i, (idx, row) in enumerate(recommendations.iterrows(), 1):
                    print(f"\n{i}. {row['title']} ({row['type']})")
                    print(f"   장르: {row['extracted_genre']}")
                    print(f"   등급: {row['rating']}")
                    print(f"   유사도: {row['similarity_score']:.1%}")
                print()
            else:
                print(f"{recommendations}\n")
                
        except KeyboardInterrupt:
            print("\n\n추천 시스템을 종료합니다.")
            break
        except Exception as e:
            print(f"오류 발생: {e}\n")


def show_all_charts(dashboard):
    """
    Parameters:
    -----------
    dashboard : dict
        대시보드 결과 딕셔너리
    """
    
    if dashboard is None:
        print("표시할 대시보드가 없습니다.")
        return
    
    analyses = [
        ('strategy_analysis', '콘텐츠 전략 분석'),
        ('global_distribution', '글로벌 콘텐츠 분포'),
        ('network_analysis', '배우-감독 네트워크'),
        ('similarity_analysis', '콘텐츠 유사도 분석'),
        ('growth_modeling', '성장 모델링')
    ]
    
    for key, name in analyses:
        if dashboard.get(key) is not None:
            print(f"{name}: 성공")
            item = dashboard[key]
            if isinstance(item, dict) and 'figure' in item:
                item['figure'].show()
            else:
                item.show()
        else:
            print(f"{name}: 실패")


# 모듈 테스트용 코드
if __name__ == "__main__":
    print("dashboard.py 모듈 테스트\n")
    
    from Python_Data.data_preprocessing import load_and_preprocess_netflix_data
    
    df = load_and_preprocess_netflix_data('netflix_titles.csv')
    
    if df is not None:
        dashboard = create_comprehensive_dashboard(df)
        
        if dashboard:
            export_analysis_report(dashboard, 'report.txt')
            save_charts_as_html(dashboard, 'charts')
            
            print("\n모듈 테스트 Done")
    else:
        print("모듈 테스트 실패: 데이터를 로드할 수 없다.")