"""
main.py
=======
Netflix 데이터 분석 메인

실행 방법:
    python main.py

기능:
1. 데이터 로드 및 전처리
2. 기본 분석 (전략, 글로벌, 네트워크, 유사도, 성장)
3. 고급 분석 (퍼널, 코호트, 시계열)
4. HTML 대시보드 생성
5. 분석 보고서 내보내기
6. 인터랙티브 추천 시스템
"""

import sys
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import load_and_preprocess_netflix_data
from dashboard import (
    create_comprehensive_dashboard,
    save_charts_as_html,
    display_individual_charts,
    export_analysis_report,
    interactive_recommendation_system,
    show_all_charts
)
from advanced_analysis import run_all_advanced_analysis


def print_menu():
    print("1. 기본 분석 실행 (전략, 글로벌, 네트워크, 유사도, 성장)")
    print("2. 고급 분석 실행 (퍼널, 코호트, 시계열)")
    print("3. 모든 분석 실행")
    print("4. 개별 차트 순차 보기")
    print("5. HTML 대시보드 생성")
    print("6. 분석 보고서 내보내기")
    print("7. 추천 시스템 사용")


def run_basic_analysis(df):
    
    dashboard = create_comprehensive_dashboard(df)
    
    if dashboard:
        show_all_charts(dashboard)
        return dashboard
    else:
        return None


def run_advanced_analysis(df):
    results = run_all_advanced_analysis(df)
    
    if results:
        import matplotlib.pyplot as plt
        
        if results.get('funnel', {}).get('figure'):
            plt.show()
        if results.get('cohort', {}).get('figure'):
            plt.show()
        if results.get('cohort_trend', {}).get('figure'):
            plt.show()
        if results.get('time_series', {}).get('figure'):
            plt.show()
        
        return results
    else:
        return None


def interactive_mode(df):
    
    dashboard = None
    advanced_results = None
    
    while True:
        print_menu()
        
        try:
            choice = input("\n선택하세요 (0-7): ").strip()
            
            if choice == '0':
                break
            
            elif choice == '1':
                dashboard = run_basic_analysis(df)
            
            elif choice == '2':
                advanced_results = run_advanced_analysis(df)
            
            elif choice == '3':
                dashboard = run_basic_analysis(df)
                advanced_results = run_advanced_analysis(df)
            
            elif choice == '4':
                if dashboard is None:
                    print("\n먼저 기본 분석을 실행해주세요. (메뉴 1)")
                else:
                    display_individual_charts(dashboard)
            
            elif choice == '5':
                if dashboard is None:
                    print("\n먼저 기본 분석을 실행해주세요. (메뉴 1)")
                else:
                    output_dir = input("출력 폴더명 (기본값: netflix_charts): ").strip()
                    if not output_dir:
                        output_dir = 'netflix_charts'
                    save_charts_as_html(dashboard, output_dir)
            
            elif choice == '6':
                if dashboard is None:
                    print("\n먼저 기본 분석을 실행해주세요. (메뉴 1)")
                else:
                    filename = input("파일명 (기본값: netflix_analysis_report.txt): ").strip()
                    if not filename:
                        filename = 'netflix_analysis_report.txt'
                    export_analysis_report(dashboard, filename)
            
            elif choice == '7':
                if dashboard is None:
                    print("\n먼저 기본 분석을 실행해주세요. (메뉴 1)")
                else:
                    interactive_recommendation_system(dashboard, df)
            
            else:
                print("\n잘못된 선택입니다. 0-7 사이의 숫자를 입력하세요.")
        
        except KeyboardInterrupt:
            print("\n\n프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"\n오류: {e}")
            import traceback
            traceback.print_exc()


def auto_mode(df):
    
    # 1. 기본 분석
    dashboard = create_comprehensive_dashboard(df)
    
    if dashboard is None:
        print("기본 분석 실패. 프로그램을 종료합니다.")
        return
    
    # 2. 고급 분석
    advanced_results = run_all_advanced_analysis(df)
    
    # 3. HTML 저장
    save_charts_as_html(dashboard, 'netflix_charts')
    
    # 4. 보고서 내보내기
    export_analysis_report(dashboard, 'netflix_analysis_report.txt')
    
    # 5. 완료
    print("\n단계 5/5: 완료")
    print("\n생성된 파일:")
    print("  - netflix_charts/ (HTML 대시보드)")
    print("  - netflix_analysis_report.txt (텍스트 보고서)")
    print("\n추천 시스템을 사용하시겠습니까? (y/n): ", end='')
    
    choice = input().strip().lower()
    if choice == 'y':
        interactive_recommendation_system(dashboard, df)


def main():
    csv_path = 'netflix_titles.csv'

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    
    df = load_and_preprocess_netflix_data(csv_path)
    
    if df is None:
        print("\n데이터를 불러올 수 없습니다.")
        return
    
    # 실행 모드 선택
    print("\n실행 모드 선택하기:")
    print("1. 인터랙티브 모드 (메뉴에서 원하는 분석 선택)")
    print("2. 자동 모드 (모든 분석 자동 실행)")
    
    try:
        mode = input("\n선택 (1 또는 2, 기본값: 1): ").strip()
        
        if mode == '2':
            auto_mode(df)
        else:
            interactive_mode(df)
    
    except KeyboardInterrupt:
        print("\n\n프로그램을 종료합니다.")
    except Exception as e:
        print(f"\n오류: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n프로그램을 종료합니다. ")


if __name__ == "__main__":
    main()