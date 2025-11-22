"""
data_preprocessing.py
=====================
1. CSV 파일 로딩
2. 날짜 데이터 변환 및 정제
3. 결측치 처리
4. 파생 변수 생성 (연도, 월, 시즌 등)
5. Duration 컬럼 분리 (값, 단위)
"""

import pandas as pd
import numpy as np
import re
import warnings
import sys
import io
warnings.filterwarnings('ignore')

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')
    
def load_and_preprocess_netflix_data(csv_path='netflix_titles.csv'):
    """
    Parameters:
    -----------
    Returns:
    --------
    pd.DataFrame - 전처리된 DataFrame 또는 오류 시 None
    
    처리 내용
    ---------
    - date_added를 datetime으로 변환
    - 날짜 정보가 없는 행 제거
    - year_added, month_added, release_decade 생성
    - 결측치를 'Unknown' 또는 빈 문자열로 대체
    - duration을 값(숫자)과 단위(문자)로 분리
    """
    
    try:
        df = pd.read_csv(csv_path)
        print(f" 원본 데이터 shape: {df.shape}")
        print(f"   - 행(Rows): {df.shape[0]:,}개")
        print(f"   - 열(Columns): {df.shape[1]}개")
        
        # 2. date_added 변환 및 NaT 처리
        print("\n 날짜 데이터 처리 중...")
        df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
        
        # 날짜 결측치 확인 및 제거
        before_drop = len(df)
        df = df.dropna(subset=['date_added'])
        dropped_rows = before_drop - len(df)
        
        if dropped_rows > 0:
            print(f"     날짜 없는 행 {dropped_rows}개 제거")
        else:
            print(f"   ✓ 모든 행에 날짜 정보 있음")
        
        # 3. 날짜 관련 파생 변수 생성
        df['year_added'] = df['date_added'].dt.year
        df['month_added'] = df['date_added'].dt.month
        df['release_decade'] = (df['release_year'] // 10) * 10
        print("   ✓ year_added, month_added, release_decade 생성 완료")
        
        # 4. 결측치 처리
        missing_info = []
        
        # country 결측치 처리
        country_missing = df['country'].isna().sum()
        df['country'] = df['country'].fillna('Unknown')
        if country_missing > 0:
            missing_info.append(f"country: {country_missing}개")
        
        # director 결측치 처리
        director_missing = df['director'].isna().sum()
        df['director'] = df['director'].fillna('Unknown')
        if director_missing > 0:
            missing_info.append(f"director: {director_missing}개")
        
        # cast 결측치 처리
        cast_missing = df['cast'].isna().sum()
        df['cast'] = df['cast'].fillna('Unknown')
        if cast_missing > 0:
            missing_info.append(f"cast: {cast_missing}개")
        
        # rating 결측치 처리
        rating_missing = df['rating'].isna().sum()
        df['rating'] = df['rating'].fillna('UNRATED')
        if rating_missing > 0:
            missing_info.append(f"rating: {rating_missing}개")
        
        # description 결측치 처리
        description_missing = df['description'].isna().sum()
        df['description'] = df['description'].fillna('')
        if description_missing > 0:
            missing_info.append(f"description: {description_missing}개")
        
        if missing_info:
            for info in missing_info:
                print(f"      - {info}")
        else:
            print("   ✓ 결측치 없음")
        
        # 5. Duration 컬럼 분리
        # 숫자 추출 (duration_value)
        df['duration_value'] = df['duration'].apply(
            lambda x: int(re.findall(r'\d+', str(x))[0]) 
            if pd.notna(x) and re.findall(r'\d+', str(x)) 
            else np.nan
        )
        
        # 단위 추출 (duration_unit)
        df['duration_unit'] = df['duration'].apply(
            lambda x: re.findall(r'[a-zA-Z]+', str(x))[0].strip() 
            if pd.notna(x) and re.findall(r'[a-zA-Z]+', str(x)) 
            else 'Unknown'
        )
        
        print("   ✓ duration_value, duration_unit 생성 완료")
        
        # 통계
        print(f"  최종 데이터 shape: {df.shape}")
        print(f"   - 행(Rows): {df.shape[0]:,}개")
        print(f"   - 열(Columns): {df.shape[1]}개")
        
        # 데이터 기간 정보
        date_range = f"{df['date_added'].min().strftime('%Y-%m-%d')} ~ {df['date_added'].max().strftime('%Y-%m-%d')}"
        print(f"\n  데이터 기간: {date_range}")
        
        # 콘텐츠 타입 분포
        type_counts = df['type'].value_counts()
        print(f"\n  콘텐츠 타입:")
        for content_type, count in type_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   - {content_type}: {count:,}개 ({percentage:.1f}%)")
        
        return df
        
    except FileNotFoundError:
        print(f" 오류: 파일 '{csv_path}'을(를) 찾을 수 없습니다.")
        print(f"   현재 작업 디렉토리를 확인하세요.")
        return None
        
    except Exception as e:
        print(f" 데이터 로딩 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_data_summary(df):
    if df is None or df.empty:
        print(" 유효한 데이터가 없습니다.")
        return
    print(" 데이터 요약 정보")

    print(f"\n 기본 정보")
    print(f"   • 전체 행 수: {len(df):,}")
    print(f"   • 전체 열 수: {len(df.columns)}")
    print(f"   • 메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 컬럼
    print(f"\n 컬럼 목록 ({len(df.columns)}개)")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        null_count = df[col].isna().sum()
        print(f"   {i:2d}. {col:20s} | {str(dtype):10s} | Non-Null: {non_null:5,} | Null: {null_count:5,}")
    
    # 중복 데이터
    duplicates = df.duplicated().sum()
    print(f"\n  중복 데이터")
    print(f"   • 중복 행 수: {duplicates:,}개")
    
    # 수치형 컬럼 통계
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n 수치형 컬럼 통계")
        print(df[numeric_cols].describe())
    
    print("="*60 + "\n")


def validate_data(df):
    """
    데이터 유효성을 검증합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        검증할 DataFrame
        
    Returns:
    --------
    dict
        검증 결과 딕셔너리
    """
    
    validation_results = {
        'is_valid': True,
        'issues': []
    }
    
    if df is None or df.empty:
        validation_results['is_valid'] = False
        validation_results['issues'].append("데이터가 비어있습니다.")
        return validation_results
    
    # 필수 컬럼 확인
    required_columns = ['show_id', 'type', 'title', 'date_added', 'release_year']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        validation_results['is_valid'] = False
        validation_results['issues'].append(f"필수 컬럼 누락: {missing_columns}")
    
    # 날짜 형식 확인
    if 'date_added' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['date_added']):
            validation_results['issues'].append("date_added가 datetime 형식이 아닙니다.")
    
    # 콘텐츠 타입 확인
    if 'type' in df.columns:
        valid_types = ['Movie', 'TV Show']
        invalid_types = df[~df['type'].isin(valid_types)]['type'].unique()
        if len(invalid_types) > 0:
            validation_results['issues'].append(f"유효하지 않은 콘텐츠 타입: {invalid_types}")
    
    if validation_results['is_valid'] and len(validation_results['issues']) == 0:
        print("  데이터가 유효합니다.")
    else:
        print("  문제:")
        for issue in validation_results['issues']:
            print(f"   - {issue}")
    print("="*50 + "\n")
    
    return validation_results


# 모듈 테스트 코드
if __name__ == "__main__":
    df = load_and_preprocess_netflix_data('netflix_titles.csv')
    
    if df is not None:
        get_data_summary(df)
        validate_data(df)
        print(" 모듈 테스트 완료함")
    else:
        print(" 데이터를 로드할 수 없음.")
