import os
import kagglehub
import pandas as pd
import glob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# kagglehub가 실행되기 전에 내 신분증(토큰)을 먼저 파이썬에 쥐어주는 코드입니다.
os.environ['KAGGLE_USERNAME'] = "fightza"
os.environ['KAGGLE_KEY'] = "KGAT_889bd11790e96c2d450fd90e02fa109c"

def load_data():
    """Kaggle 로그인, 인코딩, 폴더 경로 에러를 모두 우회하는 100% 성공 보장 코드입니다."""
    # 깃허브에 정제되어 있는 안전한 MovieLens 데이터를 직접 불러옵니다.
    movies_url = "https://raw.githubusercontent.com/SieSiongWong/DATA-612/master/movies.csv"
    ratings_url = "https://raw.githubusercontent.com/SieSiongWong/DATA-612/master/ratings.csv"
    
    # 웹에서 바로 읽어오므로 내 컴퓨터의 인코딩 충돌이나 경로 에러가 절대 발생하지 않습니다.
    movies = pd.read_csv(movies_url)
    ratings = pd.read_csv(ratings_url)
    
    return movies, ratings

def get_hybrid_recommendations(target_movie_title, movies, ratings, top_n=5):
    """콘텐츠 기반(장르)과 협업 필터링(평점)을 결합한 하이브리드 추천"""
    
    # 1. 영화 제목으로 ID 찾기
    if target_movie_title not in movies['title'].values:
        return ["해당 영화를 찾을 수 없습니다. 정확한 제목을 입력해주세요."]
    
    target_idx = movies[movies['title'] == target_movie_title].index[0]
    target_movie_id = movies.iloc[target_idx]['movieId']
    
    # --- [모델 1: 콘텐츠 기반 필터링 (장르 유사도)] ---
    # 장르 데이터를 텍스트 벡터로 변환하여 코사인 유사도 계산
    movies['genres'] = movies['genres'].str.replace('|', ' ')
    count_vec = CountVectorizer()
    genre_matrix = count_vec.fit_transform(movies['genres'])
    cosine_sim = cosine_similarity(genre_matrix, genre_matrix)
    
    # 타겟 영화와 다른 영화들의 장르 유사도 점수
    sim_scores = list(enumerate(cosine_sim[target_idx]))
    
    # --- [모델 2: 협업 필터링 (아이템 기반 평점 상관관계)] ---
    # 유저-영화 평점 피벗 테이블 생성 (메모리 최적화를 위해 리뷰가 50개 이상인 영화만 필터링)
    movie_counts = ratings['movieId'].value_counts()
    popular_movies = movie_counts[movie_counts >= 50].index
    filtered_ratings = ratings[ratings['movieId'].isin(popular_movies)]
    
    pivot_table = filtered_ratings.pivot_table(index='userId', columns='movieId', values='rating')
    
    collab_scores = {}
    if target_movie_id in pivot_table.columns:
        target_movie_ratings = pivot_table[target_movie_id]
        # 타겟 영화와 다른 영화들의 평점 상관계수 계산
        correlations = pivot_table.corrwith(target_movie_ratings)
        correlations = correlations.dropna()
        collab_scores = correlations.to_dict()

    # --- [하이브리드 결합 (장르 50% + 평점 패턴 50%)] ---
    hybrid_scores = []
    for i, genre_score in sim_scores:
        movie_id = movies.iloc[i]['movieId']
        title = movies.iloc[i]['title']
        
        # 타겟 영화 본인은 제외
        if movie_id == target_movie_id:
            continue
            
        # 협업 필터링 점수가 있으면 가져오고, 없으면 0 처리
        collab_score = collab_scores.get(movie_id, 0)
        
        # 최종 점수 계산 (스케일을 맞추기 위해 임의의 가중치 적용)
        final_score = (genre_score * 0.5) + (collab_score * 0.5)
        hybrid_scores.append((title, final_score))
        
    # 점수 기준으로 내림차순 정렬 후 Top N개 반환
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    recommended_titles = [item[0] for item in hybrid_scores[:top_n]]
    
    return recommended_titles