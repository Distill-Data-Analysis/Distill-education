import streamlit as st
from recommender import load_data, get_hybrid_recommendations

# 페이지 기본 설정
st.set_page_config(page_title="AI 영화 추천 프로그램", page_icon="🎬", layout="wide")

# UI: 제목 및 설명
st.title("🎬 AI 영화 추천 프로그램")
st.markdown("넷플릭스 알고리즘의 핵심인 '콘텐츠 기반 필터링'과 '협업 필터링'을 결합하여 나만의 영화를 추천해 드립니다.")
st.divider()

# 데이터 로딩 (st.cache_data를 사용해 매번 다운로드하지 않도록 캐싱)
@st.cache_data
def fetch_data():
    return load_data()

try:
    with st.spinner("영화 데이터베이스를 불러오는 중입니다... (최초 1회만 소요)"):
        movies, ratings = fetch_data()
    
    # UI: 영화 선택 창
    movie_list = movies['title'].tolist()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("재밌게 본 영화를 선택하세요")
        selected_movie = st.selectbox("검색하거나 목록에서 고를 수 있습니다.", movie_list)
        
    with col2:
        st.subheader("추천받을 개수")
        top_n = st.slider("몇 개의 영화를 추천받으시겠어요?", min_value=1, max_value=10, value=5)

    # 추천 실행 버튼
    if st.button("나만의 맞춤 영화 추천받기", type="primary"):
        with st.spinner("수만 개의 데이터를 분석하여 추천 알고리즘을 가동 중입니다..."):
            recommendations = get_hybrid_recommendations(selected_movie, movies, ratings, top_n)
            
            st.success(f"'{selected_movie}'와(과) 비슷한 느낌의 영화를 찾았습니다!")
            
            # 추천 결과 예쁘게 출력
            for i, rec_movie in enumerate(recommendations):
                st.info(f"**{i+1}. {rec_movie}**")

except Exception as e:
    st.error(f"데이터 로딩 중 오류가 발생했습니다: {e}")