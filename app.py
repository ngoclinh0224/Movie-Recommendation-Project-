import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sentence_transformers import util
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import rankdata

# ==========================================
# CONFIG
# ==========================================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

USER_ID = 253
ALPHA = 0.5
TOP_K = 10

# ==========================================
# CSS - NETFLIX STYLE
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #141414;
}

h1 {
    color: #E50914 !important;
    text-align: center;
}

h3 {
    color: white !important;
}

p {
    color: white;
}

.stTextInput > div > div > input {
    background-color: #222222;
    color: white;
    border-radius: 10px;
    border: 1px solid #444444;
}

.stButton button {
    background-color: #E50914;
    color: white;
    border-radius: 10px;
    border: none;
    width: 100%;
    font-weight: bold;
    height: 50px;
}

.stButton button:hover {
    background-color: #ff1e28;
}

.movie-card {
    background-color: #222222;
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    color: white;
    min-height: 140px;
    margin-bottom: 15px;
}

.movie-title {
    font-size: 18px;
    font-weight: bold;
}

.movie-score {
    color: #E50914;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODELS
# ==========================================

@st.cache_resource
def load_models():

    with open("movie_embeddings.pkl", "rb") as f:
        movie_embeddings = pickle.load(f)

    with open("als_model.pkl", "rb") as f:
        als_model = pickle.load(f)

    with open("user_encoder.pkl", "rb") as f:
        user_encoder = pickle.load(f)

    with open("movie_encoder.pkl", "rb") as f:
        movie_encoder = pickle.load(f)

    with open("user_item_matrix.pkl", "rb") as f:
        user_item_matrix = pickle.load(f)

    return (
        movie_embeddings,
        als_model,
        user_encoder,
        movie_encoder,
        user_item_matrix
    )

# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():

    df_final = pd.read_csv("df_final.csv")

    mapping_bridge = pd.read_csv(
        "movie_mapping.csv"
    )

    return df_final, mapping_bridge

(
    movie_embeddings,
    als_model,
    user_encoder,
    movie_encoder,
    user_item_matrix
) = load_models()

df_final, mapping_bridge = load_data()

df_final["id"] = df_final["id"].astype(str)

# ==========================================
# TITLE -> ID
# ==========================================

title_to_id = dict(
    zip(
        df_final["title"].str.lower(),
        df_final["id"]
    )
)

# ==========================================
# HEADER
# ==========================================

st.markdown("""
<h1>🎬 MOVIE RECOMMENDER</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align:center'>
Hybrid Recommendation System using BERT + ALS
</p>
""", unsafe_allow_html=True)

st.divider()

# ==========================================
# INPUT
# ==========================================

movie_name = st.text_input(
    "🔍 Nhập tên phim",
    placeholder="Ví dụ: Interstellar"
)

# ==========================================
# RECOMMEND
# ==========================================

if st.button("Recommend"):

    if movie_name.lower() not in title_to_id:

        st.error("❌ Không tìm thấy phim trong dữ liệu.")

    else:

        sample_seed_movie = (
            title_to_id[movie_name.lower()]
        )

        # ==========================
        # BERT
        # ==========================

        matched = df_final[
            df_final["id"]
            ==
            str(sample_seed_movie)
        ]

        seed_idx = matched.index[0]

        bert_scores = util.cos_sim(
            movie_embeddings[seed_idx],
            movie_embeddings
        )[0].cpu().numpy()

        bert_scores[seed_idx] = -1

        bert_scores_scaled = (
            MinMaxScaler()
            .fit_transform(
                bert_scores.reshape(-1, 1)
            )
            .flatten()
        )

        # ==========================
        # ALS
        # ==========================

        user_idx = (
            user_encoder
            .transform([USER_ID])[0]
        )

        user_items = (
            user_item_matrix[user_idx]
        )

        ids, als_scores_raw = (
            als_model.recommend(
                userid=user_idx,
                user_items=user_items,
                N=len(movie_encoder.classes_),
                filter_already_liked_items=False
            )
        )

        als_scores_ordered = np.zeros(
            len(df_final)
        )

        movieid_to_tmdb = dict(
            zip(
                mapping_bridge["movie_id"],
                mapping_bridge["id"]
            )
        )

        tmdb_to_dfidx = dict(
            zip(
                df_final["id"],
                df_final.index
            )
        )

        alsidx_to_movieid = dict(
            zip(
                movie_encoder.transform(
                    movie_encoder.classes_
                ),
                movie_encoder.classes_
            )
        )

        for als_idx, score in zip(
            ids,
            als_scores_raw
        ):

            movie_id = (
                alsidx_to_movieid.get(
                    als_idx
                )
            )

            if movie_id is None:
                continue

            tmdb_id = (
                movieid_to_tmdb.get(
                    movie_id
                )
            )

            if tmdb_id is None:
                continue

            df_idx = (
                tmdb_to_dfidx.get(
                    str(tmdb_id)
                )
            )

            if df_idx is None:
                continue

            als_scores_ordered[
                df_idx
            ] = score

        als_scores_scaled = (
            MinMaxScaler()
            .fit_transform(
                als_scores_ordered.reshape(-1, 1)
            )
            .flatten()
        )

        # ==========================
        # HYBRID
        # ==========================

        bert_percentile = (
            rankdata(
                bert_scores_scaled
            )
            /
            len(bert_scores_scaled)
        )

        als_percentile = (
            rankdata(
                als_scores_scaled
            )
            /
            len(als_scores_scaled)
        )

        hybrid_scores = (
            ALPHA * bert_percentile
            +
            (1 - ALPHA)
            * als_percentile
        )

        # ==========================
        # TOP K
        # ==========================

        top_indices = np.argsort(
            hybrid_scores
        )[::-1][:TOP_K]

        recommendations = (
            df_final.iloc[
                top_indices
            ][
                ["title"]
            ]
            .copy()
        )

        recommendations[
            "Score"
        ] = hybrid_scores[
            top_indices
        ]

        # ==========================
        # SHOW RESULT
        # ==========================
        import requests

        TMDB_API_KEY = "ed015a6c90538af3c627e2875d2e936a"

        @st.cache_data
        def get_movie_poster(movie_title):

            try:

                url = (
                    f"https://api.themoviedb.org/3/search/movie"
                    f"?api_key={TMDB_API_KEY}"
                    f"&query={movie_title}")

                response = requests.get(url)

                data = response.json()

                if len(data["results"]) > 0:

                    poster_path = (
                        data["results"][0]
                        .get("poster_path"))

                    if poster_path:

                        return ("https://image.tmdb.org/t/p/w500" + poster_path)

            except:
                pass

            return None
        

        recommendations = (
            recommendations
            .sort_values(
                by="Score",
                ascending=False
            )
            .reset_index(drop=True)
        )

        st.markdown(
            f"""
            <h3>
            🎬 Top {TOP_K} Recommendations
            </h3>
            """,
            unsafe_allow_html=True
        )

        NUM_COLS = 4

        for idx, row in recommendations.iterrows():

            if idx % NUM_COLS == 0:
                cols = st.columns(NUM_COLS)

            with cols[idx % NUM_COLS]:

                poster_url = get_movie_poster(
                    row["title"]
                )

                if poster_url:
                    st.image(
                        poster_url,
                        use_container_width=True
                    )

                st.markdown(
                    f"""
                    <div style='
                        text-align:center;
                        color:white;
                        font-size:22px;
                        font-weight:bold;
                        margin-top:10px;
                        margin-bottom:25px;
                    '>
                        {row['title']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )