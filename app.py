import streamlit as st
import pandas as pd
import os
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Course Recommendation System",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    file_path = "final_data.csv"

    if not os.path.exists(file_path):
        st.error("❌ final_data.csv file not found.")
        st.stop()

    df = pd.read_csv(file_path)

    # Remove completely empty rows
    df = df.dropna(how="all").reset_index(drop=True)

    # Make sure important columns exist
    required_columns = [
        "Course_ID",
        "Course_Name",
        "Category",
        "Skill_Level",
        "Skills",
        "Interest",
        "Duration_Months",
        "Rating"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        st.error(
            f"❌ Missing columns in dataset: {missing_columns}"
        )
        st.stop()

    # Fill missing values
    text_columns = [
        "Course_Name",
        "Category",
        "Skill_Level",
        "Skills",
        "Interest"
    ]

    for col in text_columns:
        df[col] = df[col].fillna("").astype(str)

    df["Rating"] = pd.to_numeric(
        df["Rating"],
        errors="coerce"
    ).fillna(0)

    df["Duration_Months"] = pd.to_numeric(
        df["Duration_Months"],
        errors="coerce"
    ).fillna(0)

    return df


df = load_data()


# ============================================================
# TITLE
# ============================================================

st.title("🎓 Course Recommendation System")

st.write(
    "Find the most suitable courses based on your "
    "category, skill level, skills and interests."
)


# ============================================================
# CREATE SEARCH TEXT
# ============================================================

df["combined_text"] = (
    df["Course_Name"].astype(str) + " " +
    df["Category"].astype(str) + " " +
    df["Skill_Level"].astype(str) + " " +
    df["Skills"].astype(str) + " " +
    df["Interest"].astype(str)
)


# ============================================================
# TF-IDF
# ============================================================

@st.cache_resource
def create_similarity_matrix(text_data):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    vectors = vectorizer.fit_transform(text_data)

    similarity_matrix = cosine_similarity(vectors)

    return vectorizer, similarity_matrix


vectorizer, similarity_matrix = create_similarity_matrix(
    df["combined_text"].tolist()
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🎯 Select Your Preferences")


categories = sorted(
    df["Category"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

skill_levels = sorted(
    df["Skill_Level"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


selected_category = st.sidebar.selectbox(
    "Category",
    categories
)


selected_skill_level = st.sidebar.selectbox(
    "Skill Level",
    skill_levels
)


selected_skills = st.sidebar.text_input(
    "Skills",
    placeholder="e.g. Python, Machine Learning"
)


selected_interest = st.sidebar.text_input(
    "Interest",
    placeholder="e.g. Artificial Intelligence"
)


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def recommend_courses(
    category,
    skill_level,
    skills,
    interest,
    number_of_courses=5
):

    # --------------------------------------------------------
    # Create user profile
    # --------------------------------------------------------

    user_profile = (
        str(category) + " " +
        str(skill_level) + " " +
        str(skills) + " " +
        str(interest)
    )

    # --------------------------------------------------------
    # Convert user profile to TF-IDF
    # --------------------------------------------------------

    user_vector = vectorizer.transform(
        [user_profile]
    )

    # --------------------------------------------------------
    # Calculate similarity
    # --------------------------------------------------------

    similarity_scores = cosine_similarity(
        user_vector,
        vectorizer.transform(df["combined_text"])
    )[0]

    # --------------------------------------------------------
    # Category matching
    # --------------------------------------------------------

    category_match = (
        df["Category"]
        .str.lower()
        .str.strip()
        .eq(str(category).lower().strip())
        .astype(float)
        .values
    )

    # --------------------------------------------------------
    # Skill level matching
    # --------------------------------------------------------

    skill_match = (
        df["Skill_Level"]
        .str.lower()
        .str.strip()
        .eq(str(skill_level).lower().strip())
        .astype(float)
        .values
    )

    # --------------------------------------------------------
    # Skill matching
    # --------------------------------------------------------

    skill_input = str(skills).lower().strip()

    if skill_input:
        skill_match_score = np.array([
            1.0 if any(
                skill.strip() in row.lower()
                for skill in skill_input.split(",")
                if skill.strip()
            )
            else 0.0
            for row in df["Skills"]
        ])
    else:
        skill_match_score = np.zeros(len(df))

    # --------------------------------------------------------
    # Interest matching
    # --------------------------------------------------------

    interest_input = str(interest).lower().strip()

    if interest_input:

        interest_match_score = np.array([
            1.0 if interest_input in row.lower()
            else 0.0
            for row in df["Interest"]
        ])

    else:
        interest_match_score = np.zeros(len(df))

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    final_score = (
        (similarity_scores * 0.50) +
        (category_match * 0.20) +
        (skill_match * 0.10) +
        (skill_match_score * 0.10) +
        (interest_match_score * 0.10)
    )

    # --------------------------------------------------------
    # Create result dataframe
    # --------------------------------------------------------

    result = df.copy()

    result["match_score"] = final_score

    # --------------------------------------------------------
    # IMPORTANT FIX
    #
    # Sort actual dataframe rows by score.
    # Do NOT use df.iloc[0] repeatedly.
    # --------------------------------------------------------

    result = result.sort_values(
        by="match_score",
        ascending=False
    )

    # --------------------------------------------------------
    # Remove duplicate Course IDs
    # --------------------------------------------------------

    result = result.drop_duplicates(
        subset=["Course_ID"]
    )

    # --------------------------------------------------------
    # Remove duplicate Course Names
    # --------------------------------------------------------

    result = result.drop_duplicates(
        subset=["Course_Name"]
    )

    # --------------------------------------------------------
    # Get top recommendations
    # --------------------------------------------------------

    result = result.head(number_of_courses)

    return result


# ============================================================
# RECOMMEND BUTTON
# ============================================================

if st.sidebar.button(
    "🔍 Recommend Courses",
    use_container_width=True
):

    if not selected_skills and not selected_interest:

        st.warning(
            "⚠️ Please enter at least Skills or Interest."
        )

    else:

        recommendations = recommend_courses(
            selected_category,
            selected_skill_level,
            selected_skills,
            selected_interest,
            5
        )

        st.subheader("🎯 Recommended Courses")

        if recommendations.empty:

            st.warning(
                "No suitable courses found."
            )

        else:

            # =================================================
            # DISPLAY TOP 5
            # =================================================

            for rank, (_, course) in enumerate(
                recommendations.iterrows(),
                start=1
            ):

                score = float(course["match_score"])

                # Keep score between 0 and 1
                score = max(0, min(score, 1))

                match_percentage = round(
                    score * 100,
                    2
                )

                st.markdown(
                    f"""
                    <div style="
                        padding:20px;
                        margin-bottom:18px;
                        border-radius:12px;
                        border:1px solid #ddd;
                        background-color:#ffffff;
                    ">

                    <h3>
                        #{rank} 🎓 {course["Course_Name"]}
                    </h3>

                    <p>
                        <b>Category:</b>
                        {course["Category"]}
                    </p>

                    <p>
                        <b>Skill Level:</b>
                        {course["Skill_Level"]}
                    </p>

                    <p>
                        <b>Skills:</b>
                        {course["Skills"]}
                    </p>

                    <p>
                        <b>Interest:</b>
                        {course["Interest"]}
                    </p>

                    <p>
                        <b>Duration:</b>
                        {course["Duration_Months"]} Months
                    </p>

                    <p>
                        <b>Rating:</b>
                        ⭐ {course["Rating"]}
                    </p>

                    <p>
                        <b>Match:</b>
                        {match_percentage}%
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# INITIAL MESSAGE
# ============================================================

else:

    st.info(
        "👈 Select your preferences from the sidebar "
        "and click **Recommend Courses**."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Course Recommendation System | "
    "TF-IDF + Cosine Similarity"
)