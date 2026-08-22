import streamlit as st
import pandas as pd
import numpy as np
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Course Recommendation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #f6f8fc;
    }

    /* Main content */
    .main .block-container {
        max-width: 1400px;
        padding-top: 30px;
        padding-bottom: 50px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #eef2ff;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        min-height: 45px;
        font-weight: 700;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: white;
        padding: 18px;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05);
    }

    /* Expander */
    [data-testid="stExpander"] {
        background-color: white;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
    }

    /* Divider */
    hr {
        margin-top: 25px;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "final_data.csv"
    )

    if not os.path.exists(file_path):

        st.error(
            "❌ final_data.csv file not found."
        )

        st.info(
            "Keep final_data.csv in the same folder as app.py."
        )

        st.stop()

    data = pd.read_csv(file_path)

    data.columns = (
        data.columns
        .astype(str)
        .str.strip()
    )

    return data


df = load_data()


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Course_ID",
    "Course_Name",
    "Category",
    "Skill_Level",
    "Skills",
    "Interest",
    "Education",
    "Career_Goal",
    "Duration_Months",
    "Rating",
    "Difficulty",
    "Job_Role",
    "Salary_Range"
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    st.error(
        "❌ Missing columns in final_data.csv:"
    )

    st.write(missing_columns)

    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

for column in required_columns:

    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )


df["Rating"] = pd.to_numeric(
    df["Rating"],
    errors="coerce"
).fillna(0)


df["Duration_Months"] = pd.to_numeric(
    df["Duration_Months"],
    errors="coerce"
).fillna(0)


# Remove empty course names

df = df[
    df["Course_Name"].str.strip() != ""
].copy()


df = df.reset_index(drop=True)


# ============================================================
# CREATE COURSE TEXT
# ============================================================

df["course_text"] = (
    df["Course_Name"] + " "
    + df["Category"] + " "
    + df["Skill_Level"] + " "
    + df["Skills"] + " "
    + df["Interest"] + " "
    + df["Education"] + " "
    + df["Career_Goal"] + " "
    + df["Difficulty"] + " "
    + df["Job_Role"]
)


# ============================================================
# TF-IDF MODEL
# ============================================================

@st.cache_resource
def create_tfidf(text_data):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000
    )

    matrix = vectorizer.fit_transform(
        text_data
    )

    return vectorizer, matrix


vectorizer, course_matrix = create_tfidf(
    df["course_text"].tolist()
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎯 Select Your Preferences")

    st.caption(
        "Tell us about your learning preferences "
        "to find suitable courses."
    )

    st.divider()

    category_options = sorted(
        df["Category"].unique().tolist()
    )

    skill_options = sorted(
        df["Skill_Level"].unique().tolist()
    )

    education_options = sorted(
        df["Education"].unique().tolist()
    )

    career_options = sorted(
        df["Career_Goal"].unique().tolist()
    )

    difficulty_options = sorted(
        df["Difficulty"].unique().tolist()
    )


    category = st.selectbox(
        "📚 Category",
        category_options
    )


    skill_level = st.selectbox(
        "📈 Skill Level",
        skill_options
    )


    skills = st.text_input(
        "🛠️ Skills",
        placeholder="Python, Machine Learning..."
    )


    interest = st.text_input(
        "💡 Interest",
        placeholder="Artificial Intelligence..."
    )


    education = st.selectbox(
        "🎓 Education",
        education_options
    )


    career_goal = st.selectbox(
        "🎯 Career Goal",
        career_options
    )


    difficulty = st.selectbox(
        "⚡ Difficulty",
        difficulty_options
    )


    st.divider()


    recommend_button = st.button(
        "🚀 Find My Courses",
        type="primary",
        use_container_width=True
    )


# ============================================================
# HEADER
# ============================================================

st.title("🎓 Course Recommendation System")

st.write(
    "Find the most suitable courses based on your "
    "category, skill level, skills, interests and career goals."
)

st.caption(
    "🤖 AI-Powered Recommendation  •  "
    "TF-IDF  •  Cosine Similarity"
)


st.divider()


# ============================================================
# PLATFORM OVERVIEW
# ============================================================

st.subheader("📊 Course Platform Overview")


stat1, stat2, stat3, stat4 = st.columns(4)


with stat1:

    st.metric(
        "📚 Total Courses",
        len(df)
    )


with stat2:

    st.metric(
        "🏷️ Categories",
        df["Category"].nunique()
    )


with stat3:

    st.metric(
        "⭐ Average Rating",
        f'{df["Rating"].mean():.1f}/5'
    )


with stat4:

    st.metric(
        "🎯 Skill Levels",
        df["Skill_Level"].nunique()
    )


# ============================================================
# KEYWORD MATCH FUNCTION
# ============================================================

def keyword_match(
    column_data,
    user_input
):

    user_input = str(
        user_input
    ).lower().strip()


    if not user_input:

        return np.zeros(
            len(column_data)
        )


    keywords = [
        word.strip()
        for word in re.split(
            r",|;|\n",
            user_input
        )
        if word.strip()
    ]


    scores = []


    for value in column_data:

        text = str(value).lower()

        matched = 0


        for keyword in keywords:

            if keyword in text:

                matched += 1


        if len(keywords) > 0:

            score = (
                matched / len(keywords)
            )

        else:

            score = 0


        scores.append(score)


    return np.array(scores)


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def get_recommendations():

    # --------------------------------------------------------
    # USER PROFILE
    # --------------------------------------------------------

    user_profile = " ".join(
        [
            category,
            skill_level,
            skills,
            interest,
            education,
            career_goal,
            difficulty
        ]
    )


    # --------------------------------------------------------
    # TF-IDF SIMILARITY
    # --------------------------------------------------------

    user_vector = vectorizer.transform(
        [user_profile]
    )


    similarity = cosine_similarity(
        user_vector,
        course_matrix
    )[0]


    # --------------------------------------------------------
    # CATEGORY MATCH
    # --------------------------------------------------------

    category_match = (
        df["Category"]
        .str.lower()
        .eq(category.lower())
        .astype(float)
        .values
    )


    # --------------------------------------------------------
    # SKILL LEVEL MATCH
    # --------------------------------------------------------

    level_match = (
        df["Skill_Level"]
        .str.lower()
        .eq(skill_level.lower())
        .astype(float)
        .values
    )


    # --------------------------------------------------------
    # EDUCATION MATCH
    # --------------------------------------------------------

    education_match = (
        df["Education"]
        .str.lower()
        .eq(education.lower())
        .astype(float)
        .values
    )


    # --------------------------------------------------------
    # CAREER MATCH
    # --------------------------------------------------------

    career_match = (
        df["Career_Goal"]
        .str.lower()
        .eq(career_goal.lower())
        .astype(float)
        .values
    )


    # --------------------------------------------------------
    # DIFFICULTY MATCH
    # --------------------------------------------------------

    difficulty_match = (
        df["Difficulty"]
        .str.lower()
        .eq(difficulty.lower())
        .astype(float)
        .values
    )


    # --------------------------------------------------------
    # SKILLS MATCH
    # --------------------------------------------------------

    skills_match = keyword_match(
        df["Skills"],
        skills
    )


    # --------------------------------------------------------
    # INTEREST MATCH
    # --------------------------------------------------------

    interest_match = keyword_match(
        df["Interest"],
        interest
    )


    # ========================================================
    # FINAL SCORE
    # ========================================================

    final_score = (

        similarity * 0.35

        + category_match * 0.18

        + level_match * 0.10

        + skills_match * 0.15

        + interest_match * 0.10

        + education_match * 0.04

        + career_match * 0.05

        + difficulty_match * 0.03

    )


    # ========================================================
    # RESULT
    # ========================================================

    result = df.copy()


    result["Match_Score"] = (
        final_score * 100
    )


    # Sort by match score

    result = result.sort_values(
        by=[
            "Match_Score",
            "Rating"
        ],
        ascending=[
            False,
            False
        ]
    )


    # ========================================================
    # REMOVE DUPLICATE COURSE NAMES
    # ========================================================

    result["Course_Key"] = (
        result["Course_Name"]
        .str.lower()
        .str.strip()
    )


    result = result.drop_duplicates(
        subset="Course_Key",
        keep="first"
    )


    # ========================================================
    # TOP 5
    # ========================================================

    result = result.head(5)

    result = result.reset_index(
        drop=True
    )


    return result


# ============================================================
# RECOMMEND COURSES
# ============================================================

if recommend_button:

    recommendations = get_recommendations()


    if len(recommendations) == 0:

        st.warning(
            "⚠️ No suitable courses found."
        )


    else:

        st.divider()

        st.subheader(
            "🏆 Your Personalized Recommendations"
        )

        st.caption(
            "Top 5 courses selected according to your preferences."
        )


        # ====================================================
        # SHOW 5 COURSES
        # ====================================================

        for index, course in recommendations.iterrows():

            rank = index + 1


            # ------------------------------------------------
            # MATCH SCORE
            # ------------------------------------------------

            match_score = float(
                course["Match_Score"]
            )


            match_score = max(
                0,
                min(
                    100,
                    round(match_score)
                )
            )


            # ------------------------------------------------
            # RANK TITLE
            # ------------------------------------------------

            if rank == 1:

                st.success(
                    f"🥇 #1 BEST MATCH  —  "
                    f"{course['Course_Name']}"
                )

            else:

                st.info(
                    f"#{rank} RECOMMENDATION  —  "
                    f"{course['Course_Name']}"
                )


            # ------------------------------------------------
            # COURSE INFORMATION
            # ------------------------------------------------

            st.subheader(
                f"🎓 {course['Course_Name']}"
            )


            info1, info2, info3 = st.columns(3)


            with info1:

                st.write(
                    f"**📚 Category:** "
                    f"{course['Category']}"
                )

                st.write(
                    f"**📈 Skill Level:** "
                    f"{course['Skill_Level']}"
                )


            with info2:

                st.write(
                    f"**🛠️ Skills:** "
                    f"{course['Skills']}"
                )

                st.write(
                    f"**💡 Interest:** "
                    f"{course['Interest']}"
                )


            with info3:

                st.write(
                    f"**🎯 Match:** "
                    f"{match_score}%"
                )

                st.write(
                    f"**⭐ Rating:** "
                    f"{course['Rating']}/5"
                )


            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            try:

                duration_value = float(
                    course["Duration_Months"]
                )

                duration_text = (
                    f"{duration_value:g} Months"
                )

            except:

                duration_text = str(
                    course["Duration_Months"]
                )


            m1, m2, m3, m4 = st.columns(4)


            with m1:

                st.metric(
                    "⭐ Rating",
                    f'{course["Rating"]}/5'
                )


            with m2:

                st.metric(
                    "⏱️ Duration",
                    duration_text
                )


            with m3:

                st.metric(
                    "📈 Level",
                    str(course["Skill_Level"])
                )


            with m4:

                job_role = str(
                    course["Job_Role"]
                )

                st.metric(
                    "💼 Job Role",
                    job_role
                )


            # ------------------------------------------------
            # MATCH PROGRESS
            # ------------------------------------------------

            st.write(
                f"**🎯 Course Match: {match_score}%**"
            )

            st.progress(
                match_score / 100
            )


            # ------------------------------------------------
            # DETAILS
            # ------------------------------------------------

            with st.expander(
                f"📋 View details — {course['Course_Name']}"
            ):

                detail1, detail2 = st.columns(2)


                with detail1:

                    st.markdown(
                        "### 💡 Interest"
                    )

                    st.write(
                        course["Interest"]
                    )


                    st.markdown(
                        "### 🎯 Career Goal"
                    )

                    st.write(
                        course["Career_Goal"]
                    )


                    st.markdown(
                        "### 🎓 Education"
                    )

                    st.write(
                        course["Education"]
                    )


                with detail2:

                    st.markdown(
                        "### ⚡ Difficulty"
                    )

                    st.write(
                        course["Difficulty"]
                    )


                    st.markdown(
                        "### 💼 Job Role"
                    )

                    st.write(
                        course["Job_Role"]
                    )


                    st.markdown(
                        "### 💰 Salary Range"
                    )

                    st.write(
                        course["Salary_Range"]
                    )


            st.divider()


else:

    # ========================================================
    # WELCOME MESSAGE
    # ========================================================

    st.info(
        "🎯 Select your preferences from the left sidebar "
        "and click **🚀 Find My Courses** to get your "
        "personalized top 5 course recommendations."
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "🎓 Course Recommendation System  •  "
    "TF-IDF + Cosine Similarity  •  "
    "Personalized Learning"
)
