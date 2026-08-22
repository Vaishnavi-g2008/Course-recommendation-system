import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import textwrap

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
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #f6f8fc;
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #eef2ff 0%,
        #f8fafc 100%
    );
    border-right: 1px solid #e2e8f0;
}

.sidebar-title {
    font-size: 24px;
    font-weight: 800;
    color: #1e293b;
    margin-bottom: 6px;
}

.sidebar-subtitle {
    font-size: 13px;
    color: #64748b;
    margin-bottom: 25px;
    line-height: 1.5;
}

/* ================= HERO ================= */

.hero {
    background: linear-gradient(
        135deg,
        #1e3a8a 0%,
        #4338ca 50%,
        #6366f1 100%
    );
    padding: 38px 42px;
    border-radius: 24px;
    margin-bottom: 28px;
    box-shadow: 0 15px 35px rgba(30, 58, 138, 0.20);
}

.hero-title {
    color: white;
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1px;
}

.hero-subtitle {
    color: #e0e7ff;
    font-size: 17px;
    margin-top: 10px;
    line-height: 1.6;
}

.hero-tag {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    color: white;
    padding: 7px 14px;
    border-radius: 50px;
    font-size: 13px;
    font-weight: 700;
    margin-top: 18px;
    border: 1px solid rgba(255,255,255,0.2);
}

/* ================= SECTION ================= */

.section-title {
    font-size: 25px;
    font-weight: 800;
    color: #172033;
    margin-top: 28px;
    margin-bottom: 15px;
}

/* ================= STAT CARDS ================= */

.stat-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 6px 18px rgba(15,23,42,0.05);
}

.stat-icon {
    font-size: 25px;
}

.stat-label {
    color: #64748b;
    font-size: 12px;
    font-weight: 700;
    margin-top: 5px;
}

.stat-value {
    color: #172033;
    font-size: 25px;
    font-weight: 800;
    margin-top: 3px;
}

/* ================= RESULT HEADER ================= */

.result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: white;
    padding: 20px 24px;
    border-radius: 18px;
    border: 1px solid #e2e8f0;
    margin-top: 30px;
    margin-bottom: 20px;
}

.result-title {
    font-size: 24px;
    font-weight: 800;
    color: #172033;
}

.result-count {
    background: #eef2ff;
    color: #4338ca;
    padding: 8px 15px;
    border-radius: 50px;
    font-size: 13px;
    font-weight: 800;
}

/* ================= COURSE CARD ================= */

.course-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 22px;
    padding: 28px;
    margin-bottom: 16px;
    box-shadow: 0 8px 25px rgba(15,23,42,0.06);
}

.course-card-best {
    background: linear-gradient(
        135deg,
        #ffffff,
        #f5f7ff
    );
    border: 2px solid #c7d2fe;
    box-shadow: 0 12px 32px rgba(67,56,202,0.10);
}

.rank {
    display: inline-block;
    background: #eef2ff;
    color: #4338ca;
    padding: 6px 12px;
    border-radius: 50px;
    font-size: 13px;
    font-weight: 800;
    margin-bottom: 12px;
}

.best-rank {
    background: #fef3c7;
    color: #92400e;
}

.course-title {
    color: #172033;
    font-size: 27px;
    font-weight: 800;
    margin-bottom: 12px;
}

.course-category {
    display: inline-block;
    background: #f1f5f9;
    color: #334155;
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 18px;
}

.match-badge {
    background: #ecfdf5;
    color: #047857;
    padding: 8px 14px;
    border-radius: 50px;
    font-size: 14px;
    font-weight: 800;
}

.detail-label {
    color: #64748b;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.detail-value {
    color: #1e293b;
    font-size: 14px;
    line-height: 1.5;
}

.skill-box {
    background: #f8fafc;
    border-radius: 12px;
    padding: 13px;
    margin-top: 12px;
    border: 1px solid #e2e8f0;
}

/* ================= FOOTER ================= */

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 13px;
    padding: 35px 0 10px;
}

</style>
""", unsafe_allow_html=True)


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
            "❌ final_data.csv not found. "
            "Please keep final_data.csv in the same folder as app.py."
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
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        f"❌ Missing columns in final_data.csv: {missing_columns}"
    )

    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

for col in required_columns:

    df[col] = (
        df[col]
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
# COURSE TEXT FOR NLP
# ============================================================

df["course_text"] = (
    df["Course_Name"] + " " +
    df["Category"] + " " +
    df["Skill_Level"] + " " +
    df["Skills"] + " " +
    df["Interest"] + " " +
    df["Education"] + " " +
    df["Career_Goal"] + " " +
    df["Difficulty"] + " " +
    df["Job_Role"]
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

    matrix = vectorizer.fit_transform(text_data)

    return vectorizer, matrix


vectorizer, course_matrix = create_tfidf(
    df["course_text"].tolist()
)


# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">
        🎓 Course Recommendation System
    </div>

    <div class="hero-subtitle">
        Discover courses that match your skills, interests,
        education and career goals.
    </div>

    <div class="hero-tag">
        🤖 AI-Powered &nbsp; • &nbsp; TF-IDF &nbsp; • &nbsp; Cosine Similarity
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# PLATFORM STATISTICS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Course Platform Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-icon">📚</div>
            <div class="stat-label">TOTAL COURSES</div>
            <div class="stat-value">{len(df)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-icon">🏷️</div>
            <div class="stat-label">CATEGORIES</div>
            <div class="stat-value">{df["Category"].nunique()}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-icon">⭐</div>
            <div class="stat-label">AVERAGE RATING</div>
            <div class="stat-value">{df["Rating"].mean():.1f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-icon">🎯</div>
            <div class="stat-label">SKILL LEVELS</div>
            <div class="stat-value">{df["Skill_Level"].nunique()}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🎯 Your Learning Profile</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'Select your preferences and find the most suitable '
        'courses for your learning journey.'
        '</div>',
        unsafe_allow_html=True
    )


    category_options = sorted(
        df["Category"].unique().tolist()
    )

    skill_level_options = sorted(
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
        skill_level_options
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


    recommend_button = st.button(
        "🚀 Find My Courses",
        use_container_width=True
    )


# ============================================================
# TEXT MATCH FUNCTION
# ============================================================

def calculate_text_match(series, user_input):

    user_input = str(user_input).lower().strip()

    if not user_input:

        return np.zeros(len(series))


    keywords = [
        item.strip()
        for item in re.split(
            r",|;|\n",
            user_input
        )
        if item.strip()
    ]


    scores = []


    for value in series:

        text = str(value).lower()

        matched = 0

        for keyword in keywords:

            if keyword in text:

                matched += 1


        if len(keywords) > 0:

            score = matched / len(keywords)

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

    user_profile = " ".join([
        category,
        skill_level,
        skills,
        interest,
        education,
        career_goal,
        difficulty
    ])


    # --------------------------------------------------------
    # USER VECTOR
    # --------------------------------------------------------

    user_vector = vectorizer.transform(
        [user_profile]
    )


    # --------------------------------------------------------
    # COSINE SIMILARITY
    # --------------------------------------------------------

    similarity_scores = cosine_similarity(
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

    skills_match = calculate_text_match(
        df["Skills"],
        skills
    )


    # --------------------------------------------------------
    # INTEREST MATCH
    # --------------------------------------------------------

    interest_match = calculate_text_match(
        df["Interest"],
        interest
    )


    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    final_score = (
        similarity_scores * 0.35
        +
        category_match * 0.18
        +
        level_match * 0.10
        +
        skills_match * 0.15
        +
        interest_match * 0.10
        +
        education_match * 0.04
        +
        career_match * 0.05
        +
        difficulty_match * 0.03
    )


    # --------------------------------------------------------
    # RESULT DATAFRAME
    # --------------------------------------------------------

    result = df.copy()

    result["Match_Score"] = (
        final_score * 100
    )


    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

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


    result = result.reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # REMOVE DUPLICATE COURSE NAMES
    # --------------------------------------------------------

    selected_courses = []

    used_names = set()


    for _, row in result.iterrows():

        course_name = (
            str(row["Course_Name"])
            .strip()
            .lower()
        )


        if course_name in used_names:

            continue


        used_names.add(
            course_name
        )


        selected_courses.append(
            row
        )


        if len(selected_courses) == 5:

            break


    # --------------------------------------------------------
    # FINAL TOP 5
    # --------------------------------------------------------

    if len(selected_courses) == 0:

        return pd.DataFrame()


    recommendations = pd.DataFrame(
        selected_courses
    )


    recommendations = recommendations.reset_index(
        drop=True
    )


    return recommendations


# ============================================================
# SHOW RECOMMENDATIONS
# ============================================================

if recommend_button:

    recommendations = get_recommendations()


    if recommendations.empty:

        st.warning(
            "⚠️ No suitable courses found."
        )


    else:

        # ====================================================
        # RESULT HEADER
        # ====================================================

        st.markdown("""
        <div class="result-header">

            <div class="result-title">
                🏆 Your Personalized Recommendations
            </div>

            <div class="result-count">
                TOP 5 COURSES
            </div>

        </div>
        """, unsafe_allow_html=True)


        # ====================================================
        # COURSE CARDS
        # ====================================================

        for rank, (_, course) in enumerate(
            recommendations.iterrows(),
            start=1
        ):

            score = float(
                course["Match_Score"]
            )


            score = max(
                0,
                min(
                    100,
                    round(score)
                )
            )


            # Duration

            try:

                duration_value = float(
                    course["Duration_Months"]
                )

                duration_text = (
                    f"{duration_value:g} Months"
                )

            except:

                duration_text = (
                    str(course["Duration_Months"])
                )


            # First course styling

            if rank == 1:

                card_class = (
                    "course-card course-card-best"
                )

                rank_class = (
                    "rank best-rank"
                )

                rank_text = (
                    "🥇 BEST MATCH"
                )

            else:

                card_class = (
                    "course-card"
                )

                rank_class = (
                    "rank"
                )

                rank_text = (
                    f"#{rank} RECOMMENDATION"
                )


            # ------------------------------------------------
            # HTML CARD
            # ------------------------------------------------

            card_html = textwrap.dedent(
                f"""
                <div class="{card_class}">

                    <div class="{rank_class}">
                        {rank_text}
                    </div>

                    <div style="
                        display:flex;
                        justify-content:space-between;
                        align-items:center;
                        gap:20px;
                        flex-wrap:wrap;
                    ">

                        <div class="course-title">
                            🎓 {course["Course_Name"]}
                        </div>

                        <div class="match-badge">
                            🎯 {score}% Match
                        </div>

                    </div>

                    <div class="course-category">
                        📚 {course["Category"]}
                    </div>

                    <div class="skill-box">

                        <div class="detail-label">
                            🛠️ Skills
                        </div>

                        <div class="detail-value">
                            {course["Skills"]}
                        </div>

                    </div>

                </div>
                """
            )


            st.markdown(
                card_html,
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # COURSE METRICS
            # ------------------------------------------------

            metric1, metric2, metric3, metric4 = st.columns(4)


            with metric1:

                st.metric(
                    "⭐ Rating",
                    f'{course["Rating"]}/5'
                )


            with metric2:

                st.metric(
                    "⏱️ Duration",
                    duration_text
                )


            with metric3:

                st.metric(
                    "📈 Level",
                    str(course["Skill_Level"])
                )


            with metric4:

                job_role = str(
                    course["Job_Role"]
                )

                if len(job_role) > 18:

                    job_role = (
                        job_role[:18] + "..."
                    )


                st.metric(
                    "💼 Job Role",
                    job_role
                )


            # ------------------------------------------------
            # COURSE DETAILS
            # ------------------------------------------------

            with st.expander(
                f'📋 View details — {course["Course_Name"]}'
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


            st.markdown(
                "<div style='height:12px'></div>",
                unsafe_allow_html=True
            )


else:

    # ========================================================
    # INITIAL SCREEN
    # ========================================================

    st.markdown("""
    <div style="
        background:white;
        border:1px solid #e2e8f0;
        border-radius:22px;
        padding:55px;
        text-align:center;
        margin-top:30px;
        box-shadow:0 8px 25px rgba(15,23,42,0.05);
    ">

        <div style="font-size:55px;">
            🎯
        </div>

        <div style="
            font-size:27px;
            font-weight:800;
            color:#172033;
            margin-top:10px;
        ">
            Find Your Perfect Course
        </div>

        <div style="
            color:#64748b;
            font-size:15px;
            margin-top:10px;
            line-height:1.6;
        ">
            Select your preferences from the sidebar
            and click <b>Find My Courses</b> to get
            personalized recommendations.
        </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    🎓 Course Recommendation System
    &nbsp; • &nbsp;
    AI-Based Personalized Learning
    &nbsp; • &nbsp;
    TF-IDF + Cosine Similarity
</div>
""", unsafe_allow_html=True)
