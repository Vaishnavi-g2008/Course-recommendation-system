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
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background: #f6f8fc;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    /* ---------- SIDEBAR ---------- */

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

    /* ---------- HERO ---------- */

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

    /* ---------- SECTION ---------- */

    .section-title {
        font-size: 25px;
        font-weight: 800;
        color: #172033;
        margin-top: 28px;
        margin-bottom: 15px;
    }

    .section-subtitle {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 20px;
    }

    /* ---------- STATS ---------- */

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

    /* ---------- RESULT HEADER ---------- */

    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: white;
        padding: 20px 24px;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        margin-top: 25px;
        margin-bottom: 18px;
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

    /* ---------- COURSE CARD ---------- */

    .course-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 22px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 8px 25px rgba(15,23,42,0.06);
        transition: 0.2s;
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

    .course-title {
        color: #172033;
        font-size: 26px;
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
        padding: 12px;
        margin-top: 15px;
        border: 1px solid #e2e8f0;
    }

    /* ---------- TOP COURSE ---------- */

    .top-course {
        background: linear-gradient(
            135deg,
            #ffffff,
            #f8faff
        );
        border: 2px solid #c7d2fe;
        border-radius: 24px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 12px 35px rgba(67,56,202,0.10);
    }

    .top-label {
        color: #4338ca;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 0.5px;
        margin-bottom: 10px;
    }

    .top-course-title {
        color: #172033;
        font-size: 32px;
        font-weight: 850;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 13px;
        padding: 30px 0 10px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
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

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    st.error(
        f"❌ Dataset मध्ये columns missing आहेत: {missing}"
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


# Remove empty course IDs
df = df[
    df["Course_ID"].str.strip() != ""
].copy()

df = df.reset_index(drop=True)


# ============================================================
# BUILD COURSE TEXT
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
# TF-IDF
# ============================================================

@st.cache_resource
def create_vectorizer(text):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000
    )

    matrix = vectorizer.fit_transform(text)

    return vectorizer, matrix


vectorizer, course_matrix = create_vectorizer(
    df["course_text"].tolist()
)


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

    <div class="hero-title">
        🎓 Course Recommendation System
    </div>

    <div class="hero-subtitle">
        Discover the courses that best match your skills,
        interests, career goals and learning profile.
    </div>

    <div class="hero-tag">
        🤖 AI-Powered • TF-IDF • Cosine Similarity
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# PLATFORM STATS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Learning Platform</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">📚</div>
        <div class="stat-label">TOTAL COURSES</div>
        <div class="stat-value">{len(df)}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">🏷️</div>
        <div class="stat-label">CATEGORIES</div>
        <div class="stat-value">{df["Category"].nunique()}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">⭐</div>
        <div class="stat-label">AVERAGE RATING</div>
        <div class="stat-value">{df["Rating"].mean():.1f}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-icon">🎯</div>
        <div class="stat-label">SKILL LEVELS</div>
        <div class="stat-value">{df["Skill_Level"].nunique()}</div>
    </div>
    """, unsafe_allow_html=True)


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
        'Tell us what you want to learn and we will find '
        'the most relevant courses for you.'
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

    recommend = st.button(
        "🚀 Find My Courses",
        use_container_width=True
    )


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def contains_text(series, user_text):

    user_text = str(user_text).lower().strip()

    if not user_text:
        return np.zeros(len(series))

    keywords = [
        x.strip()
        for x in re.split(r",|;|\n", user_text)
        if x.strip()
    ]

    scores = []

    for value in series:

        text = str(value).lower()

        matched = sum(
            1
            for keyword in keywords
            if keyword in text
        )

        if len(keywords) == 0:
            scores.append(0)

        else:
            scores.append(
                matched / len(keywords)
            )

    return np.array(scores)


def recommend_courses():

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
    # FIELD MATCHES
    # --------------------------------------------------------

    category_match = (
        df["Category"]
        .str.lower()
        .eq(category.lower())
        .astype(float)
        .values
    )

    skill_level_match = (
        df["Skill_Level"]
        .str.lower()
        .eq(skill_level.lower())
        .astype(float)
        .values
    )

    education_match = (
        df["Education"]
        .str.lower()
        .eq(education.lower())
        .astype(float)
        .values
    )

    career_match = (
        df["Career_Goal"]
        .str.lower()
        .eq(career_goal.lower())
        .astype(float)
        .values
    )

    difficulty_match = (
        df["Difficulty"]
        .str.lower()
        .eq(difficulty.lower())
        .astype(float)
        .values
    )

    skill_match = contains_text(
        df["Skills"],
        skills
    )

    interest_match = contains_text(
        df["Interest"],
        interest
    )

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    final_score = (
        similarity * 0.35 +
        category_match * 0.18 +
        skill_level_match * 0.10 +
        skill_match * 0.15 +
        interest_match * 0.10 +
        education_match * 0.04 +
        career_match * 0.05 +
        difficulty_match * 0.03
    )

    result = df.copy()

    result["Match_Score"] = (
        final_score * 100
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    result = result.sort_values(
        by=["Match_Score", "Rating"],
        ascending=[False, False]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # IMPORTANT:
    # GET DIFFERENT COURSE NAMES
    # --------------------------------------------------------

    selected_rows = []
    used_course_names = set()

    for _, row in result.iterrows():

        name = str(
            row["Course_Name"]
        ).strip().lower()

        if name in used_course_names:
            continue

        used_course_names.add(name)
        selected_rows.append(row)

        if len(selected_rows) == 5:
            break

    # --------------------------------------------------------
    # RETURN TOP 5
    # --------------------------------------------------------

    if selected_rows:

        top5 = pd.DataFrame(
            selected_rows
        ).reset_index(drop=True)

    else:

        top5 = pd.DataFrame()

    return top5


# ============================================================
# SHOW RECOMMENDATIONS
# ============================================================

if recommend:

    results = recommend_courses()

    if results.empty:

        st.error(
            "❌ No suitable courses found."
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
        # BEST MATCH
        # ====================================================

        best = results.iloc[0]

        best_score = min(
            100,
            max(
                0,
                round(float(best["Match_Score"]))
            )
        )

        st.markdown(f"""
        <div class="top-course">

            <div class="top-label">
                🥇 BEST MATCH FOR YOU
            </div>

            <div class="top-course-title">
                🎓 {best["Course_Name"]}
            </div>

            <br>

            <span class="course-category">
                📚 {best["Category"]}
            </span>

            &nbsp;&nbsp;

            <span class="match-badge">
                🎯 {best_score}% MATCH
            </span>

            <br><br>

            <div style="
                color:#64748b;
                font-size:15px;
                line-height:1.6;
            ">
                This course has the strongest overall match
                with your selected learning profile.
            </div>

        </div>
        """, unsafe_allow_html=True)


        # ====================================================
        # TOP 5
        # ====================================================

        for position, (_, course) in enumerate(
            results.iterrows(),
            start=1
        ):

            score = min(
                100,
                max(
                    0,
                    round(float(course["Match_Score"]))
                )
            )

            duration = course["Duration_Months"]

            if str(duration).strip() == "":
                duration_text = "N/A"
            else:
                duration_text = f"{duration:g} Months"

            st.markdown(f"""
            <div class="course-card">

                <div class="rank">
                    #{position} RECOMMENDATION
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
            """, unsafe_allow_html=True)

            # Metrics under card
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
                st.metric(
                    "💼 Job Role",
                    str(course["Job_Role"])
                )

            with st.expander(
                f"📋 View details — {course['Course_Name']}"
            ):

                d1, d2 = st.columns(2)

                with d1:

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

                with d2:

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

            st.markdown("<br>", unsafe_allow_html=True)


else:

    # ========================================================
    # EMPTY STATE
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
        ">
            Select your preferences from the sidebar
            and click <b>Find My Courses</b>.
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
