import streamlit as st
import pandas as pd
import os

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Course Finder",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# PREMIUM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 5% 5%, rgba(99,102,241,0.18), transparent 28%),
        radial-gradient(circle at 95% 10%, rgba(14,165,233,0.15), transparent 28%),
        radial-gradient(circle at 50% 100%, rgba(168,85,247,0.10), transparent 35%),
        linear-gradient(135deg, #f8fafc, #eef2ff, #f0f9ff);
}

/* MAIN CONTAINER */

.block-container {
    max-width: 1450px;
    padding-top: 25px;
    padding-bottom: 40px;
}

/* ============================================================
   HERO
============================================================ */

.hero {
    position: relative;
    overflow: hidden;
    padding: 55px 35px;
    border-radius: 32px;
    text-align: center;
    color: white;

    background:
        linear-gradient(
            135deg,
            #312e81 0%,
            #4f46e5 35%,
            #2563eb 70%,
            #0891b2 100%
        );

    box-shadow:
        0 25px 60px rgba(37,99,235,0.28);

    margin-bottom: 35px;
}

.hero:before {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    border-radius: 50%;
    background: rgba(255,255,255,0.08);
    top: -100px;
    left: -80px;
}

.hero:after {
    content: "";
    position: absolute;
    width: 300px;
    height: 300px;
    border-radius: 50%;
    background: rgba(255,255,255,0.07);
    right: -100px;
    bottom: -150px;
}

.hero-icon {
    font-size: 62px;
    position: relative;
    z-index: 2;
}

.hero-title {
    font-size: 44px;
    font-weight: 800;
    letter-spacing: -1.5px;
    position: relative;
    z-index: 2;
}

.hero-subtitle {
    font-size: 17px;
    opacity: 0.92;
    max-width: 750px;
    margin: 12px auto 0 auto;
    line-height: 1.7;
    position: relative;
    z-index: 2;
}

/* ============================================================
   SECTION TITLES
============================================================ */

.section-title {
    font-size: 28px;
    font-weight: 800;
    color: #172554;
    margin-top: 30px;
    margin-bottom: 6px;
}

.section-subtitle {
    color: #64748b;
    font-size: 14px;
    margin-bottom: 22px;
}

/* ============================================================
   DASHBOARD CARDS
============================================================ */

.metric-card {
    background: rgba(255,255,255,0.78);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.7);
    border-radius: 22px;

    padding: 24px;

    min-height: 135px;

    box-shadow:
        0 10px 35px rgba(15,23,42,0.07);

    transition:
        transform 0.25s ease,
        box-shadow 0.25s ease;
}

.metric-card:hover {
    transform: translateY(-7px);

    box-shadow:
        0 20px 45px rgba(15,23,42,0.13);
}

.metric-icon {
    font-size: 30px;
}

.metric-label {
    color: #64748b;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.8px;
    margin-top: 8px;
}

.metric-value {
    color: #172554;
    font-size: 29px;
    font-weight: 800;
    margin-top: 3px;
}

/* ============================================================
   PROFILE PANEL
============================================================ */

.profile-panel {
    background: rgba(255,255,255,0.76);
    backdrop-filter: blur(22px);

    border-radius: 28px;

    padding: 32px;

    border: 1px solid rgba(255,255,255,0.7);

    box-shadow:
        0 15px 45px rgba(15,23,42,0.08);

    margin-bottom: 25px;
}

/* SELECT BOX */

div[data-baseweb="select"] > div {
    min-height: 48px !important;

    border-radius: 14px !important;

    border: 1px solid #dbeafe !important;

    background: rgba(255,255,255,0.95) !important;

    box-shadow:
        0 3px 10px rgba(15,23,42,0.03);
}

div[data-baseweb="select"] > div:hover {
    border-color: #6366f1 !important;

    box-shadow:
        0 0 0 3px rgba(99,102,241,0.10);
}

label {
    font-weight: 700 !important;
    color: #334155 !important;
}

/* ============================================================
   BUTTON
============================================================ */

.stButton > button {

    width: 100%;

    min-height: 55px;

    border: none;

    border-radius: 16px;

    font-size: 17px;

    font-weight: 800;

    color: white;

    background:
        linear-gradient(
            135deg,
            #4f46e5,
            #2563eb,
            #0891b2
        );

    box-shadow:
        0 12px 28px rgba(37,99,235,0.30);

    transition:
        all 0.25s ease;
}

.stButton > button:hover {

    transform: translateY(-4px);

    box-shadow:
        0 18px 35px rgba(37,99,235,0.40);
}

/* ============================================================
   RECOMMENDATION CARD
============================================================ */

.recommend-card {

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.97),
            rgba(239,246,255,0.94)
        );

    border-radius: 28px;

    padding: 34px;

    border: 1px solid rgba(99,102,241,0.15);

    box-shadow:
        0 20px 55px rgba(30,64,175,0.12);

    margin-top: 18px;
}

.recommend-label {

    font-size: 12px;

    font-weight: 800;

    color: #6366f1;

    letter-spacing: 1px;

    text-transform: uppercase;

    margin-bottom: 8px;
}

.course-name {

    font-size: 31px;

    font-weight: 800;

    color: #172554;

    line-height: 1.3;
}

.category-badge {

    display: inline-block;

    margin-top: 14px;

    padding: 8px 16px;

    border-radius: 30px;

    background: #e0e7ff;

    color: #3730a3;

    font-size: 13px;

    font-weight: 800;
}

/* ============================================================
   MATCH SCORE
============================================================ */

.match-card {

    background: white;

    border-radius: 24px;

    padding: 25px;

    text-align: center;

    box-shadow:
        0 12px 35px rgba(15,23,42,0.08);
}

.match-label {

    font-size: 12px;

    color: #64748b;

    font-weight: 800;

    letter-spacing: 1px;
}

.match-number {

    font-size: 52px;

    font-weight: 800;

    color: #4f46e5;

    line-height: 1.1;

    margin: 8px 0;
}

.match-description {

    color: #64748b;

    font-size: 13px;
}

/* ============================================================
   DETAIL CARDS
============================================================ */

.detail-card {

    background:
        rgba(255,255,255,0.82);

    backdrop-filter: blur(15px);

    border-radius: 19px;

    padding: 21px;

    margin-bottom: 15px;

    border: 1px solid rgba(255,255,255,0.75);

    box-shadow:
        0 8px 28px rgba(15,23,42,0.06);
}

.detail-title {

    font-size: 12px;

    color: #64748b;

    font-weight: 800;

    letter-spacing: 0.6px;

    margin-bottom: 7px;
}

.detail-value {

    font-size: 16px;

    color: #172554;

    font-weight: 700;

    line-height: 1.5;
}

/* ============================================================
   TOP COURSES
============================================================ */

.top-course {

    background:
        rgba(255,255,255,0.90);

    border-radius: 20px;

    padding: 20px 23px;

    margin-bottom: 13px;

    border: 1px solid rgba(226,232,240,0.8);

    box-shadow:
        0 8px 25px rgba(15,23,42,0.05);

    transition:
        all 0.2s ease;
}

.top-course:hover {

    transform: translateX(5px);

    box-shadow:
        0 12px 32px rgba(15,23,42,0.10);
}

.rank {

    display: inline-flex;

    align-items: center;

    justify-content: center;

    width: 38px;

    height: 38px;

    border-radius: 50%;

    background: #eef2ff;

    color: #4338ca;

    font-weight: 800;

    margin-right: 10px;
}

.top-name {

    font-weight: 800;

    color: #172554;

    font-size: 16px;
}

.top-info {

    color: #64748b;

    font-size: 12px;

    margin-top: 5px;
}

.top-score {

    font-size: 16px;

    font-weight: 800;

    color: #2563eb;
}

/* ============================================================
   HOW IT WORKS
============================================================ */

.step-card {

    background:
        rgba(255,255,255,0.78);

    backdrop-filter: blur(15px);

    border-radius: 23px;

    padding: 27px;

    min-height: 185px;

    border: 1px solid rgba(255,255,255,0.75);

    box-shadow:
        0 10px 30px rgba(15,23,42,0.06);

    transition:
        all 0.25s ease;
}

.step-card:hover {

    transform: translateY(-6px);

    box-shadow:
        0 18px 38px rgba(15,23,42,0.11);
}

.step-icon {

    font-size: 34px;
}

.step-title {

    color: #172554;

    font-size: 18px;

    font-weight: 800;

    margin-top: 12px;
}

.step-text {

    color: #64748b;

    font-size: 14px;

    line-height: 1.65;

    margin-top: 8px;
}

/* ============================================================
   INFO STRIP
============================================================ */

.info-strip {

    background:
        linear-gradient(
            135deg,
            rgba(224,231,255,0.85),
            rgba(224,242,254,0.85)
        );

    border-radius: 20px;

    padding: 20px 25px;

    border: 1px solid rgba(99,102,241,0.12);

    margin: 25px 0;
}

/* ============================================================
   FOOTER
============================================================ */

.footer {

    text-align: center;

    padding: 35px 15px;

    margin-top: 45px;

    color: #64748b;

    font-size: 13px;

    border-top: 1px solid rgba(148,163,184,0.20);
}

.footer-title {

    color: #3730a3;

    font-size: 17px;

    font-weight: 800;

    margin-bottom: 8px;
}

/* ============================================================
   STREAMLIT UI
============================================================ */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* ============================================================
   MOBILE
============================================================ */

@media (max-width: 768px) {

    .hero {
        padding: 38px 20px;
    }

    .hero-title {
        font-size: 30px;
    }

    .hero-subtitle {
        font-size: 14px;
    }

    .section-title {
        font-size: 23px;
    }

    .course-name {
        font-size: 24px;
    }

    .profile-panel {
        padding: 20px;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">

    <div class="hero-icon">🎓</div>

    <div class="hero-title">
        Smart Course Finder
    </div>

    <div class="hero-subtitle">
        Find the right course for your future by matching your
        skills, interests, education and career goals.
    </div>

</div>
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

        raise FileNotFoundError(
            "final_data.csv file app.py च्या folder मध्ये सापडला नाही."
        )

    return pd.read_csv(file_path)


# ============================================================
# SAFE DATA LOADING
# ============================================================

try:

    df = load_data()

except Exception as e:

    st.error("❌ Dataset load करण्यात problem आला.")

    st.error(str(e))

    st.warning(
        "📁 app.py आणि final_data.csv एकाच folder मध्ये ठेवा."
    )

    st.stop()


# ============================================================
# DATA VALIDATION
# ============================================================

if df.empty:

    st.error(
        "❌ final_data.csv मध्ये कोणताही data नाही."
    )

    st.stop()


df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [

    "Course_ID",
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
        "❌ Dataset मध्ये required columns missing आहेत."
    )

    st.write(
        "Missing Columns:",
        missing_columns
    )

    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

df = df.copy()


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


df = df[
    df["Course_ID"].astype(str).str.strip() != ""
].copy()


if df.empty:

    st.error(
        "❌ Dataset मध्ये valid course records सापडले नाहीत."
    )

    st.stop()


# ============================================================
# DASHBOARD OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">📊 Learning Platform</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">Explore the learning opportunities available in our course database.</div>',
    unsafe_allow_html=True
)


total_courses = df["Course_ID"].nunique()

total_categories = df["Category"].nunique()

total_roles = df["Job_Role"].nunique()

average_rating = df["Rating"].mean()


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(f"""
    <div class="metric-card">

        <div class="metric-icon">🎓</div>

        <div class="metric-label">
            TOTAL COURSES
        </div>

        <div class="metric-value">
            {total_courses:,}
        </div>

    </div>
    """, unsafe_allow_html=True)


with col2:

    st.markdown(f"""
    <div class="metric-card">

        <div class="metric-icon">📚</div>

        <div class="metric-label">
            CATEGORIES
        </div>

        <div class="metric-value">
            {total_categories:,}
        </div>

    </div>
    """, unsafe_allow_html=True)


with col3:

    st.markdown(f"""
    <div class="metric-card">

        <div class="metric-icon">💼</div>

        <div class="metric-label">
            JOB ROLES
        </div>

        <div class="metric-value">
            {total_roles:,}
        </div>

    </div>
    """, unsafe_allow_html=True)


with col4:

    st.markdown(f"""
    <div class="metric-card">

        <div class="metric-icon">⭐</div>

        <div class="metric-label">
            AVG RATING
        </div>

        <div class="metric-value">
            {average_rating:.2f}/5
        </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PROFILE BUILDER
# ============================================================

st.markdown(
    '<div class="section-title">👤 Build Your Learning Profile</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">Choose your preferences and let the system find the courses that fit you best.</div>',
    unsafe_allow_html=True
)


st.markdown(
    '<div class="profile-panel">',
    unsafe_allow_html=True
)


# ============================================================
# ROW 1
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    category_options = sorted(
        df["Category"]
        .dropna()
        .unique()
        .tolist()
    )

    category = st.selectbox(
        "📚 Course Category",
        category_options
    )


with col2:

    skill_level_options = sorted(
        df["Skill_Level"]
        .dropna()
        .unique()
        .tolist()
    )

    skill_level = st.selectbox(
        "📈 Skill Level",
        skill_level_options
    )


with col3:

    interest_options = sorted(
        df["Interest"]
        .dropna()
        .unique()
        .tolist()
    )

    interest = st.selectbox(
        "💡 Area of Interest",
        interest_options
    )


# ============================================================
# ROW 2
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    education_options = sorted(
        df["Education"]
        .dropna()
        .unique()
        .tolist()
    )

    education = st.selectbox(
        "🎓 Education",
        education_options
    )


with col2:

    career_goal_options = sorted(
        df["Career_Goal"]
        .dropna()
        .unique()
        .tolist()
    )

    career_goal = st.selectbox(
        "🚀 Career Goal",
        career_goal_options
    )


with col3:

    skills_options = sorted(
        df["Skills"]
        .dropna()
        .unique()
        .tolist()
    )

    skills = st.selectbox(
        "🛠️ Skills",
        skills_options
    )


# ============================================================
# ROW 3
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    difficulty_options = sorted(
        df["Difficulty"]
        .dropna()
        .unique()
        .tolist()
    )

    difficulty = st.selectbox(
        "⚡ Difficulty",
        difficulty_options
    )


with col2:

    job_role_options = sorted(
        df["Job_Role"]
        .dropna()
        .unique()
        .tolist()
    )

    job_role = st.selectbox(
        "💼 Target Job Role",
        job_role_options
    )


with col3:

    salary_options = sorted(
        df["Salary_Range"]
        .dropna()
        .unique()
        .tolist()
    )

    salary_range = st.selectbox(
        "💰 Expected Salary Range",
        salary_options
    )


st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# BUTTON
# ============================================================

st.markdown(
    "<br>",
    unsafe_allow_html=True
)


recommend_button = st.button(
    "🚀  FIND MY PERFECT COURSE",
    use_container_width=True
)


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

if recommend_button:

    result = df.copy()

    result["Match_Score"] = 0


    # CATEGORY
    result.loc[
        result["Category"].str.lower()
        == str(category).lower(),
        "Match_Score"
    ] += 15


    # SKILL LEVEL
    result.loc[
        result["Skill_Level"].str.lower()
        == str(skill_level).lower(),
        "Match_Score"
    ] += 10


    # INTEREST
    result.loc[
        result["Interest"].str.lower()
        == str(interest).lower(),
        "Match_Score"
    ] += 15


    # EDUCATION
    result.loc[
        result["Education"].str.lower()
        == str(education).lower(),
        "Match_Score"
    ] += 10


    # CAREER GOAL
    result.loc[
        result["Career_Goal"].str.lower()
        == str(career_goal).lower(),
        "Match_Score"
    ] += 15


    # SKILLS
    result.loc[
        result["Skills"].str.lower()
        == str(skills).lower(),
        "Match_Score"
    ] += 10


    # DIFFICULTY
    result.loc[
        result["Difficulty"].str.lower()
        == str(difficulty).lower(),
        "Match_Score"
    ] += 5


    # JOB ROLE
    result.loc[
        result["Job_Role"].str.lower()
        == str(job_role).lower(),
        "Match_Score"
    ] += 10


    # SALARY
    result.loc[
        result["Salary_Range"].str.lower()
        == str(salary_range).lower(),
        "Match_Score"
    ] += 10


    # SORT
    result = result.sort_values(
        by=[
            "Match_Score",
            "Rating"
        ],
        ascending=[
            False,
            False
        ]
    ).reset_index(drop=True)


    if len(result) == 0:

        st.error(
            "❌ कोणताही recommendation मिळाला नाही."
        )

        st.stop()


    course = result.iloc[0]


    # ========================================================
    # COURSE NAME
    # ========================================================

    if "Course_Name" in result.columns:

        course_name = str(
            course["Course_Name"]
        )

        if not course_name.strip():

            course_name = (
                f"Course ID: {course['Course_ID']}"
            )

    else:

        course_name = (
            f"Course ID: {course['Course_ID']}"
        )


    # ========================================================
    # SCORE
    # ========================================================

    score = int(
        course["Match_Score"]
    )

    percentage = min(
        score,
        100
    )


    # ========================================================
    # SUCCESS
    # ========================================================

    st.success(
        "🎉 Your personalized recommendation has been generated!"
    )


    # ========================================================
    # RECOMMENDED COURSE
    # ========================================================

    st.markdown(
        '<div class="section-title">✨ Your Best Match</div>',
        unsafe_allow_html=True
    )


    st.markdown(f"""

    <div class="recommend-card">

        <div class="recommend-label">
            AI-INSPIRED COURSE MATCH
        </div>

        <div class="course-name">
            🎓 {course_name}
        </div>

        <div class="category-badge">
            📚 {course['Category']}
        </div>

        <div class="info-strip">

            <b>🎯 Why this course?</b>

            <br><br>

            This course matches your selected
            skills, interests, education and
            career preferences.

        </div>

    </div>

    """, unsafe_allow_html=True)


    # ========================================================
    # MATCH SCORE
    # ========================================================

    col1, col2 = st.columns(
        [1, 2]
    )


    with col1:

        st.markdown(f"""

        <div class="match-card">

            <div class="match-label">
                PROFILE MATCH
            </div>

            <div class="match-number">
                {percentage}%
            </div>

            <div class="match-description">
                Compatibility Score
            </div>

        </div>

        """, unsafe_allow_html=True)


    with col2:

        st.markdown(
            """
            <div style="
                margin-top:12px;
                font-weight:800;
                color:#334155;
            ">
                🎯 Recommendation Strength
            </div>
            """,
            unsafe_allow_html=True
        )


        st.progress(
            percentage / 100
        )


        if percentage >= 80:

            st.success(
                "🔥 Excellent Match — highly aligned with your profile!"
            )

        elif percentage >= 60:

            st.info(
                "👍 Good Match — this course fits many of your preferences."
            )

        else:

            st.warning(
                "💡 Moderate Match — check the Top 5 recommendations below."
            )


    # ========================================================
    # COURSE HIGHLIGHTS
    # ========================================================

    st.markdown(
        '<div class="section-title">📌 Course Highlights</div>',
        unsafe_allow_html=True
    )


    duration = float(
        course["Duration_Months"]
    )


    if duration > 0:

        duration_text = (
            f"{duration:g} Months"
        )

    else:

        duration_text = "N/A"


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.markdown(f"""

        <div class="metric-card">

            <div class="metric-icon">
                ⭐
            </div>

            <div class="metric-label">
                RATING
            </div>

            <div class="metric-value">
                {course['Rating']} / 5
            </div>

        </div>

        """, unsafe_allow_html=True)


    with col2:

        st.markdown(f"""

        <div class="metric-card">

            <div class="metric-icon">
                ⏱️
            </div>

            <div class="metric-label">
                DURATION
            </div>

            <div class="metric-value"
                 style="font-size:21px;">
                {duration_text}
            </div>

        </div>

        """, unsafe_allow_html=True)


    with col3:

        st.markdown(f"""

        <div class="metric-card">

            <div class="metric-icon">
                📈
            </div>

            <div class="metric-label">
                SKILL LEVEL
            </div>

            <div class="metric-value"
                 style="font-size:20px;">
                {course['Skill_Level']}
            </div>

        </div>

        """, unsafe_allow_html=True)


    with col4:

        st.markdown(f"""

        <div class="metric-card">

            <div class="metric-icon">
                💰
            </div>

            <div class="metric-label">
                SALARY RANGE
            </div>

            <div class="metric-value"
                 style="font-size:18px;">
                {course['Salary_Range']}
            </div>

        </div>

        """, unsafe_allow_html=True)


    # ========================================================
    # COURSE DETAILS
    # ========================================================

    st.markdown(
        '<div class="section-title">📋 Course Details</div>',
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)


    with col1:

        st.markdown(f"""

        <div class="detail-card">

            <div class="detail-title">
                🛠️ SKILLS
            </div>

            <div class="detail-value">
                {course['Skills']}
            </div>

        </div>


        <div class="detail-card">

            <div class="detail-title">
                💡 INTEREST
            </div>

            <div class="detail-value">
                {course['Interest']}
            </div>

        </div>


        <div class="detail-card">

            <div class="detail-title">
                🎯 CAREER GOAL
            </div>

            <div class="detail-value">
                {course['Career_Goal']}
            </div>

        </div>

        """, unsafe_allow_html=True)


    with col2:

        st.markdown(f"""

        <div class="detail-card">

            <div class="detail-title">
                💼 TARGET JOB ROLE
            </div>

            <div class="detail-value">
                {course['Job_Role']}
            </div>

        </div>


        <div class="detail-card">

            <div class="detail-title">
                ⚡ DIFFICULTY
            </div>

            <div class="detail-value">
                {course['Difficulty']}
            </div>

        </div>


        <div class="detail-card">

            <div class="detail-title">
                🎓 EDUCATION
            </div>

            <div class="detail-value">
                {course['Education']}
            </div>

        </div>

        """, unsafe_allow_html=True)


    # ========================================================
    # TOP 5
    # ========================================================

    st.markdown(
        '<div class="section-title">🏆 Top 5 Recommended Courses</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Best alternatives based on your profile match.</div>',
        unsafe_allow_html=True
    )


    top5 = result.head(5)


    for index, row in top5.iterrows():

        rank = index + 1


        if (
            "Course_Name" in result.columns
            and str(row["Course_Name"]).strip()
        ):

            name = str(
                row["Course_Name"]
            )

        else:

            name = (
                f"Course ID: {row['Course_ID']}"
            )


        st.markdown(f"""

        <div class="top-course">

            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                gap:15px;
            ">

                <div>

                    <span class="rank">
                        #{rank}
                    </span>

                    <span class="top-name">
                        {name}
                    </span>

                    <div class="top-info">

                        📚 {row['Category']}
                        &nbsp; • &nbsp;
                        📈 {row['Skill_Level']}
                        &nbsp; • &nbsp;
                        ⭐ {row['Rating']}

                    </div>

                </div>


                <div class="top-score">
                    {int(row['Match_Score'])}% Match
                </div>

            </div>

        </div>

        """, unsafe_allow_html=True)


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown(
    '<div class="section-title">🧠 How Smart Course Finder Works</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">Three simple steps to discover your learning path.</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


with col1:

    st.markdown("""

    <div class="step-card">

        <div class="step-icon">
            👤
        </div>

        <div class="step-title">
            01 — Build Your Profile
        </div>

        <div class="step-text">
            Select your education, skills,
            interests, difficulty level,
            career goal and expected salary.
        </div>

    </div>

    """, unsafe_allow_html=True)


with col2:

    st.markdown("""

    <div class="step-card">

        <div class="step-icon">
            🔎
        </div>

        <div class="step-title">
            02 — Smart Matching
        </div>

        <div class="step-text">
            Your preferences are compared
            against course attributes to
            calculate a personalized match score.
        </div>

    </div>

    """, unsafe_allow_html=True)


with col3:

    st.markdown("""

    <div class="step-card">

        <div class="step-icon">
            🚀
        </div>

        <div class="step-title">
            03 — Get Your Course
        </div>

        <div class="step-text">
            The highest-scoring courses are
            ranked and displayed so you can
            choose your best learning option.
        </div>

    </div>

    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""

<div class="footer">

    <div class="footer-title">
        🎓 Smart Course Finder
    </div>

    Personalized learning recommendations
    for smarter career decisions.

    <br><br>

    Built with
    <b>Python</b> •
    <b>Pandas</b> •
    <b>Streamlit</b>

</div>

""", unsafe_allow_html=True)