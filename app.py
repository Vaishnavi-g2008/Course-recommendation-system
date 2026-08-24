import os
import re
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Course Recommendation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PREMIUM CSS
# ============================================================

st.markdown("""
<style>

/* ---------- GLOBAL ---------- */

.stApp {
    background: linear-gradient(
        135deg,
        #f8fbff 0%,
        #eef4ff 50%,
        #f8faff 100%
    );
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}


/* ---------- HERO ---------- */

.hero {
    background: linear-gradient(
        135deg,
        #111827 0%,
        #1e3a8a 50%,
        #2563eb 100%
    );

    padding: 48px 55px;
    border-radius: 30px;

    color: white;

    box-shadow:
        0 25px 60px rgba(37, 99, 235, 0.20);

    margin-bottom: 35px;

    position: relative;
    overflow: hidden;
}

.hero:after {
    content: "";
    position: absolute;

    width: 300px;
    height: 300px;

    border-radius: 50%;

    background: rgba(255,255,255,0.06);

    right: -100px;
    top: -120px;
}

.hero-badge {
    display: inline-block;

    padding: 8px 16px;

    border-radius: 30px;

    background: rgba(255,255,255,0.12);

    border: 1px solid rgba(255,255,255,0.20);

    font-size: 11px;

    font-weight: 800;

    letter-spacing: 0.5px;
}

.hero h1 {
    font-size: 44px;

    font-weight: 900;

    margin: 17px 0 10px;

    line-height: 1.1;
}

.hero p {
    font-size: 15px;

    color: #dbeafe;

    max-width: 850px;

    line-height: 1.7;
}


/* ---------- SECTION ---------- */

.section-title {
    font-size: 28px;

    font-weight: 900;

    color: #0f172a;

    margin-top: 35px;

    margin-bottom: 7px;
}

.section-subtitle {
    color: #64748b;

    font-size: 14px;

    margin-bottom: 22px;
}


/* ---------- STAT CARDS ---------- */

.stat-card {
    background: rgba(255,255,255,0.95);

    border: 1px solid #e2e8f0;

    border-radius: 20px;

    padding: 24px;

    text-align: center;

    box-shadow:
        0 10px 30px rgba(15,23,42,0.05);

    transition: 0.25s;
}

.stat-card:hover {
    transform: translateY(-4px);

    box-shadow:
        0 16px 35px rgba(37,99,235,0.10);
}

.stat-label {
    color: #64748b;

    font-size: 10px;

    font-weight: 800;

    text-transform: uppercase;

    letter-spacing: 0.5px;
}

.stat-value {
    color: #0f172a;

    font-size: 29px;

    font-weight: 900;

    margin-top: 8px;
}


/* ---------- PROFILE BOX ---------- */

.profile-box {
    background: rgba(255,255,255,0.95);

    border: 1px solid #dbeafe;

    border-radius: 25px;

    padding: 28px;

    box-shadow:
        0 15px 45px rgba(30,64,175,0.07);

    margin-bottom: 20px;
}

.profile-header {
    display: flex;

    align-items: center;

    gap: 14px;

    margin-bottom: 18px;
}

.profile-icon {
    width: 50px;

    height: 50px;

    border-radius: 15px;

    background: #dbeafe;

    display: flex;

    align-items: center;

    justify-content: center;

    font-size: 25px;
}

.profile-title {
    font-size: 21px;

    font-weight: 900;

    color: #0f172a;
}

.profile-text {
    font-size: 12px;

    color: #64748b;
}


/* ---------- INPUTS ---------- */

div[data-baseweb="select"] > div {
    border-radius: 12px !important;

    border: 1px solid #cbd5e1 !important;

    background: white !important;
}

div[data-baseweb="input"] > div {
    border-radius: 12px !important;

    border: 1px solid #cbd5e1 !important;

    background: white !important;
}

.stTextInput input {
    background: white !important;
}


/* ---------- BUTTON ---------- */

.stButton > button,
.stFormSubmitButton > button {

    min-height: 52px !important;

    border-radius: 13px !important;

    border: none !important;

    background: linear-gradient(
        90deg,
        #2563eb,
        #4f46e5
    ) !important;

    color: white !important;

    font-weight: 900 !important;

    box-shadow:
        0 10px 25px rgba(37,99,235,0.20);

    transition: 0.2s;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    transform: translateY(-2px);
}


/* ---------- AI CARDS ---------- */

.ai-card {
    background: white;

    border: 1px solid #e2e8f0;

    border-radius: 22px;

    padding: 27px;

    min-height: 190px;

    box-shadow:
        0 10px 30px rgba(15,23,42,0.05);
}

.ai-number {
    color: #2563eb;

    font-size: 11px;

    font-weight: 900;

    letter-spacing: 0.5px;
}

.ai-card h3 {
    color: #0f172a;

    font-size: 20px;

    font-weight: 900;

    margin: 12px 0;
}

.ai-card p {
    color: #64748b;

    font-size: 13px;

    line-height: 1.7;
}


/* ---------- RESULT PROFILE ---------- */

.result-card {
    background: white;

    border: 1px solid #dbeafe;

    border-radius: 17px;

    padding: 16px;

    min-height: 100px;

    box-shadow:
        0 8px 25px rgba(15,23,42,0.04);
}

.result-icon {
    font-size: 20px;
}

.result-label {
    color: #94a3b8;

    font-size: 9px;

    font-weight: 900;

    text-transform: uppercase;

    margin-top: 5px;
}

.result-value {
    color: #0f172a;

    font-size: 12px;

    font-weight: 800;

    margin-top: 4px;

    word-wrap: break-word;
}


/* ---------- ANALYSIS ---------- */

.analysis-box {
    background: linear-gradient(
        135deg,
        #eef2ff,
        #eff6ff
    );

    border: 1px solid #c7d2fe;

    border-radius: 22px;

    padding: 25px;

    margin-top: 20px;
}

.analysis-title {
    color: #3730a3;

    font-size: 18px;

    font-weight: 900;
}

.analysis-text {
    color: #475569;

    font-size: 13px;

    line-height: 1.8;

    margin-top: 8px;
}


/* ---------- BEST COURSE ---------- */

.best-course {
    background: linear-gradient(
        135deg,
        #ffffff,
        #eff6ff
    );

    border: 2px solid #60a5fa;

    border-radius: 25px;

    padding: 28px;

    box-shadow:
        0 15px 45px rgba(37,99,235,0.10);
}

.best-badge {
    display: inline-block;

    background: #dbeafe;

    color: #1d4ed8;

    padding: 7px 13px;

    border-radius: 20px;

    font-size: 10px;

    font-weight: 900;
}

.best-name {
    font-size: 27px;

    font-weight: 900;

    color: #0f172a;

    margin-top: 13px;
}

.best-description {
    color: #64748b;

    font-size: 13px;

    line-height: 1.6;

    margin-top: 7px;
}


/* ---------- COURSE CARD ---------- */

.course-card {
    background: white;

    border: 1px solid #e2e8f0;

    border-radius: 21px;

    padding: 23px;

    margin: 13px 0;

    box-shadow:
        0 8px 28px rgba(15,23,42,0.05);
}

.rank-badge {
    display: inline-block;

    background: #f1f5f9;

    color: #475569;

    padding: 6px 11px;

    border-radius: 20px;

    font-size: 9px;

    font-weight: 900;
}

.course-name {
    color: #0f172a;

    font-size: 20px;

    font-weight: 900;

    margin-top: 9px;
}

.course-description {
    color: #64748b;

    font-size: 12px;

    line-height: 1.6;

    margin-top: 5px;
}

.match-card {
    background: #eff6ff;

    border: 1px solid #bfdbfe;

    border-radius: 17px;

    padding: 11px;

    text-align: center;
}

.match-score {
    color: #2563eb;

    font-size: 26px;

    font-weight: 900;
}

.match-label {
    color: #64748b;

    font-size: 8px;

    font-weight: 900;
}


/* ---------- INFO ---------- */

.info-card {
    background: #f8fafc;

    border: 1px solid #e2e8f0;

    border-radius: 13px;

    padding: 12px;

    min-height: 65px;
}

.info-label {
    color: #94a3b8;

    font-size: 9px;

    font-weight: 900;

    text-transform: uppercase;
}

.info-value {
    color: #0f172a;

    font-size: 11px;

    font-weight: 800;

    margin-top: 5px;
}


/* ---------- FOOTER ---------- */

.footer {
    text-align: center;

    color: #64748b;

    font-size: 12px;

    padding: 40px 0 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# DATASET
# ============================================================

DATASET_FILES = [
    "course_recommendation_dataset_1000.csv",
    "final_data.csv",
    "cleaned_data.csv"
]


@st.cache_data
def load_data():

    for file in DATASET_FILES:

        if os.path.exists(file):

            data = pd.read_csv(file)

            data.columns = [
                str(c)
                .strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
                for c in data.columns
            ]

            data = data.replace(
                [np.inf, -np.inf],
                np.nan
            )

            data = data.fillna("")

            return data, file

    return None, None


df, dataset_used = load_data()


if df is None:

    st.error(
        "❌ Dataset not found!"
    )

    st.info(
        "Keep your CSV file in the same folder as app.py."
    )

    st.stop()


# ============================================================
# COLUMN DETECTION
# ============================================================

def find_column(names):

    for name in names:

        if name in df.columns:
            return name

    return None


COURSE_COL = find_column([
    "course_name",
    "course",
    "course_title",
    "title"
])

CATEGORY_COL = find_column([
    "category",
    "domain",
    "field"
])

LEVEL_COL = find_column([
    "skill_level",
    "level",
    "difficulty"
])

SKILLS_COL = find_column([
    "skills",
    "skill",
    "required_skills"
])

INTEREST_COL = find_column([
    "interest",
    "interests"
])

EDUCATION_COL = find_column([
    "education",
    "qualification",
    "education_level"
])

CAREER_COL = find_column([
    "career_goal",
    "career",
    "career_path"
])

DURATION_COL = find_column([
    "duration_months",
    "duration",
    "course_duration"
])

RATING_COL = find_column([
    "rating",
    "course_rating"
])

JOB_COL = find_column([
    "job_role",
    "job",
    "role"
])

SALARY_COL = find_column([
    "salary_range",
    "salary",
    "expected_salary"
])


if COURSE_COL is None:

    st.error(
        "Course_Name column not found."
    )

    st.write(
        "Available columns:",
        list(df.columns)
    )

    st.stop()


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    value = str(value).lower()

    value = value.replace(
        "&",
        " and "
    )

    value = re.sub(
        r"[^a-zA-Z0-9+#.\s]",
        " ",
        value
    )

    return re.sub(
        r"\s+",
        " ",
        value
    ).strip()


def get_value(row, column):

    if column is None:
        return ""

    value = row.get(
        column,
        ""
    )

    if pd.isna(value):
        return ""

    return str(value).strip()


# ============================================================
# AI RECOMMENDATION ENGINE
# ============================================================

def recommend_courses(
    category,
    level,
    interest,
    education,
    career,
    skills
):

    data = df.copy()

    data = data.drop_duplicates(
        subset=[COURSE_COL]
    ).reset_index(drop=True)

    course_text = []

    for _, row in data.iterrows():

        text = " ".join([
            get_value(row, COURSE_COL),
            get_value(row, CATEGORY_COL),
            get_value(row, LEVEL_COL),
            get_value(row, SKILLS_COL),
            get_value(row, INTEREST_COL),
            get_value(row, EDUCATION_COL),
            get_value(row, CAREER_COL),
            get_value(row, JOB_COL)
        ])

        course_text.append(
            clean_text(text)
        )


    user_text = " ".join([
        category,
        level,
        interest,
        education,
        career,
        skills
    ])

    user_text = clean_text(
        user_text
    )


    # ========================================================
    # TF-IDF
    # ========================================================

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000
    )

    all_text = [
        user_text
    ] + course_text

    matrix = vectorizer.fit_transform(
        all_text
    )


    # ========================================================
    # COSINE SIMILARITY
    # ========================================================

    similarity = cosine_similarity(
        matrix[0:1],
        matrix[1:]
    )[0]


    scores = []


    # ========================================================
    # HYBRID SCORING
    # ========================================================

    user_skills = [
        clean_text(x)
        for x in re.split(
            r",|;|\|",
            skills
        )
        if clean_text(x)
    ]


    for index, (_, row) in enumerate(
        data.iterrows()
    ):

        score = similarity[index] * 50


        course_category = clean_text(
            get_value(
                row,
                CATEGORY_COL
            )
        )

        course_level = clean_text(
            get_value(
                row,
                LEVEL_COL
            )
        )

        course_interest = clean_text(
            get_value(
                row,
                INTEREST_COL
            )
        )

        course_education = clean_text(
            get_value(
                row,
                EDUCATION_COL
            )
        )

        course_career = clean_text(
            get_value(
                row,
                CAREER_COL
            )
        )

        course_skills = clean_text(
            get_value(
                row,
                SKILLS_COL
            )
        )


        # Category
        if (
            category
            and clean_text(category)
            in course_category
        ):
            score += 10


        # Skill level
        if (
            level
            and clean_text(level)
            in course_level
        ):
            score += 7


        # Interest
        interest_text = (
            course_interest
            + " "
            + course_category
            + " "
            + course_skills
        )

        if (
            interest
            and clean_text(interest)
            in interest_text
        ):
            score += 12


        # Education
        if (
            education
            and clean_text(education)
            in course_education
        ):
            score += 5


        # Career
        career_text = (
            course_career
            + " "
            + course_skills
            + " "
            + course_category
        )

        if (
            career
            and clean_text(career)
            in career_text
        ):
            score += 10


        # Skills
        if user_skills:

            matched = 0

            for skill in user_skills:

                if skill in course_skills:

                    matched += 1

            skill_score = (
                matched /
                len(user_skills)
            ) * 6

            score += skill_score


        score = min(
            max(score, 0),
            99
        )

        scores.append(
            round(score, 1)
        )


    data["AI_Match"] = scores


    # Sort
    data = data.sort_values(
        "AI_Match",
        ascending=False
    )


    # Ensure different courses
    data = data.drop_duplicates(
        subset=[COURSE_COL]
    )


    return data.head(5).reset_index(
        drop=True
    )


# ============================================================
# HOME HERO
# ============================================================

st.markdown("""
<div class="hero">

    <span class="hero-badge">
        🤖 AI-POWERED RECOMMENDATION ENGINE
    </span>

    <h1>
        🎓 Course Recommendation System
    </h1>

    <p>
        Discover the right courses based on your interests,
        career goals, skills, education and experience level.
        Let AI build a personalized learning path for you.
    </p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SMART DISCOVERY
# ============================================================

st.markdown(
    '<div class="section-title">🚀 Smart Course Discovery</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-subtitle">
        Build your profile and let our AI engine discover
        the courses that best match your career journey.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# STATISTICS
# ============================================================

unique_courses = df[COURSE_COL].astype(str).nunique()

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.markdown(
        f"""
        <div class="stat-card">

            <div class="stat-label">
                Dataset Records
            </div>

            <div class="stat-value">
                {len(df):,}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c2:

    st.markdown(
        f"""
        <div class="stat-card">

            <div class="stat-label">
                Unique Courses
            </div>

            <div class="stat-value">
                {unique_courses}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c3:

    st.markdown(
        """
        <div class="stat-card">

            <div class="stat-label">
                AI Technology
            </div>

            <div class="stat-value">
                NLP
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with c4:

    st.markdown(
        """
        <div class="stat-card">

            <div class="stat-label">
                Recommendations
            </div>

            <div class="stat-value">
                TOP 5
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PROFILE SECTION
# ============================================================

st.markdown(
    '<div class="section-title">👤 Build Your AI Profile</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-subtitle">
        Enter your information below. Your personalized
        recommendations will appear after you submit your profile.
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="profile-box">

        <div class="profile-header">

            <div class="profile-icon">
                👤
            </div>

            <div>

                <div class="profile-title">
                    Student Profile
                </div>

                <div class="profile-text">
                    Tell our AI about your learning interests and career goals.
                </div>

            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROFILE FORM
# ============================================================

with st.form("student_profile"):

    col1, col2 = st.columns(2)


    with col1:

        category_options = [
            "Select Category"
        ]

        if CATEGORY_COL:

            values = (
                df[CATEGORY_COL]
                .astype(str)
                .str.strip()
                .unique()
            )

            values = sorted([
                value
                for value in values
                if value
                and value.lower() != "nan"
            ])

            category_options.extend(
                values
            )


        category = st.selectbox(
            "📚 Category",
            category_options
        )


        interest = st.text_input(
            "💡 Interest",
            placeholder="e.g. Artificial Intelligence"
        )


        education = st.text_input(
            "🎓 Education",
            placeholder="e.g. B.Sc Computer Science"
        )


    with col2:

        level = st.selectbox(
            "📊 Skill Level",
            [
                "Beginner",
                "Intermediate",
                "Advanced"
            ]
        )


        career = st.text_input(
            "🎯 Career Goal",
            placeholder="e.g. AI Engineer"
        )


        skills = st.text_input(
            "🛠️ Skills",
            placeholder="e.g. Python, SQL, Pandas"
        )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    submitted = st.form_submit_button(
        "🤖  FIND MY COURSES",
        use_container_width=True
    )


# ============================================================
# BEFORE SUBMIT
# ============================================================

if not submitted:

    st.markdown(
        '<div class="section-title">🧠 How Our AI Works</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-subtitle">
            Our AI engine analyzes your profile and finds
            the most relevant learning opportunities.
        </div>
        """,
        unsafe_allow_html=True
    )


    a, b, c = st.columns(3)


    with a:

        st.markdown(
            """
            <div class="ai-card">

                <div class="ai-number">
                    01 • PROFILE ANALYSIS
                </div>

                <h3>
                    👤 Understand Your Profile
                </h3>

                <p>
                    AI analyzes your interests, career goals,
                    skills, education and current skill level.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with b:

        st.markdown(
            """
            <div class="ai-card">

                <div class="ai-number">
                    02 • SMART MATCHING
                </div>

                <h3>
                    🧠 Semantic Matching
                </h3>

                <p>
                    TF-IDF and Cosine Similarity compare your
                    profile with available course information.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c:

        st.markdown(
            """
            <div class="ai-card">

                <div class="ai-number">
                    03 • SMART RANKING
                </div>

                <h3>
                    🎯 Personalized Results
                </h3>

                <p>
                    Courses receive AI match scores and the
                    five most relevant courses are ranked for you.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        """
        <div class="footer">
            🎓 Course Recommendation System
            <br>
            AI-Powered Personalized Learning
        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# VALIDATION
# ============================================================

if category == "Select Category":
    category = ""


missing = []


if not interest.strip():
    missing.append("Interest")

if not education.strip():
    missing.append("Education")

if not career.strip():
    missing.append("Career Goal")

if not skills.strip():
    missing.append("Skills")


if missing:

    st.warning(
        "⚠️ Please fill: "
        + ", ".join(missing)
    )

    st.stop()


# ============================================================
# AI RECOMMENDATION
# ============================================================

with st.spinner(
    "🤖 AI is analyzing your profile..."
):

    recommendations = recommend_courses(
        category,
        level,
        interest,
        education,
        career,
        skills
    )


# ============================================================
# RESULT HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <span class="hero-badge">
            ✨ AI ANALYSIS COMPLETE
        </span>

        <h1>
            🎯 Your Personalized Results
        </h1>

        <p>
            Our AI has analyzed your profile and ranked
            the courses that best match your learning journey.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# USER PROFILE RESULT
# ============================================================

st.markdown(
    '<div class="section-title">👤 Your AI Profile</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-subtitle">
        These preferences were used to generate your recommendations.
    </div>
    """,
    unsafe_allow_html=True
)


profile_items = [
    ("📚", "Category", category or "All Categories"),
    ("💡", "Interest", interest),
    ("🎯", "Career Goal", career),
    ("🎓", "Education", education),
    ("🛠️", "Skills", skills),
    ("📊", "Skill Level", level)
]


profile_columns = st.columns(6)


for column, item in zip(
    profile_columns,
    profile_items
):

    icon, label, value = item

    with column:

        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-icon">
                    {icon}
                </div>

                <div class="result-label">
                    {label}
                </div>

                <div class="result-value">
                    {value}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# AI ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">🤖 AI Profile Analysis</div>',
    unsafe_allow_html=True
)


st.markdown(
    f"""
    <div class="analysis-box">

        <div class="analysis-title">
            ✨ Personalized AI Analysis
        </div>

        <div class="analysis-text">

            Your profile indicates an interest in
            <b>{interest}</b> with a career goal of
            <b>{career}</b>.

            You are currently at a
            <b>{level}</b> skill level with skills including
            <b>{skills}</b>.

            The recommendation engine compares your profile
            with course information using
            <b>TF-IDF + Cosine Similarity</b>
            and ranks the most relevant courses.

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# BEST COURSE
# ============================================================

if len(recommendations) > 0:

    best = recommendations.iloc[0]

    best_name = get_value(
        best,
        COURSE_COL
    )

    best_score = float(
        best["AI_Match"]
    )


    st.markdown(
        '<div class="section-title">🏆 Best Course Match</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
        <div class="best-course">

            <span class="best-badge">
                🥇 #1 AI RECOMMENDATION
            </span>

            <div class="best-name">
                {best_name}
            </div>

            <div class="best-description">
                This course achieved the highest compatibility
                score with your profile.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown("<br>", unsafe_allow_html=True)


    b1, b2, b3, b4 = st.columns(4)


    with b1:

        st.markdown(
            f"""
            <div class="match-card">

                <div class="match-score">
                    {best_score:.0f}%
                </div>

                <div class="match-label">
                    AI MATCH
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with b2:

        st.markdown(
            f"""
            <div class="info-card">

                <div class="info-label">
                    ⭐ Rating
                </div>

                <div class="info-value">
                    {get_value(best, RATING_COL) or "N/A"}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with b3:

        st.markdown(
            f"""
            <div class="info-card">

                <div class="info-label">
                    ⏱ Duration
                </div>

                <div class="info-value">
                    {get_value(best, DURATION_COL) or "N/A"}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with b4:

        st.markdown(
            f"""
            <div class="info-card">

                <div class="info-label">
                    💼 Job Role
                </div>

                <div class="info-value">
                    {get_value(best, JOB_COL) or "N/A"}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# TOP 5 RECOMMENDATIONS
# ============================================================

st.markdown(
    '<div class="section-title">✨ Top 5 AI Recommendations</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="section-subtitle">
        Ranked from highest to lowest based on your AI match score.
    </div>
    """,
    unsafe_allow_html=True
)


for rank, (_, row) in enumerate(
    recommendations.iterrows(),
    start=1
):

    course_name = get_value(
        row,
        COURSE_COL
    )

    score = float(
        row["AI_Match"]
    )


    st.markdown(
        f"""
        <div class="course-card">

            <span class="rank-badge">
                #{rank} RECOMMENDED
            </span>

            <div class="course-name">
                {course_name}
            </div>

            <div class="course-description">
                🤖 AI selected this course based on
                the similarity between your profile
                and the course information.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    p1, p2 = st.columns(
        [6, 1]
    )


    with p1:

        st.progress(
            min(score / 100, 1.0)
        )


    with p2:

        st.markdown(
            f"""
            <div class="match-card">

                <div class="match-score">
                    {score:.0f}%
                </div>

                <div class="match-label">
                    MATCH
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    d1, d2, d3, d4 = st.columns(4)


    details = [
        (
            "📚 Category",
            get_value(
                row,
                CATEGORY_COL
            )
        ),

        (
            "🛠 Skills",
            get_value(
                row,
                SKILLS_COL
            )
        ),

        (
            "⏱ Duration",
            get_value(
                row,
                DURATION_COL
            )
        ),

        (
            "💰 Salary",
            get_value(
                row,
                SALARY_COL
            )
        )
    ]


    for column, (label, value) in zip(
        [d1, d2, d3, d4],
        details
    ):

        with column:

            st.markdown(
                f"""
                <div class="info-card">

                    <div class="info-label">
                        {label}
                    </div>

                    <div class="info-value">
                        {value or "N/A"}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


# ============================================================
# HOW AI WORKS
# ============================================================

st.markdown(
    '<div class="section-title">🧠 How Your Recommendation Was Generated</div>',
    unsafe_allow_html=True
)


x1, x2, x3 = st.columns(3)


with x1:

    st.markdown(
        """
        <div class="ai-card">

            <div class="ai-number">
                STEP 01
            </div>

            <h3>
                👤 Profile Analysis
            </h3>

            <p>
                Your interest, education, career goal,
                skills and skill level are collected
                as your personalized profile.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


with x2:

    st.markdown(
        """
        <div class="ai-card">

            <div class="ai-number">
                STEP 02
            </div>

            <h3>
                🧠 AI Similarity
            </h3>

            <p>
                TF-IDF converts profile information into
                vectors and Cosine Similarity measures
                course relevance.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


with x3:

    st.markdown(
        """
        <div class="ai-card">

            <div class="ai-number">
                STEP 03
            </div>

            <h3>
                🎯 Smart Ranking
            </h3>

            <p>
                Courses receive AI match scores and are
                ranked to generate your Top 5 recommendations.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        🎓 <b>Course Recommendation System</b>

        <br><br>

        AI-Powered Personalized Learning
        • Smart Course Discovery
        • Career-Oriented Recommendations

    </div>
    """,
    unsafe_allow_html=True
)
