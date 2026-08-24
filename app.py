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
    page_title="AI Course Finder",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PREMIUM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(99,102,241,.10), transparent 25%),
        radial-gradient(circle at 90% 20%, rgba(14,165,233,.10), transparent 25%),
        linear-gradient(135deg,#f8fafc,#eef2ff,#f8fafc);
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ---------------- SIDEBAR ---------------- */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0f172a,#172554 55%,#111827);
    border-right: 1px solid rgba(255,255,255,.08);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.sidebar-logo {
    text-align:center;
    padding:12px 5px 22px;
}

.sidebar-logo .icon {
    font-size:52px;
}

.sidebar-logo h2 {
    margin:5px 0 2px;
    font-size:23px;
    font-weight:800;
}

.sidebar-logo p {
    color:#94a3b8 !important;
    font-size:12px;
}

.sidebar-title {
    font-size:13px;
    font-weight:800;
    letter-spacing:.5px;
    margin-bottom:10px;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background:rgba(255,255,255,.08) !important;
    border:1px solid rgba(255,255,255,.15) !important;
    border-radius:12px !important;
}

.stButton > button {
    border-radius:13px !important;
    min-height:48px !important;
    font-weight:800 !important;
    border:0 !important;
}

/* ---------------- HERO ---------------- */

.hero {
    position:relative;
    overflow:hidden;
    border-radius:28px;
    padding:42px 45px;
    background:
        radial-gradient(circle at 85% 15%,rgba(255,255,255,.20),transparent 25%),
        linear-gradient(135deg,#0f172a,#1e3a8a 55%,#2563eb);
    color:white;
    box-shadow:0 20px 55px rgba(30,64,175,.22);
    margin-bottom:22px;
}

.hero:after {
    content:"";
    position:absolute;
    width:220px;
    height:220px;
    border-radius:50%;
    right:-70px;
    bottom:-100px;
    background:rgba(255,255,255,.08);
}

.hero-badge {
    display:inline-block;
    padding:7px 14px;
    border-radius:30px;
    background:rgba(255,255,255,.13);
    border:1px solid rgba(255,255,255,.18);
    font-size:11px;
    font-weight:800;
    letter-spacing:.7px;
}

.hero h1 {
    font-size:42px;
    line-height:1.1;
    font-weight:900;
    margin:14px 0 10px;
}

.hero p {
    margin:0;
    color:#dbeafe;
    font-size:16px;
    max-width:750px;
}

/* ---------------- SECTION ---------------- */

.section-heading {
    display:flex;
    align-items:center;
    gap:10px;
    font-size:25px;
    font-weight:900;
    color:#0f172a;
    margin:30px 0 15px;
}

.section-subtitle {
    color:#64748b;
    font-size:13px;
    margin-top:-8px;
    margin-bottom:18px;
}

/* ---------------- PROFILE ---------------- */

.profile-box {
    background:rgba(255,255,255,.82);
    backdrop-filter:blur(14px);
    border:1px solid rgba(148,163,184,.20);
    border-radius:22px;
    padding:20px;
    box-shadow:0 10px 35px rgba(15,23,42,.06);
}

.profile-card {
    background:white;
    border:1px solid #e2e8f0;
    border-radius:16px;
    padding:17px;
    min-height:95px;
    box-shadow:0 5px 18px rgba(15,23,42,.04);
}

.profile-icon {
    font-size:20px;
}

.profile-label {
    color:#64748b;
    font-size:10px;
    text-transform:uppercase;
    font-weight:800;
    letter-spacing:.5px;
    margin-top:7px;
}

.profile-value {
    color:#0f172a;
    font-size:14px;
    font-weight:700;
    margin-top:3px;
    word-break:break-word;
}

/* ---------------- AI ANALYSIS ---------------- */

.ai-analysis {
    border-radius:22px;
    padding:24px;
    background:linear-gradient(135deg,#eef2ff,#eff6ff);
    border:1px solid #c7d2fe;
    box-shadow:0 8px 30px rgba(79,70,229,.07);
}

.ai-analysis-title {
    color:#3730a3;
    font-weight:900;
    font-size:17px;
}

.ai-analysis-text {
    color:#475569;
    font-size:13px;
    margin-top:6px;
}

/* ---------------- BEST MATCH ---------------- */

.best-card {
    background:linear-gradient(135deg,#ffffff,#eff6ff);
    border:2px solid #60a5fa;
    border-radius:26px;
    padding:27px;
    box-shadow:0 15px 45px rgba(37,99,235,.12);
}

.best-badge {
    display:inline-block;
    padding:6px 12px;
    background:#dbeafe;
    color:#1d4ed8;
    border-radius:30px;
    font-size:10px;
    font-weight:900;
    letter-spacing:.5px;
}

.best-title {
    font-size:28px;
    font-weight:900;
    color:#0f172a;
    margin:12px 0 7px;
}

.best-description {
    color:#64748b;
    font-size:13px;
    line-height:1.6;
}

/* ---------------- SCORE ---------------- */

.score-circle {
    text-align:center;
    padding:17px 10px;
    border-radius:20px;
    background:#eff6ff;
    border:1px solid #bfdbfe;
}

.score-number {
    color:#2563eb;
    font-size:36px;
    font-weight:900;
}

.score-text {
    color:#64748b;
    font-size:10px;
    font-weight:900;
    letter-spacing:.5px;
}

/* ---------------- INFO CARDS ---------------- */

.info-card {
    background:white;
    border:1px solid #e2e8f0;
    border-radius:15px;
    padding:14px;
    min-height:72px;
}

.info-label {
    color:#94a3b8;
    font-size:9px;
    font-weight:900;
    text-transform:uppercase;
}

.info-value {
    color:#0f172a;
    font-size:13px;
    font-weight:700;
    margin-top:5px;
}

/* ---------------- COURSE CARD ---------------- */

.course-card {
    background:rgba(255,255,255,.95);
    border:1px solid #e2e8f0;
    border-radius:22px;
    padding:22px;
    margin:13px 0;
    box-shadow:0 8px 28px rgba(15,23,42,.055);
    transition:.2s;
}

.course-card:hover {
    transform:translateY(-3px);
    box-shadow:0 15px 38px rgba(15,23,42,.10);
}

.rank-badge {
    display:inline-block;
    background:#f1f5f9;
    color:#475569;
    padding:5px 11px;
    border-radius:20px;
    font-size:10px;
    font-weight:900;
}

.course-name {
    color:#0f172a;
    font-size:20px;
    font-weight:900;
    margin:8px 0 5px;
}

.course-reason {
    color:#64748b;
    font-size:12px;
    line-height:1.5;
}

/* ---------------- MATCH ---------------- */

.match-box {
    text-align:center;
    background:#f8fafc;
    border:1px solid #e2e8f0;
    border-radius:17px;
    padding:12px;
}

.match-number {
    color:#2563eb;
    font-size:25px;
    font-weight:900;
}

.match-label {
    color:#64748b;
    font-size:9px;
    font-weight:900;
}

/* ---------------- STAT ---------------- */

.stat {
    background:white;
    border:1px solid #e2e8f0;
    border-radius:18px;
    padding:18px;
    text-align:center;
    box-shadow:0 5px 20px rgba(15,23,42,.04);
}

.stat-icon {
    font-size:24px;
}

.stat-value {
    font-size:25px;
    font-weight:900;
    color:#0f172a;
    margin-top:4px;
}

.stat-label {
    color:#64748b;
    font-size:10px;
    font-weight:800;
    text-transform:uppercase;
}

/* ---------------- FOOTER ---------------- */

.footer {
    text-align:center;
    padding:35px 10px 10px;
    color:#64748b;
    font-size:12px;
}

#MainMenu {
    visibility:hidden;
}

footer {
    visibility:hidden;
}

header {
    visibility:hidden;
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


def find_dataset():

    for file in DATASET_FILES:
        if os.path.exists(file):
            return file

    for file in os.listdir("."):
        if file.endswith(".csv"):
            return file

    return None


@st.cache_data
def load_data():

    file = find_dataset()

    if file is None:
        return None, None

    data = pd.read_csv(file)

    data.columns = [
        str(c).strip().lower()
        .replace(" ", "_")
        .replace("-", "_")
        for c in data.columns
    ]

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna("")

    return data, file


df, dataset_file = load_data()


if df is None:
    st.error("❌ CSV dataset not found.")
    st.info(
        "Keep course_recommendation_dataset_1000.csv "
        "in the same folder as app.py."
    )
    st.stop()


# ============================================================
# COLUMN DETECTION
# ============================================================

def get_col(names):

    for name in names:

        name = (
            name.lower()
            .strip()
            .replace(" ", "_")
            .replace("-", "_")
        )

        if name in df.columns:
            return name

    return None


COURSE_COL = get_col([
    "course_name",
    "course",
    "course_title",
    "title",
    "program"
])

INTEREST_COL = get_col([
    "interest",
    "interests",
    "category",
    "domain",
    "field"
])

CAREER_COL = get_col([
    "career_goal",
    "career",
    "career_path",
    "job_role",
    "role"
])

SKILLS_COL = get_col([
    "skills",
    "skill",
    "required_skills",
    "technical_skills"
])

EDUCATION_COL = get_col([
    "education",
    "qualification",
    "eligibility",
    "education_level"
])

LEVEL_COL = get_col([
    "skill_level",
    "level",
    "difficulty",
    "difficulty_level"
])

DURATION_COL = get_col([
    "duration",
    "duration_months",
    "course_duration",
    "course_length"
])

RATING_COL = get_col([
    "rating",
    "course_rating",
    "ratings"
])

SALARY_COL = get_col([
    "salary_range",
    "salary",
    "expected_salary",
    "salary_package",
    "package"
])


if COURSE_COL is None:
    st.error("❌ Course Name column not found.")
    st.write(list(df.columns))
    st.stop()


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    text = str(value).lower()

    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")

    text = re.sub(
        r"[^a-zA-Z0-9+#.\s]",
        " ",
        text
    )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def value(row, col):

    if col is None:
        return ""

    if col not in row.index:
        return ""

    x = row[col]

    if pd.isna(x):
        return ""

    return str(x).strip()


# ============================================================
# RELATED AI TERMS
# ============================================================

RELATED = {

    "ai": [
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "data science",
        "neural network",
        "computer vision",
        "nlp"
    ],

    "artificial intelligence": [
        "ai",
        "machine learning",
        "deep learning",
        "data science",
        "computer vision",
        "natural language processing"
    ],

    "machine learning": [
        "machine learning",
        "artificial intelligence",
        "ai",
        "deep learning",
        "data science",
        "predictive analytics"
    ],

    "data science": [
        "data science",
        "machine learning",
        "artificial intelligence",
        "python",
        "statistics",
        "data analytics"
    ],

    "python": [
        "python",
        "machine learning",
        "data science",
        "artificial intelligence",
        "automation"
    ],

    "cloud": [
        "cloud computing",
        "aws",
        "azure",
        "google cloud",
        "devops"
    ],

    "cyber security": [
        "cybersecurity",
        "ethical hacking",
        "network security",
        "information security"
    ],

    "web development": [
        "frontend",
        "backend",
        "full stack",
        "javascript",
        "react",
        "html",
        "css"
    ]
}


def expand(text):

    text = clean_text(text)

    result = [text]

    for key, terms in RELATED.items():

        if key in text:
            result.extend(terms)

    return " ".join(result)


# ============================================================
# UNIQUE DATA
# ============================================================

@st.cache_data
def unique_courses():

    data = df.copy()

    data["_key"] = (
        data[COURSE_COL]
        .astype(str)
        .apply(clean_text)
    )

    data = data[data["_key"] != ""]

    data = data.drop_duplicates(
        "_key"
    )

    return data.reset_index(drop=True)


# ============================================================
# AI RECOMMENDER
# ============================================================

def recommend(
    interest,
    career,
    education,
    skills,
    level
):

    data = unique_courses()

    profiles = []

    for _, row in data.iterrows():

        profile = " ".join([
            expand(value(row, COURSE_COL)),
            expand(value(row, INTEREST_COL)),
            expand(value(row, CAREER_COL)),
            expand(value(row, SKILLS_COL)),
            expand(value(row, EDUCATION_COL)),
            expand(value(row, LEVEL_COL))
        ])

        profiles.append(profile)

    user_profile = " ".join([
        expand(interest),
        expand(career),
        expand(skills),
        expand(education),
        expand(level)
    ])

    documents = [user_profile] + profiles

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=12000,
        sublinear_tf=True
    )

    matrix = vectorizer.fit_transform(
        documents
    )

    semantic = cosine_similarity(
        matrix[0:1],
        matrix[1:]
    )[0]

    final_scores = []
    reasons = []

    for i, (_, row) in enumerate(
        data.iterrows()
    ):

        course_text = profiles[i]

        interest_score = (
            1 if any(
                word in course_text
                for word in expand(interest).split()
                if len(word) > 2
            )
            else 0
        )

        career_score = (
            1 if any(
                word in course_text
                for word in expand(career).split()
                if len(word) > 2
            )
            else 0
        )

        skill_words = [
            clean_text(x)
            for x in re.split(
                r",|;|\|",
                skills
            )
            if clean_text(x)
        ]

        skill_matches = sum(
            1
            for skill in skill_words
            if skill and skill in course_text
        )

        skill_score = (
            skill_matches / len(skill_words)
            if skill_words
            else 0
        )

        education_score = (
            1
            if clean_text(education)
            and clean_text(education)
            in course_text
            else 0
        )

        level_score = (
            1
            if clean_text(level)
            and clean_text(level)
            in course_text
            else 0
        )

        score = (
            float(semantic[i]) * 35
            + interest_score * 20
            + career_score * 20
            + skill_score * 15
            + education_score * 5
            + level_score * 5
        )

        score = min(
            max(score, 0),
            100
        )

        final_scores.append(
            round(score, 1)
        )

        why = []

        if interest_score:
            why.append("interest")

        if career_score:
            why.append("career goal")

        if skill_score > 0:
            why.append("skills")

        if education_score:
            why.append("education")

        if level_score:
            why.append("skill level")

        if semantic[i] >= .20:
            why.append("AI semantic similarity")

        if why:
            reason = (
                "Strong match with your "
                + ", ".join(why)
                + "."
            )
        else:
            reason = (
                "Recommended based on overall "
                "AI profile similarity."
            )

        reasons.append(reason)

    data["AI_Score"] = final_scores
    data["AI_Reason"] = reasons

    data = data.sort_values(
        "AI_Score",
        ascending=False
    )

    data = data.drop_duplicates(
        subset=[
            data[COURSE_COL]
            .astype(str)
            .apply(clean_text)
            .name
        ]
        if False else COURSE_COL
    )

    return data.reset_index(drop=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-logo">
        <div class="icon">🎓</div>
        <h2>AI Course Finder</h2>
        <p>Intelligent Learning Recommendation</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown(
        '<div class="sidebar-title">👤 BUILD YOUR AI PROFILE</div>',
        unsafe_allow_html=True
    )

    interest = st.text_input(
        "💡 Interest",
        placeholder="Artificial Intelligence"
    )

    career = st.text_input(
        "🎯 Career Goal",
        placeholder="AI Engineer"
    )

    education = st.text_input(
        "🎓 Education",
        placeholder="B.Sc Computer Science"
    )

    skills = st.text_input(
        "🛠️ Skills",
        placeholder="Python, SQL, Pandas"
    )

    level = st.selectbox(
        "📊 Skill Level",
        [
            "Beginner",
            "Intermediate",
            "Advanced"
        ]
    )

    st.markdown("<br>", unsafe_allow_html=True)

    generate = st.button(
        "🤖  FIND MY COURSES",
        type="primary",
        use_container_width=True
    )

    st.divider()

    st.caption("🧠 AI: TF-IDF + Cosine Similarity")
    st.caption(f"📚 Courses: {len(df):,}")
    st.caption(f"📂 {dataset_file}")


# ============================================================
# LANDING PAGE
# ============================================================

if not generate:

    st.markdown("""
    <div class="hero">

        <span class="hero-badge">
        🤖 AI-POWERED • PERSONALIZED • SMART
        </span>

        <h1>Find the Right Course<br>for Your Future.</h1>

        <p>
        Tell us about your interests, career goals and skills.
        Our AI analyzes your profile and discovers the courses
        that best match your learning journey.
        </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">✨ Why use AI Course Finder?</div>',
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    with a:
        st.markdown("""
        <div class="course-card">
            <div style="font-size:30px;">🧠</div>
            <h3>AI-Powered Matching</h3>
            <p class="course-reason">
            NLP analyzes your profile and compares it
            with available courses.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="course-card">
            <div style="font-size:30px;">🎯</div>
            <h3>Personalized Results</h3>
            <p class="course-reason">
            Recommendations are based on your interests,
            skills and career goals.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown("""
        <div class="course-card">
            <div style="font-size:30px;">📊</div>
            <h3>Explainable AI</h3>
            <p class="course-reason">
            See the AI match percentage and understand
            why each course was recommended.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">🚀 Get Started</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Fill your profile using the left panel and click "
        "'FIND MY COURSES' to generate personalized recommendations."
    )

    st.stop()


# ============================================================
# VALIDATION
# ============================================================

if not interest.strip():
    st.warning("⚠️ Please enter your Interest.")
    st.stop()

if not career.strip():
    st.warning("⚠️ Please enter your Career Goal.")
    st.stop()

if not education.strip():
    st.warning("⚠️ Please enter your Education.")
    st.stop()

if not skills.strip():
    st.warning("⚠️ Please enter your Skills.")
    st.stop()


# ============================================================
# AI RECOMMENDATION
# ============================================================

with st.spinner(
    "🤖 AI is analyzing your profile..."
):

    results = recommend(
        interest,
        career,
        education,
        skills,
        level
    ).head(5)


# ============================================================
# RESULTS HERO
# ============================================================

st.markdown("""
<div class="hero">

    <span class="hero-badge">
    ✨ AI ANALYSIS COMPLETE
    </span>

    <h1>Your Personalized Learning Path</h1>

    <p>
    We analyzed your profile and ranked the most relevant
    courses using our AI recommendation engine.
    </p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# USER PROFILE DISPLAY
# ============================================================

st.markdown(
    '<div class="section-heading">👤 Your AI Profile</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">The information used by AI to generate your recommendations.</div>',
    unsafe_allow_html=True
)

p1, p2, p3, p4, p5 = st.columns(5)

profile_items = [
    ("💡", "Interest", interest),
    ("🎯", "Career Goal", career),
    ("🎓", "Education", education),
    ("🛠️", "Skills", skills),
    ("📊", "Level", level)
]

for col, item in zip(
    [p1, p2, p3, p4, p5],
    profile_items
):

    icon, label, val = item

    with col:

        st.markdown(
            f"""
            <div class="profile-card">

                <div class="profile-icon">{icon}</div>

                <div class="profile-label">
                {label}
                </div>

                <div class="profile-value">
                {val}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# AI ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-heading">🤖 AI Profile Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="ai-analysis">

        <div class="ai-analysis-title">
        ✨ AI has analyzed your learning profile
        </div>

        <div class="ai-analysis-text">
        Your profile focuses on <b>{interest}</b>,
        with a career goal of <b>{career}</b>.
        Based on your <b>{skills}</b> skills,
        <b>{education}</b> background and
        <b>{level}</b> experience level,
        the AI ranked the most relevant courses below.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# BEST COURSE
# ============================================================

if len(results) > 0:

    best = results.iloc[0]

    best_name = value(
        best,
        COURSE_COL
    )

    best_score = float(
        best["AI_Score"]
    )

    st.markdown(
        '<div class="section-heading">🏆 Best Course Match</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="best-card">

            <span class="best-badge">
            🥇 TOP AI RECOMMENDATION
            </span>

            <div class="best-title">
            {best_name}
            </div>

            <div class="best-description">
            {value(best, "AI_Reason")}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    q1, q2, q3, q4 = st.columns(4)

    with q1:

        st.markdown(
            f"""
            <div class="score-circle">
                <div class="score-number">{best_score:.0f}%</div>
                <div class="score-text">AI MATCH</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with q2:

        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-label">💼 Career Role</div>
                <div class="info-value">
                {value(best, CAREER_COL) or "Not specified"}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with q3:

        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-label">⭐ Rating</div>
                <div class="info-value">
                {value(best, RATING_COL) or "Not specified"}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with q4:

        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-label">⏱ Duration</div>
                <div class="info-value">
                {value(best, DURATION_COL) or "Not specified"}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# TOP 5
# ============================================================

st.markdown(
    '<div class="section-heading">✨ Top 5 AI Recommendations</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">Courses ranked according to your personalized AI match score.</div>',
    unsafe_allow_html=True
)


for rank, (_, row) in enumerate(
    results.iterrows(),
    start=1
):

    course_name = value(
        row,
        COURSE_COL
    )

    score = float(
        row["AI_Score"]
    )

    reason = value(
        row,
        "AI_Reason"
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

            <div class="course-reason">
            💡 {reason}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    x1, x2 = st.columns([5, 1])

    with x1:

        st.progress(
            min(score / 100, 1.0)
        )

    with x2:

        st.markdown(
            f"""
            <div class="match-box">

                <div class="match-number">
                {score:.0f}%
                </div>

                <div class="match-label">
                AI MATCH
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    y1, y2, y3, y4 = st.columns(4)

    info = [
        ("💼 Career", value(row, CAREER_COL)),
        ("🛠️ Skills", value(row, SKILLS_COL)),
        ("⏱ Duration", value(row, DURATION_COL)),
        ("⭐ Rating", value(row, RATING_COL))
    ]

    for col, (label, val) in zip(
        [y1, y2, y3, y4],
        info
    ):

        with col:

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="info-label">{label}</div>
                    <div class="info-value">
                    {val or "Not specified"}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# STATISTICS
# ============================================================

st.markdown(
    '<div class="section-heading">📊 Recommendation Insights</div>',
    unsafe_allow_html=True
)

average_score = results["AI_Score"].mean()

highest_score = results["AI_Score"].max()

total_results = len(results)

s1, s2, s3, s4 = st.columns(4)

stats = [
    ("🤖", f"{average_score:.0f}%", "Average AI Match"),
    ("🏆", f"{highest_score:.0f}%", "Best Match"),
    ("📚", str(total_results), "Courses Found"),
    ("🧠", "NLP", "AI Technology")
]

for col, stat in zip(
    [s1, s2, s3, s4],
    stats
):

    icon, number, label = stat

    with col:

        st.markdown(
            f"""
            <div class="stat">

                <div class="stat-icon">{icon}</div>

                <div class="stat-value">
                {number}
                </div>

                <div class="stat-label">
                {label}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# MATCH CHART
# ============================================================

st.markdown(
    '<div class="section-heading">📈 AI Match Analytics</div>',
    unsafe_allow_html=True
)

chart = pd.DataFrame({

    "Course": [
        value(row, COURSE_COL)
        for _, row in results.iterrows()
    ],

    "AI Match": [
        float(row["AI_Score"])
        for _, row in results.iterrows()
    ]

})

st.bar_chart(
    chart.set_index("Course"),
    height=350
)


# ============================================================
# HOW AI WORKS
# ============================================================

st.markdown(
    '<div class="section-heading">🧠 How the AI Makes Recommendations</div>',
    unsafe_allow_html=True
)

h1, h2, h3 = st.columns(3)

with h1:

    st.markdown("""
    <div class="course-card">

        <div style="font-size:28px;">01️⃣</div>

        <h3>Profile Understanding</h3>

        <p class="course-reason">
        AI analyzes your interests, career goal,
        education, skills and experience level.
        </p>

    </div>
    """, unsafe_allow_html=True)


with h2:

    st.markdown("""
    <div class="course-card">

        <div style="font-size:28px;">02️⃣</div>

        <h3>Semantic Matching</h3>

        <p class="course-reason">
        TF-IDF converts profile and course information
        into vectors and Cosine Similarity measures
        their relevance.
        </p>

    </div>
    """, unsafe_allow_html=True)


with h3:

    st.markdown("""
    <div class="course-card">

        <div style="font-size:28px;">03️⃣</div>

        <h3>Smart Ranking</h3>

        <p class="course-reason">
        The hybrid AI score ranks the most suitable
        courses and displays the Top 5 recommendations.
        </p>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

    🎓 <b>AI Course Recommendation System</b>
    <br>
    Personalized Learning • Intelligent Matching • Explainable AI
    <br><br>
    Built with Python • Streamlit • Scikit-learn

</div>
""", unsafe_allow_html=True)
