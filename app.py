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
    page_title="AI Course Recommendation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS - PROFESSIONAL UI
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #f8faff 0%, #eef3ff 50%, #f8faff 100%);
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #1e293b 100%);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

section[data-testid="stSidebar"] input {
    background-color: rgba(255,255,255,0.10) !important;
    border: 1px solid rgba(255,255,255,0.20) !important;
    border-radius: 10px !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.10) !important;
    border-radius: 10px !important;
}

/* Hero */
.hero {
    padding: 38px 42px;
    border-radius: 24px;
    background: linear-gradient(
        135deg,
        #111827 0%,
        #1d4ed8 55%,
        #2563eb 100%
    );
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 15px 40px rgba(37, 99, 235, 0.20);
}

.hero h1 {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    font-size: 17px;
    opacity: 0.90;
    margin-bottom: 0;
}

/* AI Badge */
.ai-badge {
    display: inline-block;
    padding: 7px 15px;
    border-radius: 30px;
    background: rgba(255,255,255,0.16);
    border: 1px solid rgba(255,255,255,0.25);
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 15px;
}

/* Metric cards */
.metric-card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 7px 25px rgba(15, 23, 42, 0.06);
    text-align: center;
}

.metric-title {
    color: #64748b;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
}

.metric-value {
    color: #111827;
    font-size: 28px;
    font-weight: 800;
    margin-top: 5px;
}

/* Recommendation card */
.course-card {
    background: white;
    border-radius: 20px;
    padding: 24px;
    margin: 15px 0;
    border: 1px solid #e2e8f0;
    box-shadow: 0 8px 30px rgba(15, 23, 42, 0.07);
    transition: 0.2s ease;
}

.course-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 35px rgba(15, 23, 42, 0.12);
}

.rank {
    display: inline-block;
    background: #eff6ff;
    color: #2563eb;
    border-radius: 20px;
    padding: 6px 13px;
    font-size: 12px;
    font-weight: 700;
}

.course-title {
    color: #111827;
    font-size: 23px;
    font-weight: 800;
    margin: 10px 0 8px;
}

.course-description {
    color: #64748b;
    font-size: 14px;
}

/* Match score */
.score-box {
    text-align: center;
    background: #eff6ff;
    border-radius: 16px;
    padding: 15px 10px;
}

.score-number {
    color: #2563eb;
    font-size: 29px;
    font-weight: 800;
}

.score-label {
    color: #64748b;
    font-size: 11px;
    font-weight: 700;
}

/* Info pills */
.info-pill {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 12px;
    height: 100%;
}

.info-label {
    color: #64748b;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
}

.info-value {
    color: #111827;
    font-size: 14px;
    font-weight: 600;
    margin-top: 4px;
}

/* Best match */
.best-match {
    background: linear-gradient(
        135deg,
        #ffffff 0%,
        #eff6ff 100%
    );
    border: 2px solid #3b82f6;
    border-radius: 22px;
    padding: 28px;
    margin: 20px 0;
    box-shadow: 0 10px 35px rgba(37,99,235,0.12);
}

.best-label {
    color: #2563eb;
    font-size: 13px;
    font-weight: 800;
    text-transform: uppercase;
}

/* Section heading */
.section-title {
    color: #111827;
    font-size: 25px;
    font-weight: 800;
    margin-top: 25px;
}

/* Button */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 700 !important;
    min-height: 48px !important;
}

/* Footer */
.footer {
    text-align: center;
    padding: 25px;
    color: #64748b;
    font-size: 13px;
}

/* Hide Streamlit branding */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONFIGURATION
# ============================================================

TOP_N = 5

DATASET_FILES = [
    "course_recommendation_dataset_1000.csv",
    "final_data.csv",
    "cleaned_data.csv"
]


# ============================================================
# FIND DATASET
# ============================================================

def find_dataset_file():

    for file_name in DATASET_FILES:

        if os.path.exists(file_name):
            return file_name

    try:

        for file_name in os.listdir("."):

            if file_name.lower().endswith(".csv"):
                return file_name

    except Exception:
        pass

    return None


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    dataset_file = find_dataset_file()

    if dataset_file is None:
        return None, None

    try:

        data = pd.read_csv(dataset_file)

        data.columns = [
            str(col)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            for col in data.columns
        ]

        data = data.replace(
            [np.inf, -np.inf],
            np.nan
        )

        data = data.fillna("")

        return data, dataset_file

    except Exception:

        return None, dataset_file


df, loaded_dataset_file = load_data()


# ============================================================
# DATASET ERROR
# ============================================================

if df is None:

    st.error("❌ Dataset could not be loaded.")

    st.info(
        "Keep your CSV file in the same folder as app.py."
    )

    st.stop()


# ============================================================
# COLUMN FINDER
# ============================================================

def find_column(possible_names):

    for name in possible_names:

        name = (
            name.lower()
            .strip()
            .replace(" ", "_")
            .replace("-", "_")
        )

        if name in df.columns:
            return name

    return None


# ============================================================
# DETECT COLUMNS
# ============================================================

COURSE_COL = find_column([
    "course",
    "course_name",
    "course_title",
    "title",
    "program",
    "program_name"
])

INTEREST_COL = find_column([
    "interest",
    "interests",
    "domain",
    "category",
    "field",
    "area"
])

CAREER_COL = find_column([
    "career_goal",
    "career",
    "career_path",
    "job_role",
    "role",
    "recommended_role"
])

SKILLS_COL = find_column([
    "skills",
    "skill",
    "required_skills",
    "technical_skills",
    "course_skills"
])

EDUCATION_COL = find_column([
    "education",
    "qualification",
    "eligibility",
    "education_level",
    "educational_qualification"
])

LEVEL_COL = find_column([
    "skill_level",
    "level",
    "difficulty",
    "difficulty_level",
    "experience_level"
])

DURATION_COL = find_column([
    "duration",
    "course_duration",
    "duration_months",
    "course_length"
])

RATING_COL = find_column([
    "rating",
    "course_rating",
    "ratings"
])

SALARY_COL = find_column([
    "salary",
    "salary_range",
    "expected_salary",
    "salary_package",
    "package"
])


# ============================================================
# COURSE COLUMN CHECK
# ============================================================

if COURSE_COL is None:

    st.error("❌ Course Name column not found.")

    st.write("Available columns:")
    st.write(list(df.columns))

    st.stop()


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize(text):

    if text is None:
        return ""

    text = str(text).lower().strip()

    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")

    text = re.sub(
        r"[^a-zA-Z0-9+#.\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# ROW VALUE
# ============================================================

def row_value(row, column):

    if column is None:
        return ""

    if column not in row.index:
        return ""

    value = row[column]

    if pd.isna(value):
        return ""

    return str(value).strip()


# ============================================================
# AI RELATED TERMS
# ============================================================

RELATED_TERMS = {

    "ai": [
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "neural network",
        "computer vision",
        "natural language processing",
        "nlp",
        "generative ai",
        "data science"
    ],

    "artificial intelligence": [
        "ai",
        "machine learning",
        "deep learning",
        "neural network",
        "computer vision",
        "natural language processing",
        "nlp",
        "generative ai",
        "data science"
    ],

    "machine learning": [
        "machine learning",
        "artificial intelligence",
        "ai",
        "deep learning",
        "neural network",
        "data science",
        "predictive analytics"
    ],

    "data science": [
        "data science",
        "machine learning",
        "artificial intelligence",
        "ai",
        "data analytics",
        "statistics",
        "python",
        "pandas",
        "sql"
    ],

    "data analytics": [
        "data analytics",
        "data analysis",
        "data science",
        "sql",
        "python",
        "statistics",
        "business intelligence"
    ],

    "cloud": [
        "cloud computing",
        "aws",
        "azure",
        "google cloud",
        "cloud engineer",
        "devops"
    ],

    "cyber security": [
        "cyber security",
        "cybersecurity",
        "ethical hacking",
        "network security",
        "information security"
    ],

    "web development": [
        "web development",
        "frontend",
        "backend",
        "full stack",
        "javascript",
        "react",
        "html",
        "css"
    ],

    "python": [
        "python",
        "data science",
        "machine learning",
        "artificial intelligence",
        "automation",
        "pandas",
        "numpy"
    ],

    "sql": [
        "sql",
        "database",
        "data analytics",
        "data science",
        "business intelligence"
    ]
}


# ============================================================
# EXPAND TERMS
# ============================================================

def expand_terms(text):

    text = normalize(text)

    expanded = [text]

    for key, values in RELATED_TERMS.items():

        if key in text:

            expanded.extend(values)

    return " ".join(expanded)


# ============================================================
# PREPARE UNIQUE COURSES
# ============================================================

@st.cache_data
def prepare_unique_courses():

    data = df.copy()

    data["_course_key"] = (
        data[COURSE_COL]
        .astype(str)
        .apply(normalize)
    )

    data = data[
        data["_course_key"] != ""
    ]

    data = data.drop_duplicates(
        subset=["_course_key"],
        keep="first"
    )

    return data.reset_index(drop=True)


# ============================================================
# COURSE PROFILE
# ============================================================

def create_course_profile(row):

    values = [

        row_value(row, COURSE_COL),
        row_value(row, INTEREST_COL),
        row_value(row, CAREER_COL),
        row_value(row, SKILLS_COL),
        row_value(row, EDUCATION_COL),
        row_value(row, LEVEL_COL)

    ]

    return " ".join(
        expand_terms(value)
        for value in values
        if value
    )


# ============================================================
# MATCH FUNCTIONS
# ============================================================

def keyword_match(user_text, course_text):

    user_text = expand_terms(user_text)
    course_text = expand_terms(course_text)

    if not user_text or not course_text:
        return 0.0

    user_words = set(user_text.split())
    course_words = set(course_text.split())

    if not user_words:
        return 0.0

    common = user_words.intersection(course_words)

    return min(
        len(common) / len(user_words),
        1.0
    )


def skill_match(user_skills, course_skills):

    user_skills = normalize(user_skills)
    course_skills = normalize(course_skills)

    if not user_skills or not course_skills:
        return 0.0

    user_list = re.split(
        r",|;|\|",
        user_skills
    )

    user_list = [
        normalize(x)
        for x in user_list
        if normalize(x)
    ]

    if not user_list:
        return 0.0

    matched = 0

    for skill in user_list:

        if skill in course_skills:
            matched += 1

    return min(
        matched / len(user_list),
        1.0
    )


def exact_match(user_value, course_value):

    user = normalize(user_value)
    course = normalize(course_value)

    if not user or not course:
        return 0.0

    if user == course:
        return 1.0

    if user in course or course in user:
        return 1.0

    return 0.0


# ============================================================
# 🤖 AI RECOMMENDATION ENGINE
# ============================================================

def recommend_courses(
    interest,
    career_goal,
    education,
    skills,
    skill_level
):

    data = prepare_unique_courses()

    if data.empty:
        return data

    # --------------------------------------------------------
    # Create AI Course Profiles
    # --------------------------------------------------------

    course_profiles = [
        create_course_profile(row)
        for _, row in data.iterrows()
    ]

    # --------------------------------------------------------
    # Create User Profile
    # --------------------------------------------------------

    user_profile = " ".join([
        expand_terms(interest),
        expand_terms(career_goal),
        expand_terms(skills),
        normalize(education),
        normalize(skill_level)
    ])

    documents = [
        user_profile
    ] + course_profiles

    # --------------------------------------------------------
    # TF-IDF AI MODEL
    # --------------------------------------------------------

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True
        )

        matrix = vectorizer.fit_transform(
            documents
        )

        semantic_scores = cosine_similarity(
            matrix[0:1],
            matrix[1:]
        )[0]

    except Exception:

        semantic_scores = np.zeros(
            len(data)
        )

    # --------------------------------------------------------
    # HYBRID AI SCORING
    # --------------------------------------------------------

    scores = []

    explanations = []

    for index, (_, row) in enumerate(
        data.iterrows()
    ):

        course_name = row_value(
            row,
            COURSE_COL
        )

        course_interest = row_value(
            row,
            INTEREST_COL
        )

        course_career = row_value(
            row,
            CAREER_COL
        )

        course_skills = row_value(
            row,
            SKILLS_COL
        )

        course_education = row_value(
            row,
            EDUCATION_COL
        )

        course_level = row_value(
            row,
            LEVEL_COL
        )

        # ----------------------------------------------------
        # Individual AI Features
        # ----------------------------------------------------

        interest_score = keyword_match(
            interest,
            course_name + " " + course_interest
        )

        career_score = keyword_match(
            career_goal,
            course_name + " " + course_career
        )

        skills_score = skill_match(
            skills,
            course_skills
        )

        education_score = exact_match(
            education,
            course_education
        )

        level_score = exact_match(
            skill_level,
            course_level
        )

        semantic_score = float(
            semantic_scores[index]
        )

        # ----------------------------------------------------
        # Final Hybrid AI Score
        #
        # Semantic AI       = 35%
        # Interest           = 20%
        # Career             = 20%
        # Skills             = 15%
        # Education          = 5%
        # Level              = 5%
        # ----------------------------------------------------

        final_score = (
            semantic_score * 35
            + interest_score * 20
            + career_score * 20
            + skills_score * 15
            + education_score * 5
            + level_score * 5
        )

        final_score = min(
            max(final_score, 0),
            100
        )

        scores.append(
            round(final_score, 2)
        )

        # ----------------------------------------------------
        # Explainable AI
        # ----------------------------------------------------

        reasons = []

        if interest_score >= 0.25:
            reasons.append("your interest")

        if career_score >= 0.25:
            reasons.append("your career goal")

        if skills_score >= 0.25:
            reasons.append("your skills")

        if education_score > 0:
            reasons.append("your education")

        if level_score > 0:
            reasons.append("your skill level")

        if semantic_score >= 0.20:
            reasons.append("AI semantic similarity")

        if reasons:

            explanation = (
                "Recommended because it matches "
                + ", ".join(reasons)
                + "."
            )

        else:

            explanation = (
                "Recommended based on overall "
                "AI profile similarity."
            )

        explanations.append(
            explanation
        )

    data["AI_Score"] = scores
    data["AI_Explanation"] = explanations

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    data = data.sort_values(
        by="AI_Score",
        ascending=False
    )

    # --------------------------------------------------------
    # Ensure unique course names
    # --------------------------------------------------------

    data["_course_unique"] = (
        data[COURSE_COL]
        .astype(str)
        .apply(normalize)
    )

    data = data.drop_duplicates(
        subset="_course_unique",
        keep="first"
    )

    return data.reset_index(
        drop=True
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
        text-align:center;
        padding:10px 0 20px 0;
        ">
        <div style="font-size:48px;">🎓</div>
        <h2 style="margin:0;">AI Course Finder</h2>
        <p style="font-size:13px;opacity:.7;">
        Smart • Personalized • AI Powered
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        "### 👤 Build Your Profile"
    )

    interest = st.text_input(
        "💡 Your Interest",
        placeholder="e.g. Artificial Intelligence"
    )

    career_goal = st.text_input(
        "🎯 Career Goal",
        placeholder="e.g. AI Engineer"
    )

    education = st.text_input(
        "🎓 Education",
        placeholder="e.g. B.Sc Computer Science"
    )

    skills = st.text_input(
        "🛠️ Your Skills",
        placeholder="e.g. Python, SQL, Pandas"
    )

    skill_level = st.selectbox(
        "📊 Skill Level",
        [
            "Beginner",
            "Intermediate",
            "Advanced"
        ]
    )

    st.markdown("<br>", unsafe_allow_html=True)

    recommend_button = st.button(
        "🤖  GET AI RECOMMENDATIONS",
        type="primary",
        use_container_width=True
    )

    st.divider()

    st.caption(
        "🧠 AI Engine: TF-IDF + Cosine Similarity"
    )

    st.caption(
        f"📂 Dataset: {loaded_dataset_file}"
    )

    st.caption(
        f"📚 Records: {len(df):,}"
    )


# ============================================================
# HOME PAGE
# ============================================================

if not recommend_button:

    st.markdown(
        """
        <div class="hero">

        <div class="ai-badge">
        🤖 AI-POWERED RECOMMENDATION ENGINE
        </div>

        <h1>🎓 Course Recommendation System</h1>

        <p>
        Discover the right courses based on your interests,
        career goals, skills, education and experience level.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">🚀 Smart Course Discovery</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Our AI engine analyzes your profile and finds "
        "the courses that best match your career journey."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Metrics

    unique_courses = (
        df[COURSE_COL]
        .astype(str)
        .apply(normalize)
        .nunique()
    )

    categories = (
        df[INTEREST_COL].nunique()
        if INTEREST_COL
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Dataset Records</div>
                <div class="metric-value">{len(df):,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Unique Courses</div>
                <div class="metric-value">{unique_courses:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">AI Technology</div>
                <div class="metric-value">NLP</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Recommendations</div>
                <div class="metric-value">Top 5</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # AI process

    st.markdown(
        '<div class="section-title">🧠 How Our AI Works</div>',
        unsafe_allow_html=True
    )

    p1, p2, p3 = st.columns(3)

    with p1:

        st.markdown(
            """
            <div class="course-card">

            <h3>01 • Profile Analysis</h3>

            <p class="course-description">
            AI analyzes your interests, career goal,
            skills, education and skill level.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with p2:

        st.markdown(
            """
            <div class="course-card">

            <h3>02 • Semantic Matching</h3>

            <p class="course-description">
            TF-IDF and Cosine Similarity compare
            your profile with course information.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with p3:

        st.markdown(
            """
            <div class="course-card">

            <h3>03 • Smart Ranking</h3>

            <p class="course-description">
            Courses are scored and ranked to provide
            the top 5 personalized recommendations.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="footer">
        🎓 AI Course Recommendation System
        • Personalized Learning • Smart Career Planning
        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# VALIDATION
# ============================================================

if not interest.strip():

    st.warning("⚠️ Please enter your Interest.")
    st.stop()

if not career_goal.strip():

    st.warning("⚠️ Please enter your Career Goal.")
    st.stop()

if not education.strip():

    st.warning("⚠️ Please enter your Education.")
    st.stop()

if not skills.strip():

    st.warning("⚠️ Please enter your Skills.")
    st.stop()


# ============================================================
# AI PROCESSING
# ============================================================

with st.spinner(
    "🤖 AI is analyzing your profile and finding the best courses..."
):

    recommendations = recommend_courses(
        interest,
        career_goal,
        education,
        skills,
        skill_level
    ).head(TOP_N)


# ============================================================
# RESULT HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

    <div class="ai-badge">
    ✨ PERSONALIZED AI RESULTS
    </div>

    <h1>🎯 Your Recommended Courses</h1>

    <p>
    Based on your profile, our AI identified the
    most relevant learning opportunities for you.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# BEST MATCH
# ============================================================

if not recommendations.empty:

    best_course = recommendations.iloc[0]

    best_name = row_value(
        best_course,
        COURSE_COL
    )

    best_score = float(
        best_course["AI_Score"]
    )

    st.markdown(
        """
        <div class="section-title">
        🏆 Best Match For You
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="best-match">

        <div class="best-label">
        🥇 AI TOP RECOMMENDATION
        </div>

        <h2 style="margin:8px 0;color:#111827;">
        {best_name}
        </h2>

        <p style="color:#64748b;">
        {best_course["AI_Explanation"]}
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    b1, b2, b3 = st.columns(3)

    with b1:

        st.metric(
            "🤖 AI Match",
            f"{best_score:.0f}%"
        )

    with b2:

        st.metric(
            "🏆 Rank",
            "#1"
        )

    with b3:

        rating = row_value(
            best_course,
            RATING_COL
        )

        st.metric(
            "⭐ Rating",
            rating if rating else "N/A"
        )


# ============================================================
# RECOMMENDATIONS
# ============================================================

st.markdown(
    """
    <div class="section-title">
    📚 Top 5 AI Course Recommendations
    </div>
    """,
    unsafe_allow_html=True
)


for index, (_, row) in enumerate(
    recommendations.iterrows()
):

    course_name = row_value(
        row,
        COURSE_COL
    )

    score = float(
        row["AI_Score"]
    )

    career_role = row_value(
        row,
        CAREER_COL
    )

    duration = row_value(
        row,
        DURATION_COL
    )

    rating = row_value(
        row,
        RATING_COL
    )

    salary = row_value(
        row,
        SALARY_COL
    )

    course_skills = row_value(
        row,
        SKILLS_COL
    )

    explanation = row_value(
        row,
        "AI_Explanation"
    )

    st.markdown(
        f"""
        <div class="course-card">

        <span class="rank">
        ⭐ RANK {index + 1}
        </span>

        <div class="course-title">
        {course_name}
        </div>

        <p class="course-description">
        {explanation}
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns([5, 1])

    with c1:

        st.progress(
            min(score / 100, 1.0)
        )

    with c2:

        st.markdown(
            f"""
            <div class="score-box">

            <div class="score-number">
            {score:.0f}%
            </div>

            <div class="score-label">
            AI MATCH
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    d1, d2, d3, d4 = st.columns(4)

    with d1:

        st.markdown(
            f"""
            <div class="info-pill">

            <div class="info-label">
            💼 Career
            </div>

            <div class="info-value">
            {career_role if career_role else "Not specified"}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with d2:

        st.markdown(
            f"""
            <div class="info-pill">

            <div class="info-label">
            ⏱️ Duration
            </div>

            <div class="info-value">
            {duration if duration else "Not specified"}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with d3:

        st.markdown(
            f"""
            <div class="info-pill">

            <div class="info-label">
            ⭐ Rating
            </div>

            <div class="info-value">
            {rating if rating else "Not specified"}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with d4:

        st.markdown(
            f"""
            <div class="info-pill">

            <div class="info-label">
            💰 Salary
            </div>

            <div class="info-value">
            {salary if salary else "Not specified"}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    if course_skills:

        st.markdown(
            f"""
            <div style="
            background:#f8fafc;
            padding:12px 16px;
            border-radius:12px;
            margin-top:12px;
            color:#475569;
            font-size:13px;
            ">
            🛠️ <b>Recommended Skills:</b> {course_skills}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# SUMMARY
# ============================================================

st.markdown(
    """
    <div class="section-title">
    📊 Recommendation Summary
    </div>
    """,
    unsafe_allow_html=True
)

summary_data = []

for index, (_, row) in enumerate(
    recommendations.iterrows()
):

    summary_data.append({

        "Rank": index + 1,

        "Course": row_value(
            row,
            COURSE_COL
        ),

        "AI Match": (
            f"{float(row['AI_Score']):.0f}%"
        ),

        "Career": row_value(
            row,
            CAREER_COL
        ),

        "Duration": row_value(
            row,
            DURATION_COL
        ),

        "Rating": row_value(
            row,
            RATING_COL
        )

    })


summary_df = pd.DataFrame(
    summary_data
)

st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# AI MATCH CHART
# ============================================================

st.markdown(
    """
    <div class="section-title">
    📈 AI Match Comparison
    </div>
    """,
    unsafe_allow_html=True
)

chart_data = pd.DataFrame({

    "Course": [
        row_value(
            row,
            COURSE_COL
        )
        for _, row in recommendations.iterrows()
    ],

    "AI Match %": [
        round(
            float(row["AI_Score"]),
            1
        )
        for _, row in recommendations.iterrows()
    ]

})

st.bar_chart(
    chart_data.set_index("Course")
)


# ============================================================
# AI TECHNOLOGY INFORMATION
# ============================================================

st.markdown(
    """
    <div class="course-card">

    <h3>🤖 AI Recommendation Technology</h3>

    <p class="course-description">

    <b>TF-IDF:</b>
    Converts user profile and course information
    into numerical feature vectors.

    <br><br>

    <b>Cosine Similarity:</b>
    Measures semantic similarity between the
    student's profile and available courses.

    <br><br>

    <b>Hybrid Scoring:</b>
    Combines semantic similarity with interest,
    career goal, skills, education and skill level.

    <br><br>

    <b>Explainable AI:</b>
    Shows why each course was recommended.

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

    🎓 <b>AI Course Recommendation System</b>

    <br>

    Personalized Learning • Intelligent Matching •
    Explainable AI

    </div>
    """,
    unsafe_allow_html=True
)
