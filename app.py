import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Course Recommendation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS - IMPRESSIVE UI
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(99,102,241,0.12), transparent 25%),
        radial-gradient(circle at 90% 20%, rgba(168,85,247,0.10), transparent 25%),
        linear-gradient(135deg, #f8faff 0%, #eef2ff 100%);
}

/* Main container */
.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Header */
.hero {
    background: linear-gradient(
        135deg,
        #4f46e5 0%,
        #7c3aed 50%,
        #9333ea 100%
    );
    padding: 38px 42px;
    border-radius: 26px;
    color: white;
    margin-bottom: 28px;
    box-shadow: 0 18px 45px rgba(79,70,229,0.25);
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    margin: 0;
    letter-spacing: -1px;
}

.hero-subtitle {
    font-size: 17px;
    margin-top: 10px;
    opacity: 0.92;
}

.ai-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.28);
    padding: 8px 15px;
    border-radius: 30px;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 15px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827, #1e1b4b);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.sidebar-title {
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 4px;
}

.sidebar-subtitle {
    font-size: 13px;
    opacity: 0.75;
    margin-bottom: 25px;
}

/* Section title */
.section-title {
    font-size: 27px;
    font-weight: 800;
    color: #111827;
    margin-top: 25px;
    margin-bottom: 18px;
}

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 8px 25px rgba(31,41,55,0.07);
    height: 100%;
}

.metric-label {
    color: #6b7280;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #111827;
    font-size: 25px;
    font-weight: 800;
    margin-top: 5px;
}

/* Course card */
.course-card {
    background: rgba(255,255,255,0.96);
    border: 1px solid #e5e7eb;
    border-radius: 22px;
    padding: 26px;
    margin: 15px 0;
    box-shadow: 0 12px 35px rgba(31,41,55,0.08);
    transition: all 0.25s ease;
}

.course-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 18px 45px rgba(79,70,229,0.15);
}

.rank {
    display: inline-block;
    background: linear-gradient(135deg, #f59e0b, #f97316);
    color: white;
    font-size: 12px;
    font-weight: 800;
    padding: 6px 12px;
    border-radius: 20px;
    margin-bottom: 12px;
}

.course-title {
    font-size: 28px;
    font-weight: 800;
    color: #111827;
    margin-bottom: 8px;
}

.job-role {
    color: #6366f1;
    font-weight: 700;
    font-size: 15px;
    margin-bottom: 20px;
}

/* Match score */
.match-box {
    background: linear-gradient(135deg, #eef2ff, #f5f3ff);
    border: 1px solid #ddd6fe;
    border-radius: 16px;
    padding: 17px;
    text-align: center;
}

.match-number {
    font-size: 30px;
    font-weight: 800;
    color: #4f46e5;
}

.match-label {
    font-size: 12px;
    color: #6b7280;
    font-weight: 600;
}

/* Info boxes */
.info-box {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 15px;
    margin-top: 10px;
}

.info-label {
    color: #6b7280;
    font-size: 12px;
    font-weight: 600;
}

.info-value {
    color: #111827;
    font-size: 15px;
    font-weight: 700;
    margin-top: 4px;
}

/* Skill badge */
.skill-badge {
    display: inline-block;
    background: #eef2ff;
    color: #4338ca;
    border: 1px solid #c7d2fe;
    padding: 6px 10px;
    border-radius: 20px;
    margin: 4px 3px 0 0;
    font-size: 12px;
    font-weight: 600;
}

/* Button */
.stButton > button {
    width: 100%;
    height: 54px;
    border-radius: 14px;
    border: none;
    color: white;
    font-size: 16px;
    font-weight: 800;
    background: linear-gradient(
        135deg,
        #4f46e5,
        #7c3aed,
        #9333ea
    );
    box-shadow: 0 8px 22px rgba(79,70,229,0.25);
    transition: 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(79,70,229,0.35);
}

/* Divider */
.custom-divider {
    height: 1px;
    background: linear-gradient(
        90deg,
        transparent,
        #c7d2fe,
        transparent
    );
    margin: 28px 0;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">

    <div class="ai-badge">
        ✨ Intelligent Recommendation Engine
    </div>

    <div class="hero-title">
        🎓 Course Recommendation System
    </div>

    <div class="hero-subtitle">
        Discover courses that match your interests, career goals,
        skills and educational background using intelligent AI-based matching.
    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATASET
# =========================================================

FILE_OPTIONS = [
    "course_recommendation_dataset_1000.csv",
    "final_data.csv",
    "cleaned_data.csv"
]

df = None
used_file = None

for file_name in FILE_OPTIONS:
    try:
        temp_df = pd.read_csv(file_name)

        if len(temp_df) > 0:
            df = temp_df.copy()
            used_file = file_name
            break

    except FileNotFoundError:
        continue
    except Exception:
        continue


if df is None:
    st.error(
        "❌ Dataset not found. Please keep your CSV file in the same "
        "folder as app.py."
    )

    st.info(
        "Expected file name: course_recommendation_dataset_1000.csv"
    )

    st.stop()


# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)


# =========================================================
# COLUMN ALIASES
# =========================================================

COLUMN_ALIASES = {
    "Category": ["Category", "category"],
    "Skill_Level": ["Skill_Level", "Skill Level", "skill_level"],
    "Interest": ["Interest", "interest"],
    "Education": ["Education", "education"],
    "Career_Goal": ["Career_Goal", "Career Goal", "career_goal"],
    "Skills": ["Skills", "skills"],
    "Course_Name": [
        "Course_Name",
        "Course Name",
        "course_name",
        "Course"
    ],
    "Duration_Months": [
        "Duration_Months",
        "Duration Months",
        "duration_months",
        "Duration"
    ],
    "Rating": ["Rating", "rating"],
    "Difficulty": ["Difficulty", "difficulty"],
    "Job_Role": ["Job_Role", "Job Role", "job_role"],
    "Salary_Range": [
        "Salary_Range",
        "Salary Range",
        "salary_range"
    ]
}


def find_column(possible_names):

    for name in possible_names:
        if name in df.columns:
            return name

    return None


COLUMN_MAP = {}

for standard_name, possible_names in COLUMN_ALIASES.items():

    found = find_column(possible_names)

    if found:
        COLUMN_MAP[standard_name] = found


# =========================================================
# REQUIRED COLUMN CHECK
# =========================================================

required = [
    "Interest",
    "Career_Goal",
    "Skills",
    "Education",
    "Course_Name"
]

missing = [
    col for col in required
    if col not in COLUMN_MAP
]

if missing:

    st.error(
        "❌ Required columns are missing from your dataset."
    )

    st.write("Missing:", missing)

    st.write("Available columns:")
    st.write(list(df.columns))

    st.stop()


# =========================================================
# SAFE VALUE FUNCTION
# =========================================================

def safe_value(row, column_name, default="Not specified"):

    actual_column = COLUMN_MAP.get(column_name)

    if actual_column is None:
        return default

    value = row.get(actual_column, default)

    if pd.isna(value):
        return default

    return str(value).strip()


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower()

    value = re.sub(
        r"[^a-z0-9+#.\s]",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


# =========================================================
# CREATE COURSE TEXT
# =========================================================

df["_interest_text"] = df[
    COLUMN_MAP["Interest"]
].fillna("").astype(str).apply(normalize_text)

df["_career_text"] = df[
    COLUMN_MAP["Career_Goal"]
].fillna("").astype(str).apply(normalize_text)

df["_skills_text"] = df[
    COLUMN_MAP["Skills"]
].fillna("").astype(str).apply(normalize_text)

df["_education_text"] = df[
    COLUMN_MAP["Education"]
].fillna("").astype(str).apply(normalize_text)


# =========================================================
# SIDEBAR - USER PROFILE
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🧠 AI Profile</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'Tell us about yourself to find the best courses.'
        '</div>',
        unsafe_allow_html=True
    )

    # -------------------------
    # INTEREST
    # -------------------------

    interest_values = sorted(
        df[COLUMN_MAP["Interest"]]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    selected_interest = st.selectbox(
        "💡 Your Interest",
        interest_values
    )


    # -------------------------
    # CAREER GOAL
    # -------------------------

    career_values = sorted(
        df[COLUMN_MAP["Career_Goal"]]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    selected_career = st.selectbox(
        "🎯 Career Goal",
        career_values
    )


    # -------------------------
    # EDUCATION
    # -------------------------

    education_values = sorted(
        df[COLUMN_MAP["Education"]]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    selected_education = st.selectbox(
        "🎓 Education",
        education_values
    )


    # -------------------------
    # SKILLS
    # -------------------------

    skill_input = st.text_input(
        "🛠️ Your Skills",
        placeholder="Example: Python, Pandas, SQL"
    )


    # -------------------------
    # SKILL LEVEL
    # -------------------------

    if "Skill_Level" in COLUMN_MAP:

        skill_level_values = sorted(
            df[COLUMN_MAP["Skill_Level"]]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        selected_skill_level = st.selectbox(
            "📊 Skill Level",
            skill_level_values
        )

    else:

        selected_skill_level = ""


    st.markdown("<br>", unsafe_allow_html=True)

    recommend_button = st.button(
        "✨ Get AI Recommendations"
    )


# =========================================================
# DEFAULT PROFILE
# =========================================================

if "recommendations" not in st.session_state:

    st.session_state.recommendations = None


# =========================================================
# AI RECOMMENDATION FUNCTION
# =========================================================

def generate_recommendations(
    interest,
    career,
    education,
    skills,
    skill_level
):

    user_interest = normalize_text(interest)
    user_career = normalize_text(career)
    user_education = normalize_text(education)
    user_skills = normalize_text(skills)
    user_level = normalize_text(skill_level)


    # ---------------------------------------------
    # COURSE TEXT FOR NLP
    # ---------------------------------------------

    course_text = (
        df["_interest_text"] + " "
        + df["_career_text"] + " "
        + df["_skills_text"] + " "
        + df["_education_text"]
    )


    # ---------------------------------------------
    # USER PROFILE TEXT
    # ---------------------------------------------

    user_text = (
        user_interest + " "
        + user_career + " "
        + user_education + " "
        + user_skills + " "
        + user_level
    )


    # ---------------------------------------------
    # TF-IDF NLP MATCHING
    # ---------------------------------------------

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english"
    )

    combined_text = pd.concat(
        [
            course_text,
            pd.Series([user_text])
        ],
        ignore_index=True
    )

    tfidf_matrix = vectorizer.fit_transform(
        combined_text
    )

    user_vector = tfidf_matrix[-1]

    course_vectors = tfidf_matrix[:-1]

    semantic_scores = cosine_similarity(
        user_vector,
        course_vectors
    )[0]


    # ---------------------------------------------
    # WEIGHTED FIELD MATCHING
    # ---------------------------------------------

    final_scores = []


    for index, row in df.iterrows():

        score = 0.0


        # ============================
        # INTEREST - 30%
        # ============================

        row_interest = normalize_text(
            safe_value(row, "Interest", "")
        )

        if user_interest and row_interest:

            if user_interest == row_interest:
                score += 30

            elif (
                user_interest in row_interest
                or row_interest in user_interest
            ):
                score += 22


        # ============================
        # CAREER GOAL - 25%
        # ============================

        row_career = normalize_text(
            safe_value(row, "Career_Goal", "")
        )

        if user_career and row_career:

            if user_career == row_career:
                score += 25

            elif (
                user_career in row_career
                or row_career in user_career
            ):
                score += 18


        # ============================
        # SKILLS - 20%
        # ============================

        row_skills = normalize_text(
            safe_value(row, "Skills", "")
        )

        if user_skills and row_skills:

            user_skill_words = set(
                user_skills.split()
            )

            course_skill_words = set(
                row_skills.split()
            )

            common_words = (
                user_skill_words
                & course_skill_words
            )

            if common_words:

                skill_ratio = (
                    len(common_words)
                    / max(len(user_skill_words), 1)
                )

                score += min(
                    20,
                    skill_ratio * 20
                )


        # ============================
        # EDUCATION - 10%
        # ============================

        row_education = normalize_text(
            safe_value(row, "Education", "")
        )

        if user_education and row_education:

            if user_education == row_education:

                score += 10

            elif (
                user_education in row_education
                or row_education in user_education
            ):

                score += 7


        # ============================
        # SKILL LEVEL - 5%
        # ============================

        if skill_level and "Skill_Level" in COLUMN_MAP:

            row_level = normalize_text(
                safe_value(row, "Skill_Level", "")
            )

            if row_level == user_level:

                score += 5


        # ============================
        # NLP SEMANTIC SCORE - 10%
        # ============================

        score += float(
            semantic_scores[index]
        ) * 10


        final_scores.append(score)


    result = df.copy()

    result["AI_Match_Score"] = final_scores


    # ---------------------------------------------
    # NORMALIZE SCORE
    # ---------------------------------------------

    max_score = result[
        "AI_Match_Score"
    ].max()

    min_score = result[
        "AI_Match_Score"
    ].min()


    if max_score > min_score:

        result["AI_Match_Score"] = (
            (
                result["AI_Match_Score"]
                - min_score
            )
            /
            (
                max_score
                - min_score
            )
        ) * 100

    else:

        result["AI_Match_Score"] = 0


    # ---------------------------------------------
    # SORT
    # ---------------------------------------------

    result = result.sort_values(
        by="AI_Match_Score",
        ascending=False
    )


    # ---------------------------------------------
    # REMOVE DUPLICATE COURSES
    # ---------------------------------------------

    course_col = COLUMN_MAP["Course_Name"]

    result = result.drop_duplicates(
        subset=[course_col]
    )


    return result.head(3)


# =========================================================
# GENERATE RECOMMENDATIONS
# =========================================================

if recommend_button:

    recommendations = generate_recommendations(
        selected_interest,
        selected_career,
        selected_education,
        skill_input,
        selected_skill_level
    )

    st.session_state.recommendations = recommendations


# =========================================================
# USER PROFILE SUMMARY
# =========================================================

st.markdown(
    '<div class="section-title">👤 Your Learning Profile</div>',
    unsafe_allow_html=True
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">INTEREST</div>
            <div class="metric-value">
                {selected_interest}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c2:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">CAREER GOAL</div>
            <div class="metric-value">
                {selected_career}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c3:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">EDUCATION</div>
            <div class="metric-value">
                {selected_education}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c4:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">SKILLS</div>
            <div class="metric-value">
                {skill_input if skill_input else "Not added"}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# RECOMMENDATION SECTION
# =========================================================

if st.session_state.recommendations is not None:

    recommendations = (
        st.session_state.recommendations
    )

    st.markdown(
        '<div class="custom-divider"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">'
        '✨ AI Recommended Courses'
        '</div>',
        unsafe_allow_html=True
    )


    st.caption(
        "Recommendations are generated using weighted profile matching "
        "and NLP-based similarity."
    )


    # =====================================================
    # COURSE CARDS
    # =====================================================

    for rank, (_, row) in enumerate(
        recommendations.iterrows(),
        start=1
    ):

        course_name = safe_value(
            row,
            "Course_Name"
        )

        job_role = safe_value(
            row,
            "Job_Role"
        )

        category = safe_value(
            row,
            "Category"
        )

        skills = safe_value(
            row,
            "Skills"
        )

        education = safe_value(
            row,
            "Education"
        )

        duration = safe_value(
            row,
            "Duration_Months"
        )

        rating = safe_value(
            row,
            "Rating"
        )

        difficulty = safe_value(
            row,
            "Difficulty"
        )

        salary = safe_value(
            row,
            "Salary_Range"
        )

        match_score = float(
            row["AI_Match_Score"]
        )

        # -------------------------------------------------
        # Rank label
        # -------------------------------------------------

        if rank == 1:
            rank_text = "🏆 TOP MATCH"

        elif rank == 2:
            rank_text = "🥈 SECOND BEST"

        else:
            rank_text = "🥉 THIRD BEST"


        # -------------------------------------------------
        # Card
        # -------------------------------------------------

        st.markdown(
            f"""
            <div class="course-card">

                <div class="rank">
                    {rank_text}
                </div>

                <div class="course-title">
                    {course_name}
                </div>

                <div class="job-role">
                    💼 {job_role}
                </div>

                <div class="match-box">
                    <div class="match-number">
                        {match_score:.0f}%
                    </div>

                    <div class="match-label">
                        AI MATCH SCORE
                    </div>
                </div>

                <br>

                <div class="info-box">
                    <div class="info-label">
                        📚 CATEGORY
                    </div>

                    <div class="info-value">
                        {category}
                    </div>
                </div>

                <div class="info-box">
                    <div class="info-label">
                        🛠️ SKILLS COVERED
                    </div>

                    <div class="info-value">
                        {skills}
                    </div>
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # -------------------------------------------------
        # Details row
        # -------------------------------------------------

        d1, d2, d3, d4 = st.columns(4)


        with d1:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        🎓 EDUCATION
                    </div>
                    <div class="metric-value"
                         style="font-size:17px;">
                        {education}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with d2:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        ⏱️ DURATION
                    </div>
                    <div class="metric-value"
                         style="font-size:17px;">
                        {duration} Months
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with d3:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        ⭐ RATING
                    </div>
                    <div class="metric-value"
                         style="font-size:17px;">
                        {rating}/5
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with d4:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        💰 SALARY RANGE
                    </div>
                    <div class="metric-value"
                         style="font-size:17px;">
                        {salary}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        st.markdown("<br>", unsafe_allow_html=True)

        st.info(
            f"⚡ Difficulty: {difficulty}  |  "
            f"💼 Recommended Role: {job_role}"
        )


# =========================================================
# INITIAL SCREEN
# =========================================================

else:

    st.markdown(
        '<div class="custom-divider"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            background:rgba(255,255,255,0.9);
            padding:35px;
            border-radius:22px;
            text-align:center;
            border:1px solid #e5e7eb;
            box-shadow:0 10px 30px rgba(0,0,0,0.06);
        ">

            <div style="
                font-size:48px;
                margin-bottom:10px;
            ">
                🤖
            </div>

            <div style="
                font-size:27px;
                font-weight:800;
                color:#111827;
            ">
                Ready to find your perfect course?
            </div>

            <div style="
                color:#6b7280;
                font-size:15px;
                margin-top:10px;
            ">
                Select your profile information from the sidebar
                and click <b>Get AI Recommendations</b>.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#6b7280;
        padding:35px 0 10px 0;
        font-size:13px;
    ">
        🎓 Course Recommendation System
        <br>
        Intelligent NLP + Weighted Matching Engine
    </div>
    """,
    unsafe_allow_html=True
)
