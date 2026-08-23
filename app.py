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
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_FILE = "course_recommendation_dataset_1000.csv"
TOP_N = 5


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    if not os.path.exists(DATASET_FILE):
        return None

    data = pd.read_csv(DATASET_FILE)

    # Clean column names
    data.columns = [
        str(col)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        for col in data.columns
    ]

    data = data.fillna("")

    return data


df = load_data()


# ============================================================
# DATASET NOT FOUND
# ============================================================

if df is None:

    st.error(
        "Dataset file not found."
    )

    st.info(
        "Place course_recommendation_dataset_1000.csv "
        "in the same folder as app.py."
    )

    st.stop()


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(possible_names):

    for name in possible_names:

        name = (
            name
            .lower()
            .strip()
            .replace(" ", "_")
            .replace("-", "_")
        )

        if name in df.columns:
            return name

    return None


# ============================================================
# DETECT DATASET COLUMNS
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
# COURSE COLUMN REQUIRED
# ============================================================

if COURSE_COL is None:

    st.error(
        "Course column was not found in the dataset."
    )

    st.write(
        "Available dataset columns:"
    )

    st.write(
        list(df.columns)
    )

    st.stop()


# ============================================================
# TEXT NORMALIZATION
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
# VALUE FROM ROW
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
# RELATED AI TERMS
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

    "machine learning": [
        "machine learning",
        "artificial intelligence",
        "ai",
        "deep learning",
        "neural network",
        "data science",
        "predictive analytics"
    ],

    "deep learning": [
        "deep learning",
        "machine learning",
        "artificial intelligence",
        "ai",
        "neural network",
        "computer vision",
        "natural language processing",
        "nlp"
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
        "cloud architecture",
        "devops"
    ],

    "cloud computing": [
        "cloud computing",
        "aws",
        "azure",
        "google cloud",
        "cloud engineer",
        "cloud architecture",
        "devops"
    ],

    "cyber security": [
        "cyber security",
        "cybersecurity",
        "ethical hacking",
        "network security",
        "information security",
        "penetration testing"
    ],

    "cybersecurity": [
        "cyber security",
        "cybersecurity",
        "ethical hacking",
        "network security",
        "information security",
        "penetration testing"
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
    ],

    "devops": [
        "devops",
        "cloud computing",
        "aws",
        "azure",
        "docker",
        "kubernetes",
        "ci cd"
    ]
}


# ============================================================
# EXPAND RELATED TERMS
# ============================================================

def expand_terms(text):

    text = normalize(text)

    expanded = [text]

    for key, values in RELATED_TERMS.items():

        if key in text:

            expanded.extend(values)

    return " ".join(expanded)


# ============================================================
# MATCH TEXT
# ============================================================

def text_match(user_text, course_text):

    user_text = normalize(user_text)
    course_text = normalize(course_text)

    if not user_text or not course_text:
        return 0.0

    if user_text == course_text:
        return 1.0

    if user_text in course_text:
        return 1.0

    user_words = set(
        user_text.split()
    )

    course_words = set(
        course_text.split()
    )

    if not user_words:
        return 0.0

    common_words = (
        user_words
        .intersection(course_words)
    )

    return len(common_words) / len(user_words)


# ============================================================
# RELATED MATCH
# ============================================================

def related_match(user_text, course_text):

    normal_score = text_match(
        user_text,
        course_text
    )

    expanded_score = text_match(
        expand_terms(user_text),
        course_text
    )

    return max(
        normal_score,
        expanded_score
    )


# ============================================================
# SKILL MATCH
# ============================================================

def calculate_skill_match(
    user_skills,
    course_skills
):

    user_skills = normalize(
        user_skills
    )

    course_skills = normalize(
        course_skills
    )

    if not user_skills or not course_skills:
        return 0.0

    user_skill_list = re.split(
        r",|;|\|",
        user_skills
    )

    user_skill_list = [
        skill.strip()
        for skill in user_skill_list
        if skill.strip()
    ]

    if not user_skill_list:
        return 0.0

    matched = 0

    for skill in user_skill_list:

        if skill in course_skills:

            matched += 1

            continue

        expanded = expand_terms(skill)

        expanded_words = expanded.split()

        if any(
            word in course_skills
            for word in expanded_words
        ):

            matched += 1

    return min(
        matched / len(user_skill_list),
        1.0
    )


# ============================================================
# EDUCATION MATCH
# ============================================================

def calculate_education_match(
    user_education,
    course_education
):

    user = normalize(
        user_education
    )

    course = normalize(
        course_education
    )

    if not user or not course:
        return 0.0

    if user == course:
        return 1.0

    if user in course:
        return 1.0

    if course in user:
        return 1.0

    education_groups = {

        "10th": [
            "10th",
            "secondary"
        ],

        "12th": [
            "12th",
            "hsc",
            "higher secondary"
        ],

        "diploma": [
            "diploma"
        ],

        "graduate": [
            "graduate",
            "graduation",
            "bachelor",
            "btech",
            "b.e",
            "be",
            "bca",
            "bsc"
        ],

        "postgraduate": [
            "postgraduate",
            "master",
            "masters",
            "mtech",
            "mca",
            "msc",
            "mba"
        ]
    }

    user_group = None
    course_group = None

    for group, words in education_groups.items():

        if any(
            word in user
            for word in words
        ):

            user_group = group

        if any(
            word in course
            for word in words
        ):

            course_group = group

    if (
        user_group is not None
        and
        user_group == course_group
    ):

        return 1.0

    return 0.0


# ============================================================
# LEVEL MATCH
# ============================================================

def calculate_level_match(
    user_level,
    course_level
):

    user = normalize(
        user_level
    )

    course = normalize(
        course_level
    )

    if not user or not course:
        return 0.0

    if user == course:
        return 1.0

    if user in course:
        return 1.0

    return 0.0


# ============================================================
# COURSE PROFILE
# ============================================================

def create_course_profile(row):

    values = [

        row_value(
            row,
            COURSE_COL
        ),

        row_value(
            row,
            INTEREST_COL
        ),

        row_value(
            row,
            CAREER_COL
        ),

        row_value(
            row,
            SKILLS_COL
        ),

        row_value(
            row,
            EDUCATION_COL
        ),

        row_value(
            row,
            LEVEL_COL
        )
    ]

    return " ".join(
        normalize(value)
        for value in values
        if value
    )


# ============================================================
# REMOVE DUPLICATE COURSE RECORDS
# ============================================================

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

    # Keep only one record for every course
    data = data.drop_duplicates(
        subset=["_course_key"],
        keep="first"
    )

    data = data.reset_index(
        drop=True
    )

    return data


# ============================================================
# AI RECOMMENDATION ENGINE
# ============================================================

def recommend_courses(
    interest,
    career_goal,
    education,
    skills,
    skill_level
):

    data = prepare_unique_courses()

    # --------------------------------------------------------
    # COURSE TEXT
    # --------------------------------------------------------

    course_profiles = [
        create_course_profile(row)
        for _, row in data.iterrows()
    ]

    # --------------------------------------------------------
    # USER PROFILE
    # --------------------------------------------------------

    user_profile = " ".join([

        expand_terms(interest),

        expand_terms(career_goal),

        normalize(education),

        expand_terms(skills),

        normalize(skill_level)

    ])

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    documents = [
        user_profile
    ] + course_profiles

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        matrix = vectorizer.fit_transform(
            documents
        )

        nlp_scores = cosine_similarity(
            matrix[0:1],
            matrix[1:]
        )[0]

    except Exception:

        nlp_scores = np.zeros(
            len(data)
        )

    # --------------------------------------------------------
    # CALCULATE SCORE
    # --------------------------------------------------------

    scores = []

    interest_scores = []
    career_scores = []
    skill_scores = []
    education_scores = []
    level_scores = []
    nlp_scores_percent = []

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
        # INTEREST
        # ----------------------------------------------------

        interest_text = (
            course_name
            + " "
            + course_interest
        )

        interest_score = related_match(
            interest,
            interest_text
        )

        # ----------------------------------------------------
        # CAREER
        # ----------------------------------------------------

        career_text = (
            course_name
            + " "
            + course_career
        )

        career_score = related_match(
            career_goal,
            career_text
        )

        # ----------------------------------------------------
        # SKILLS
        # ----------------------------------------------------

        skill_score = calculate_skill_match(
            skills,
            course_skills
        )

        # ----------------------------------------------------
        # EDUCATION
        # ----------------------------------------------------

        education_score = calculate_education_match(
            education,
            course_education
        )

        # ----------------------------------------------------
        # LEVEL
        # ----------------------------------------------------

        level_score = calculate_level_match(
            skill_level,
            course_level
        )

        # ----------------------------------------------------
        # NLP
        # ----------------------------------------------------

        nlp_score = float(
            nlp_scores[index]
        )

        # ----------------------------------------------------
        # WEIGHTED SCORE
        # ----------------------------------------------------

        final_score = (

            interest_score * 30

            +

            career_score * 30

            +

            skill_score * 15

            +

            education_score * 10

            +

            level_score * 5

            +

            nlp_score * 10
        )

        # ----------------------------------------------------
        # MATCH BONUSES
        # ----------------------------------------------------

        if interest_score >= 1.0:
            final_score += 2

        if career_score >= 1.0:
            final_score += 2

        if skill_score >= 1.0:
            final_score += 1

        if education_score >= 1.0:
            final_score += 1

        final_score = min(
            final_score,
            100
        )

        scores.append(
            final_score
        )

        interest_scores.append(
            interest_score * 100
        )

        career_scores.append(
            career_score * 100
        )

        skill_scores.append(
            skill_score * 100
        )

        education_scores.append(
            education_score * 100
        )

        level_scores.append(
            level_score * 100
        )

        nlp_scores_percent.append(
            nlp_score * 100
        )

    # --------------------------------------------------------
    # ADD SCORE COLUMNS
    # --------------------------------------------------------

    data["AI_Score"] = scores

    data["Interest_Match"] = interest_scores

    data["Career_Match"] = career_scores

    data["Skill_Match"] = skill_scores

    data["Education_Match"] = education_scores

    data["Level_Match"] = level_scores

    data["NLP_Match"] = nlp_scores_percent

    # --------------------------------------------------------
    # SORT BY SCORE
    # --------------------------------------------------------

    data = data.sort_values(
        by="AI_Score",
        ascending=False
    )

    data = data.reset_index(
        drop=True
    )

    return data


# ============================================================
# DIFFERENT COURSE SELECTION
# ============================================================

def select_different_courses(
    scored_data,
    top_n=5
):

    if scored_data.empty:
        return scored_data

    # Since duplicate course names are already removed,
    # this selection will never return the same course twice.

    selected = []

    used_courses = set()

    # --------------------------------------------------------
    # FIRST: Best matching course
    # --------------------------------------------------------

    for index, row in scored_data.iterrows():

        course_name = normalize(
            row_value(
                row,
                COURSE_COL
            )
        )

        if course_name not in used_courses:

            selected.append(index)

            used_courses.add(
                course_name
            )

            break

    # --------------------------------------------------------
    # NEXT: Different courses
    # --------------------------------------------------------

    for index, row in scored_data.iterrows():

        if len(selected) >= top_n:
            break

        course_name = normalize(
            row_value(
                row,
                COURSE_COL
            )
        )

        if course_name in used_courses:
            continue

        selected.append(index)

        used_courses.add(
            course_name
        )

    result = scored_data.loc[
        selected
    ].copy()

    result = result.reset_index(
        drop=True
    )

    return result


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "👤 Your Profile"
    )

    st.caption(
        "Tell us about yourself and find the best courses."
    )

    st.divider()

    # IMPORTANT:
    # These are TEXT INPUTS.
    # User types their own values.

    interest = st.text_input(
        "💡 Your Interest",
        placeholder="Example: Artificial Intelligence"
    )

    career_goal = st.text_input(
        "🎯 Career Goal",
        placeholder="Example: AI Engineer"
    )

    education = st.text_input(
        "🎓 Education",
        placeholder="Example: Diploma"
    )

    skills = st.text_input(
        "🛠️ Your Skills",
        placeholder="Example: Python, Pandas, SQL"
    )

    skill_level = st.selectbox(
        "📊 Skill Level",
        [
            "Beginner",
            "Intermediate",
            "Advanced"
        ]
    )

    st.divider()

    recommend_button = st.button(
        "✨ Get AI Recommendations",
        type="primary",
        use_container_width=True
    )

    st.caption(
        f"{len(df):,} records loaded"
    )


# ============================================================
# HOME SCREEN
# ============================================================

if not recommend_button:

    st.title(
        "🎓 Course Recommendation System"
    )

    st.subheader(
        "🤖 AI-Powered Personalized Course Recommendations"
    )

    st.write(
        "Enter your profile information from the sidebar "
        "and get personalized course recommendations."
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Dataset Records",
            f"{len(df):,}"
        )

    with col2:

        unique_courses = (
            df[COURSE_COL]
            .astype(str)
            .apply(normalize)
            .nunique()
        )

        st.metric(
            "Unique Courses",
            f"{unique_courses:,}"
        )

    with col3:

        st.metric(
            "AI Matching",
            "Weighted + NLP"
        )

    with col4:

        st.metric(
            "Recommendations",
            "Top 5"
        )

    st.divider()

    st.subheader(
        "Recommendation Method"
    )

    a, b, c = st.columns(3)

    with a:

        st.write(
            "### 🎯 Profile Matching"
        )

        st.write(
            "Interest, career goal, education, "
            "skills and skill level are compared."
        )

    with b:

        st.write(
            "### 🧠 NLP Matching"
        )

        st.write(
            "Related course concepts are detected "
            "using TF-IDF similarity."
        )

    with c:

        st.write(
            "### 🔀 Course Diversity"
        )

        st.write(
            "Duplicate course recommendations are removed."
        )

    st.stop()


# ============================================================
# VALIDATION
# ============================================================

if not interest.strip():

    st.warning(
        "Please enter your Interest."
    )

    st.stop()


if not career_goal.strip():

    st.warning(
        "Please enter your Career Goal."
    )

    st.stop()


if not education.strip():

    st.warning(
        "Please enter your Education."
    )

    st.stop()


if not skills.strip():

    st.warning(
        "Please enter your Skills."
    )

    st.stop()


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

with st.spinner(
    "AI is analyzing your profile..."
):

    scored_courses = recommend_courses(

        interest=interest,

        career_goal=career_goal,

        education=education,

        skills=skills,

        skill_level=skill_level
    )

    recommendations = select_different_courses(
        scored_courses,
        TOP_N
    )


# ============================================================
# RESULT HEADER
# ============================================================

st.title(
    "✨ AI Recommended Courses"
)

st.caption(
    "Recommendations are generated using weighted profile "
    "matching and NLP-based similarity."
)


# ============================================================
# PROFILE SUMMARY
# ============================================================

st.subheader(
    "👤 Your Profile"
)

p1, p2, p3, p4, p5 = st.columns(5)

with p1:

    st.write("💡 **Interest**")
    st.write(interest)

with p2:

    st.write("🎯 **Career Goal**")
    st.write(career_goal)

with p3:

    st.write("🎓 **Education**")
    st.write(education)

with p4:

    st.write("🛠️ **Skills**")
    st.write(skills)

with p5:

    st.write("📊 **Skill Level**")
    st.write(skill_level)


st.divider()


# ============================================================
# NO RESULTS
# ============================================================

if recommendations.empty:

    st.warning(
        "No suitable courses were found."
    )

    st.stop()


# ============================================================
# BEST MATCH
# ============================================================

best_course = recommendations.iloc[0]

best_name = row_value(
    best_course,
    COURSE_COL
)

best_score = float(
    best_course["AI_Score"]
)

st.subheader(
    "🏆 Best Match"
)

best1, best2 = st.columns(
    [4, 1]
)

with best1:

    st.success(
        f"Best Recommended Course: {best_name}"
    )

with best2:

    st.metric(
        "AI Match",
        f"{best_score:.0f}%"
    )


st.divider()


# ============================================================
# COURSE RECOMMENDATIONS
# ============================================================

st.subheader(
    "📚 Top 5 Different Course Recommendations"
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

    education_value = row_value(
        row,
        EDUCATION_COL
    )

    duration_value = row_value(
        row,
        DURATION_COL
    )

    rating_value = row_value(
        row,
        RATING_COL
    )

    salary_value = row_value(
        row,
        SALARY_COL
    )

    course_skills = row_value(
        row,
        SKILLS_COL
    )

    # --------------------------------------------------------
    # RANK
    # --------------------------------------------------------

    if index == 0:

        rank_text = "🥇 TOP MATCH"

    elif index == 1:

        rank_text = "🥈 SECOND BEST"

    elif index == 2:

        rank_text = "🥉 THIRD BEST"

    else:

        rank_text = f"⭐ RANK {index + 1}"

    # --------------------------------------------------------
    # COURSE TITLE
    # --------------------------------------------------------

    st.markdown(
        f"## {rank_text}"
    )

    st.markdown(
        f"### {course_name}"
    )

    # --------------------------------------------------------
    # AI SCORE
    # --------------------------------------------------------

    score_col1, score_col2 = st.columns(
        [5, 1]
    )

    with score_col1:

        st.progress(
            min(
                score / 100,
                1.0
            )
        )

    with score_col2:

        st.metric(
            "AI Match",
            f"{score:.0f}%"
        )

    # --------------------------------------------------------
    # COURSE DETAILS
    # --------------------------------------------------------

    d1, d2, d3, d4 = st.columns(4)

    with d1:

        st.write(
            "🎓 **Education**"
        )

        st.write(
            education_value
            if education_value
            else "Not specified"
        )

    with d2:

        st.write(
            "⏱️ **Duration**"
        )

        st.write(
            duration_value
            if duration_value
            else "Not specified"
        )

    with d3:

        st.write(
            "⭐ **Rating**"
        )

        st.write(
            rating_value
            if rating_value
            else "Not specified"
        )

    with d4:

        st.write(
            "💼 **Career Role**"
        )

        st.write(
            career_role
            if career_role
            else "Not specified"
        )

    if salary_value:

        st.write(
            f"💰 **Salary Range:** {salary_value}"
        )

    if course_skills:

        st.write(
            f"🛠️ **Course Skills:** {course_skills}"
        )

    # --------------------------------------------------------
    # AI MATCH DETAILS
    # --------------------------------------------------------

    with st.expander(
        "🔍 View AI Match Details"
    ):

        x1, x2, x3, x4, x5 = st.columns(5)

        with x1:

            st.metric(
                "Interest",
                f"{float(row['Interest_Match']):.0f}%"
            )

        with x2:

            st.metric(
                "Career",
                f"{float(row['Career_Match']):.0f}%"
            )

        with x3:

            st.metric(
                "Skills",
                f"{float(row['Skill_Match']):.0f}%"
            )

        with x4:

            st.metric(
                "Education",
                f"{float(row['Education_Match']):.0f}%"
            )

        with x5:

            st.metric(
                "NLP",
                f"{float(row['NLP_Match']):.0f}%"
            )

    st.divider()


# ============================================================
# SUMMARY TABLE
# ============================================================

st.subheader(
    "📊 Recommendation Summary"
)

summary_data = []

for index, (_, row) in enumerate(
    recommendations.iterrows()
):

    summary_data.append({

        "Rank":
            index + 1,

        "Course":
            row_value(
                row,
                COURSE_COL
            ),

        "AI Match":
            f"{float(row['AI_Score']):.0f}%",

        "Career Role":
            row_value(
                row,
                CAREER_COL
            ) or "Not specified",

        "Education":
            row_value(
                row,
                EDUCATION_COL
            ) or "Not specified",

        "Duration":
            row_value(
                row,
                DURATION_COL
            ) or "Not specified",

        "Rating":
            row_value(
                row,
                RATING_COL
            ) or "Not specified"
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
# AI SCORE CHART
# ============================================================

st.subheader(
    "📈 AI Match Comparison"
)

chart_data = pd.DataFrame({

    "Course": [
        row_value(
            row,
            COURSE_COL
        )
        for _, row
        in recommendations.iterrows()
    ],

    "AI Match": [
        round(
            float(row["AI_Score"]),
            1
        )
        for _, row
        in recommendations.iterrows()
    ]
})


st.bar_chart(
    chart_data.set_index(
        "Course"
    )
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Course Recommendation System | "
    "Weighted Profile Matching + NLP Similarity + "
    "Duplicate-Free Recommendations"
)
