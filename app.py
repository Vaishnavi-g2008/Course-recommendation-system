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

TOP_N = 5

# Dataset possible names
DATASET_FILES = [
    "course_recommendation_dataset_1000.csv",
    "course_recommendation_dataset_1000.cs"
]


# ============================================================
# FIND DATASET FILE
# ============================================================

def find_dataset_file():

    # Check exact expected names
    for file_name in DATASET_FILES:

        if os.path.exists(file_name):
            return file_name

    # Automatically find CSV or CS files
    try:

        for file_name in os.listdir("."):

            if (
                file_name.lower().endswith(".csv")
                or file_name.lower().endswith(".cs")
            ):

                return file_name

    except Exception:
        pass

    return None


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    dataset_file = find_dataset_file()

    if dataset_file is None:
        return None, None

    try:

        data = pd.read_csv(dataset_file)

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

        return data, dataset_file

    except Exception:
        return None, dataset_file


# ============================================================
# LOAD DATA
# ============================================================

df, loaded_dataset_file = load_data()


# ============================================================
# DATASET NOT FOUND
# ============================================================

if df is None:

    st.error("❌ Dataset file not found or cannot be read.")

    st.write("Files available in project folder:")

    try:
        st.write(os.listdir("."))
    except Exception:
        pass

    st.info(
        "Keep the dataset file in the same folder as app.py."
    )

    st.stop()


# ============================================================
# FIND COLUMN
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
# COURSE COLUMN CHECK
# ============================================================

if COURSE_COL is None:

    st.error("❌ Course column was not found.")

    st.write("Available columns:")

    st.write(list(df.columns))

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
# GET VALUE FROM ROW
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
# RELATED TERMS
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
# TEXT MATCH
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

    user_words = set(user_text.split())
    course_words = set(course_text.split())

    if not user_words:
        return 0.0

    common_words = user_words.intersection(course_words)

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

def calculate_skill_match(user_skills, course_skills):

    user_skills = normalize(user_skills)
    course_skills = normalize(course_skills)

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

    matched = 0

    for skill in user_skill_list:

        if skill in course_skills:
            matched += 1

    return min(
        matched / len(user_skill_list),
        1.0
    )


# ============================================================
# EDUCATION MATCH
# ============================================================

def calculate_education_match(user_education, course_education):

    user = normalize(user_education)
    course = normalize(course_education)

    if not user or not course:
        return 0.0

    if user == course:
        return 1.0

    if user in course or course in user:
        return 1.0

    return 0.0


# ============================================================
# LEVEL MATCH
# ============================================================

def calculate_level_match(user_level, course_level):

    user = normalize(user_level)
    course = normalize(course_level)

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

        row_value(row, COURSE_COL),
        row_value(row, INTEREST_COL),
        row_value(row, CAREER_COL),
        row_value(row, SKILLS_COL),
        row_value(row, EDUCATION_COL),
        row_value(row, LEVEL_COL)

    ]

    return " ".join(
        normalize(value)
        for value in values
        if value
    )


# ============================================================
# PREPARE UNIQUE COURSES
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

    data = data.drop_duplicates(
        subset=["_course_key"],
        keep="first"
    )

    return data.reset_index(drop=True)


# ============================================================
# RECOMMEND COURSES
# ============================================================

def recommend_courses(
    interest,
    career_goal,
    education,
    skills,
    skill_level
):

    data = prepare_unique_courses()

    course_profiles = [
        create_course_profile(row)
        for _, row in data.iterrows()
    ]

    user_profile = " ".join([

        expand_terms(interest),
        expand_terms(career_goal),
        normalize(education),
        expand_terms(skills),
        normalize(skill_level)

    ])

    documents = [user_profile] + course_profiles

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

        nlp_scores = np.zeros(len(data))

    scores = []

    for index, (_, row) in enumerate(data.iterrows()):

        course_name = row_value(row, COURSE_COL)

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

        interest_score = related_match(
            interest,
            course_name + " " + course_interest
        )

        career_score = related_match(
            career_goal,
            course_name + " " + course_career
        )

        skill_score = calculate_skill_match(
            skills,
            course_skills
        )

        education_score = calculate_education_match(
            education,
            course_education
        )

        level_score = calculate_level_match(
            skill_level,
            course_level
        )

        nlp_score = float(
            nlp_scores[index]
        )

        final_score = (

            interest_score * 30
            + career_score * 30
            + skill_score * 15
            + education_score * 10
            + level_score * 5
            + nlp_score * 10

        )

        final_score = min(
            final_score,
            100
        )

        scores.append(final_score)

    data["AI_Score"] = scores

    data = data.sort_values(
        by="AI_Score",
        ascending=False
    )

    return data.reset_index(drop=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("👤 Your Profile")

    st.caption(
        "Tell us about yourself and find the best courses."
    )

    st.divider()

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

    st.success(
        f"Dataset Loaded: {loaded_dataset_file}"
    )

    st.caption(
        f"{len(df):,} records loaded"
    )


# ============================================================
# HOME SCREEN
# ============================================================

if not recommend_button:

    st.title("🎓 Course Recommendation System")

    st.subheader(
        "🤖 AI-Powered Personalized Course Recommendations"
    )

    st.write(
        "Enter your profile information from the sidebar "
        "and click Get AI Recommendations."
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
            "NLP + Profile"
        )

    with col4:
        st.metric(
            "Recommendations",
            "Top 5"
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
    "🤖 AI is analyzing your profile..."
):

    recommendations = recommend_courses(

        interest,
        career_goal,
        education,
        skills,
        skill_level

    ).head(TOP_N)


# ============================================================
# RESULTS
# ============================================================

st.title("✨ AI Recommended Courses")

st.caption(
    "Personalized recommendations based on your profile."
)

st.divider()


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

    st.subheader("🏆 Best Match")

    c1, c2 = st.columns([4, 1])

    with c1:

        st.success(
            f"Best Recommended Course: {best_name}"
        )

    with c2:

        st.metric(
            "AI Match",
            f"{best_score:.0f}%"
        )


# ============================================================
# COURSE RECOMMENDATIONS
# ============================================================

st.divider()

st.subheader("📚 Top 5 Course Recommendations")

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

    duration_value = row_value(
        row,
        DURATION_COL
    )

    rating_value = row_value(
        row,
        RATING_COL
    )

    course_skills = row_value(
        row,
        SKILLS_COL
    )

    st.markdown(
        f"### ⭐ Rank {index + 1}: {course_name}"
    )

    col1, col2 = st.columns([5, 1])

    with col1:

        st.progress(
            min(score / 100, 1.0)
        )

    with col2:

        st.metric(
            "Match",
            f"{score:.0f}%"
        )

    d1, d2, d3 = st.columns(3)

    with d1:

        st.write("💼 **Career Role**")

        st.write(
            career_role
            if career_role
            else "Not specified"
        )

    with d2:

        st.write("⏱️ **Duration**")

        st.write(
            duration_value
            if duration_value
            else "Not specified"
        )

    with d3:

        st.write("⭐ **Rating**")

        st.write(
            rating_value
            if rating_value
            else "Not specified"
        )

    if course_skills:

        st.write(
            f"🛠️ **Skills:** {course_skills}"
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

        "Rank": index + 1,

        "Course": row_value(
            row,
            COURSE_COL
        ),

        "AI Match": (
            f"{float(row['AI_Score']):.0f}%"
        ),

        "Career Role": row_value(
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
# AI SCORE CHART
# ============================================================

st.subheader(
    "📈 AI Match Comparison"
)

chart_data = pd.DataFrame({

    "Course": [
        row_value(row, COURSE_COL)
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
    chart_data.set_index("Course")
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 Course Recommendation System | "
    "AI-Powered Personalized Recommendations"
)
