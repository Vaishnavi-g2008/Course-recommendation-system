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
    layout="wide"
)


# ============================================================
# DATASET
# ============================================================

DATASET_FILE = "course_recommendation_dataset_1000.csv"


@st.cache_data
def load_data():

    if not os.path.exists(DATASET_FILE):
        return None

    data = pd.read_csv(DATASET_FILE)

    data.columns = [
        str(col).strip().lower().replace(" ", "_").replace("-", "_")
        for col in data.columns
    ]

    data = data.fillna("")

    return data


df = load_data()


if df is None:

    st.error(
        "course_recommendation_dataset_1000.csv was not found."
    )

    st.info(
        "Keep the CSV file in the same folder as app.py."
    )

    st.stop()


# ============================================================
# COLUMN DETECTION
# ============================================================

def detect_column(names):

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


course_col = detect_column([
    "course",
    "course_name",
    "course_title",
    "title",
    "program",
    "program_name"
])

interest_col = detect_column([
    "interest",
    "interests",
    "domain",
    "category",
    "field",
    "area",
    "course_domain"
])

career_col = detect_column([
    "career_goal",
    "career",
    "career_path",
    "job_role",
    "role",
    "recommended_role",
    "job",
    "target_role"
])

skills_col = detect_column([
    "skills",
    "skill",
    "required_skills",
    "technical_skills",
    "course_skills",
    "skill_set"
])

education_col = detect_column([
    "education",
    "qualification",
    "eligibility",
    "educational_qualification",
    "education_level",
    "required_education"
])

level_col = detect_column([
    "skill_level",
    "level",
    "difficulty",
    "difficulty_level",
    "experience_level"
])

duration_col = detect_column([
    "duration",
    "course_duration",
    "duration_months",
    "course_length"
])

rating_col = detect_column([
    "rating",
    "course_rating",
    "ratings",
    "review_rating"
])

salary_col = detect_column([
    "salary",
    "salary_range",
    "expected_salary",
    "salary_package",
    "package",
    "average_salary"
])


if course_col is None:

    st.error(
        "Course column could not be detected."
    )

    st.write(
        "Available columns:"
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
# GET VALUE
# ============================================================

def get_value(row, column):

    if column is None:
        return ""

    if column not in row.index:
        return ""

    value = row[column]

    if pd.isna(value):
        return ""

    return str(value).strip()


# ============================================================
# RELATED AI DOMAINS
# ============================================================

RELATED_TERMS = {

    "artificial intelligence": [
        "artificial intelligence",
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

    "ai": [
        "artificial intelligence",
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

    "deep learning": [
        "deep learning",
        "machine learning",
        "artificial intelligence",
        "ai",
        "neural network",
        "computer vision",
        "nlp",
        "natural language processing"
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

    result = [text]

    for key, values in RELATED_TERMS.items():

        if key in text:

            result.extend(values)

    return " ".join(result)


# ============================================================
# WORD MATCH
# ============================================================

def word_match(user_text, course_text):

    user_text = normalize(user_text)
    course_text = normalize(course_text)

    if not user_text or not course_text:
        return 0.0

    user_words = set(user_text.split())
    course_words = set(course_text.split())

    if not user_words:
        return 0.0

    common = user_words.intersection(course_words)

    return len(common) / len(user_words)


# ============================================================
# RELATED MATCH
# ============================================================

def related_match(user_text, course_text):

    expanded = expand_terms(user_text)

    return word_match(
        expanded,
        course_text
    )


# ============================================================
# EXACT / PARTIAL MATCH
# ============================================================

def direct_match(user_text, course_text):

    user_text = normalize(user_text)
    course_text = normalize(course_text)

    if not user_text or not course_text:
        return 0.0

    if user_text == course_text:
        return 1.0

    if user_text in course_text:
        return 1.0

    user_words = user_text.split()

    if len(user_words) == 1:

        if user_words[0] in course_text.split():
            return 1.0

    return word_match(
        user_text,
        course_text
    )


# ============================================================
# SKILL MATCH
# ============================================================

def skill_match(user_skills, course_skills):

    user_skills = normalize(
        user_skills
    )

    course_skills = normalize(
        course_skills
    )

    if not user_skills or not course_skills:
        return 0.0

    user_list = re.split(
        r",|;|\|",
        user_skills
    )

    user_list = [
        x.strip()
        for x in user_list
        if x.strip()
    ]

    if not user_list:
        return 0.0

    matched = 0

    for skill in user_list:

        if skill in course_skills:

            matched += 1

            continue

        related = expand_terms(skill)

        if any(
            term in course_skills
            for term in related.split()
        ):

            matched += 1

    return min(
        matched / len(user_list),
        1.0
    )


# ============================================================
# EDUCATION MATCH
# ============================================================

def education_match(user_education, course_education):

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
            "higher secondary",
            "hsc"
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
            "post graduation",
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

    if user_group is None or course_group is None:
        return 0.0

    if user_group == course_group:
        return 1.0

    return 0.0


# ============================================================
# SKILL LEVEL MATCH
# ============================================================

def level_match(user_level, course_level):

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

    level_map = {

        "beginner": [
            "beginner",
            "basic",
            "foundation",
            "entry"
        ],

        "intermediate": [
            "intermediate",
            "medium"
        ],

        "advanced": [
            "advanced",
            "expert",
            "professional"
        ]
    }

    user_group = None
    course_group = None

    for group, words in level_map.items():

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

    if user_group == course_group:
        return 1.0

    return 0.0


# ============================================================
# NLP SIMILARITY
# ============================================================

def calculate_nlp_score(
    user_profile,
    course_profiles
):

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

        scores = cosine_similarity(
            matrix[0:1],
            matrix[1:]
        )[0]

        return scores

    except Exception:

        return np.zeros(
            len(course_profiles)
        )


# ============================================================
# COURSE PROFILE
# ============================================================

def course_profile(row):

    values = [

        get_value(row, course_col),

        get_value(row, interest_col),

        get_value(row, career_col),

        get_value(row, skills_col),

        get_value(row, education_col),

        get_value(row, level_col)
    ]

    return " ".join(
        normalize(x)
        for x in values
        if x
    )


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

def recommend_courses(
    interest,
    career_goal,
    education,
    skills,
    skill_level
):

    result = df.copy()

    course_profiles = [

        course_profile(row)

        for _, row in result.iterrows()
    ]

    user_profile = " ".join([

        expand_terms(interest),

        expand_terms(career_goal),

        normalize(education),

        expand_terms(skills),

        normalize(skill_level)
    ])

    nlp_scores = calculate_nlp_score(
        user_profile,
        course_profiles
    )

    final_scores = []

    interest_scores = []
    career_scores = []
    skill_scores = []
    education_scores = []
    level_scores = []

    for i, (_, row) in enumerate(
        result.iterrows()
    ):

        course_name = get_value(
            row,
            course_col
        )

        course_interest = get_value(
            row,
            interest_col
        )

        course_career = get_value(
            row,
            career_col
        )

        course_skills = get_value(
            row,
            skills_col
        )

        course_education = get_value(
            row,
            education_col
        )

        course_level = get_value(
            row,
            level_col
        )

        # ----------------------------------------------------
        # INTEREST
        # ----------------------------------------------------

        interest_text = (
            course_interest
            + " "
            + course_name
        )

        interest_score = max(

            direct_match(
                interest,
                interest_text
            ),

            related_match(
                interest,
                interest_text
            )
        )

        # ----------------------------------------------------
        # CAREER
        # ----------------------------------------------------

        career_text = (
            course_career
            + " "
            + course_name
        )

        career_score = max(

            direct_match(
                career_goal,
                career_text
            ),

            related_match(
                career_goal,
                career_text
            )
        )

        # ----------------------------------------------------
        # SKILLS
        # ----------------------------------------------------

        skill_score = skill_match(
            skills,
            course_skills
        )

        # ----------------------------------------------------
        # EDUCATION
        # ----------------------------------------------------

        education_score = education_match(
            education,
            course_education
        )

        # ----------------------------------------------------
        # LEVEL
        # ----------------------------------------------------

        level_score = level_match(
            skill_level,
            course_level
        )

        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        nlp_score = nlp_scores[i]

        # Strong profile-based weighting
        score = (

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
        # EXACT MATCH BONUS
        # ----------------------------------------------------

        exact_matches = 0

        if direct_match(
            interest,
            interest_text
        ) >= 1:
            exact_matches += 1

        if direct_match(
            career_goal,
            career_text
        ) >= 1:
            exact_matches += 1

        if education_score >= 1:
            exact_matches += 1

        if skill_score >= 1:
            exact_matches += 1

        # Give a bonus when multiple profile fields
        # directly match the course.
        score += exact_matches * 3

        score = min(
            score,
            100
        )

        final_scores.append(
            score
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

    result["AI_Score"] = final_scores

    result["Interest_Match"] = interest_scores

    result["Career_Match"] = career_scores

    result["Skill_Match"] = skill_scores

    result["Education_Match"] = education_scores

    result["Level_Match"] = level_scores

    result = result.sort_values(
        "AI_Score",
        ascending=False
    )

    result = result.reset_index(
        drop=True
    )

    return result


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "👤 Your Profile"
    )

    st.write(
        "Enter your profile details. "
        "Course information is automatically loaded "
        "from the dataset."
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
        placeholder="Example: Python, SQL"
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
        "Enter your Interest, Career Goal, Education, "
        "Skills and Skill Level from the sidebar."
    )

    st.info(
        "Course name, duration, rating, salary, "
        "career role and other course information "
        "are automatically taken from the CSV dataset."
    )

    st.divider()

    st.subheader(
        "How It Works"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.write(
            "💡 Interest"
        )

        st.caption(
            "Identifies courses related to your interest."
        )

    with c2:

        st.write(
            "🎯 Career"
        )

        st.caption(
            "Matches your target career role."
        )

    with c3:

        st.write(
            "🛠️ Skills"
        )

        st.caption(
            "Checks your current skills."
        )

    with c4:

        st.write(
            "🧠 AI + NLP"
        )

        st.caption(
            "Finds related courses using NLP similarity."
        )

    st.stop()


# ============================================================
# VALIDATION
# ============================================================

if not interest.strip():

    st.error(
        "Please enter your Interest."
    )

    st.stop()


if not career_goal.strip():

    st.error(
        "Please enter your Career Goal."
    )

    st.stop()


if not education.strip():

    st.error(
        "Please enter your Education."
    )

    st.stop()


if not skills.strip():

    st.error(
        "Please enter your Skills."
    )

    st.stop()


# ============================================================
# GENERATE
# ============================================================

with st.spinner(
    "AI is analyzing your profile..."
):

    recommendations = recommend_courses(

        interest,

        career_goal,

        education,

        skills,

        skill_level
    )


# ============================================================
# RESULT HEADER
# ============================================================

st.title(
    "✨ AI Recommended Courses"
)

st.write(
    "These recommendations are generated from your profile "
    "and the course dataset."
)


# ============================================================
# USER PROFILE SUMMARY
# ============================================================

st.subheader(
    "👤 Your Profile"
)

p1, p2, p3, p4, p5 = st.columns(5)

with p1:
    st.write("💡 Interest")
    st.write(interest)

with p2:
    st.write("🎯 Career Goal")
    st.write(career_goal)

with p3:
    st.write("🎓 Education")
    st.write(education)

with p4:
    st.write("🛠️ Skills")
    st.write(skills)

with p5:
    st.write("📊 Skill Level")
    st.write(skill_level)


st.divider()


# ============================================================
# TOP RECOMMENDATION
# ============================================================

best = recommendations.iloc[0]

best_course = get_value(
    best,
    course_col
)

best_score = float(
    best["AI_Score"]
)

st.subheader(
    "🏆 Best Match"
)

st.success(
    f"{best_course} — {best_score:.0f}% AI Match"
)


# ============================================================
# TOP 5 RECOMMENDATIONS
# ============================================================

st.subheader(
    "📚 Recommended Courses"
)

top_n = min(
    5,
    len(recommendations)
)


for i in range(top_n):

    row = recommendations.iloc[i]

    course_name = get_value(
        row,
        course_col
    )

    score = float(
        row["AI_Score"]
    )

    role = get_value(
        row,
        career_col
    )

    st.markdown(
        f"### {i + 1}. {course_name}"
    )

    st.progress(
        min(score / 100, 1.0)
    )

    st.write(
        f"AI Match Score: **{score:.0f}%**"
    )

    st.write(
        f"Recommended Role: **{role if role else 'Not specified'}**"
    )

    # --------------------------------------------------------
    # INFORMATION COMES AUTOMATICALLY FROM DATASET
    # --------------------------------------------------------

    info1, info2, info3, info4 = st.columns(4)

    with info1:

        st.write("🎓 Education")

        st.write(
            get_value(
                row,
                education_col
            ) or "Not specified"
        )

    with info2:

        st.write("⏱️ Duration")

        st.write(
            get_value(
                row,
                duration_col
            ) or "Not specified"
        )

    with info3:

        st.write("⭐ Rating")

        st.write(
            get_value(
                row,
                rating_col
            ) or "Not specified"
        )

    with info4:

        st.write("💰 Salary")

        st.write(
            get_value(
                row,
                salary_col
            ) or "Not specified"
        )

    st.write("🛠️ Course Skills")

    st.write(
        get_value(
            row,
            skills_col
        ) or "Not specified"
    )

    with st.expander(
        "View AI Match Details"
    ):

        b1, b2, b3 = st.columns(3)

        with b1:

            st.metric(
                "Interest",
                f"{row['Interest_Match']:.0f}%"
            )

        with b2:

            st.metric(
                "Career",
                f"{row['Career_Match']:.0f}%"
            )

        with b3:

            st.metric(
                "Skills",
                f"{row['Skill_Match']:.0f}%"
            )

        b4, b5 = st.columns(2)

        with b4:

            st.metric(
                "Education",
                f"{row['Education_Match']:.0f}%"
            )

        with b5:

            st.metric(
                "Skill Level",
                f"{row['Level_Match']:.0f}%"
            )

    st.divider()


# ============================================================
# RECOMMENDATION TABLE
# ============================================================

st.subheader(
    "📊 Recommendation Summary"
)

table_data = []

for i in range(top_n):

    row = recommendations.iloc[i]

    table_data.append({

        "Rank":
            i + 1,

        "Course":
            get_value(
                row,
                course_col
            ),

        "AI Match":
            f"{float(row['AI_Score']):.0f}%",

        "Career Role":
            get_value(
                row,
                career_col
            ) or "Not specified",

        "Education":
            get_value(
                row,
                education_col
            ) or "Not specified",

        "Duration":
            get_value(
                row,
                duration_col
            ) or "Not specified",

        "Rating":
            get_value(
                row,
                rating_col
            ) or "Not specified",

        "Salary":
            get_value(
                row,
                salary_col
            ) or "Not specified"
    })


summary = pd.DataFrame(
    table_data
)

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SIMPLE SCORE DISPLAY
# ============================================================

st.subheader(
    "📈 AI Match Scores"
)

score_df = pd.DataFrame({

    "Course":
        [
            get_value(
                recommendations.iloc[i],
                course_col
            )
            for i in range(top_n)
        ],

    "AI Match Score":
        [
            round(
                float(
                    recommendations.iloc[i]["AI_Score"]
                ),
                1
            )
            for i in range(top_n)
        ]
})


st.bar_chart(
    score_df.set_index(
        "Course"
    )
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Course Recommendation System | "
    "Weighted Intelligent Matching + NLP Similarity"
)
