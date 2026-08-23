import streamlit as st
import pandas as pd
import numpy as np
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Course Recommendation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONSTANTS
# ============================================================

DATASET_FILE = "course_recommendation_dataset_1000.csv"
NUMBER_OF_RECOMMENDATIONS = 5


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    if not os.path.exists(DATASET_FILE):
        return None

    data = pd.read_csv(DATASET_FILE)

    data.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        for column in data.columns
    ]

    data = data.fillna("")

    return data


df = load_dataset()


# ============================================================
# DATASET ERROR
# ============================================================

if df is None:

    st.error(
        "Dataset file not found."
    )

    st.info(
        "Keep course_recommendation_dataset_1000.csv "
        "in the same folder as app.py."
    )

    st.stop()


# ============================================================
# COLUMN DETECTION
# ============================================================

def find_column(names):

    for name in names:

        normalized_name = (
            name
            .lower()
            .strip()
            .replace(" ", "_")
            .replace("-", "_")
        )

        if normalized_name in df.columns:
            return normalized_name

    return None


course_column = find_column([
    "course",
    "course_name",
    "course_title",
    "title",
    "program",
    "program_name"
])

interest_column = find_column([
    "interest",
    "interests",
    "domain",
    "category",
    "field",
    "area"
])

career_column = find_column([
    "career_goal",
    "career",
    "career_path",
    "job_role",
    "role",
    "recommended_role"
])

skills_column = find_column([
    "skills",
    "skill",
    "required_skills",
    "technical_skills",
    "course_skills"
])

education_column = find_column([
    "education",
    "qualification",
    "eligibility",
    "education_level",
    "educational_qualification"
])

level_column = find_column([
    "skill_level",
    "level",
    "difficulty",
    "difficulty_level",
    "experience_level"
])

duration_column = find_column([
    "duration",
    "course_duration",
    "duration_months",
    "course_length"
])

rating_column = find_column([
    "rating",
    "course_rating",
    "ratings"
])

salary_column = find_column([
    "salary",
    "salary_range",
    "expected_salary",
    "salary_package",
    "package"
])


if course_column is None:

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

def normalize(value):

    if value is None:
        return ""

    value = str(value).lower().strip()

    value = value.replace("&", " and ")
    value = value.replace("/", " ")

    value = re.sub(
        r"[^a-zA-Z0-9+#.\s]",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


# ============================================================
# GET COLUMN VALUE
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
# RELATED CONCEPTS
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

def expand_text(text):

    text = normalize(text)

    expanded = [text]

    for key, values in RELATED_TERMS.items():

        if key in text:

            expanded.extend(values)

    return " ".join(expanded)


# ============================================================
# TOKEN MATCH
# ============================================================

def token_match(user_text, course_text):

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

    matched_words = (
        user_words.intersection(
            course_words
        )
    )

    return len(
        matched_words
    ) / len(
        user_words
    )


# ============================================================
# RELATED MATCH
# ============================================================

def related_match(user_text, course_text):

    normal_score = token_match(
        user_text,
        course_text
    )

    expanded_score = token_match(
        expand_text(user_text),
        course_text
    )

    return max(
        normal_score,
        expanded_score
    )


# ============================================================
# SKILL MATCH
# ============================================================

def skill_match(user_skills, course_skills):

    user_skills = normalize(user_skills)
    course_skills = normalize(course_skills)

    if not user_skills or not course_skills:
        return 0.0

    skills = re.split(
        r",|;|\|",
        user_skills
    )

    skills = [
        skill.strip()
        for skill in skills
        if skill.strip()
    ]

    if not skills:
        return 0.0

    matched = 0

    for skill in skills:

        if skill in course_skills:

            matched += 1
            continue

        expanded = expand_text(skill)

        if any(
            word in course_skills
            for word in expanded.split()
        ):

            matched += 1

    return min(
        matched / len(skills),
        1.0
    )


# ============================================================
# EDUCATION MATCH
# ============================================================

def education_match(
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

    groups = {

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

    for group, words in groups.items():

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
# LEVEL MATCH
# ============================================================

def level_match(
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

        get_value(
            row,
            course_column
        ),

        get_value(
            row,
            interest_column
        ),

        get_value(
            row,
            career_column
        ),

        get_value(
            row,
            skills_column
        ),

        get_value(
            row,
            education_column
        ),

        get_value(
            row,
            level_column
        )
    ]

    return " ".join(
        normalize(value)
        for value in values
        if value
    )


# ============================================================
# UNIQUE COURSE DATA
# ============================================================

def create_unique_courses():

    working = df.copy()

    working["Course_Key"] = (
        working[course_column]
        .apply(normalize)
    )

    working = working[
        working["Course_Key"] != ""
    ]

    return working


# ============================================================
# AI RECOMMENDATION ENGINE
# ============================================================

def generate_recommendations(
    interest,
    career_goal,
    education,
    skills,
    skill_level
):

    data = create_unique_courses()

    # --------------------------------------------------------
    # CREATE COURSE PROFILES
    # --------------------------------------------------------

    profiles = [
        create_course_profile(row)
        for _, row in data.iterrows()
    ]

    # --------------------------------------------------------
    # USER PROFILE
    # --------------------------------------------------------

    user_profile = " ".join([

        expand_text(interest),

        expand_text(career_goal),

        normalize(education),

        expand_text(skills),

        normalize(skill_level)

    ])

    # --------------------------------------------------------
    # NLP TF-IDF
    # --------------------------------------------------------

    documents = [
        user_profile
    ] + profiles

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
    # SCORE EACH COURSE
    # --------------------------------------------------------

    final_scores = []

    interest_scores = []
    career_scores = []
    skill_scores = []
    education_scores = []
    level_scores = []
    nlp_percentages = []

    for index, (_, row) in enumerate(
        data.iterrows()
    ):

        course_name = get_value(
            row,
            course_column
        )

        course_interest = get_value(
            row,
            interest_column
        )

        course_career = get_value(
            row,
            career_column
        )

        course_skills = get_value(
            row,
            skills_column
        )

        course_education = get_value(
            row,
            education_column
        )

        course_level = get_value(
            row,
            level_column
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
        # NLP
        # ----------------------------------------------------

        nlp_score = float(
            nlp_scores[index]
        )

        # ----------------------------------------------------
        # WEIGHTED AI SCORE
        #
        # Interest     = 30%
        # Career Goal  = 30%
        # Skills       = 15%
        # Education    = 10%
        # Skill Level  = 5%
        # NLP          = 10%
        # ----------------------------------------------------

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
        # SMALL EXACT MATCH BONUS
        # ----------------------------------------------------

        if interest_score >= 1:
            score += 2

        if career_score >= 1:
            score += 2

        if skill_score >= 1:
            score += 1

        if education_score >= 1:
            score += 1

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

        nlp_percentages.append(
            nlp_score * 100
        )

    # --------------------------------------------------------
    # ADD SCORES
    # --------------------------------------------------------

    data["AI_Score"] = final_scores

    data["Interest_Match"] = interest_scores

    data["Career_Match"] = career_scores

    data["Skill_Match"] = skill_scores

    data["Education_Match"] = education_scores

    data["Level_Match"] = level_scores

    data["NLP_Match"] = nlp_percentages

    # --------------------------------------------------------
    # SORT
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
# DIVERSIFIED RECOMMENDATION SELECTION
# ============================================================

def select_diverse_courses(
    scored_data,
    number_of_courses=5
):

    if scored_data.empty:
        return scored_data

    # --------------------------------------------------------
    # COURSE NAMES
    # --------------------------------------------------------

    course_names = [
        normalize(
            get_value(
                row,
                course_column
            )
        )
        for _, row
        in scored_data.iterrows()
    ]

    # --------------------------------------------------------
    # COURSE TITLE SIMILARITY
    # --------------------------------------------------------

    try:

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2)
        )

        title_matrix = vectorizer.fit_transform(
            course_names
        )

        title_similarity = cosine_similarity(
            title_matrix
        )

    except Exception:

        title_similarity = np.zeros(
            (
                len(course_names),
                len(course_names)
            )
        )

    # --------------------------------------------------------
    # GREEDY DIVERSIFICATION
    # --------------------------------------------------------

    selected_indices = []

    remaining = list(
        range(
            len(scored_data)
        )
    )

    # Always select the best course first
    first_index = 0

    selected_indices.append(
        first_index
    )

    if first_index in remaining:
        remaining.remove(
            first_index
        )

    # --------------------------------------------------------
    # SELECT NEXT DIFFERENT COURSES
    # --------------------------------------------------------

    while (
        remaining
        and
        len(selected_indices)
        < number_of_courses
    ):

        best_index = None
        best_value = -999

        for candidate in remaining:

            original_score = float(
                scored_data.iloc[
                    candidate
                ]["AI_Score"]
            )

            # ------------------------------------------------
            # Similarity penalty
            # ------------------------------------------------

            if selected_indices:

                maximum_similarity = max(

                    title_similarity[
                        candidate,
                        selected
                    ]

                    for selected
                    in selected_indices
                )

            else:

                maximum_similarity = 0

            # ------------------------------------------------
            # Diversity score
            # ------------------------------------------------

            diversity_value = (

                original_score

                -

                (
                    maximum_similarity
                    * 20
                )
            )

            # ------------------------------------------------
            # Prefer completely different course names
            # ------------------------------------------------

            candidate_name = course_names[
                candidate
            ]

            same_as_existing = any(

                candidate_name
                ==
                course_names[selected]

                for selected
                in selected_indices
            )

            if same_as_existing:
                continue

            if diversity_value > best_value:

                best_value = diversity_value

                best_index = candidate

        if best_index is None:
            break

        selected_indices.append(
            best_index
        )

        remaining.remove(
            best_index
        )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    selected = scored_data.iloc[
        selected_indices
    ].copy()

    selected = selected.reset_index(
        drop=True
    )

    return selected


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

    recommend = st.button(
        "✨ Get AI Recommendations",
        type="primary",
        use_container_width=True
    )

    st.caption(
        f"{len(df):,} course records loaded"
    )


# ============================================================
# WELCOME SCREEN
# ============================================================

if not recommend:

    st.title(
        "🎓 Course Recommendation System"
    )

    st.subheader(
        "🤖 AI-Powered Personalized Course Recommendations"
    )

    st.write(
        "Enter your profile details from the sidebar "
        "to discover courses that match your interests, "
        "career goals and skills."
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Course Records",
            f"{len(df):,}"
        )

    with col2:

        unique_count = (
            df[course_column]
            .astype(str)
            .apply(normalize)
            .nunique()
        )

        st.metric(
            "Unique Courses",
            f"{unique_count:,}"
        )

    with col3:

        st.metric(
            "Recommendation Type",
            "AI + NLP"
        )

    with col4:

        st.metric(
            "Results",
            "Top 5"
        )

    st.divider()

    st.subheader(
        "How It Works"
    )

    a, b, c = st.columns(3)

    with a:

        st.write(
            "### 🎯 Profile Matching"
        )

        st.write(
            "Your interest, career goal, education "
            "and skills are compared with course data."
        )

    with b:

        st.write(
            "### 🧠 NLP Similarity"
        )

        st.write(
            "Related concepts are identified using "
            "TF-IDF based text similarity."
        )

    with c:

        st.write(
            "### 🔀 Course Diversity"
        )

        st.write(
            "Repeated courses are removed and "
            "different relevant courses are selected."
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

    scored_courses = generate_recommendations(

        interest,
        career_goal,
        education,
        skills,
        skill_level
    )

    recommendations = select_diverse_courses(
        scored_courses,
        NUMBER_OF_RECOMMENDATIONS
    )


# ============================================================
# RESULT HEADER
# ============================================================

st.title(
    "✨ AI Recommended Courses"
)

st.caption(
    "Recommendations are generated using weighted profile "
    "matching, related-concept matching and NLP similarity."
)


# ============================================================
# PROFILE SUMMARY
# ============================================================

st.subheader(
    "👤 Your Profile"
)

p1, p2, p3, p4, p5 = st.columns(5)

with p1:

    st.metric(
        "Interest",
        interest
    )

with p2:

    st.metric(
        "Career Goal",
        career_goal
    )

with p3:

    st.metric(
        "Education",
        education
    )

with p4:

    st.metric(
        "Skills",
        skills
    )

with p5:

    st.metric(
        "Skill Level",
        skill_level
    )


st.divider()


# ============================================================
# BEST MATCH
# ============================================================

if not recommendations.empty:

    best = recommendations.iloc[0]

    best_name = get_value(
        best,
        course_column
    )

    best_score = float(
        best["AI_Score"]
    )

    st.subheader(
        "🏆 Best Match"
    )

    best_col1, best_col2 = st.columns(
        [3, 1]
    )

    with best_col1:

        st.success(
            f"Recommended Course: {best_name}"
        )

    with best_col2:

        st.metric(
            "AI Match",
            f"{best_score:.0f}%"
        )


st.divider()


# ============================================================
# RECOMMENDED COURSE CARDS
# ============================================================

st.subheader(
    "📚 Top Different Course Recommendations"
)

st.caption(
    "Each recommendation is a different course."
)


for index, (_, row) in enumerate(
    recommendations.iterrows()
):

    course_name = get_value(
        row,
        course_column
    )

    score = float(
        row["AI_Score"]
    )

    role = get_value(
        row,
        career_column
    )

    education_value = get_value(
        row,
        education_column
    )

    duration_value = get_value(
        row,
        duration_column
    )

    rating_value = get_value(
        row,
        rating_column
    )

    salary_value = get_value(
        row,
        salary_column
    )

    course_skills = get_value(
        row,
        skills_column
    )

    # --------------------------------------------------------
    # COURSE HEADER
    # --------------------------------------------------------

    if index == 0:

        st.markdown(
            f"### 🥇 {course_name}"
        )

    elif index == 1:

        st.markdown(
            f"### 🥈 {course_name}"
        )

    elif index == 2:

        st.markdown(
            f"### 🥉 {course_name}"
        )

    else:

        st.markdown(
            f"### #{index + 1} {course_name}"
        )

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    score_col1, score_col2 = st.columns(
        [4, 1]
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
    # COURSE INFORMATION
    # --------------------------------------------------------

    info1, info2, info3, info4 = st.columns(4)

    with info1:

        st.write(
            "🎓 **Education**"
        )

        st.write(
            education_value
            if education_value
            else "Not specified"
        )

    with info2:

        st.write(
            "⏱️ **Duration**"
        )

        st.write(
            duration_value
            if duration_value
            else "Not specified"
        )

    with info3:

        st.write(
            "⭐ **Rating**"
        )

        st.write(
            rating_value
            if rating_value
            else "Not specified"
        )

    with info4:

        st.write(
            "💼 **Career Role**"
        )

        st.write(
            role
            if role
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

        d1, d2, d3, d4, d5 = st.columns(5)

        with d1:

            st.metric(
                "Interest",
                f"{float(row['Interest_Match']):.0f}%"
            )

        with d2:

            st.metric(
                "Career",
                f"{float(row['Career_Match']):.0f}%"
            )

        with d3:

            st.metric(
                "Skills",
                f"{float(row['Skill_Match']):.0f}%"
            )

        with d4:

            st.metric(
                "Education",
                f"{float(row['Education_Match']):.0f}%"
            )

        with d5:

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

summary_rows = []

for index, (_, row) in enumerate(
    recommendations.iterrows()
):

    summary_rows.append({

        "Rank":
            index + 1,

        "Course":
            get_value(
                row,
                course_column
            ),

        "AI Match":
            f"{float(row['AI_Score']):.0f}%",

        "Career Role":
            get_value(
                row,
                career_column
            ) or "Not specified",

        "Education":
            get_value(
                row,
                education_column
            ) or "Not specified",

        "Duration":
            get_value(
                row,
                duration_column
            ) or "Not specified",

        "Rating":
            get_value(
                row,
                rating_column
            ) or "Not specified"
    })


summary_df = pd.DataFrame(
    summary_rows
)


st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# AI MATCH CHART
# ============================================================

st.subheader(
    "📈 AI Match Comparison"
)

chart_df = pd.DataFrame({

    "Course": [
        get_value(
            row,
            course_column
        )
        for _, row
        in recommendations.iterrows()
    ],

    "AI Match Score": [
        round(
            float(row["AI_Score"]),
            1
        )
        for _, row
        in recommendations.iterrows()
    ]
})


st.bar_chart(
    chart_df.set_index(
        "Course"
    )
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Course Recommendation System | "
    "Weighted Matching + NLP Similarity + Course Diversification"
)
