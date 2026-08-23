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
# CUSTOM STREAMLIT SETTINGS
# ============================================================

st.markdown(
    """
    # 🎓 Course Recommendation System
    ### 🤖 AI-Powered Personalized Learning Recommendations

    Find the most suitable courses based on your **Interest, Career Goal,
    Education, Skills and Skill Level**.
    """
)

st.divider()


# ============================================================
# DATASET FILE
# ============================================================

DATASET_FILE = "course_recommendation_dataset_1000.csv"


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    if not os.path.exists(DATASET_FILE):
        return None

    try:

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

        # Remove completely empty columns
        data = data.dropna(axis=1, how="all")

        # Fill missing values
        data = data.fillna("")

        return data

    except Exception as error:

        st.error(
            f"Dataset load error: {error}"
        )

        return None


df = load_dataset()


# ============================================================
# DATASET NOT FOUND
# ============================================================

if df is None:

    st.error(
        f"❌ `{DATASET_FILE}` सापडली नाही."
    )

    st.info(
        "कृपया `course_recommendation_dataset_1000.csv` "
        "ही file `app.py` च्या same folder मध्ये ठेवा."
    )

    st.stop()


# ============================================================
# COLUMN DETECTION FUNCTION
# ============================================================

def find_column(possible_names):

    for name in possible_names:

        normalized = (
            name.lower()
            .strip()
            .replace(" ", "_")
            .replace("-", "_")
        )

        if normalized in df.columns:
            return normalized

    return None


# ============================================================
# DETECT IMPORTANT COLUMNS
# ============================================================

course_col = find_column([
    "course",
    "course_name",
    "coursename",
    "title",
    "course_title",
    "program",
    "program_name"
])


interest_col = find_column([
    "interest",
    "interests",
    "domain",
    "category",
    "field",
    "area",
    "course_domain"
])


career_col = find_column([
    "career_goal",
    "career",
    "career_path",
    "job_role",
    "role",
    "recommended_role",
    "job",
    "target_role"
])


skills_col = find_column([
    "skills",
    "skill",
    "required_skills",
    "technical_skills",
    "course_skills",
    "skill_set"
])


education_col = find_column([
    "education",
    "qualification",
    "eligibility",
    "educational_qualification",
    "education_level",
    "required_education"
])


skill_level_col = find_column([
    "skill_level",
    "level",
    "difficulty",
    "difficulty_level",
    "experience_level"
])


duration_col = find_column([
    "duration",
    "course_duration",
    "duration_months",
    "course_length"
])


rating_col = find_column([
    "rating",
    "course_rating",
    "ratings",
    "review_rating"
])


salary_col = find_column([
    "salary",
    "salary_range",
    "expected_salary",
    "salary_package",
    "package",
    "average_salary"
])


# ============================================================
# FALLBACK COURSE COLUMN
# ============================================================

if course_col is None:

    # Try to find a suitable text column
    for column in df.columns:

        if df[column].dtype == "object":

            course_col = column

            break


# ============================================================
# VALIDATE COURSE COLUMN
# ============================================================

if course_col is None:

    st.error(
        "❌ Dataset मध्ये Course Name column सापडला नाही."
    )

    st.write(
        "Available columns:"
    )

    st.write(
        list(df.columns)
    )

    st.stop()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    value = str(value).lower()

    value = value.replace("&", " and ")

    value = value.replace("/", " ")

    value = value.replace(",", " ")

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
# GET SAFE VALUE
# ============================================================

def get_value(row, column):

    if column is None:
        return ""

    if column not in row.index:
        return ""

    return clean_text(
        row[column]
    )


# ============================================================
# ORIGINAL VALUE
# ============================================================

def get_original_value(row, column):

    if column is None:
        return "Not specified"

    if column not in row.index:
        return "Not specified"

    value = row[column]

    if pd.isna(value):
        return "Not specified"

    value = str(value).strip()

    if value == "":
        return "Not specified"

    return value


# ============================================================
# RELATED DOMAIN KNOWLEDGE
# ============================================================

RELATED_DOMAINS = {

    "artificial intelligence": [
        "artificial intelligence",
        "ai",
        "machine learning",
        "deep learning",
        "data science",
        "neural networks",
        "natural language processing",
        "nlp",
        "computer vision",
        "generative ai"
    ],

    "ai": [
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "data science",
        "neural networks",
        "nlp",
        "computer vision",
        "generative ai"
    ],

    "machine learning": [
        "machine learning",
        "artificial intelligence",
        "deep learning",
        "data science",
        "predictive analytics",
        "neural networks"
    ],

    "deep learning": [
        "deep learning",
        "machine learning",
        "artificial intelligence",
        "neural networks",
        "computer vision",
        "natural language processing",
        "nlp"
    ],

    "data science": [
        "data science",
        "machine learning",
        "artificial intelligence",
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
        "business intelligence",
        "python",
        "sql",
        "power bi",
        "tableau"
    ],

    "web development": [
        "web development",
        "frontend development",
        "backend development",
        "full stack development",
        "javascript",
        "react",
        "html",
        "css"
    ],

    "full stack development": [
        "full stack development",
        "web development",
        "frontend",
        "backend",
        "javascript",
        "react",
        "node",
        "database"
    ],

    "cloud computing": [
        "cloud computing",
        "aws",
        "azure",
        "google cloud",
        "cloud engineer",
        "devops",
        "cloud architecture"
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

def expand_related_terms(text):

    text = clean_text(text)

    if not text:
        return ""

    terms = [text]

    for key, related_terms in RELATED_DOMAINS.items():

        if key in text:

            terms.extend(
                related_terms
            )

    return " ".join(terms)


# ============================================================
# TOKENIZATION
# ============================================================

def get_tokens(text):

    text = clean_text(text)

    if not text:
        return set()

    return set(
        text.split()
    )


# ============================================================
# TOKEN SIMILARITY
# ============================================================

def token_similarity(
    user_text,
    course_text
):

    user_tokens = get_tokens(
        user_text
    )

    course_tokens = get_tokens(
        course_text
    )

    if not user_tokens:
        return 0.0

    if not course_tokens:
        return 0.0

    common_tokens = (
        user_tokens
        .intersection(course_tokens)
    )

    return (
        len(common_tokens)
        /
        len(user_tokens)
    )


# ============================================================
# EXACT / PHRASE MATCH
# ============================================================

def phrase_match(
    user_text,
    course_text
):

    user_text = clean_text(
        user_text
    )

    course_text = clean_text(
        course_text
    )

    if not user_text or not course_text:
        return 0.0

    if user_text in course_text:
        return 1.0

    user_words = user_text.split()

    if len(user_words) == 1:

        if user_words[0] in course_text.split():
            return 1.0

    return 0.0


# ============================================================
# SKILL MATCHING
# ============================================================

def skill_matching(
    user_skills,
    course_skills
):

    user_skill_list = [
        x.strip()
        for x in re.split(
            r",|;|\|",
            clean_text(user_skills)
        )
        if x.strip()
    ]

    course_skill_list = [
        x.strip()
        for x in re.split(
            r",|;|\|",
            clean_text(course_skills)
        )
        if x.strip()
    ]

    if not user_skill_list:
        return 0.0

    if not course_skill_list:
        return 0.0

    matched = 0

    for user_skill in user_skill_list:

        for course_skill in course_skill_list:

            if (
                user_skill in course_skill
                or
                course_skill in user_skill
            ):

                matched += 1

                break

    return min(
        matched / len(user_skill_list),
        1.0
    )


# ============================================================
# EDUCATION MATCH
# ============================================================

def education_matching(
    user_education,
    course_education
):

    user_text = clean_text(
        user_education
    )

    course_text = clean_text(
        course_education
    )

    if not user_text or not course_text:
        return 0.0

    # Exact phrase
    if user_text in course_text:
        return 1.0

    # Common education keywords
    education_keywords = [
        "10th",
        "12th",
        "diploma",
        "graduate",
        "graduation",
        "bachelor",
        "btech",
        "b.e",
        "be",
        "bsc",
        "bca",
        "master",
        "masters",
        "mtech",
        "mca",
        "msc",
        "postgraduate",
        "pg"
    ]

    user_found = [
        word
        for word in education_keywords
        if word in user_text
    ]

    course_found = [
        word
        for word in education_keywords
        if word in course_text
    ]

    if not user_found or not course_found:
        return token_similarity(
            user_text,
            course_text
        )

    if set(user_found).intersection(
        set(course_found)
    ):

        return 1.0

    return 0.0


# ============================================================
# SKILL LEVEL MATCH
# ============================================================

def level_matching(
    user_level,
    course_level
):

    user_level = clean_text(
        user_level
    )

    course_level = clean_text(
        course_level
    )

    if not user_level or not course_level:
        return 0.0

    if user_level == course_level:
        return 1.0

    level_aliases = {

        "beginner": [
            "beginner",
            "basic",
            "entry",
            "foundation"
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

    for group, values in level_aliases.items():

        if any(
            value in user_level
            for value in values
        ):

            user_group = group

        if any(
            value in course_level
            for value in values
        ):

            course_group = group

    if user_group == course_group:
        return 1.0

    if user_group == "advanced" and course_group == "intermediate":
        return 0.75

    if user_group == "intermediate" and course_group == "beginner":
        return 0.70

    if user_group == "beginner" and course_group == "intermediate":
        return 0.55

    return 0.25


# ============================================================
# CREATE COURSE PROFILE
# ============================================================

def create_course_profile(row):

    values = [

        get_value(
            row,
            course_col
        ),

        get_value(
            row,
            interest_col
        ),

        get_value(
            row,
            career_col
        ),

        get_value(
            row,
            skills_col
        ),

        get_value(
            row,
            education_col
        ),

        get_value(
            row,
            skill_level_col
        )
    ]

    return " ".join(
        value
        for value in values
        if value
    )


# ============================================================
# CREATE USER PROFILE
# ============================================================

def create_user_profile(
    interest,
    career_goal,
    education,
    skills,
    skill_level
):

    return " ".join([

        expand_related_terms(
            interest
        ),

        expand_related_terms(
            career_goal
        ),

        clean_text(
            education
        ),

        expand_related_terms(
            skills
        ),

        clean_text(
            skill_level
        )
    ])


# ============================================================
# NLP SIMILARITY
# ============================================================

def calculate_nlp_similarity(
    user_profile,
    course_profiles
):

    if not user_profile.strip():
        return np.zeros(
            len(course_profiles)
        )

    documents = [
        user_profile
    ]

    documents.extend(
        course_profiles
    )

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1
        )

        matrix = vectorizer.fit_transform(
            documents
        )

        similarities = cosine_similarity(
            matrix[0:1],
            matrix[1:]
        )[0]

        return similarities

    except Exception:

        return np.zeros(
            len(course_profiles)
        )


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

def generate_recommendations(
    interest,
    career_goal,
    education,
    skills,
    skill_level
):

    results = df.copy()

    # --------------------------------------------------------
    # User Profile
    # --------------------------------------------------------

    user_profile = create_user_profile(

        interest,

        career_goal,

        education,

        skills,

        skill_level
    )

    # --------------------------------------------------------
    # Course Profiles
    # --------------------------------------------------------

    course_profiles = []

    for _, row in results.iterrows():

        course_profiles.append(
            create_course_profile(
                row
            )
        )

    # --------------------------------------------------------
    # NLP Scores
    # --------------------------------------------------------

    nlp_scores = calculate_nlp_similarity(
        user_profile,
        course_profiles
    )

    results["NLP_Score"] = (
        nlp_scores * 100
    )

    # --------------------------------------------------------
    # Individual Scores
    # --------------------------------------------------------

    interest_scores = []
    career_scores = []
    skill_scores = []
    education_scores = []
    level_scores = []
    related_scores = []

    # --------------------------------------------------------
    # Calculate Each Course
    # --------------------------------------------------------

    for _, row in results.iterrows():

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
            skill_level_col
        )

        # ----------------------------------------------------
        # Interest
        # ----------------------------------------------------

        expanded_user_interest = (
            expand_related_terms(
                interest
            )
        )

        interest_text = (
            course_interest
            + " "
            + course_name
        )

        interest_score = max(

            token_similarity(
                expanded_user_interest,
                interest_text
            ),

            phrase_match(
                clean_text(interest),
                interest_text
            )

        )

        # ----------------------------------------------------
        # Career
        # ----------------------------------------------------

        expanded_user_career = (
            expand_related_terms(
                career_goal
            )
        )

        career_text = (
            course_career
            + " "
            + course_name
        )

        career_score = max(

            token_similarity(
                expanded_user_career,
                career_text
            ),

            phrase_match(
                clean_text(career_goal),
                career_text
            )

        )

        # ----------------------------------------------------
        # Skills
        # ----------------------------------------------------

        skill_score = skill_matching(

            skills,

            course_skills
            + " "
            + course_name

        )

        # Also use token similarity for skills
        skill_score = max(

            skill_score,

            token_similarity(
                expand_related_terms(skills),
                course_skills
                + " "
                + course_name
            )

        )

        # ----------------------------------------------------
        # Education
        # ----------------------------------------------------

        education_score = education_matching(

            education,

            course_education

        )

        # ----------------------------------------------------
        # Skill Level
        # ----------------------------------------------------

        level_score = level_matching(

            skill_level,

            course_level

        )

        # ----------------------------------------------------
        # Related Course Score
        # ----------------------------------------------------

        related_score = max(

            token_similarity(
                expanded_user_interest,
                create_course_profile(row)
            ),

            token_similarity(
                expanded_user_career,
                create_course_profile(row)
            )

        )

        # ----------------------------------------------------
        # Store Scores
        # ----------------------------------------------------

        interest_scores.append(
            interest_score
        )

        career_scores.append(
            career_score
        )

        skill_scores.append(
            skill_score
        )

        education_scores.append(
            education_score
        )

        level_scores.append(
            level_score
        )

        related_scores.append(
            related_score
        )

    # --------------------------------------------------------
    # Add Individual Scores
    # --------------------------------------------------------

    results["Interest_Score"] = (
        np.array(interest_scores) * 100
    )

    results["Career_Score"] = (
        np.array(career_scores) * 100
    )

    results["Skill_Score"] = (
        np.array(skill_scores) * 100
    )

    results["Education_Score"] = (
        np.array(education_scores) * 100
    )

    results["Level_Score"] = (
        np.array(level_scores) * 100
    )

    results["Related_Score"] = (
        np.array(related_scores) * 100
    )

    # --------------------------------------------------------
    # FINAL WEIGHTED SCORE
    #
    # Interest       = 25%
    # Career Goal    = 25%
    # Skills         = 20%
    # Education      = 10%
    # Skill Level    = 5%
    # NLP Similarity = 15%
    # --------------------------------------------------------

    base_score = (

        results["Interest_Score"] * 0.25

        +

        results["Career_Score"] * 0.25

        +

        results["Skill_Score"] * 0.20

        +

        results["Education_Score"] * 0.10

        +

        results["Level_Score"] * 0.05

        +

        results["NLP_Score"] * 0.15

    )

    # --------------------------------------------------------
    # Related Domain Boost
    # --------------------------------------------------------

    related_boost = (
        results["Related_Score"] * 0.05
    )

    final_score = (
        base_score
        +
        related_boost
    )

    # --------------------------------------------------------
    # Prevent Unrealistic 100% Scores
    # --------------------------------------------------------

    final_score = np.clip(
        final_score,
        0,
        99.0
    )

    results["AI_Score"] = (
        final_score
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    results = results.sort_values(
        by="AI_Score",
        ascending=False
    )

    results = results.reset_index(
        drop=True
    )

    return results


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("👤 Your Profile")

    st.caption(
        "Tell us about yourself and AI will find the best courses."
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
        placeholder="Example: 12th Pass"
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

    get_recommendations = st.button(
        "✨ Get AI Recommendations",
        use_container_width=True,
        type="primary"
    )

    st.divider()

    st.caption(
        "🤖 AI Engine"
    )

    st.caption(
        "Weighted Matching + NLP Similarity"
    )


# ============================================================
# INITIAL SCREEN
# ============================================================

if not get_recommendations:

    st.info(
        "👈 Enter your profile details from the sidebar "
        "and click **✨ Get AI Recommendations**."
    )

    st.subheader(
        "🧠 How Recommendation Works"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            "💡 **Interest Matching**\n\n"
            "Your interests are matched with course domains "
            "and related technologies."
        )

    with col2:

        st.info(
            "🎯 **Career Matching**\n\n"
            "Your career goal is compared with the recommended "
            "career role for every course."
        )

    with col3:

        st.info(
            "🧠 **NLP Matching**\n\n"
            "TF-IDF and cosine similarity identify related "
            "course profiles."
        )

    st.divider()

    st.subheader(
        "📊 Dataset Information"
    )

    info1, info2, info3 = st.columns(3)

    with info1:

        st.metric(
            "📚 Total Courses",
            len(df)
        )

    with info2:

        st.metric(
            "🧾 Dataset Columns",
            len(df.columns)
        )

    with info3:

        st.metric(
            "🤖 AI Engine",
            "NLP + Weighted"
        )

    st.stop()


# ============================================================
# INPUT VALIDATION
# ============================================================

if not interest.strip():

    st.error(
        "⚠️ Please enter your Interest."
    )

    st.stop()


if not career_goal.strip():

    st.error(
        "⚠️ Please enter your Career Goal."
    )

    st.stop()


if not education.strip():

    st.error(
        "⚠️ Please enter your Education."
    )

    st.stop()


if not skills.strip():

    st.error(
        "⚠️ Please enter your Skills."
    )

    st.stop()


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

with st.spinner(
    "🤖 AI is analyzing your profile and finding the best courses..."
):

    recommendations = generate_recommendations(

        interest,

        career_goal,

        education,

        skills,

        skill_level

    )


# ============================================================
# RESULT HEADER
# ============================================================

st.success(
    "✨ AI recommendations generated successfully!"
)

st.header(
    "✨ AI Recommended Courses"
)

st.write(
    "Recommendations are generated using weighted profile "
    "matching and NLP-based similarity."
)


# ============================================================
# USER PROFILE SUMMARY
# ============================================================

with st.expander(
    "👤 View Your Profile"
):

    profile_col1, profile_col2 = st.columns(2)

    with profile_col1:

        st.write(
            f"💡 **Interest:** {interest}"
        )

        st.write(
            f"🎯 **Career Goal:** {career_goal}"
        )

        st.write(
            f"🎓 **Education:** {education}"
        )

    with profile_col2:

        st.write(
            f"🛠️ **Skills:** {skills}"
        )

        st.write(
            f"📊 **Skill Level:** {skill_level}"
        )


st.divider()


# ============================================================
# TOP MATCH
# ============================================================

top_course = recommendations.iloc[0]

top_score = float(
    top_course["AI_Score"]
)

st.subheader(
    "🏆 Best Course For You"
)

st.success(
    f"🎯 **{get_original_value(top_course, course_col)}** "
    f"is your Top AI Match with **{top_score:.0f}%** compatibility."
)


# ============================================================
# RECOMMENDATION CARDS
# ============================================================

top_n = min(
    5,
    len(recommendations)
)


for index in range(top_n):

    row = recommendations.iloc[index]

    rank = index + 1

    score = float(
        row["AI_Score"]
    )

    course_name = get_original_value(
        row,
        course_col
    )

    role = get_original_value(
        row,
        career_col
    )

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    if rank == 1:

        rank_title = "🏆 TOP MATCH"

    elif rank == 2:

        rank_title = "🥈 SECOND BEST"

    elif rank == 3:

        rank_title = "🥉 THIRD BEST"

    else:

        rank_title = f"⭐ RECOMMENDATION #{rank}"

    # --------------------------------------------------------
    # Course Container
    # --------------------------------------------------------

    with st.container(
        border=True
    ):

        st.subheader(
            rank_title
        )

        st.markdown(
            f"### 📚 {course_name}"
        )

        st.write(
            f"💼 **Recommended Role:** {role}"
        )

        # ----------------------------------------------------
        # Score
        # ----------------------------------------------------

        st.progress(
            min(score / 100, 1.0)
        )

        score_col1, score_col2 = st.columns(2)

        with score_col1:

            st.metric(
                "🤖 AI Match Score",
                f"{score:.0f}%"
            )

        with score_col2:

            if score >= 85:

                st.success(
                    "🔥 Excellent Match"
                )

            elif score >= 70:

                st.info(
                    "👍 Strong Match"
                )

            elif score >= 50:

                st.warning(
                    "👌 Good Match"
                )

            else:

                st.error(
                    "🔎 Low Match"
                )

        st.divider()

        # ----------------------------------------------------
        # Course Information
        # ----------------------------------------------------

        info_col1, info_col2, info_col3, info_col4 = st.columns(4)

        with info_col1:

            st.write(
                "🎓 **Education**"
            )

            st.write(
                get_original_value(
                    row,
                    education_col
                )
            )

        with info_col2:

            st.write(
                "⏱️ **Duration**"
            )

            st.write(
                get_original_value(
                    row,
                    duration_col
                )
            )

        with info_col3:

            st.write(
                "⭐ **Rating**"
            )

            st.write(
                get_original_value(
                    row,
                    rating_col
                )
            )

        with info_col4:

            st.write(
                "💰 **Salary Range**"
            )

            st.write(
                get_original_value(
                    row,
                    salary_col
                )
            )

        st.divider()

        # ----------------------------------------------------
        # More Information
        # ----------------------------------------------------

        more_col1, more_col2 = st.columns(2)

        with more_col1:

            st.write(
                "⚡ **Difficulty / Skill Level**"
            )

            st.write(
                get_original_value(
                    row,
                    skill_level_col
                )
            )

        with more_col2:

            st.write(
                "💼 **Career Role**"
            )

            st.write(
                get_original_value(
                    row,
                    career_col
                )
            )

        # ----------------------------------------------------
        # Skills
        # ----------------------------------------------------

        st.write(
            "🛠️ **Course Skills**"
        )

        st.info(
            get_original_value(
                row,
                skills_col
            )
        )

        # ----------------------------------------------------
        # AI Score Breakdown
        # ----------------------------------------------------

        with st.expander(
            "🧠 View AI Match Breakdown"
        ):

            breakdown_col1, breakdown_col2 = st.columns(2)

            with breakdown_col1:

                st.write(
                    f"💡 Interest Match: "
                    f"**{row['Interest_Score']:.1f}%**"
                )

                st.write(
                    f"🎯 Career Match: "
                    f"**{row['Career_Score']:.1f}%**"
                )

                st.write(
                    f"🛠️ Skills Match: "
                    f"**{row['Skill_Score']:.1f}%**"
                )

            with breakdown_col2:

                st.write(
                    f"🎓 Education Match: "
                    f"**{row['Education_Score']:.1f}%**"
                )

                st.write(
                    f"📊 Level Match: "
                    f"**{row['Level_Score']:.1f}%**"
                )

                st.write(
                    f"🧠 NLP Similarity: "
                    f"**{row['NLP_Score']:.1f}%**"
                )

        st.write("")


# ============================================================
# RECOMMENDATION SUMMARY
# ============================================================

st.divider()

st.subheader(
    "📊 Recommendation Summary"
)


summary_rows = []


for index in range(top_n):

    row = recommendations.iloc[index]

    summary_rows.append({

        "Rank":
            index + 1,

        "Course":
            get_original_value(
                row,
                course_col
            ),

        "AI Match":
            f"{row['AI_Score']:.0f}%",

        "Career Role":
            get_original_value(
                row,
                career_col
            ),

        "Education":
            get_original_value(
                row,
                education_col
            ),

        "Duration":
            get_original_value(
                row,
                duration_col
            ),

        "Rating":
            get_original_value(
                row,
                rating_col
            )

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
# AI MATCH VISUALIZATION
# ============================================================

st.divider()

st.subheader(
    "📈 AI Match Comparison"
)


chart_data = recommendations[
    [
        course_col,
        "AI_Score"
    ]
].head(top_n).copy()


chart_data = chart_data.rename(
    columns={
        course_col: "Course",
        "AI_Score": "AI Match Score"
    }
)


chart_data = chart_data.set_index(
    "Course"
)


st.bar_chart(
    chart_data,
    y="AI Match Score"
)


# ============================================================
# DATASET INFORMATION
# ============================================================

st.divider()

st.subheader(
    "📚 Dataset Information"
)

data_col1, data_col2, data_col3 = st.columns(3)

with data_col1:

    st.metric(
        "Total Courses",
        len(df)
    )

with data_col2:

    st.metric(
        "Dataset Columns",
        len(df.columns)
    )

with data_col3:

    st.metric(
        "Top Match",
        f"{top_score:.0f}%"
    )


# ============================================================
# DETECTED DATASET COLUMNS
# ============================================================

with st.expander(
    "🔍 View Detected Dataset Columns"
):

    column_info = {

        "Course":
            course_col,

        "Interest":
            interest_col,

        "Career Goal":
            career_col,

        "Skills":
            skills_col,

        "Education":
            education_col,

        "Skill Level":
            skill_level_col,

        "Duration":
            duration_col,

        "Rating":
            rating_col,

        "Salary":
            salary_col

    }

    detected_df = pd.DataFrame(
        list(
            column_info.items()
        ),
        columns=[
            "Field",
            "Dataset Column"
        ]
    )

    st.dataframe(
        detected_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 Course Recommendation System"
)

st.caption(
    "🤖 AI Recommendation Engine • "
    "Weighted Profile Matching • "
    "TF-IDF NLP Similarity"
)
