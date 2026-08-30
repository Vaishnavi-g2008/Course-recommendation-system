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
# COLORFUL UI THEME
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #eef2ff 0%,
        #fdf4ff 45%,
        #ecfeff 100%
    );
}

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #312e81 0%,
        #4c1d95 50%,
        #831843 100%
    );
}

[data-testid="stSidebar"] label {
    color: white !important;
    font-weight: 600 !important;
}

/* =========================
   SIDEBAR TEXT - WHITE
   ========================= */

[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5,
[data-testid="stSidebar"] h6 {
    color: white !important;
}

/* Keep typed input text dark on white input boxes */
[data-testid="stSidebar"] input {
    color: #111827 !important;
    background-color: white !important;
    border-radius: 10px !important;
}

[data-testid="stSidebar"] input::placeholder {
    color: #6b7280 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] {
    background-color: white !important;
    border-radius: 10px !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: white !important;
    color: #111827 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] span {
    color: #111827 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] input {
    color: #111827 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: #111827 !important;
    color: #111827 !important;
}

div[data-baseweb="popover"] {
    background-color: white !important;
}

div[data-baseweb="popover"] li {
    color: #111827 !important;
    background-color: white !important;
}

div[data-baseweb="popover"] li:hover {
    background-color: #ede9fe !important;
    color: #312e81 !important;
}

.stButton > button {
    background: linear-gradient(
        90deg,
        #2563eb,
        #7c3aed,
        #db2777
    ) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: bold !important;
    padding: 12px !important;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.20);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0px 8px 20px rgba(0,0,0,0.25);
}

div[data-testid="stMetric"] {
    background: white !important;
    border-radius: 18px !important;
    padding: 18px !important;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.10);
}

h1 {
    color: #312e81 !important;
    font-weight: 800 !important;
}

h2 {
    color: #4c1d95 !important;
}

h3 {
    color: #6d28d9 !important;
}

div[data-testid="stProgress"] > div > div {
    background: linear-gradient(
        90deg,
        #06b6d4,
        #6366f1,
        #ec4899
    ) !important;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.25) !important;
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
    "course_recommendation_dataset_1000.cs"
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
# LOAD DATASET
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

        data = data.fillna("")

        return data, dataset_file

    except Exception:
        return None, dataset_file


df, loaded_dataset_file = load_data()


# ============================================================
# DATASET CHECK
# ============================================================

if df is None:

    st.error("❌ Dataset file not found or cannot be read.")

    st.write("Files available in project folder:")

    try:
        st.write(os.listdir("."))
    except Exception:
        pass

    st.info("Keep the CSV dataset in the same folder as app.py.")

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
# DATASET COLUMNS
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
# NORMALIZATION
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

    "cloud engineer": [
        "cloud",
        "cloud computing",
        "aws",
        "azure",
        "google cloud",
        "devops"
    ],

    "cyber security": [
        "cyber security",
        "cybersecurity",
        "ethical hacking",
        "network security",
        "information security",
        "penetration testing",
        "cyber defense"
    ],

    "cybersecurity": [
        "cyber security",
        "cybersecurity",
        "ethical hacking",
        "network security",
        "information security",
        "penetration testing",
        "cyber defense"
    ],

    "ethical hacking": [
        "ethical hacking",
        "cyber security",
        "cybersecurity",
        "penetration testing",
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
    ],

    "devops": [
        "devops",
        "cloud",
        "cloud computing",
        "aws",
        "azure",
        "docker",
        "kubernetes",
        "ci cd"
    ]
}


# ============================================================
# EXPAND TERMS
# ============================================================

def expand_terms(text):

    text = normalize(text)

    if not text:
        return ""

    expanded = [text]

    for key, values in RELATED_TERMS.items():

        if key in text:

            expanded.extend(values)

    return " ".join(expanded)


# ============================================================
# TOKENIZE SKILLS
# ============================================================

def skill_tokens(text):

    text = normalize(text)

    if not text:
        return []

    text = text.replace("|", ",")
    text = text.replace(";", ",")

    parts = text.split(",")

    tokens = []

    for part in parts:

        part = normalize(part)

        if part:
            tokens.append(part)

    return tokens


# ============================================================
# SKILL MATCH
# ============================================================

def calculate_skill_match(user_skills, course_skills):

    user_list = skill_tokens(user_skills)
    course_text = normalize(course_skills)

    if not user_list or not course_text:
        return 0.0

    matched = 0

    for skill in user_list:

        if skill in course_text:

            matched += 1
            continue

        related = RELATED_TERMS.get(skill, [])

        if any(
            normalize(term) in course_text
            for term in related
        ):
            matched += 1

    return min(
        matched / len(user_list),
        1.0
    )


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
# EDUCATION MATCH
# ============================================================

def calculate_education_match(
    user_education,
    course_education
):

    user = normalize(user_education)
    course = normalize(course_education)

    if not user or not course:
        return 0.0

    if user == course:
        return 1.0

    if user in course or course in user:
        return 1.0

    education_keywords = [
        "btech",
        "b.e",
        "be",
        "mtech",
        "m.e",
        "me",
        "bca",
        "mca",
        "bsc",
        "msc",
        "diploma",
        "12th",
        "10th"
    ]

    for keyword in education_keywords:

        if keyword in user and keyword in course:
            return 1.0

    return 0.0


# ============================================================
# LEVEL MATCH
# ============================================================

def calculate_level_match(
    user_level,
    course_level
):

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

        expand_terms(value)
        for value in values
        if value

    )


# ============================================================
# UNIQUE COURSES
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

    if data.empty:
        return data

    # --------------------------------------------------------
    # COURSE PROFILES
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

        expand_terms(skills),

        normalize(education),

        normalize(skill_level)

    ])

    # --------------------------------------------------------
    # TF-IDF + COSINE SIMILARITY
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
    # CALCULATE FINAL SCORE
    # --------------------------------------------------------

    scores = []

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
        # INDIVIDUAL MATCHES
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # WEIGHTED HYBRID SCORE
        # ----------------------------------------------------
        #
        # Interest       = 30%
        # Career Goal    = 30%
        # Skills         = 20%
        # Education      = 10%
        # Skill Level    = 5%
        # TF-IDF         = 5%
        #
        # TOTAL          = 100%
        #

        final_score = (

            interest_score * 30

            + career_score * 30

            + skill_score * 20

            + education_score * 10

            + level_score * 5

            + nlp_score * 5

        )

        final_score = float(
            np.clip(
                final_score,
                0,
                100
            )
        )

        scores.append(final_score)

    # --------------------------------------------------------
    # ADD SCORE
    # --------------------------------------------------------

    data["AI_Score"] = scores

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    data = data.sort_values(
        by="AI_Score",
        ascending=False
    )

    return data.reset_index(
        drop=True
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("👤 Your Profile")

    st.write(
        "✨ Tell us about yourself and find the best courses."
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
        placeholder="Example: B.Tech"
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

    st.success(
        f"📂 Dataset: {loaded_dataset_file}"
    )

    st.caption(
        f"📊 {len(df):,} records loaded"
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

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "📚 Dataset Records",
            f"{len(df):,}"
        )

    with c2:

        unique_courses = (
            df[COURSE_COL]
            .astype(str)
            .apply(normalize)
            .nunique()
        )

        st.metric(
            "🎓 Unique Courses",
            f"{unique_courses:,}"
        )

    with c3:

        st.metric(
            "🤖 AI Matching",
            "Hybrid AI"
        )

    with c4:

        st.metric(
            "🏆 Recommendations",
            "Top 5"
        )

    st.divider()

    f1, f2, f3 = st.columns(3)

    with f1:

        st.info(
            "🧠 **Smart Matching**\n\n"
            "AI analyzes your interest, career goal and skills."
        )

    with f2:

        st.success(
            "🎯 **Personalized Results**\n\n"
            "Courses are selected according to your profile."
        )

    with f3:

        st.warning(
            "📈 **Smart Ranking**\n\n"
            "Courses are ranked using a hybrid AI matching score."
        )

    st.stop()


# ============================================================
# VALIDATION
# ============================================================

if not interest.strip():

    st.warning(
        "⚠️ Please enter your Interest."
    )

    st.stop()


if not career_goal.strip():

    st.warning(
        "⚠️ Please enter your Career Goal."
    )

    st.stop()


if not education.strip():

    st.warning(
        "⚠️ Please enter your Education."
    )

    st.stop()


if not skills.strip():

    st.warning(
        "⚠️ Please enter your Skills."
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
# RESULTS HEADER
# ============================================================

st.title(
    "🎓 Course Recommendation System"
)

st.subheader(
    "🤖 AI-Powered Personalized Course Recommendations"
)

st.write(
    "🎯 Personalized course recommendations based on your profile."
)

st.divider()


# ============================================================
# PROFILE SUMMARY
# ============================================================

st.subheader(
    "👤 Your Learning Profile"
)

p1, p2, p3, p4, p5 = st.columns(5)

with p1:

    st.info(
        f"💡 **Interest**\n\n{interest}"
    )

with p2:

    st.info(
        f"🎯 **Career Goal**\n\n{career_goal}"
    )

with p3:

    st.info(
        f"🎓 **Education**\n\n{education}"
    )

with p4:

    st.info(
        f"🛠️ **Skills**\n\n{skills}"
    )

with p5:

    st.info(
        f"📊 **Level**\n\n{skill_level}"
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

    st.divider()

    st.subheader(
        "🏆 Best Match"
    )

    b1, b2 = st.columns([4, 1])

    with b1:

        st.success(
            f"🥇 **{best_name}**\n\n"
            "This is the strongest course match for your profile."
        )

    with b2:

        st.metric(
            "🎯 AI Match",
            f"{best_score:.0f}%"
        )

    st.progress(
        min(best_score / 100, 1.0)
    )


# ============================================================
# TOP 5 RECOMMENDATIONS
# ============================================================

st.divider()

st.subheader(
    "📚 Top 5 Course Recommendations"
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
        f"## ⭐ Rank {index + 1}: {course_name}"
    )

    c1, c2 = st.columns([5, 1])

    with c1:

        st.progress(
            min(score / 100, 1.0)
        )

    with c2:

        st.metric(
            "AI Match",
            f"{score:.0f}%"
        )

    d1, d2, d3 = st.columns(3)

    with d1:

        st.info(
            f"💼 **Career Role**\n\n"
            f"{career_role if career_role else 'Not specified'}"
        )

    with d2:

        st.warning(
            f"⏱️ **Duration**\n\n"
            f"{duration_value if duration_value else 'Not specified'}"
        )

    with d3:

        st.success(
            f"⭐ **Rating**\n\n"
            f"{rating_value if rating_value else 'Not specified'}"
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
            ),

        "Duration":
            row_value(
                row,
                DURATION_COL
            ),

        "Rating":
            row_value(
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

        row_value(
            row,
            COURSE_COL
        )

        for _, row
        in recommendations.iterrows()

    ],

    "AI Match": [

        round(
            float(
                row["AI_Score"]
            ),
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
# METHODOLOGY
# ============================================================

st.divider()

st.subheader(
    "🧠 Recommendation Method"
)

m1, m2, m3 = st.columns(3)

with m1:

    st.info(
        "🔎 **Profile Matching**\n\n"
        "Interest, career goal, skills, education and skill level are analyzed."
    )

with m2:

    st.success(
        "📚 **TF-IDF + Cosine Similarity**\n\n"
        "Text similarity is used to understand relationships between the user profile and courses."
    )

with m3:

    st.warning(
        "🏆 **Weighted Ranking**\n\n"
        "All matching factors are combined into a final AI Match score from 0–100%."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.success(
    "🎓 Course Recommendation System | "
    "🤖 Hybrid AI-Powered Personalized Recommendations"
)
