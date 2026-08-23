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
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# TITLE
# ============================================================

st.title("🎓 Course Recommendation System")

st.caption(
    "🤖 AI-powered course recommendations using weighted profile matching "
    "and NLP-based similarity."
)

st.divider()


# ============================================================
# DATASET LOADER
# ============================================================

@st.cache_data
def load_dataset():

    possible_files = [
        "courses.csv",
        "course_dataset.csv",
        "Course.csv",
        "courses.xlsx",
        "course_dataset.xlsx"
    ]

    file_path = None

    for file in possible_files:
        if os.path.exists(file):
            file_path = file
            break

    if file_path is None:
        return None

    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        return df

    except Exception as e:
        st.error(f"Dataset load error: {e}")
        return None


df = load_dataset()


# ============================================================
# IF DATASET NOT FOUND
# ============================================================

if df is None:

    st.warning("⚠️ Dataset file सापडली नाही.")

    st.info(
        "तुझ्या project folder मध्ये `courses.csv` किंवा "
        "`course_dataset.csv` नावाची file ठेवा."
    )

    st.stop()


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = [
    str(col).strip().lower().replace(" ", "_")
    for col in df.columns
]


# ============================================================
# COLUMN FINDER
# ============================================================

def find_column(possible_names):

    for name in possible_names:

        name = name.lower().replace(" ", "_")

        if name in df.columns:
            return name

    return None


course_col = find_column([
    "course",
    "course_name",
    "coursename",
    "title",
    "course_title"
])

interest_col = find_column([
    "interest",
    "interests",
    "domain",
    "category",
    "field"
])

career_col = find_column([
    "career_goal",
    "career",
    "job_role",
    "role",
    "career_path",
    "recommended_role"
])

skills_col = find_column([
    "skills",
    "skill",
    "required_skills",
    "technical_skills"
])

education_col = find_column([
    "education",
    "qualification",
    "eligibility",
    "educational_qualification"
])

level_col = find_column([
    "skill_level",
    "level",
    "difficulty",
    "difficulty_level"
])

duration_col = find_column([
    "duration",
    "course_duration"
])

rating_col = find_column([
    "rating",
    "course_rating"
])

salary_col = find_column([
    "salary",
    "salary_range",
    "expected_salary"
])


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

if course_col is None:

    st.error(
        "❌ Course column सापडला नाही. Dataset मध्ये "
        "`Course`, `Course_Name` किंवा `Title` column असणे आवश्यक आहे."
    )

    st.stop()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower()

    value = value.replace("&", " and ")

    value = re.sub(r"[^a-zA-Z0-9+#.\s]", " ", value)

    value = re.sub(r"\s+", " ", value)

    return value.strip()


def get_value(row, column):

    if column is None:
        return ""

    return clean_text(row.get(column, ""))


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🎯 Your Profile")

    st.write(
        "Enter your profile details to get personalized "
        "AI course recommendations."
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
        placeholder="Example: 12th Pass / B.Tech / Graduate"
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
        use_container_width=True,
        type="primary"
    )


# ============================================================
# PROFILE TEXT
# ============================================================

user_interest = clean_text(interest)
user_career = clean_text(career_goal)
user_education = clean_text(education)
user_skills = clean_text(skills)
user_level = clean_text(skill_level)


# ============================================================
# RELATED DOMAIN KNOWLEDGE
# ============================================================

related_domains = {

    "artificial intelligence": [
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "data science",
        "neural network",
        "nlp",
        "computer vision",
        "generative ai"
    ],

    "machine learning": [
        "machine learning",
        "artificial intelligence",
        "deep learning",
        "data science",
        "neural network",
        "predictive analytics"
    ],

    "deep learning": [
        "deep learning",
        "machine learning",
        "artificial intelligence",
        "neural network",
        "computer vision",
        "nlp"
    ],

    "data science": [
        "data science",
        "machine learning",
        "artificial intelligence",
        "data analytics",
        "python",
        "statistics",
        "pandas"
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

    "cloud computing": [
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
        "network security",
        "ethical hacking",
        "information security"
    ],

    "python": [
        "python",
        "data science",
        "machine learning",
        "artificial intelligence",
        "automation",
        "pandas"
    ]
}


def get_related_terms(text):

    terms = [text]

    for key, values in related_domains.items():

        if key in text:

            terms.extend(values)

    return " ".join(terms)


expanded_interest = get_related_terms(user_interest)
expanded_career = get_related_terms(user_career)
expanded_skills = get_related_terms(user_skills)


# ============================================================
# PROFILE MATCH FUNCTION
# ============================================================

def token_similarity(user_text, course_text):

    user_text = clean_text(user_text)
    course_text = clean_text(course_text)

    if not user_text or not course_text:
        return 0.0

    user_tokens = set(user_text.split())
    course_tokens = set(course_text.split())

    if not user_tokens or not course_tokens:
        return 0.0

    common = user_tokens.intersection(course_tokens)

    return len(common) / max(len(user_tokens), 1)


# ============================================================
# SCORE CALCULATION
# ============================================================

def calculate_score(row):

    course_name = get_value(row, course_col)
    course_interest = get_value(row, interest_col)
    course_career = get_value(row, career_col)
    course_skills = get_value(row, skills_col)
    course_education = get_value(row, education_col)
    course_level = get_value(row, level_col)

    # --------------------------------------------------------
    # Create course profile
    # --------------------------------------------------------

    course_profile = " ".join([
        course_name,
        course_interest,
        course_career,
        course_skills,
        course_education,
        course_level
    ])

    # --------------------------------------------------------
    # User profile
    # --------------------------------------------------------

    user_profile = " ".join([
        expanded_interest,
        expanded_career,
        user_education,
        expanded_skills,
        user_level
    ])

    # --------------------------------------------------------
    # NLP similarity
    # --------------------------------------------------------

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        vectors = vectorizer.fit_transform(
            [user_profile, course_profile]
        )

        nlp_score = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0]

    except Exception:

        nlp_score = 0.0

    # --------------------------------------------------------
    # Individual weighted scores
    # --------------------------------------------------------

    interest_score = token_similarity(
        expanded_interest,
        course_interest + " " + course_name
    )

    career_score = token_similarity(
        expanded_career,
        course_career + " " + course_name
    )

    skill_score = token_similarity(
        expanded_skills,
        course_skills + " " + course_name
    )

    education_score = token_similarity(
        user_education,
        course_education
    )

    level_score = token_similarity(
        user_level,
        course_level
    )

    # --------------------------------------------------------
    # Extra semantic matching
    # --------------------------------------------------------

    interest_course_similarity = token_similarity(
        expanded_interest,
        course_profile
    )

    career_course_similarity = token_similarity(
        expanded_career,
        course_profile
    )

    # --------------------------------------------------------
    # Weighted AI score
    #
    # Interest       = 25%
    # Career Goal    = 25%
    # Skills         = 20%
    # Education      = 10%
    # Skill Level    = 5%
    # NLP Similarity = 15%
    # --------------------------------------------------------

    final_score = (

        interest_score * 0.25 +

        career_score * 0.25 +

        skill_score * 0.20 +

        education_score * 0.10 +

        level_score * 0.05 +

        nlp_score * 0.15

    )

    # --------------------------------------------------------
    # Related-course boost
    # --------------------------------------------------------

    if interest_course_similarity > 0:
        final_score += interest_course_similarity * 0.05

    if career_course_similarity > 0:
        final_score += career_course_similarity * 0.05

    # --------------------------------------------------------
    # Cap score
    # --------------------------------------------------------

    final_score = min(final_score, 1.0)

    return final_score * 100


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

def generate_recommendations():

    results = df.copy()

    results["AI_Score"] = results.apply(
        calculate_score,
        axis=1
    )

    results = results.sort_values(
        by="AI_Score",
        ascending=False
    )

    results = results.reset_index(drop=True)

    return results


# ============================================================
# COURSE CARD
# ============================================================

def show_course_card(row, rank):

    course_name = row.get(course_col, "Course")

    role = (
        row.get(career_col, "Career Opportunity")
        if career_col
        else "Career Opportunity"
    )

    score = float(row["AI_Score"])

    if rank == 1:
        rank_text = "🏆 TOP MATCH"

    elif rank == 2:
        rank_text = "🥈 SECOND BEST"

    elif rank == 3:
        rank_text = "🥉 THIRD BEST"

    else:
        rank_text = f"⭐ RECOMMENDATION #{rank}"

    with st.container(border=True):

        st.subheader(f"{rank_text}")

        st.markdown(
            f"## 📚 {course_name}"
        )

        st.write(
            f"💼 **Recommended Role:** {role}"
        )

        st.progress(
            min(score / 100, 1.0)
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "🤖 AI Match Score",
                f"{score:.0f}%"
            )

        with col2:

            if score >= 85:

                st.success("🔥 Excellent Match")

            elif score >= 70:

                st.info("👍 Strong Match")

            elif score >= 50:

                st.warning("👌 Good Match")

            else:

                st.write("🔎 Related Course")

        st.divider()

        info1, info2, info3, info4 = st.columns(4)

        with info1:

            if education_col:

                st.write("🎓 **Education**")

                st.write(
                    str(row.get(education_col, "Not specified"))
                )

        with info2:

            if duration_col:

                st.write("⏱️ **Duration**")

                st.write(
                    str(row.get(duration_col, "Not specified"))
                )

        with info3:

            if rating_col:

                st.write("⭐ **Rating**")

                st.write(
                    str(row.get(rating_col, "Not specified"))
                )

        with info4:

            if salary_col:

                st.write("💰 **Salary Range**")

                st.write(
                    str(row.get(salary_col, "Not specified"))
                )

        st.divider()

        extra1, extra2 = st.columns(2)

        with extra1:

            if level_col:

                st.write(
                    f"⚡ **Difficulty:** "
                    f"{row.get(level_col, 'Not specified')}"
                )

        with extra2:

            if career_col:

                st.write(
                    f"💼 **Career:** "
                    f"{row.get(career_col, 'Not specified')}"
                )


# ============================================================
# MAIN CONTENT
# ============================================================

if not recommend_button:

    st.info(
        "👈 Enter your Interest, Career Goal, Education and Skills "
        "from the sidebar, then click **Get AI Recommendations**."
    )

    st.divider()

    st.subheader("🤖 How the AI Recommendation Works")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.write("💡 **Interest Matching**")

        st.caption(
            "Your interests are compared with course domains."
        )

    with col2:

        st.write("🎯 **Career Matching**")

        st.caption(
            "Your career goal is matched with recommended roles."
        )

    with col3:

        st.write("🛠️ **Skill Matching**")

        st.caption(
            "Your current skills are compared with course skills."
        )

    with col4:

        st.write("🧠 **NLP Similarity**")

        st.caption(
            "TF-IDF based similarity identifies related courses."
        )

    st.stop()


# ============================================================
# INPUT VALIDATION
# ============================================================

if not interest:

    st.error("⚠️ Please enter your Interest.")

    st.stop()

if not career_goal:

    st.error("⚠️ Please enter your Career Goal.")

    st.stop()

if not education:

    st.error("⚠️ Please enter your Education.")

    st.stop()

if not skills:

    st.error("⚠️ Please enter your Skills.")

    st.stop()


# ============================================================
# GENERATE RESULTS
# ============================================================

with st.spinner("🤖 AI is analyzing your profile..."):

    recommendations = generate_recommendations()


# ============================================================
# RESULT HEADER
# ============================================================

st.success(
    "✨ AI recommendations generated successfully!"
)

st.header("✨ AI Recommended Courses")

st.caption(
    "Recommendations are generated using weighted profile matching "
    "and NLP-based similarity."
)


# ============================================================
# USER PROFILE SUMMARY
# ============================================================

with st.expander("👤 View Your Profile"):

    p1, p2, p3, p4 = st.columns(4)

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
        st.write("📊 **Skill Level**")
        st.write(skill_level)

    st.write(
        f"🛠️ **Skills:** {skills}"
    )


st.divider()


# ============================================================
# TOP RECOMMENDATIONS
# ============================================================

top_n = min(5, len(recommendations))

for i in range(top_n):

    show_course_card(
        recommendations.iloc[i],
        i + 1
    )

    st.write("")


# ============================================================
# RECOMMENDATION TABLE
# ============================================================

st.divider()

st.subheader("📊 Recommendation Summary")

summary_data = []

for i in range(top_n):

    row = recommendations.iloc[i]

    summary_data.append({

        "Rank":
            i + 1,

        "Course":
            row.get(course_col, ""),

        "AI Match":
            f"{row['AI_Score']:.0f}%",

        "Career Role":
            row.get(career_col, "")
            if career_col
            else "",

        "Education":
            row.get(education_col, "")
            if education_col
            else "",

        "Duration":
            row.get(duration_col, "")
            if duration_col
            else "",

        "Rating":
            row.get(rating_col, "")
            if rating_col
            else ""

    })


summary_df = pd.DataFrame(summary_data)

st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 Course Recommendation System | "
    "AI + Weighted Matching + NLP"
)