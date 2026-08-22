import os
import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart Course Recommendation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS - IMPRESSIVE UI
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #f5f7ff 0%,
        #ffffff 50%,
        #f3f6ff 100%
    );
}


/* Main container */

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* Hero section */

.hero {
    padding: 35px;
    border-radius: 25px;
    background: linear-gradient(
        135deg,
        #667eea,
        #764ba2
    );
    color: white;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0px 10px 30px rgba(80, 80, 150, 0.25);
}

.hero h1 {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero p {
    font-size: 18px;
    opacity: 0.95;
}


/* Section title */

.section-title {
    font-size: 28px;
    font-weight: 750;
    margin-top: 25px;
    margin-bottom: 15px;
}


/* Preference cards */

.pref-card {
    background: white;
    padding: 18px;
    border-radius: 18px;
    border: 1px solid #e7e9f2;
    box-shadow: 0px 5px 18px rgba(0,0,0,0.06);
    min-height: 90px;
}

.pref-label {
    font-size: 13px;
    color: #777;
}

.pref-value {
    font-size: 17px;
    font-weight: 700;
    color: #222;
    margin-top: 6px;
}


/* Course card */

.course-card {
    background: white;
    padding: 25px;
    border-radius: 22px;
    margin-top: 18px;
    margin-bottom: 18px;
    border: 1px solid #e5e7ef;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.07);
}


.best-card {
    border: 3px solid #667eea;
    box-shadow: 0px 10px 35px rgba(102,126,234,0.22);
}


/* Badge */

.badge {
    display: inline-block;
    padding: 7px 15px;
    border-radius: 25px;
    background: #667eea;
    color: white;
    font-size: 13px;
    font-weight: 700;
}


/* Course title */

.course-title {
    font-size: 28px;
    font-weight: 800;
    color: #202124;
    margin-top: 15px;
    margin-bottom: 12px;
}


/* Information box */

.info-box {
    background: #f7f8fc;
    padding: 15px;
    border-radius: 14px;
    margin-top: 10px;
}


/* Sidebar */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #f5f7ff,
        #ffffff
    );
}


/* Footer */

.footer {
    text-align: center;
    padding: 30px;
    margin-top: 40px;
    color: #777;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# FIND DATASET
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

csv_path = os.path.join(
    BASE_DIR,
    "course_recommendation_dataset_1000.csv"
)


# =========================================================
# CHECK DATASET
# =========================================================

if not os.path.exists(csv_path):

    st.error(
        "❌ course_recommendation_dataset_1000.csv file सापडली नाही!"
    )

    st.write(
        "ही CSV file app.py च्या same folder मध्ये ठेवा."
    )

    st.write("Expected location:")

    st.code(csv_path)

    st.stop()


# =========================================================
# LOAD DATA
# =========================================================

try:

    df = pd.read_csv(csv_path)

except Exception as e:

    st.error("❌ CSV file read करता आली नाही.")

    st.write(e)

    st.stop()


# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

df.columns = (
    df.columns
    .str.strip()
)


# =========================================================
# CLEAN DATA
# =========================================================

df = df.dropna(
    how="all"
)

df = df.reset_index(
    drop=True
)


# =========================================================
# CHECK REQUIRED COLUMNS
# =========================================================

required_columns = [
    "Course",
    "Category",
    "Skill Level",
    "Skills",
    "Interest",
    "Education",
    "Career Goal",
    "Difficulty",
    "Rating",
    "Job Role",
    "Duration"
]


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    st.error(
        "❌ Dataset मध्ये required columns missing आहेत."
    )

    st.write("Missing columns:")

    for col in missing_columns:

        st.write(
            "• " + col
        )

    st.write("")

    st.write(
        "तुझ्या CSV मधील actual columns:"
    )

    st.code(
        ", ".join(df.columns.tolist())
    )

    st.stop()


# =========================================================
# TEXT CLEANING
# =========================================================

for col in required_columns:

    if col != "Rating":

        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )


# =========================================================
# REMOVE EMPTY COURSES
# =========================================================

df = df[
    df["Course"] != ""
]


# =========================================================
# REMOVE DUPLICATE COURSES
# =========================================================

df = df.drop_duplicates(
    subset=["Course"],
    keep="first"
)


# =========================================================
# RATING
# =========================================================

df["Rating"] = pd.to_numeric(
    df["Rating"],
    errors="coerce"
)

df["Rating"] = df["Rating"].fillna(0)


# =========================================================
# HERO HEADER
# =========================================================

st.markdown("""
<div class="hero">

<h1>🎓 Smart Course Recommendation System</h1>

<p>
Discover the right course based on your
skills, interests, education and career goals 🚀
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    "## 🎯 Your Preferences"
)

st.sidebar.write(
    "Select your preferences to get personalized course recommendations."
)


# =========================================================
# CATEGORY
# =========================================================

category_list = sorted(
    df["Category"]
    .dropna()
    .unique()
    .tolist()
)

category = st.sidebar.selectbox(
    "📚 Category",
    category_list
)


# =========================================================
# SKILL LEVEL
# =========================================================

skill_level_list = sorted(
    df["Skill Level"]
    .dropna()
    .unique()
    .tolist()
)

skill_level = st.sidebar.selectbox(
    "📈 Skill Level",
    skill_level_list
)


# =========================================================
# SKILLS
# =========================================================

skill_list = sorted(
    df["Skills"]
    .dropna()
    .unique()
    .tolist()
)

skill = st.sidebar.selectbox(
    "🛠️ Skills",
    skill_list
)


# =========================================================
# INTEREST
# =========================================================

interest_list = sorted(
    df["Interest"]
    .dropna()
    .unique()
    .tolist()
)

interest = st.sidebar.selectbox(
    "💡 Interest",
    interest_list
)


# =========================================================
# EDUCATION
# =========================================================

education_list = sorted(
    df["Education"]
    .dropna()
    .unique()
    .tolist()
)

education = st.sidebar.selectbox(
    "🎓 Education",
    education_list
)


# =========================================================
# CAREER GOAL
# =========================================================

career_list = sorted(
    df["Career Goal"]
    .dropna()
    .unique()
    .tolist()
)

career_goal = st.sidebar.selectbox(
    "🎯 Career Goal",
    career_list
)


# =========================================================
# DIFFICULTY
# =========================================================

difficulty_list = sorted(
    df["Difficulty"]
    .dropna()
    .unique()
    .tolist()
)

difficulty = st.sidebar.selectbox(
    "⚡ Difficulty",
    difficulty_list
)


# =========================================================
# USER PREFERENCE DISPLAY
# =========================================================

st.markdown(
    '<div class="section-title">🔎 Your Selected Preferences</div>',
    unsafe_allow_html=True
)


p1, p2, p3, p4 = st.columns(4)


with p1:

    st.markdown(
        f"""
        <div class="pref-card">

        <div class="pref-label">
        📚 Category
        </div>

        <div class="pref-value">
        {category}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with p2:

    st.markdown(
        f"""
        <div class="pref-card">

        <div class="pref-label">
        🛠️ Skills
        </div>

        <div class="pref-value">
        {skill}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with p3:

    st.markdown(
        f"""
        <div class="pref-card">

        <div class="pref-label">
        💡 Interest
        </div>

        <div class="pref-value">
        {interest}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with p4:

    st.markdown(
        f"""
        <div class="pref-card">

        <div class="pref-label">
        🎯 Career Goal
        </div>

        <div class="pref-value">
        {career_goal}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# RECOMMENDATION SCORE FUNCTION
# =========================================================

def calculate_score(row):

    score = 0


    # ---------------------------------------------
    # CATEGORY
    # ---------------------------------------------

    if (
        str(row["Category"]).lower()
        ==
        str(category).lower()
    ):

        score += 25


    # ---------------------------------------------
    # SKILL LEVEL
    # ---------------------------------------------

    if (
        str(row["Skill Level"]).lower()
        ==
        str(skill_level).lower()
    ):

        score += 10


    # ---------------------------------------------
    # SKILLS
    # ---------------------------------------------

    selected_skill = (
        str(skill)
        .lower()
        .strip()
    )

    course_skills = (
        str(row["Skills"])
        .lower()
    )

    if selected_skill in course_skills:

        score += 15


    # ---------------------------------------------
    # INTEREST
    # ---------------------------------------------

    selected_interest = (
        str(interest)
        .lower()
        .strip()
    )

    course_interest = (
        str(row["Interest"])
        .lower()
    )

    if selected_interest in course_interest:

        score += 20


    # ---------------------------------------------
    # EDUCATION
    # ---------------------------------------------

    if (
        str(row["Education"]).lower()
        ==
        str(education).lower()
    ):

        score += 10


    # ---------------------------------------------
    # CAREER GOAL
    # ---------------------------------------------

    selected_goal = (
        str(career_goal)
        .lower()
        .strip()
    )

    course_goal = (
        str(row["Career Goal"])
        .lower()
    )

    if selected_goal in course_goal:

        score += 15


    # ---------------------------------------------
    # DIFFICULTY
    # ---------------------------------------------

    if (
        str(row["Difficulty"]).lower()
        ==
        str(difficulty).lower()
    ):

        score += 5


    return score


# =========================================================
# CALCULATE SCORE
# =========================================================

df["Match Score"] = df.apply(
    calculate_score,
    axis=1
)


# =========================================================
# ADD SMALL RATING BONUS
# =========================================================

df["Final Score"] = (
    df["Match Score"]
    +
    (df["Rating"] * 0.5)
)


# =========================================================
# SORT COURSES
# =========================================================

df = df.sort_values(
    by=[
        "Final Score",
        "Rating"
    ],
    ascending=[
        False,
        False
    ]
)


# =========================================================
# GET DIFFERENT COURSES
# =========================================================

recommendations = []

used_courses = set()


for _, row in df.iterrows():

    course_name = (
        str(row["Course"])
        .strip()
        .lower()
    )


    if course_name in used_courses:

        continue


    used_courses.add(
        course_name
    )


    recommendations.append(
        row
    )


    if len(recommendations) == 5:

        break


recommendations = pd.DataFrame(
    recommendations
)


# =========================================================
# SHOW RECOMMENDATIONS
# =========================================================

st.markdown("---")


st.markdown(
    '<div class="section-title">🏆 Your Personalized Recommendations</div>',
    unsafe_allow_html=True
)


st.write(
    "Top 5 different courses selected according to your preferences."
)


# =========================================================
# NO RESULT
# =========================================================

if recommendations.empty:

    st.warning(
        "⚠️ No recommendations found."
    )

    st.info(
        "Please change your preferences."
    )

    st.stop()


# =========================================================
# BEST COURSE
# =========================================================

best = recommendations.iloc[0]


st.markdown(
    f"""
    <div class="course-card best-card">

    <span class="badge">
    🥇 #1 BEST MATCH
    </span>

    <div class="course-title">
    🎓 {best["Course"]}
    </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# BEST COURSE METRICS
# =========================================================

b1, b2, b3, b4 = st.columns(4)


with b1:

    st.metric(
        "🎯 Match",
        f"{int(best['Match Score'])}%"
    )


with b2:

    st.metric(
        "⭐ Rating",
        f"{best['Rating']}/5"
    )


with b3:

    st.metric(
        "⏱️ Duration",
        str(best["Duration"])
    )


with b4:

    st.metric(
        "💼 Job Role",
        str(best["Job Role"])
    )


# Progress bar

st.progress(
    min(
        int(best["Match Score"]),
        100
    ) / 100
)


# =========================================================
# BEST COURSE DETAILS
# =========================================================

d1, d2, d3 = st.columns(3)


with d1:

    st.markdown(
        f"""
        <div class="info-box">

        📚 <b>Category</b><br>
        {best["Category"]}

        </div>
        """,
        unsafe_allow_html=True
    )


with d2:

    st.markdown(
        f"""
        <div class="info-box">

        🛠️ <b>Skills</b><br>
        {best["Skills"]}

        </div>
        """,
        unsafe_allow_html=True
    )


with d3:

    st.markdown(
        f"""
        <div class="info-box">

        ⚡ <b>Difficulty</b><br>
        {best["Difficulty"]}

        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")


st.info(
    f"💡 Interest: {best['Interest']}   |   "
    f"🎯 Career Goal: {best['Career Goal']}   |   "
    f"🎓 Education: {best['Education']}"
)


# =========================================================
# OTHER DIFFERENT COURSES
# =========================================================

st.markdown("---")


st.markdown(
    '<div class="section-title">✨ More Recommended Courses</div>',
    unsafe_allow_html=True
)


for position in range(
    1,
    len(recommendations)
):

    course = recommendations.iloc[
        position
    ]


    # Badge

    if position == 1:

        badge = "🥈 #2 RECOMMENDATION"

    elif position == 2:

        badge = "🥉 #3 RECOMMENDATION"

    else:

        badge = (
            f"⭐ #{position + 1} RECOMMENDATION"
        )


    # Course Card

    st.markdown(
        f"""
        <div class="course-card">

        <span class="badge">
        {badge}
        </span>

        <div class="course-title">
        🎓 {course["Course"]}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # Metrics

    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "🎯 Match",
            f"{int(course['Match Score'])}%"
        )


    with c2:

        st.metric(
            "⭐ Rating",
            f"{course['Rating']}/5"
        )


    with c3:

        st.metric(
            "⏱️ Duration",
            str(course["Duration"])
        )


    with c4:

        st.metric(
            "💼 Job Role",
            str(course["Job Role"])
        )


    # Progress

    st.progress(
        min(
            int(course["Match Score"]),
            100
        ) / 100
    )


    # Details

    x1, x2, x3 = st.columns(3)


    with x1:

        st.write(
            f"📚 **Category:** "
            f"{course['Category']}"
        )


    with x2:

        st.write(
            f"🛠️ **Skills:** "
            f"{course['Skills']}"
        )


    with x3:

        st.write(
            f"⚡ **Difficulty:** "
            f"{course['Difficulty']}"
        )


    st.write(
        f"💡 **Interest:** "
        f"{course['Interest']}"
    )


    st.write(
        f"🎯 **Career Goal:** "
        f"{course['Career Goal']}"
    )


    st.divider()


# =========================================================
# DATASET INFORMATION
# =========================================================

st.markdown("---")


with st.expander("📊 Dataset Information"):

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Total Courses",
            len(df)
        )


    with col2:

        st.metric(
            "Categories",
            df["Category"].nunique()
        )


    with col3:

        st.metric(
            "Career Goals",
            df["Career Goal"].nunique()
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

    🎓 <b>Smart Course Recommendation System</b>

    <br><br>

    🤖 AI Based Recommendation
    &nbsp; | &nbsp;
    🎯 Career Guidance
    &nbsp; | &nbsp;
    🚀 Personalized Learning

    </div>
    """,
    unsafe_allow_html=True
)
