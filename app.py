import streamlit as st
import pandas as pd
import os

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Course Recommendation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "final_data.csv"
    )

    if not os.path.exists(file_path):
        return None

    return pd.read_csv(file_path)


df = load_data()


# ============================================================
# DATA ERROR CHECK
# ============================================================

if df is None:

    st.error("❌ final_data.csv not found.")

    st.info(
        "Please keep final_data.csv in the same folder as app.py."
    )

    st.stop()


if df.empty:

    st.error("❌ final_data.csv is empty.")

    st.stop()


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Course_ID",
    "Category",
    "Skill_Level",
    "Skills",
    "Interest",
    "Education",
    "Career_Goal",
    "Duration_Months",
    "Rating",
    "Difficulty",
    "Job_Role",
    "Salary_Range"
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    st.error("❌ Required columns are missing.")

    st.write(
        "Missing columns:",
        missing_columns
    )

    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

df = df.copy()


for column in required_columns:

    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )


df["Rating"] = pd.to_numeric(
    df["Rating"],
    errors="coerce"
).fillna(0)


df["Duration_Months"] = pd.to_numeric(
    df["Duration_Months"],
    errors="coerce"
).fillna(0)


df = df[
    df["Course_ID"].str.strip() != ""
].copy()


if df.empty:

    st.error("❌ No valid courses found.")

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🎓 Smart Course Finder")

st.subheader(
    "Find the right course for your skills, interests and career goals."
)

st.write(
    "Select your learning preferences below and get personalized "
    "course recommendations."
)

st.divider()


# ============================================================
# DASHBOARD
# ============================================================

st.header("📊 Learning Platform")

st.caption(
    "Explore the learning opportunities available in the course database."
)


total_courses = df["Course_ID"].nunique()

total_categories = df["Category"].nunique()

total_roles = df["Job_Role"].nunique()

average_rating = df["Rating"].mean()


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        label="🎓 Total Courses",
        value=f"{total_courses:,}"
    )


with col2:

    st.metric(
        label="📚 Categories",
        value=f"{total_categories:,}"
    )


with col3:

    st.metric(
        label="💼 Job Roles",
        value=f"{total_roles:,}"
    )


with col4:

    st.metric(
        label="⭐ Average Rating",
        value=f"{average_rating:.2f}/5"
    )


st.divider()


# ============================================================
# PROFILE
# ============================================================

st.header("👤 Build Your Learning Profile")

st.caption(
    "Choose your preferences to find the courses that best match your profile."
)


# ============================================================
# CATEGORY OPTIONS
# ============================================================

category_options = sorted(
    [
        x for x in df["Category"].unique()
        if str(x).strip()
    ]
)


skill_level_options = sorted(
    [
        x for x in df["Skill_Level"].unique()
        if str(x).strip()
    ]
)


interest_options = sorted(
    [
        x for x in df["Interest"].unique()
        if str(x).strip()
    ]
)


education_options = sorted(
    [
        x for x in df["Education"].unique()
        if str(x).strip()
    ]
)


career_goal_options = sorted(
    [
        x for x in df["Career_Goal"].unique()
        if str(x).strip()
    ]
)


skills_options = sorted(
    [
        x for x in df["Skills"].unique()
        if str(x).strip()
    ]
)


difficulty_options = sorted(
    [
        x for x in df["Difficulty"].unique()
        if str(x).strip()
    ]
)


job_role_options = sorted(
    [
        x for x in df["Job_Role"].unique()
        if str(x).strip()
    ]
)


salary_options = sorted(
    [
        x for x in df["Salary_Range"].unique()
        if str(x).strip()
    ]
)


# ============================================================
# PROFILE ROW 1
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    category = st.selectbox(
        "📚 Course Category",
        category_options
    )


with col2:

    skill_level = st.selectbox(
        "📈 Skill Level",
        skill_level_options
    )


with col3:

    interest = st.selectbox(
        "💡 Area of Interest",
        interest_options
    )


# ============================================================
# PROFILE ROW 2
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    education = st.selectbox(
        "🎓 Education",
        education_options
    )


with col2:

    career_goal = st.selectbox(
        "🚀 Career Goal",
        career_goal_options
    )


with col3:

    skills = st.selectbox(
        "🛠️ Skills",
        skills_options
    )


# ============================================================
# PROFILE ROW 3
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    difficulty = st.selectbox(
        "⚡ Difficulty",
        difficulty_options
    )


with col2:

    job_role = st.selectbox(
        "💼 Target Job Role",
        job_role_options
    )


with col3:

    salary_range = st.selectbox(
        "💰 Expected Salary Range",
        salary_options
    )


st.write("")


# ============================================================
# RECOMMEND BUTTON
# ============================================================

recommend_button = st.button(
    "🚀 FIND MY PERFECT COURSE",
    type="primary",
    use_container_width=True
)


# ============================================================
# RECOMMENDATION SYSTEM
# ============================================================

if recommend_button:

    result = df.copy()

    result["Match_Score"] = 0


    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    result.loc[
        result["Category"].str.lower()
        == str(category).lower(),
        "Match_Score"
    ] += 15


    # --------------------------------------------------------
    # SKILL LEVEL
    # --------------------------------------------------------

    result.loc[
        result["Skill_Level"].str.lower()
        == str(skill_level).lower(),
        "Match_Score"
    ] += 10


    # --------------------------------------------------------
    # INTEREST
    # --------------------------------------------------------

    result.loc[
        result["Interest"].str.lower()
        == str(interest).lower(),
        "Match_Score"
    ] += 15


    # --------------------------------------------------------
    # EDUCATION
    # --------------------------------------------------------

    result.loc[
        result["Education"].str.lower()
        == str(education).lower(),
        "Match_Score"
    ] += 10


    # --------------------------------------------------------
    # CAREER GOAL
    # --------------------------------------------------------

    result.loc[
        result["Career_Goal"].str.lower()
        == str(career_goal).lower(),
        "Match_Score"
    ] += 15


    # --------------------------------------------------------
    # SKILLS
    # --------------------------------------------------------

    result.loc[
        result["Skills"].str.lower()
        == str(skills).lower(),
        "Match_Score"
    ] += 10


    # --------------------------------------------------------
    # DIFFICULTY
    # --------------------------------------------------------

    result.loc[
        result["Difficulty"].str.lower()
        == str(difficulty).lower(),
        "Match_Score"
    ] += 5


    # --------------------------------------------------------
    # JOB ROLE
    # --------------------------------------------------------

    result.loc[
        result["Job_Role"].str.lower()
        == str(job_role).lower(),
        "Match_Score"
    ] += 10


    # --------------------------------------------------------
    # SALARY
    # --------------------------------------------------------

    result.loc[
        result["Salary_Range"].str.lower()
        == str(salary_range).lower(),
        "Match_Score"
    ] += 10


    # ========================================================
    # SORT RESULTS
    # ========================================================

    result = result.sort_values(
        by=["Match_Score", "Rating"],
        ascending=[False, False]
    ).reset_index(drop=True)


    if result.empty:

        st.error(
            "❌ No recommendation found."
        )

        st.stop()


    course = result.iloc[0]


    # ========================================================
    # COURSE NAME
    # ========================================================

    if "Course_Name" in result.columns:

        course_name = str(
            course["Course_Name"]
        ).strip()

        if not course_name:

            course_name = (
                f"Course ID: {course['Course_ID']}"
            )

    else:

        course_name = (
            f"Course ID: {course['Course_ID']}"
        )


    # ========================================================
    # SCORE
    # ========================================================

    score = int(
        course["Match_Score"]
    )


    percentage = min(
        score,
        100
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.header("✨ Your Best Match")

    st.success(
        "🎉 Your personalized course recommendation has been generated!"
    )


    # ========================================================
    # BEST COURSE
    # ========================================================

    col1, col2 = st.columns(
        [2, 1]
    )


    with col1:

        st.subheader(
            f"🎓 {course_name}"
        )

        st.write(
            f"**Course ID:** {course['Course_ID']}"
        )

        st.write(
            f"**Category:** {course['Category']}"
        )

        st.write(
            f"**Career Goal:** {course['Career_Goal']}"
        )

        st.write(
            f"**Target Job Role:** {course['Job_Role']}"
        )


    with col2:

        st.metric(
            "🎯 Profile Match",
            f"{percentage}%"
        )

        st.progress(
            percentage / 100
        )


    # ========================================================
    # MATCH MESSAGE
    # ========================================================

    if percentage >= 80:

        st.success(
            "🔥 Excellent Match — this course is highly aligned with your profile!"
        )

    elif percentage >= 60:

        st.info(
            "👍 Good Match — this course fits many of your preferences."
        )

    else:

        st.warning(
            "💡 Moderate Match — explore the Top 5 courses below."
        )


    st.divider()


    # ========================================================
    # COURSE HIGHLIGHTS
    # ========================================================

    st.header("📌 Course Highlights")


    duration = float(
        course["Duration_Months"]
    )


    if duration > 0:

        duration_text = (
            f"{duration:g} Months"
        )

    else:

        duration_text = "N/A"


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "⭐ Rating",
            f"{course['Rating']} / 5"
        )


    with col2:

        st.metric(
            "⏱️ Duration",
            duration_text
        )


    with col3:

        st.metric(
            "📈 Skill Level",
            course["Skill_Level"]
        )


    with col4:

        st.metric(
            "💰 Salary Range",
            course["Salary_Range"]
        )


    st.divider()


    # ========================================================
    # DETAILS
    # ========================================================

    st.header("📋 Course Details")


    tab1, tab2, tab3 = st.tabs(
        [
            "🛠️ Skills & Interest",
            "🎯 Career",
            "🎓 Education"
        ]
    )


    with tab1:

        st.subheader("🛠️ Skills")

        st.info(
            course["Skills"]
        )

        st.subheader("💡 Interest")

        st.info(
            course["Interest"]
        )


    with tab2:

        st.subheader("🎯 Career Goal")

        st.success(
            course["Career_Goal"]
        )

        st.subheader("💼 Target Job Role")

        st.info(
            course["Job_Role"]
        )

        st.subheader("⚡ Difficulty")

        st.warning(
            course["Difficulty"]
        )


    with tab3:

        st.subheader("🎓 Education")

        st.info(
            course["Education"]
        )

        st.subheader("📚 Category")

        st.info(
            course["Category"]
        )


    st.divider()


    # ========================================================
    # TOP 5 RECOMMENDATIONS
    # ========================================================

    st.header("🏆 Top 5 Recommended Courses")

    st.caption(
        "Best alternative courses based on your profile."
    )


    top5 = result.head(5)


    for index, row in top5.iterrows():

        rank = index + 1


        if (
            "Course_Name" in result.columns
            and str(row["Course_Name"]).strip()
        ):

            name = str(
                row["Course_Name"]
            )

        else:

            name = (
                f"Course ID: {row['Course_ID']}"
            )


        with st.container(border=True):

            col1, col2, col3 = st.columns(
                [0.7, 4, 1.2]
            )


            with col1:

                st.subheader(
                    f"#{rank}"
                )


            with col2:

                st.write(
                    f"### 🎓 {name}"
                )

                st.caption(
                    f"📚 {row['Category']}  |  "
                    f"📈 {row['Skill_Level']}  |  "
                    f"⭐ {row['Rating']}"
                )


            with col3:

                st.metric(
                    "Match",
                    f"{int(row['Match_Score'])}%"
                )


# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

st.header("🧠 How Smart Course Finder Works")

st.caption(
    "Three simple steps to discover your learning path."
)


col1, col2, col3 = st.columns(3)


with col1:

    with st.container(border=True):

        st.subheader("01 👤")

        st.write(
            "### Build Your Profile"
        )

        st.write(
            "Select your education, skills, "
            "interests and career preferences."
        )


with col2:

    with st.container(border=True):

        st.subheader("02 🔎")

        st.write(
            "### Smart Matching"
        )

        st.write(
            "Your preferences are compared "
            "with course attributes to calculate "
            "a compatibility score."
        )


with col3:

    with st.container(border=True):

        st.subheader("03 🚀")

        st.write(
            "### Get Your Recommendation"
        )

        st.write(
            "The highest-scoring courses are "
            "ranked and displayed for you."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 Smart Course Finder | Built with Python, Pandas & Streamlit"
)
