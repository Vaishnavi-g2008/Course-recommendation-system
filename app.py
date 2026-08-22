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
# ERROR CHECK
# ============================================================

if df is None:
    st.error("❌ final_data.csv not found.")
    st.info("Please keep final_data.csv in the same folder as app.py.")
    st.stop()

if df.empty:
    st.error("❌ final_data.csv is empty.")
    st.stop()

# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.astype(str).str.strip()

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
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error("❌ Required columns are missing.")
    st.write("Missing columns:", missing_columns)
    st.stop()

# ============================================================
# CLEAN DATA
# ============================================================

df = df.copy()

for col in required_columns:
    df[col] = (
        df[col]
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
# HEADER
# ============================================================

st.title("🎓 Course Recommendation System")

st.subheader(
    "✨ Smart Course Finder"
)

st.write(
    "Find the right learning path based on your "
    "skills, interests, education and career goals."
)

st.info(
    "💡 Select your preferences below and let the system "
    "find the most suitable courses for you."
)

# ============================================================
# DASHBOARD
# ============================================================

st.divider()

st.header("📊 Learning Platform")

st.caption(
    "Explore the course database and discover available learning opportunities."
)

total_courses = df["Course_ID"].nunique()
total_categories = df["Category"].nunique()
total_roles = df["Job_Role"].nunique()
average_rating = df["Rating"].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🎓 Total Courses",
        f"{total_courses:,}"
    )

with col2:
    st.metric(
        "📚 Categories",
        f"{total_categories:,}"
    )

with col3:
    st.metric(
        "💼 Job Roles",
        f"{total_roles:,}"
    )

with col4:
    st.metric(
        "⭐ Average Rating",
        f"{average_rating:.2f}/5"
    )

# ============================================================
# CATEGORY OVERVIEW
# ============================================================

st.divider()

st.header("📚 Course Categories")

category_count = (
    df["Category"]
    .value_counts()
    .reset_index()
)

category_count.columns = [
    "Category",
    "Courses"
]

cols = st.columns(len(category_count))

for i, row in category_count.iterrows():

    with cols[i]:
        st.metric(
            f"📘 {row['Category']}",
            int(row["Courses"])
        )

# ============================================================
# PROFILE SECTION
# ============================================================

st.divider()

st.header("👤 Build Your Learning Profile")

st.caption(
    "Tell us about your learning preferences to get a personalized recommendation."
)

# ============================================================
# OPTIONS
# ============================================================

def get_options(column):

    return sorted([
        x for x in df[column].unique()
        if str(x).strip()
    ])


category_options = get_options("Category")
skill_level_options = get_options("Skill_Level")
interest_options = get_options("Interest")
education_options = get_options("Education")
career_goal_options = get_options("Career_Goal")
skills_options = get_options("Skills")
difficulty_options = get_options("Difficulty")
job_role_options = get_options("Job_Role")
salary_options = get_options("Salary_Range")

# ============================================================
# PROFILE CARD 1
# ============================================================

st.subheader("📘 Learning Preferences")

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
# PROFILE CARD 2
# ============================================================

st.subheader("🎓 Background & Career")

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

    job_role = st.selectbox(
        "💼 Target Job Role",
        job_role_options
    )

# ============================================================
# PROFILE CARD 3
# ============================================================

st.subheader("🛠️ Skills & Course Preferences")

col1, col2, col3 = st.columns(3)

with col1:

    skills = st.selectbox(
        "🛠️ Skills",
        skills_options
    )

with col2:

    difficulty = st.selectbox(
        "⚡ Difficulty",
        difficulty_options
    )

with col3:

    salary_range = st.selectbox(
        "💰 Expected Salary Range",
        salary_options
    )

# ============================================================
# FIND BUTTON
# ============================================================

st.write("")

find_course = st.button(
    "🚀 FIND MY PERFECT COURSE",
    type="primary",
    use_container_width=True
)

# ============================================================
# RECOMMENDATION SYSTEM
# ============================================================

if find_course:

    result = df.copy()

    result["Match_Score"] = 0

    # Category
    result.loc[
        result["Category"].str.lower() ==
        str(category).lower(),
        "Match_Score"
    ] += 15

    # Skill Level
    result.loc[
        result["Skill_Level"].str.lower() ==
        str(skill_level).lower(),
        "Match_Score"
    ] += 10

    # Interest
    result.loc[
        result["Interest"].str.lower() ==
        str(interest).lower(),
        "Match_Score"
    ] += 15

    # Education
    result.loc[
        result["Education"].str.lower() ==
        str(education).lower(),
        "Match_Score"
    ] += 10

    # Career Goal
    result.loc[
        result["Career_Goal"].str.lower() ==
        str(career_goal).lower(),
        "Match_Score"
    ] += 15

    # Skills
    result.loc[
        result["Skills"].str.lower() ==
        str(skills).lower(),
        "Match_Score"
    ] += 10

    # Difficulty
    result.loc[
        result["Difficulty"].str.lower() ==
        str(difficulty).lower(),
        "Match_Score"
    ] += 5

    # Job Role
    result.loc[
        result["Job_Role"].str.lower() ==
        str(job_role).lower(),
        "Match_Score"
    ] += 10

    # Salary
    result.loc[
        result["Salary_Range"].str.lower() ==
        str(salary_range).lower(),
        "Match_Score"
    ] += 10

    # ========================================================
    # SORT
    # ========================================================

    result = result.sort_values(
        by=["Match_Score", "Rating"],
        ascending=[False, False]
    ).reset_index(drop=True)

    course = result.iloc[0]

    # ========================================================
    # COURSE NAME
    # ========================================================

    if "Course_Name" in result.columns:

        course_name = str(
            course["Course_Name"]
        ).strip()

        if not course_name:
            course_name = f"Course ID: {course['Course_ID']}"

    else:

        course_name = f"Course ID: {course['Course_ID']}"

    score = int(course["Match_Score"])
    percentage = min(score, 100)

    # ========================================================
    # RESULT HEADER
    # ========================================================

    st.divider()

    st.header("🏆 Your Course Recommendation")

    st.success(
        "🎉 We found a course that matches your learning profile!"
    )

    # ========================================================
    # BEST COURSE
    # ========================================================

    st.subheader(
        f"🎓 {course_name}"
    )

    col1, col2 = st.columns([2, 1])

    with col1:

        st.write(
            f"**Course ID:** {course['Course_ID']}"
        )

        st.write(
            f"**📚 Category:** {course['Category']}"
        )

        st.write(
            f"**📈 Skill Level:** {course['Skill_Level']}"
        )

        st.write(
            f"**💡 Interest:** {course['Interest']}"
        )

        st.write(
            f"**🎯 Career Goal:** {course['Career_Goal']}"
        )

        st.write(
            f"**💼 Job Role:** {course['Job_Role']}"
        )

    with col2:

        st.metric(
            "🎯 Profile Match",
            f"{percentage}%"
        )

        st.progress(
            percentage / 100
        )

        st.metric(
            "⭐ Rating",
            f"{course['Rating']} / 5"
        )

    # ========================================================
    # MATCH MESSAGE
    # ========================================================

    if percentage >= 80:

        st.success(
            "🔥 Excellent Match! This course strongly matches your profile."
        )

    elif percentage >= 60:

        st.info(
            "👍 Good Match! This course matches many of your preferences."
        )

    else:

        st.warning(
            "💡 Moderate Match. Check the Top 5 recommendations below."
        )

    # ========================================================
    # COURSE HIGHLIGHTS
    # ========================================================

    st.divider()

    st.header("📌 Course Highlights")

    duration = float(course["Duration_Months"])

    duration_text = (
        f"{duration:g} Months"
        if duration > 0
        else "N/A"
    )

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
            "⚡ Difficulty",
            course["Difficulty"]
        )

    with col4:

        st.metric(
            "💰 Salary",
            course["Salary_Range"]
        )

    # ========================================================
    # DETAILS TABS
    # ========================================================

    st.divider()

    st.header("📋 Course Details")

    tab1, tab2, tab3 = st.tabs(
        [
            "🛠️ Skills & Interest",
            "🎯 Career",
            "🎓 Education"
        ]
    )

    with tab1:

        st.subheader("🛠️ Required Skills")

        st.info(
            course["Skills"]
        )

        st.subheader("💡 Area of Interest")

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

        st.subheader("⚡ Difficulty Level")

        st.warning(
            course["Difficulty"]
        )

    with tab3:

        st.subheader("🎓 Required Education")

        st.info(
            course["Education"]
        )

        st.subheader("📚 Course Category")

        st.info(
            course["Category"]
        )

    # ========================================================
    # TOP 5
    # ========================================================

    st.divider()

    st.header("🥇 Top 5 Recommended Courses")

    st.caption(
        "Alternative courses ranked according to your profile."
    )

    top5 = result.head(5)

    for index, row in top5.iterrows():

        rank = index + 1

        if (
            "Course_Name" in result.columns
            and str(row["Course_Name"]).strip()
        ):

            name = str(row["Course_Name"])

        else:

            name = f"Course ID: {row['Course_ID']}"

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
                    f"**🎓 {name}**"
                )

                st.caption(
                    f"📚 {row['Category']}  •  "
                    f"📈 {row['Skill_Level']}  •  "
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

st.header("🧠 How Course Recommendation Works")

st.caption(
    "Our recommendation process works in three simple steps."
)

col1, col2, col3 = st.columns(3)

with col1:

    with st.container(border=True):

        st.subheader("01 👤")

        st.write(
            "### Build Your Profile"
        )

        st.write(
            "Choose your education, skills, interests "
            "and career preferences."
        )

with col2:

    with st.container(border=True):

        st.subheader("02 🔎")

        st.write(
            "### Smart Matching"
        )

        st.write(
            "The system compares your preferences "
            "with the available course information."
        )

with col3:

    with st.container(border=True):

        st.subheader("03 🚀")

        st.write(
            "### Get Recommendations"
        )

        st.write(
            "Courses are ranked using a matching score "
            "and the best options are displayed."
        )

# ============================================================
# FINAL INFORMATION
# ============================================================

st.divider()

st.header("✨ Why Use Course Recommendation System?")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info(
        "🎯\n\n**Personalized**\n\nRecommendations based on your profile."
    )

with col2:
    st.success(
        "⚡\n\n**Fast**\n\nFind suitable courses quickly."
    )

with col3:
    st.warning(
        "📊\n\n**Data Driven**\n\nCourses ranked using profile matching."
    )

with col4:
    st.error(
        "🚀\n\n**Career Focused**\n\nDesigned around your career goals."
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 Course Recommendation System | "
    "Built with Python, Pandas & Streamlit"
)
