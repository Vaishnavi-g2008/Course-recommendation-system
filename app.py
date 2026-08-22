import streamlit as st
import pandas as pd
import os

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Course Recommendation System",
    page_icon="🎓",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.title-box {
    text-align: center;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}

.big-title {
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# STREAMLIT TEST
# ============================================================

st.title("🎓 Course Recommendation System")

st.write(
    "Find the best course based on your skills, interests and career goals."
)

st.success("✅ Streamlit application started successfully!")


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
        raise FileNotFoundError(
            "final_data.csv file app.py च्या folder मध्ये सापडला नाही."
        )

    data = pd.read_csv(file_path)

    return data


# ============================================================
# LOAD DATA SAFELY
# ============================================================

try:

    df = load_data()

except Exception as e:

    st.error("❌ Dataset load करण्यात problem आला.")

    st.error(str(e))

    st.warning(
        "📁 app.py आणि final_data.csv हे दोन्ही एकाच folder मध्ये आहेत का ते check करा."
    )

    st.stop()


# ============================================================
# DATASET CHECK
# ============================================================

if df.empty:

    st.error("❌ final_data.csv मध्ये कोणताही data नाही.")

    st.stop()


# ============================================================
# REMOVE EXTRA SPACES FROM COLUMN NAMES
# ============================================================

df.columns = df.columns.astype(str).str.strip()


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


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

missing_columns = [

    col
    for col in required_columns
    if col not in df.columns

]


if missing_columns:

    st.error(
        "❌ Dataset मध्ये required columns missing आहेत."
    )

    st.write(
        "### Missing Columns"
    )

    st.write(missing_columns)

    st.write(
        "### Available Columns"
    )

    st.write(list(df.columns))

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


# ============================================================
# NUMERIC COLUMNS
# ============================================================

df["Rating"] = pd.to_numeric(
    df["Rating"],
    errors="coerce"
).fillna(0)


df["Duration_Months"] = pd.to_numeric(
    df["Duration_Months"],
    errors="coerce"
).fillna(0)


# ============================================================
# REMOVE COMPLETELY EMPTY ROWS
# ============================================================

df = df[
    df["Course_ID"].astype(str).str.strip() != ""
].copy()


if df.empty:

    st.error(
        "❌ Dataset मध्ये valid course records सापडले नाहीत."
    )

    st.stop()


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.divider()

st.header("📊 Course Platform Overview")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "🎓 Total Courses",
        df["Course_ID"].nunique()
    )


with col2:

    st.metric(
        "📚 Categories",
        df["Category"].nunique()
    )


with col3:

    st.metric(
        "💼 Job Roles",
        df["Job_Role"].nunique()
    )


with col4:

    avg_rating = df["Rating"].mean()

    st.metric(
        "⭐ Average Rating",
        round(avg_rating, 2)
    )


# ============================================================
# USER PROFILE
# ============================================================

st.divider()

st.header("🎯 Build Your Learning Profile")


# ============================================================
# FIRST ROW
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    category_options = sorted(
        df["Category"].dropna().unique().tolist()
    )

    category = st.selectbox(
        "📚 Course Category",
        category_options
    )


with col2:

    skill_level_options = sorted(
        df["Skill_Level"].dropna().unique().tolist()
    )

    skill_level = st.selectbox(
        "📈 Skill Level",
        skill_level_options
    )


with col3:

    interest_options = sorted(
        df["Interest"].dropna().unique().tolist()
    )

    interest = st.selectbox(
        "💡 Area of Interest",
        interest_options
    )


# ============================================================
# SECOND ROW
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    education_options = sorted(
        df["Education"].dropna().unique().tolist()
    )

    education = st.selectbox(
        "🎓 Education",
        education_options
    )


with col2:

    career_goal_options = sorted(
        df["Career_Goal"].dropna().unique().tolist()
    )

    career_goal = st.selectbox(
        "🚀 Career Goal",
        career_goal_options
    )


with col3:

    skills_options = sorted(
        df["Skills"].dropna().unique().tolist()
    )

    skills = st.selectbox(
        "🛠️ Skills",
        skills_options
    )


# ============================================================
# THIRD ROW
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    difficulty_options = sorted(
        df["Difficulty"].dropna().unique().tolist()
    )

    difficulty = st.selectbox(
        "⚡ Difficulty",
        difficulty_options
    )


with col2:

    job_role_options = sorted(
        df["Job_Role"].dropna().unique().tolist()
    )

    job_role = st.selectbox(
        "💼 Target Job Role",
        job_role_options
    )


with col3:

    salary_options = sorted(
        df["Salary_Range"].dropna().unique().tolist()
    )

    salary_range = st.selectbox(
        "💰 Expected Salary Range",
        salary_options
    )


# ============================================================
# RECOMMENDATION
# ============================================================

st.divider()

st.header("🔍 Course Recommendation")


recommend_button = st.button(
    "🚀 Recommend My Course",
    use_container_width=True
)


# ============================================================
# RECOMMENDATION LOGIC
# ============================================================

if recommend_button:

    result = df.copy()

    # --------------------------------------------------------
    # START SCORE
    # --------------------------------------------------------

    result["Match_Score"] = 0


    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    result.loc[
        result["Category"].str.lower() == str(category).lower(),
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
    # SALARY RANGE
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


    # ========================================================
    # BEST COURSE
    # ========================================================

    if len(result) == 0:

        st.error(
            "❌ कोणताही course recommendation मिळाला नाही."
        )

        st.stop()


    course = result.iloc[0]


    # ========================================================
    # SUCCESS
    # ========================================================

    st.success(
        "🎉 Course recommendation generated successfully!"
    )


    st.divider()


    # ========================================================
    # COURSE NAME
    # ========================================================

    st.header("✨ Recommended Course")


    if "Course_Name" in result.columns:

        course_name = str(
            course["Course_Name"]
        )

        if course_name.strip() == "":
            course_name = f"Course ID: {course['Course_ID']}"

    else:

        course_name = f"Course ID: {course['Course_ID']}"


    st.subheader(
        f"🎓 {course_name}"
    )


    st.info(
        f"📚 Category: {course['Category']}"
    )


    # ========================================================
    # MATCH SCORE
    # ========================================================

    score = int(
        course["Match_Score"]
    )

    percentage = min(
        score,
        100
    )


    st.subheader("🎯 Profile Match")


    st.progress(
        percentage / 100
    )


    st.write(
        f"### {percentage}% Match"
    )


    # ========================================================
    # COURSE HIGHLIGHTS
    # ========================================================

    st.subheader(
        "📌 Course Highlights"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "⭐ Rating",
            f"{course['Rating']} / 5"
        )


    with col2:

        duration = float(
            course["Duration_Months"]
        )


        if duration > 0:

            duration_text = (
                f"{duration:g} Months"
            )

        else:

            duration_text = "N/A"


        st.metric(
            "⏱️ Duration",
            duration_text
        )


    with col3:

        st.metric(
            "📈 Skill Level",
            str(course["Skill_Level"])
        )


    with col4:

        st.metric(
            "💰 Salary",
            str(course["Salary_Range"])
        )


    # ========================================================
    # COURSE DETAILS
    # ========================================================

    st.divider()

    st.subheader(
        "📋 Course Details"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.write("### 🛠️ Skills")

        st.info(
            str(course["Skills"])
        )


        st.write("### 💡 Interest")

        st.info(
            str(course["Interest"])
        )


        st.write("### 🎯 Career Goal")

        st.info(
            str(course["Career_Goal"])
        )


    with col2:

        st.write("### 💼 Job Role")

        st.info(
            str(course["Job_Role"])
        )


        st.write("### ⚡ Difficulty")

        st.info(
            str(course["Difficulty"])
        )


        st.write("### 🎓 Education")

        st.info(
            str(course["Education"])
        )


    # ========================================================
    # TOP 5 COURSES
    # ========================================================

    st.divider()

    st.subheader(
        "🏆 Top 5 Recommended Courses"
    )


    top5 = result.head(5)


    top5_columns = [
        "Course_ID",
        "Category",
        "Skill_Level",
        "Rating",
        "Match_Score"
    ]


    st.dataframe(
        top5[top5_columns],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

st.header("🧠 How It Works")


col1, col2, col3 = st.columns(3)


with col1:

    st.subheader(
        "1️⃣ Select Profile"
    )

    st.write(
        "Select your education, skills, interest "
        "and career preferences."
    )


with col2:

    st.subheader(
        "2️⃣ Match Courses"
    )

    st.write(
        "The system compares your preferences "
        "with available courses."
    )


with col3:

    st.subheader(
        "3️⃣ Get Recommendation"
    )

    st.write(
        "The course with the highest matching score "
        "is recommended."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 Course Recommendation System | "
    "Python + Pandas + Streamlit"
)