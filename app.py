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
# PROFESSIONAL UI STYLE
# ============================================================

st.markdown("""
<style>

    /* ========================================================
       MAIN APPLICATION
       ======================================================== */

    .stApp {
        background: linear-gradient(
            135deg,
            #f8faff 0%,
            #eef4ff 50%,
            #f8fbff 100%
        );
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ========================================================
       MAIN TITLE
       ======================================================== */

    .main-title {
        font-size: 46px;
        font-weight: 800;
        color: #172554;
        letter-spacing: -1px;
        margin-bottom: 5px;
    }

    .main-subtitle {
        font-size: 24px;
        font-weight: 700;
        color: #2563eb;
        margin-bottom: 8px;
    }

    .main-description {
        font-size: 17px;
        color: #64748b;
        margin-bottom: 18px;
    }

    /* ========================================================
       SECTION HEADINGS
       ======================================================== */

    h1, h2, h3 {
        color: #172554 !important;
    }

    h2 {
        font-weight: 750 !important;
    }

    h3 {
        font-weight: 700 !important;
    }

    /* ========================================================
       METRIC CARDS
       ======================================================== */

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid #dbe4f0;
        border-radius: 18px;
        padding: 20px;
        min-height: 120px;
        box-shadow:
            0 8px 25px rgba(15, 23, 42, 0.07);
        transition: all 0.25s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow:
            0 14px 30px rgba(37, 99, 235, 0.13);
    }

    div[data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #475569;
    }

    div[data-testid="stMetricValue"] {
        color: #172554;
        font-weight: 800;
    }

    /* ========================================================
       SELECT BOX
       ======================================================== */

    div[data-baseweb="select"] > div {
        background: white;
        border-radius: 12px;
        border: 1px solid #dbe4f0;
        min-height: 48px;
    }

    div[data-baseweb="select"] > div:hover {
        border-color: #2563eb;
    }

    /* ========================================================
       BUTTON
       ======================================================== */

    .stButton > button {
        width: 100%;
        height: 55px;
        border-radius: 14px;
        font-size: 17px;
        font-weight: 750;
        border: none;
        box-shadow:
            0 8px 20px rgba(37, 99, 235, 0.20);
        transition: all 0.25s ease;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow:
            0 12px 28px rgba(37, 99, 235, 0.28);
    }

    /* ========================================================
       CONTAINERS / CARDS
       ======================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        box-shadow:
            0 8px 25px rgba(15, 23, 42, 0.06);
        padding: 8px;
        transition: all 0.25s ease;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow:
            0 12px 30px rgba(37, 99, 235, 0.10);
    }

    /* ========================================================
       ALERT BOXES
       ======================================================== */

    div[data-testid="stAlert"] {
        border-radius: 14px;
    }

    /* ========================================================
       TABS
       ======================================================== */

    button[data-baseweb="tab"] {
        font-weight: 650;
        font-size: 15px;
    }

    /* ========================================================
       PROGRESS BAR
       ======================================================== */

    div[data-testid="stProgress"] > div {
        border-radius: 20px;
    }

    /* ========================================================
       DIVIDER
       ======================================================== */

    hr {
        border-color: #dbe4f0;
        margin-top: 25px;
        margin-bottom: 25px;
    }

    /* ========================================================
       CAPTION
       ======================================================== */

    .stCaption {
        color: #64748b;
    }

</style>
""", unsafe_allow_html=True)


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
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    st.error(
        "❌ Required columns are missing."
    )

    st.write(
        "Missing columns:",
        missing_columns
    )

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

    st.error(
        "❌ No valid courses found."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎓 Course Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">✨ Smart Course Finder</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-description">'
    'Find the right course based on your skills, interests, '
    'education and career goals.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# FIND YOUR LEARNING PATH
# ============================================================

st.header("🚀 Find Your Perfect Learning Path")

st.caption(
    "Choose your learning preferences and get personalized "
    "course recommendations."
)


# ============================================================
# DASHBOARD
# ============================================================

st.header("📊 Learning Platform")

st.caption(
    "Explore our course database and available learning opportunities."
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

st.caption(
    "Explore how courses are distributed across different categories."
)


category_count = (
    df["Category"]
    .value_counts()
    .reset_index()
)

category_count.columns = [
    "Category",
    "Courses"
]


category_columns = st.columns(
    max(1, len(category_count))
)


for i, row in category_count.iterrows():

    with category_columns[i]:

        with st.container(border=True):

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
    "Tell us about yourself and select your learning preferences "
    "to receive the most suitable course recommendations."
)


# ============================================================
# OPTIONS FUNCTION
# ============================================================

def get_options(column):

    return sorted([
        x
        for x in df[column].unique()
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
# FIND COURSE BUTTON
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
        by=[
            "Match_Score",
            "Rating"
        ],
        ascending=[
            False,
            False
        ]
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
    # MATCH SCORE
    # ========================================================

    score = int(
        course["Match_Score"]
    )

    percentage = min(
        score,
        100
    )


    # ========================================================
    # RECOMMENDATION RESULT
    # ========================================================

    st.divider()

    st.header("🏆 Your Course Recommendation")

    st.success(
        "🎉 We found a course that matches your learning profile!"
    )


    # ========================================================
    # BEST COURSE CARD
    # ========================================================

    with st.container(border=True):

        st.subheader(
            f"🎓 {course_name}"
        )

        st.write("")


        col1, col2 = st.columns(
            [2, 1]
        )


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
            "⚡ Difficulty",
            course["Difficulty"]
        )


    with col4:

        st.metric(
            "💰 Salary",
            course["Salary_Range"]
        )


    # ========================================================
    # COURSE DETAILS
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

        st.subheader(
            "🛠️ Required Skills"
        )

        st.info(
            course["Skills"]
        )

        st.subheader(
            "💡 Area of Interest"
        )

        st.info(
            course["Interest"]
        )


    with tab2:

        st.subheader(
            "🎯 Career Goal"
        )

        st.success(
            course["Career_Goal"]
        )

        st.subheader(
            "💼 Target Job Role"
        )

        st.info(
            course["Job_Role"]
        )

        st.subheader(
            "⚡ Difficulty Level"
        )

        st.warning(
            course["Difficulty"]
        )


    with tab3:

        st.subheader(
            "🎓 Required Education"
        )

        st.info(
            course["Education"]
        )

        st.subheader(
            "📚 Course Category"
        )

        st.info(
            course["Category"]
        )


    # ========================================================
    # TOP 5 RECOMMENDATIONS
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

        st.subheader(
            "01 👤"
        )

        st.write(
            "### Build Your Profile"
        )

        st.write(
            "Choose your education, skills, interests "
            "and career preferences."
        )


with col2:

    with st.container(border=True):

        st.subheader(
            "02 🔎"
        )

        st.write(
            "### Smart Matching"
        )

        st.write(
            "The system compares your preferences "
            "with the available course information."
        )


with col3:

    with st.container(border=True):

        st.subheader(
            "03 🚀"
        )

        st.write(
            "### Get Recommendations"
        )

        st.write(
            "Courses are ranked using a matching score "
            "and the best options are displayed."
        )


# ============================================================
# WHY USE THE SYSTEM
# ============================================================

st.divider()

st.header(
    "✨ Why Use Course Recommendation System?"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    with st.container(border=True):

        st.subheader("🎯 Personalized")

        st.write(
            "Recommendations are based on "
            "your learning profile."
        )


with col2:

    with st.container(border=True):

        st.subheader("⚡ Fast")

        st.write(
            "Find suitable courses quickly "
            "without searching manually."
        )


with col3:

    with st.container(border=True):

        st.subheader("📊 Data Driven")

        st.write(
            "Courses are ranked using "
            "profile matching."
        )


with col4:

    with st.container(border=True):

        st.subheader("🚀 Career Focused")

        st.write(
            "Recommendations are aligned "
            "with your career goals."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#64748b;
        padding:20px;
        font-size:14px;
    ">
        🎓 <b>Course Recommendation System</b>
        <br>
        Built with Python • Pandas • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)