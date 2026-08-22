import os
import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Smart Course Recommendation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #f5f7ff, #ffffff, #eef2ff);
}

.block-container {
    padding-top: 2rem;
}


/* Header */

.hero {
    padding: 35px;
    border-radius: 25px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 10px 30px rgba(80,80,150,0.25);
}

.hero h1 {
    font-size: 40px;
    font-weight: 800;
}

.hero p {
    font-size: 18px;
}


/* Course Card */

.course-card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    margin-top: 18px;
    margin-bottom: 18px;
    border: 1px solid #e4e7ef;
    box-shadow: 0 8px 25px rgba(0,0,0,0.07);
}

.best-card {
    border: 3px solid #667eea;
    box-shadow: 0 10px 35px rgba(102,126,234,0.25);
}


/* Badge */

.badge {
    display: inline-block;
    padding: 7px 15px;
    border-radius: 20px;
    background: #667eea;
    color: white;
    font-weight: 700;
    font-size: 13px;
}


/* Course title */

.course-title {
    font-size: 28px;
    font-weight: 800;
    margin-top: 15px;
    margin-bottom: 15px;
}


/* Info */

.info-box {
    background: #f7f8fc;
    padding: 15px;
    border-radius: 14px;
    margin-top: 10px;
}


/* Sidebar */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f5f7ff, #ffffff);
}


/* Footer */

.footer {
    text-align: center;
    padding: 30px;
    color: #777;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# =========================================================
# CSV FILE PATH
# =========================================================

csv_path = os.path.join(
    BASE_DIR,
    "course_recommendation_dataset_1000.csv"
)


# =========================================================
# CHECK CSV
# =========================================================

if not os.path.exists(csv_path):

    st.error(
        "❌ course_recommendation_dataset_1000.csv file सापडली नाही!"
    )

    st.write(
        "CSV file आणि app.py एकाच folder मध्ये ठेवा."
    )

    st.code(csv_path)

    st.stop()


# =========================================================
# LOAD CSV
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
# ACTUAL REQUIRED COLUMNS
# =========================================================

required_columns = [
    "Course_ID",
    "Course_Name",
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


# =========================================================
# CHECK COLUMNS
# =========================================================

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]


if missing_columns:

    st.error(
        "❌ Dataset मध्ये काही columns missing आहेत."
    )

    st.write("Missing columns:")

    for col in missing_columns:

        st.write("•", col)

    st.write("")

    st.write("तुझ्या CSV मधील actual columns:")

    st.code(
        ", ".join(df.columns.tolist())
    )

    st.stop()


# =========================================================
# CLEAN DATA
# =========================================================

for col in required_columns:

    if col not in ["Rating", "Duration_Months"]:

        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )


df["Rating"] = pd.to_numeric(
    df["Rating"],
    errors="coerce"
)

df["Rating"] = df["Rating"].fillna(0)


df["Duration_Months"] = pd.to_numeric(
    df["Duration_Months"],
    errors="coerce"
)

df["Duration_Months"] = (
    df["Duration_Months"]
    .fillna(0)
)


# =========================================================
# REMOVE EMPTY COURSE
# =========================================================

df = df[
    df["Course_Name"] != ""
]


# =========================================================
# REMOVE DUPLICATE COURSES
# =========================================================

df = df.drop_duplicates(
    subset=["Course_Name"]
)


df = df.reset_index(
    drop=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">

<h1>🎓 Smart Course Recommendation System</h1>

<p>
Find the best courses based on your skills,
interests and career goals 🚀
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "🎯 Your Preferences"
)

st.sidebar.write(
    "Select your preferences below."
)


# =========================================================
# CATEGORY
# =========================================================

categories = sorted(
    df["Category"]
    .unique()
    .tolist()
)

category = st.sidebar.selectbox(
    "📚 Category",
    categories
)


# =========================================================
# SKILL LEVEL
# =========================================================

skill_levels = sorted(
    df["Skill_Level"]
    .unique()
    .tolist()
)

skill_level = st.sidebar.selectbox(
    "📈 Skill Level",
    skill_levels
)


# =========================================================
# SKILLS
# =========================================================

skills = sorted(
    df["Skills"]
    .unique()
    .tolist()
)

skill = st.sidebar.selectbox(
    "🛠️ Skills",
    skills
)


# =========================================================
# INTEREST
# =========================================================

interests = sorted(
    df["Interest"]
    .unique()
    .tolist()
)

interest = st.sidebar.selectbox(
    "💡 Interest",
    interests
)


# =========================================================
# EDUCATION
# =========================================================

educations = sorted(
    df["Education"]
    .unique()
    .tolist()
)

education = st.sidebar.selectbox(
    "🎓 Education",
    educations
)


# =========================================================
# CAREER GOAL
# =========================================================

career_goals = sorted(
    df["Career_Goal"]
    .unique()
    .tolist()
)

career_goal = st.sidebar.selectbox(
    "🎯 Career Goal",
    career_goals
)


# =========================================================
# DIFFICULTY
# =========================================================

difficulties = sorted(
    df["Difficulty"]
    .unique()
    .tolist()
)

difficulty = st.sidebar.selectbox(
    "⚡ Difficulty",
    difficulties
)


# =========================================================
# SELECTED PREFERENCES
# =========================================================

st.subheader(
    "🔎 Your Selected Preferences"
)


p1, p2, p3, p4 = st.columns(4)


with p1:

    st.info(
        f"📚 Category\n\n{category}"
    )


with p2:

    st.info(
        f"🛠️ Skill\n\n{skill}"
    )


with p3:

    st.info(
        f"💡 Interest\n\n{interest}"
    )


with p4:

    st.info(
        f"🎯 Career Goal\n\n{career_goal}"
    )


# =========================================================
# RECOMMENDATION SCORE
# =========================================================

def calculate_score(row):

    score = 0


    # Category = 25
    if str(row["Category"]).lower() == str(category).lower():

        score += 25


    # Skill Level = 10
    if str(row["Skill_Level"]).lower() == str(skill_level).lower():

        score += 10


    # Skills = 15
    if str(skill).lower() in str(row["Skills"]).lower():

        score += 15


    # Interest = 20
    if str(interest).lower() in str(row["Interest"]).lower():

        score += 20


    # Education = 10
    if str(row["Education"]).lower() == str(education).lower():

        score += 10


    # Career Goal = 15
    if str(career_goal).lower() in str(row["Career_Goal"]).lower():

        score += 15


    # Difficulty = 5
    if str(row["Difficulty"]).lower() == str(difficulty).lower():

        score += 5


    return score


# =========================================================
# CALCULATE SCORE
# =========================================================

df["Match_Score"] = df.apply(
    calculate_score,
    axis=1
)


# =========================================================
# FINAL SCORE
# =========================================================

df["Final_Score"] = (
    df["Match_Score"]
    +
    (df["Rating"] * 0.5)
)


# =========================================================
# SORT
# =========================================================

df = df.sort_values(
    by=["Final_Score", "Rating"],
    ascending=[False, False]
)


# =========================================================
# GET DIFFERENT TOP 5 COURSES
# =========================================================

recommendations = []

used_courses = set()


for _, row in df.iterrows():

    course_name = (
        str(row["Course_Name"])
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
# RECOMMENDATION SECTION
# =========================================================

st.divider()

st.header(
    "🏆 Your Personalized Recommendations"
)

st.write(
    "Top 5 different courses selected according to your preferences."
)


# =========================================================
# IF NO COURSE
# =========================================================

if recommendations.empty:

    st.warning(
        "⚠️ No recommendations found."
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
🎓 {best["Course_Name"]}
</div>

</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# BEST COURSE METRICS
# =========================================================

a, b, c, d = st.columns(4)


with a:

    st.metric(
        "🎯 Match",
        f"{int(best['Match_Score'])}%"
    )


with b:

    st.metric(
        "⭐ Rating",
        f"{best['Rating']}/5"
    )


with c:

    st.metric(
        "⏱️ Duration",
        f"{int(best['Duration_Months'])} Months"
    )


with d:

    st.metric(
        "💼 Job Role",
        best["Job_Role"]
    )


# =========================================================
# PROGRESS
# =========================================================

st.progress(
    min(
        int(best["Match_Score"]),
        100
    ) / 100
)


# =========================================================
# BEST COURSE DETAILS
# =========================================================

x1, x2, x3 = st.columns(3)


with x1:

    st.markdown(
        f"""
        <div class="info-box">

        📚 <b>Category</b><br>
        {best["Category"]}

        </div>
        """,
        unsafe_allow_html=True
    )


with x2:

    st.markdown(
        f"""
        <div class="info-box">

        🛠️ <b>Skills</b><br>
        {best["Skills"]}

        </div>
        """,
        unsafe_allow_html=True
    )


with x3:

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

st.success(
    f"💡 Interest: {best['Interest']}   |   "
    f"🎯 Career Goal: {best['Career_Goal']}   |   "
    f"🎓 Education: {best['Education']}"
)


st.info(
    f"💰 Salary Range: {best['Salary_Range']}"
)


# =========================================================
# OTHER COURSES
# =========================================================

st.divider()

st.header(
    "✨ More Recommended Courses"
)


for i in range(
    1,
    len(recommendations)
):

    course = recommendations.iloc[i]


    # Badge

    if i == 1:

        badge = "🥈 #2 RECOMMENDATION"

    elif i == 2:

        badge = "🥉 #3 RECOMMENDATION"

    else:

        badge = f"⭐ #{i + 1} RECOMMENDATION"


    # Card

    st.markdown(
        f"""
<div class="course-card">

<span class="badge">
{badge}
</span>

<div class="course-title">
🎓 {course["Course_Name"]}
</div>

</div>
""",
        unsafe_allow_html=True
    )


    # Metrics

    a, b, c, d = st.columns(4)


    with a:

        st.metric(
            "🎯 Match",
            f"{int(course['Match_Score'])}%"
        )


    with b:

        st.metric(
            "⭐ Rating",
            f"{course['Rating']}/5"
        )


    with c:

        st.metric(
            "⏱️ Duration",
            f"{int(course['Duration_Months'])} Months"
        )


    with d:

        st.metric(
            "💼 Job Role",
            course["Job_Role"]
        )


    # Progress

    st.progress(
        min(
            int(course["Match_Score"]),
            100
        ) / 100
    )


    # Details

    c1, c2 = st.columns(2)


    with c1:

        st.write(
            f"📚 **Category:** {course['Category']}"
        )

        st.write(
            f"🛠️ **Skills:** {course['Skills']}"
        )

        st.write(
            f"💡 **Interest:** {course['Interest']}"
        )


    with c2:

        st.write(
            f"🎯 **Career Goal:** {course['Career_Goal']}"
        )

        st.write(
            f"🎓 **Education:** {course['Education']}"
        )

        st.write(
            f"⚡ **Difficulty:** {course['Difficulty']}"
        )


    st.write(
        f"💰 **Salary Range:** {course['Salary_Range']}"
    )


    st.divider()


# =========================================================
# DATASET INFORMATION
# =========================================================

with st.expander("📊 Dataset Information"):

    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "📚 Total Courses",
            len(df)
        )


    with c2:

        st.metric(
            "📂 Categories",
            df["Category"].nunique()
        )


    with c3:

        st.metric(
            "🎯 Career Goals",
            df["Career_Goal"].nunique()
        )


    with c4:

        st.metric(
            "💡 Interests",
            df["Interest"].nunique()
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

🎓 <b>Smart Course Recommendation System</b>

<br><br>

🤖 AI Based Recommendation
&nbsp; | &nbsp;
🎯 Career Guidance
&nbsp; | &nbsp;
🚀 Personalized Learning

</div>
""", unsafe_allow_html=True)
