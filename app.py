import os
import re
import pandas as pd
import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Smart Course Recommendation System",
    page_icon="🎓",
    layout="wide"
)

# =========================================================
# CUSTOM CSS - ONLY FOR UI
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.hero {
    padding: 35px;
    border-radius: 25px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 10px;
}

.hero p {
    font-size: 18px;
}

.section-title {
    font-size: 28px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 15px;
}

.course-card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    margin: 18px 0;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.10);
    border: 1px solid #e5e7eb;
}

.course-name {
    font-size: 28px;
    font-weight: 700;
    color: #273469;
}

.badge {
    display: inline-block;
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: white;
    padding: 7px 15px;
    border-radius: 20px;
    font-weight: 600;
    margin-bottom: 12px;
}

.match {
    font-size: 30px;
    font-weight: 700;
    color: #1976d2;
}

.info {
    font-size: 16px;
    margin-top: 8px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# FIND CSV FILE
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

possible_files = [
    "course_recommendation_dataset_1000.csv",
    "final_data.csv",
    "cleaned_data.csv",
    "courses.csv"
]

csv_path = None

for file_name in possible_files:
    path = os.path.join(BASE_DIR, file_name)

    if os.path.exists(path):
        csv_path = path
        break

# If CSV not found
if csv_path is None:

    st.error("❌ CSV file सापडली नाही!")

    st.write("खालीलपैकी कोणतीही CSV file app.py च्या same folder मध्ये ठेवा:")

    for file_name in possible_files:
        st.write("•", file_name)

    st.write("Expected folder:")
    st.code(BASE_DIR)

    st.stop()

# =========================================================
# READ DATASET
# =========================================================

try:
    df = pd.read_csv(csv_path)

except Exception as e:

    st.error("❌ CSV file read करताना error आला.")
    st.write(e)
    st.stop()


# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

df.columns = df.columns.str.strip()

# =========================================================
# REQUIRED COLUMNS
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

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error("❌ Dataset मध्ये required columns missing आहेत.")

    st.write("Missing columns:")

    for col in missing_columns:
        st.write("•", col)

    st.write("तुझ्या CSV मधील actual columns:")

    st.write(list(df.columns))

    st.stop()


# =========================================================
# REMOVE EMPTY VALUES
# =========================================================

for col in required_columns:
    df[col] = df[col].fillna("").astype(str).str.strip()


# =========================================================
# HELPER FUNCTION
# =========================================================

def get_unique_values(column):

    values = []

    for value in df[column]:

        if value == "":
            continue

        if value not in values:
            values.append(value)

    return sorted(values)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🎯 Your Preferences")

    st.write("Select all your preferences below.")

    # -----------------------------------------------------
    # CATEGORY
    # -----------------------------------------------------

    category_options = ["-- Select Category --"] + get_unique_values(
        "Category"
    )

    category = st.selectbox(
        "📚 Category",
        category_options
    )

    # -----------------------------------------------------
    # SKILL LEVEL
    # -----------------------------------------------------

    skill_level_options = ["-- Select Skill Level --"] + get_unique_values(
        "Skill_Level"
    )

    skill_level = st.selectbox(
        "📊 Skill Level",
        skill_level_options
    )

    # -----------------------------------------------------
    # SKILLS
    # -----------------------------------------------------

    skill_options = ["-- Select Skills --"] + get_unique_values(
        "Skills"
    )

    skills = st.selectbox(
        "🛠️ Skills",
        skill_options
    )

    # -----------------------------------------------------
    # INTEREST
    # -----------------------------------------------------

    interest_options = ["-- Select Interest --"] + get_unique_values(
        "Interest"
    )

    interest = st.selectbox(
        "💡 Interest",
        interest_options
    )

    # -----------------------------------------------------
    # EDUCATION
    # -----------------------------------------------------

    education_options = ["-- Select Education --"] + get_unique_values(
        "Education"
    )

    education = st.selectbox(
        "🎓 Education",
        education_options
    )

    # -----------------------------------------------------
    # CAREER GOAL
    # -----------------------------------------------------

    career_options = ["-- Select Career Goal --"] + get_unique_values(
        "Career_Goal"
    )

    career_goal = st.selectbox(
        "🎯 Career Goal",
        career_options
    )


# =========================================================
# HERO SECTION
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
# SELECTED PREFERENCES
# =========================================================

st.markdown(
    '<div class="section-title">🔎 Your Selected Preferences</div>',
    unsafe_allow_html=True
)


# Check whether all information is selected

all_selected = (
    category != "-- Select Category --"
    and skill_level != "-- Select Skill Level --"
    and skills != "-- Select Skills --"
    and interest != "-- Select Interest --"
    and education != "-- Select Education --"
    and career_goal != "-- Select Career Goal --"
)


# =========================================================
# SHOW SELECTED INFORMATION
# =========================================================

if all_selected:

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(f"📚 **Category**\n\n{category}")

    with col2:
        st.info(f"📊 **Skill Level**\n\n{skill_level}")

    with col3:
        st.info(f"🛠️ **Skills**\n\n{skills}")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.info(f"💡 **Interest**\n\n{interest}")

    with col5:
        st.info(f"🎓 **Education**\n\n{education}")

    with col6:
        st.info(f"🎯 **Career Goal**\n\n{career_goal}")

    st.markdown("---")

    # =====================================================
    # RECOMMENDATION BUTTON
    # =====================================================

    button = st.button(
        "🎯 Get Recommendations",
        use_container_width=True,
        type="primary"
    )

else:

    st.warning(
        "👆 कृपया Category, Skill Level, Skills, Interest, "
        "Education आणि Career Goal ही सर्व information select करा."
    )

    button = False


# =========================================================
# RECOMMENDATION LOGIC
# =========================================================

if button:

    data = df.copy()

    # -----------------------------------------------------
    # SCORE CALCULATION
    # -----------------------------------------------------

    scores = []

    for index, row in data.iterrows():

        score = 0

        # Category
        if str(row["Category"]).lower() == category.lower():
            score += 20

        # Skill Level
        if str(row["Skill_Level"]).lower() == skill_level.lower():
            score += 15

        # Skills
        if skills.lower() in str(row["Skills"]).lower():
            score += 20

        # Interest
        if str(row["Interest"]).lower() == interest.lower():
            score += 15

        # Education
        if str(row["Education"]).lower() == education.lower():
            score += 10

        # Career Goal
        if str(row["Career_Goal"]).lower() == career_goal.lower():
            score += 20

        scores.append(score)

    data["Match_Score"] = scores

    # =====================================================
    # SORT BY SCORE
    # =====================================================

    data = data.sort_values(
        by=["Match_Score", "Rating"],
        ascending=[False, False]
    )

    # =====================================================
    # REMOVE DUPLICATE COURSES
    # =====================================================

    data = data.drop_duplicates(
        subset=["Course_Name"]
    )

    # =====================================================
    # GET TOP 5 DIFFERENT COURSES
    # =====================================================

    recommendations = data.head(5)

    # =====================================================
    # RESULT
    # =====================================================

    st.success(
        f"🎉 Top {len(recommendations)} different courses found for you!"
    )

    st.markdown(
        '<div class="section-title">🏆 Your Personalized Recommendations</div>',
        unsafe_allow_html=True
    )

    # =====================================================
    # DISPLAY COURSES
    # =====================================================

    for rank, (_, course) in enumerate(
        recommendations.iterrows(),
        start=1
    ):

        match_score = int(course["Match_Score"])

        # Course Card
        st.markdown(
            f"""
            <div class="course-card">

                <div class="badge">
                    ⭐ #{rank} RECOMMENDATION
                </div>

                <div class="course-name">
                    🎓 {course["Course_Name"]}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        # -------------------------------------------------
        # COURSE INFORMATION
        # -------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "🎯 Match",
                f"{match_score}%"
            )

        with col2:

            try:
                rating = float(course["Rating"])
                rating_text = f"{rating:.1f}/5"
            except:
                rating_text = str(course["Rating"])

            st.metric(
                "⭐ Rating",
                rating_text
            )

        with col3:

            st.metric(
                "⏱️ Duration",
                f'{course["Duration_Months"]} Months'
            )

        with col4:

            st.metric(
                "💼 Job Role",
                str(course["Job_Role"])
            )

        # -------------------------------------------------
        # PROGRESS BAR
        # -------------------------------------------------

        st.progress(
            min(match_score / 100, 1.0)
        )

        # -------------------------------------------------
        # COURSE DETAILS
        # -------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f"📚 **Category:** {course['Category']}"
            )

            st.write(
                f"🛠️ **Skills:** {course['Skills']}"
            )

            st.write(
                f"💡 **Interest:** {course['Interest']}"
            )

            st.write(
                f"💰 **Salary Range:** {course['Salary_Range']}"
            )

        with col2:

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
                f"💼 **Job Role:** {course['Job_Role']}"
            )

        st.markdown("---")


# =========================================================
# DATASET INFORMATION
# =========================================================

with st.expander("📊 Dataset Information"):

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📚 Total Courses",
            len(df)
        )

    with col2:
        st.metric(
            "📂 Categories",
            df["Category"].nunique()
        )

    with col3:
        st.metric(
            "🎯 Career Goals",
            df["Career_Goal"].nunique()
        )

    with col4:
        st.metric(
            "💡 Interests",
            df["Interest"].nunique()
        )

    st.write(
        f"📄 Dataset: **{os.path.basename(csv_path)}**"
    )
