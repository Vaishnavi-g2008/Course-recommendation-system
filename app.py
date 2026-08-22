import streamlit as st
import pandas as pd

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Course Recommendation System",
    page_icon="🎓",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("🎓 Course Recommendation System")

st.write(
    "Select your information and get personalized course recommendations."
)

# =========================================================
# LOAD CSV FILE
# =========================================================

try:
    df = pd.read_csv("courses.csv")

except FileNotFoundError:
    st.error("❌ courses.csv file not found!")
    st.stop()

# =========================================================
# REMOVE EXTRA SPACES FROM COLUMN NAMES
# =========================================================

df.columns = df.columns.str.strip()

# =========================================================
# REQUIRED COLUMNS
# =========================================================

required_columns = [
    "Category",
    "Skill_Level",
    "Interest",
    "Education",
    "Career_Goal",
    "Skills",
    "Course_Name",
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
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        "❌ The following columns are missing from courses.csv:"
    )

    st.write(missing_columns)

    st.stop()

# =========================================================
# CLEAN DATA
# =========================================================

df = df.copy()

for column in required_columns:
    df[column] = df[column].fillna("")

# =========================================================
# USER INFORMATION
# =========================================================

st.header("👤 Select Your Information")

# =========================================================
# CATEGORY + SKILL LEVEL
# =========================================================

col1, col2 = st.columns(2)

with col1:

    category_options = sorted(
        df["Category"]
        .astype(str)
        .str.strip()
        .unique()
    )

    category = st.selectbox(
        "📚 Select Category",
        category_options
    )

with col2:

    skill_level_options = sorted(
        df["Skill_Level"]
        .astype(str)
        .str.strip()
        .unique()
    )

    skill_level = st.selectbox(
        "📊 Select Skill Level",
        skill_level_options
    )

# =========================================================
# INTEREST + EDUCATION
# =========================================================

col1, col2 = st.columns(2)

with col1:

    interest_options = sorted(
        df["Interest"]
        .astype(str)
        .str.strip()
        .unique()
    )

    interest = st.selectbox(
        "💡 Select Interest",
        interest_options
    )

with col2:

    education_options = sorted(
        df["Education"]
        .astype(str)
        .str.strip()
        .unique()
    )

    education = st.selectbox(
        "🎓 Select Education",
        education_options
    )

# =========================================================
# CAREER GOAL
# =========================================================

career_options = sorted(
    df["Career_Goal"]
    .astype(str)
    .str.strip()
    .unique()
)

career_goal = st.selectbox(
    "🎯 Select Career Goal",
    career_options
)

# =========================================================
# SKILLS
# =========================================================

st.subheader("🛠️ Select Your Skills")

all_skills = set()

for value in df["Skills"]:

    skill_list = str(value).split(",")

    for skill in skill_list:

        skill = skill.strip()

        if skill:
            all_skills.add(skill)

all_skills = sorted(all_skills)

skills = st.multiselect(
    "Choose your skills",
    all_skills
)

# =========================================================
# GET RECOMMENDATIONS BUTTON
# =========================================================

if st.button(
    "🎯 Get Recommendations",
    use_container_width=True
):

    results = []

    # =====================================================
    # SELECTED SKILLS
    # =====================================================

    selected_skills = [
        str(skill).strip().lower()
        for skill in skills
    ]

    # =====================================================
    # CALCULATE COURSE SCORE
    # =====================================================

    for _, row in df.iterrows():

        score = 0

        # -------------------------------------------------
        # CATEGORY
        # -------------------------------------------------

        if (
            str(row["Category"])
            .strip()
            .lower()
            ==
            str(category)
            .strip()
            .lower()
        ):

            score += 20

        # -------------------------------------------------
        # SKILL LEVEL
        # -------------------------------------------------

        if (
            str(row["Skill_Level"])
            .strip()
            .lower()
            ==
            str(skill_level)
            .strip()
            .lower()
        ):

            score += 15

        # -------------------------------------------------
        # INTEREST
        # -------------------------------------------------

        if (
            str(row["Interest"])
            .strip()
            .lower()
            ==
            str(interest)
            .strip()
            .lower()
        ):

            score += 15

        # -------------------------------------------------
        # EDUCATION
        # -------------------------------------------------

        if (
            str(row["Education"])
            .strip()
            .lower()
            ==
            str(education)
            .strip()
            .lower()
        ):

            score += 10

        # -------------------------------------------------
        # CAREER GOAL
        # -------------------------------------------------

        if (
            str(row["Career_Goal"])
            .strip()
            .lower()
            ==
            str(career_goal)
            .strip()
            .lower()
        ):

            score += 25

        # -------------------------------------------------
        # SKILLS
        # -------------------------------------------------

        course_skills = (
            str(row["Skills"])
            .strip()
            .lower()
        )

        for skill in selected_skills:

            if skill and skill in course_skills:

                score += 5

        # -------------------------------------------------
        # STORE RESULT
        # -------------------------------------------------

        results.append({

            "score": score,

            "course": row["Course_Name"],

            "category": row["Category"],

            "skills": row["Skills"],

            "interest": row["Interest"],

            "career": row["Career_Goal"],

            "education": row["Education"],

            "duration": row["Duration_Months"],

            "rating": row["Rating"],

            "difficulty": row["Difficulty"],

            "job": row["Job_Role"],

            "salary": row["Salary_Range"]

        })

    # =====================================================
    # SORT COURSES BY SCORE
    # =====================================================

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    # =====================================================
    # GET TOP 5 DIFFERENT COURSES
    # =====================================================

    recommendations = []

    used_courses = set()

    for item in results:

        course_name = (
            str(item["course"])
            .strip()
        )

        if course_name not in used_courses:

            recommendations.append(item)

            used_courses.add(course_name)

        if len(recommendations) == 5:

            break

    # =====================================================
    # CHECK RESULTS
    # =====================================================

    if len(recommendations) == 0:

        st.warning(
            "❌ No course recommendations found."
        )

        st.stop()

    # =====================================================
    # SUCCESS MESSAGE
    # =====================================================

    st.success(
        f"🎉 Top {len(recommendations)} "
        "different courses found for you!"
    )

    # =====================================================
    # RESULT TITLE
    # =====================================================

    st.subheader(
        "🏆 Your Personalized Course Recommendations"
    )

    # =====================================================
    # DISPLAY COURSES
    # =====================================================

    for i, course in enumerate(
        recommendations,
        1
    ):

        st.markdown("---")

        # -------------------------------------------------
        # COURSE NAME
        # -------------------------------------------------

        st.markdown(
            f"## 🎓 #{i} {course['course']}"
        )

        # -------------------------------------------------
        # COURSE METRICS
        # -------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "🎯 Match",
                f"{course['score']}%"
            )

        with col2:

            st.metric(
                "⭐ Rating",
                f"{course['rating']}/5"
            )

        with col3:

            st.metric(
                "⏱️ Duration",
                f"{course['duration']} Months"
            )

        with col4:

            st.metric(
                "💼 Job Role",
                str(course["job"])
            )

        # -------------------------------------------------
        # MATCH PROGRESS
        # -------------------------------------------------

        st.progress(
            min(
                course["score"] / 100,
                1.0
            )
        )

        # -------------------------------------------------
        # COURSE DETAILS
        # -------------------------------------------------

        col1, col2 = st.columns(2)

        # LEFT SIDE
        with col1:

            st.write(
                f"📚 **Category:** "
                f"{course['category']}"
            )

            st.write(
                f"🛠️ **Skills:** "
                f"{course['skills']}"
            )

            st.write(
                f"💡 **Interest:** "
                f"{course['interest']}"
            )

            st.write(
                f"💰 **Salary Range:** "
                f"{course['salary']}"
            )

        # RIGHT SIDE
        with col2:

            st.write(
                f"🎯 **Career Goal:** "
                f"{course['career']}"
            )

            st.write(
                f"🎓 **Education:** "
                f"{course['education']}"
            )

            st.write(
                f"⚡ **Difficulty:** "
                f"{course['difficulty']}"
            )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "🎓 Course Recommendation System"
)
