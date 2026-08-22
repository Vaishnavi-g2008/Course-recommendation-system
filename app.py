import streamlit as st
import pandas as pd

# =========================================================
# PAGE CONFIG
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
    "Select your information to get personalized course recommendations."
)

# =========================================================
# LOAD DATASET
# =========================================================

FILE_NAME = "course_recommendation_dataset_1000.csv"

try:
    df = pd.read_csv(FILE_NAME)

except FileNotFoundError:
    st.error(f"❌ Dataset not found: {FILE_NAME}")
    st.info("Make sure the CSV file is in the same folder as app.py.")
    st.stop()

# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

df.columns = df.columns.astype(str).str.strip()

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
# CHECK REQUIRED COLUMNS
# =========================================================

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error("❌ Required columns are missing from the dataset.")

    st.write("Missing columns:")

    for column in missing_columns:
        st.write(f"- {column}")

    st.write("Columns available in your CSV:")

    st.write(list(df.columns))

    st.stop()

# =========================================================
# CLEAN DATA
# =========================================================

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

    category = st.selectbox(
        "📚 Select Category",
        sorted(
            df["Category"]
            .astype(str)
            .str.strip()
            .unique()
        )
    )

with col2:

    skill_level = st.selectbox(
        "📊 Select Skill Level",
        sorted(
            df["Skill_Level"]
            .astype(str)
            .str.strip()
            .unique()
        )
    )

# =========================================================
# INTEREST + EDUCATION
# =========================================================

col1, col2 = st.columns(2)

with col1:

    interest = st.selectbox(
        "💡 Select Interest",
        sorted(
            df["Interest"]
            .astype(str)
            .str.strip()
            .unique()
        )
    )

with col2:

    education = st.selectbox(
        "🎓 Select Education",
        sorted(
            df["Education"]
            .astype(str)
            .str.strip()
            .unique()
        )
    )

# =========================================================
# CAREER GOAL
# =========================================================

career_goal = st.selectbox(
    "🎯 Select Career Goal",
    sorted(
        df["Career_Goal"]
        .astype(str)
        .str.strip()
        .unique()
    )
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

skills = st.multiselect(
    "Choose your skills",
    sorted(all_skills)
)

# =========================================================
# GET RECOMMENDATIONS
# =========================================================

if st.button(
    "🎯 Get Recommendations",
    use_container_width=True
):

    results = []

    # -----------------------------------------------------
    # SELECTED SKILLS
    # -----------------------------------------------------

    selected_skills = [
        str(skill).strip().lower()
        for skill in skills
    ]

    # -----------------------------------------------------
    # SCORE EACH COURSE
    # -----------------------------------------------------

    for _, row in df.iterrows():

        score = 0

        # Category
        if (
            str(row["Category"]).strip().lower()
            ==
            str(category).strip().lower()
        ):
            score += 20

        # Skill Level
        if (
            str(row["Skill_Level"]).strip().lower()
            ==
            str(skill_level).strip().lower()
        ):
            score += 15

        # Interest
        if (
            str(row["Interest"]).strip().lower()
            ==
            str(interest).strip().lower()
        ):
            score += 15

        # Education
        if (
            str(row["Education"]).strip().lower()
            ==
            str(education).strip().lower()
        ):
            score += 10

        # Career Goal
        if (
            str(row["Career_Goal"]).strip().lower()
            ==
            str(career_goal).strip().lower()
        ):
            score += 25

        # Skills
        course_skills = (
            str(row["Skills"])
            .strip()
            .lower()
        )

        for skill in selected_skills:

            if skill and skill in course_skills:
                score += 5

        # Store result
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
    # SORT BY SCORE
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

        course_name = str(
            item["course"]
        ).strip()

        if course_name not in used_courses:

            recommendations.append(item)
            used_courses.add(course_name)

        if len(recommendations) == 5:
            break

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    if recommendations:

        st.success(
            f"🎉 Top {len(recommendations)} "
            "different courses found for you!"
        )

        st.subheader(
            "🏆 Your Personalized Course Recommendations"
        )

        # -------------------------------------------------
        # DISPLAY EACH COURSE
        # -------------------------------------------------

        for i, course in enumerate(
            recommendations,
            1
        ):

            st.markdown("---")

            st.markdown(
                f"## 🎓 #{i} {course['course']}"
            )

            # Metrics
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

            # Progress bar
            st.progress(
                min(
                    course["score"] / 100,
                    1.0
                )
            )

            # Details
            col1, col2 = st.columns(2)

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

    else:

        st.warning(
            "❌ No course recommendations found."
        )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption("🎓 Course Recommendation System")
