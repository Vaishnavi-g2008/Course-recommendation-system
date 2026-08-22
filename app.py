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
# CUSTOM CSS - IMPRESSIVE UI
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Main Header */
.header-box {
    background: linear-gradient(135deg, #667eea, #764ba2);
    padding: 35px;
    border-radius: 25px;
    text-align: center;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.15);
}

.header-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 10px;
}

.header-subtitle {
    font-size: 18px;
}

/* Section title */
.section-title {
    font-size: 28px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 15px;
}

/* Preference cards */
.preference-card {
    background: #eef4ff;
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid #dce6ff;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.05);
}

.preference-title {
    font-size: 15px;
    color: #4267a9;
    font-weight: 600;
}

.preference-value {
    font-size: 19px;
    font-weight: 700;
    margin-top: 8px;
}

/* Recommendation card */
.course-card {
    background: white;
    padding: 28px;
    border-radius: 22px;
    margin-top: 20px;
    margin-bottom: 25px;
    border: 1px solid #e3e7ef;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.08);
}

.course-title {
    font-size: 28px;
    font-weight: 800;
    color: #222;
}

.recommendation-badge {
    display: inline-block;
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: white;
    padding: 8px 18px;
    border-radius: 20px;
    font-weight: 700;
    margin-bottom: 15px;
}

.match-box {
    background: #eef7ff;
    padding: 15px;
    border-radius: 15px;
    text-align: center;
}

.match-number {
    font-size: 30px;
    font-weight: 800;
    color: #2878d8;
}

.info-label {
    font-weight: 700;
    color: #555;
}

.info-value {
    color: #222;
    font-size: 16px;
}

.small-divider {
    margin-top: 15px;
    margin-bottom: 15px;
    border-top: 1px solid #e5e5e5;
}

/* Button */
.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 15px;
    font-size: 20px;
    font-weight: 700;
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: white;
    border: none;
}

.stButton > button:hover {
    transform: scale(1.02);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #f8f9fc;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD CSV
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(
    BASE_DIR,
    "course_recommendation_dataset_1000.csv"
)


# =========================================================
# FILE CHECK
# =========================================================

if not os.path.exists(csv_path):

    st.error("❌ Dataset file सापडली नाही!")

    st.write(
        "कृपया `course_recommendation_dataset_1000.csv` "
        "ही file `app.py` च्या same folder मध्ये ठेवा."
    )

    st.write("Expected location:")
    st.code(csv_path)

    st.stop()


# =========================================================
# READ DATASET
# =========================================================

try:

    df = pd.read_csv(csv_path)

except Exception as e:

    st.error("❌ CSV file read करता आली नाही.")
    st.write(e)
    st.stop()


# Remove unwanted spaces from column names

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
# CLEAN DATA
# =========================================================

for col in required_columns:

    if df[col].dtype == "object":

        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    else:

        df[col] = df[col].fillna(0)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="header-box">

<div class="header-title">
🎓 Smart Course Recommendation System
</div>

<div class="header-subtitle">
Find the best courses based on your skills, interests and career goals 🚀
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## 🎯 Your Preferences")

st.sidebar.write(
    "Select all your preferences below."
)


# =========================================================
# GET OPTIONS
# =========================================================

def get_options(column):

    values = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = values[values != ""]

    return sorted(values.unique().tolist())


categories = get_options("Category")
skill_levels = get_options("Skill_Level")
skills = get_options("Skills")
interests = get_options("Interest")
educations = get_options("Education")
career_goals = get_options("Career_Goal")


# =========================================================
# SIDEBAR INPUTS
# =========================================================

category = st.sidebar.selectbox(
    "📚 Category",
    ["Select Category"] + categories
)


skill_level = st.sidebar.selectbox(
    "📈 Skill Level",
    ["Select Skill Level"] + skill_levels
)


selected_skill = st.sidebar.selectbox(
    "🛠️ Skills",
    ["Select Skills"] + skills
)


interest = st.sidebar.selectbox(
    "💡 Interest",
    ["Select Interest"] + interests
)


education = st.sidebar.selectbox(
    "🎓 Education",
    ["Select Education"] + educations
)


career_goal = st.sidebar.selectbox(
    "🎯 Career Goal",
    ["Select Career Goal"] + career_goals
)


# =========================================================
# CHECK ALL INFORMATION SELECTED
# =========================================================

all_selected = (
    category != "Select Category"
    and
    skill_level != "Select Skill Level"
    and
    selected_skill != "Select Skills"
    and
    interest != "Select Interest"
    and
    education != "Select Education"
    and
    career_goal != "Select Career Goal"
)


# =========================================================
# SELECTED PREFERENCES
# =========================================================

st.markdown(
    '<div class="section-title">🔎 Your Selected Preferences</div>',
    unsafe_allow_html=True
)


if all_selected:

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(f"""
        <div class="preference-card">
            <div class="preference-title">📚 Category</div>
            <div class="preference-value">{category}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class="preference-card">
            <div class="preference-title">📈 Skill Level</div>
            <div class="preference-value">{skill_level}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown(f"""
        <div class="preference-card">
            <div class="preference-title">🛠️ Skills</div>
            <div class="preference-value">{selected_skill}</div>
        </div>
        """, unsafe_allow_html=True)


    col4, col5, col6 = st.columns(3)

    with col4:

        st.markdown(f"""
        <div class="preference-card">
            <div class="preference-title">💡 Interest</div>
            <div class="preference-value">{interest}</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:

        st.markdown(f"""
        <div class="preference-card">
            <div class="preference-title">🎓 Education</div>
            <div class="preference-value">{education}</div>
        </div>
        """, unsafe_allow_html=True)

    with col6:

        st.markdown(f"""
        <div class="preference-card">
            <div class="preference-title">🎯 Career Goal</div>
            <div class="preference-value">{career_goal}</div>
        </div>
        """, unsafe_allow_html=True)


    st.write("")


    # =====================================================
    # RECOMMENDATION BUTTON
    # =====================================================

    st.markdown(
        '<div class="section-title">🚀 Ready?</div>',
        unsafe_allow_html=True
    )

    get_recommendation = st.button(
        "🎯 GET COURSE RECOMMENDATIONS"
    )


    # =====================================================
    # RECOMMENDATION LOGIC
    # =====================================================

    if get_recommendation:

        recommendation_df = df.copy()

        # -----------------------------------------------
        # Calculate matching score
        # -----------------------------------------------

        recommendation_df["Score"] = 0

        # Category - highest importance
        recommendation_df["Score"] += (
            recommendation_df["Category"]
            .str.lower()
            .eq(category.lower())
            .astype(int) * 30
        )

        # Skill Level
        recommendation_df["Score"] += (
            recommendation_df["Skill_Level"]
            .str.lower()
            .eq(skill_level.lower())
            .astype(int) * 15
        )

        # Skills
        recommendation_df["Score"] += (
            recommendation_df["Skills"]
            .str.lower()
            .str.contains(
                selected_skill.lower(),
                regex=False,
                na=False
            )
            .astype(int) * 15
        )

        # Interest
        recommendation_df["Score"] += (
            recommendation_df["Interest"]
            .str.lower()
            .eq(interest.lower())
            .astype(int) * 15
        )

        # Education
        recommendation_df["Score"] += (
            recommendation_df["Education"]
            .str.lower()
            .eq(education.lower())
            .astype(int) * 10
        )

        # Career Goal
        recommendation_df["Score"] += (
            recommendation_df["Career_Goal"]
            .str.lower()
            .eq(career_goal.lower())
            .astype(int) * 15
        )


        # ===============================================
        # SORT BY SCORE
        # ===============================================

        recommendation_df = recommendation_df.sort_values(
            by=["Score", "Rating"],
            ascending=[False, False]
        )


        # ===============================================
        # REMOVE DUPLICATE COURSE NAMES
        # ===============================================

        recommendation_df = (
            recommendation_df
            .drop_duplicates(
                subset=["Course_Name"]
            )
        )


        # ===============================================
        # TOP 5 DIFFERENT COURSES
        # ===============================================

        recommendations = recommendation_df.head(5)


        # ===============================================
        # SHOW RESULT
        # ===============================================

        st.markdown(
            '<div class="section-title">🏆 Your Personalized Recommendations</div>',
            unsafe_allow_html=True
        )

        st.success(
            f"🎉 Top {len(recommendations)} different courses found for you!"
        )


        # ===============================================
        # COURSE CARDS
        # ===============================================

        for index, (_, course) in enumerate(
            recommendations.iterrows(),
            start=1
        ):

            score = int(course["Score"])

            rating = course["Rating"]

            duration = course["Duration_Months"]

            job_role = course["Job_Role"]

            course_name = course["Course_Name"]

            course_category = course["Category"]

            course_skills = course["Skills"]

            course_interest = course["Interest"]

            course_education = course["Education"]

            difficulty = course["Difficulty"]

            salary = course["Salary_Range"]


            # -------------------------------------------
            # COURSE CARD
            # -------------------------------------------

            st.markdown(f"""
            <div class="course-card">

                <div class="recommendation-badge">
                    ⭐ #{index} RECOMMENDATION
                </div>

                <div class="course-title">
                    🎓 {course_name}
                </div>

                <div class="small-divider"></div>

            </div>
            """, unsafe_allow_html=True)


            # -------------------------------------------
            # MAIN INFO
            # -------------------------------------------

            c1, c2, c3, c4 = st.columns(4)


            with c1:

                st.markdown(
                    f"""
                    <div class="match-box">
                        <div>🎯 Match</div>
                        <div class="match-number">
                            {score}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with c2:

                st.metric(
                    "⭐ Rating",
                    f"{rating}/5"
                )


            with c3:

                st.metric(
                    "⏱️ Duration",
                    f"{duration} Months"
                )


            with c4:

                st.metric(
                    "💼 Job Role",
                    str(job_role)
                )


            # -------------------------------------------
            # PROGRESS BAR
            # -------------------------------------------

            st.progress(
                min(score, 100) / 100
            )


            # -------------------------------------------
            # DETAILS
            # -------------------------------------------

            d1, d2 = st.columns(2)


            with d1:

                st.markdown(
                    f"""
                    **📚 Category:** {course_category}

                    **🛠️ Skills:** {course_skills}

                    **💡 Interest:** {course_interest}

                    **💰 Salary Range:** {salary}
                    """
                )


            with d2:

                st.markdown(
                    f"""
                    **🎯 Career Goal:** {course["Career_Goal"]}

                    **🎓 Education:** {course_education}

                    **⚡ Difficulty:** {difficulty}

                    **🆔 Course ID:** {course["Course_ID"]}
                    """
                )


            st.divider()


# =========================================================
# IF INFORMATION IS NOT COMPLETE
# =========================================================

else:

    st.info(
        "👈 कृपया Category, Skill Level, Skills, Interest, "
        "Education आणि Career Goal ही सर्व information select करा."
    )


# =========================================================
# DATASET INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">📊 Dataset Information</div>',
    unsafe_allow_html=True
)


info1, info2, info3, info4 = st.columns(4)


with info1:

    st.metric(
        "📚 Total Courses",
        len(df)
    )


with info2:

    st.metric(
        "📂 Categories",
        df["Category"].nunique()
    )


with info3:

    st.metric(
        "🎯 Career Goals",
        df["Career_Goal"].nunique()
    )


with info4:

    st.metric(
        "💡 Interests",
        df["Interest"].nunique()
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#777;">
        🎓 Smart Course Recommendation System
        <br>
        Built with Python, Pandas & Streamlit 🚀
    </div>
    """,
    unsafe_allow_html=True
)