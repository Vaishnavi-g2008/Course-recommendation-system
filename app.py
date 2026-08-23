import streamlit as st
import pandas as pd

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Course Recommendation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* ---------- Main App ---------- */

    .stApp {
        background: #ffffff;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    /* ---------- Title ---------- */

    .main-title {
        font-size: 48px;
        font-weight: 800;
        color: #292b3a;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }

    .subtitle {
        font-size: 18px;
        color: #555b70;
        margin-bottom: 35px;
    }

    /* ---------- Section Header ---------- */

    .section-title {
        font-size: 32px;
        font-weight: 750;
        color: #292b3a;
        margin-top: 15px;
        margin-bottom: 22px;
    }

    /* ---------- Input Labels ---------- */

    label {
        font-weight: 600 !important;
        color: #26304a !important;
        font-size: 16px !important;
    }

    /* ---------- Selectbox ---------- */

    div[data-baseweb="select"] > div {
        background-color: #f1f3f7 !important;
        border: 1px solid #e1e4ea !important;
        border-radius: 12px !important;
        min-height: 52px !important;
    }

    div[data-baseweb="select"] > div:hover {
        border: 1px solid #7b61ff !important;
    }

    /* ---------- Recommendation Button ---------- */

    div.stButton > button {
        width: 100%;
        height: 55px;
        border-radius: 12px;
        border: none;
        background: linear-gradient(90deg, #5b3cc4, #7648d8);
        color: white;
        font-size: 18px;
        font-weight: 700;
        margin-top: 18px;
        transition: 0.2s;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0px 8px 20px rgba(91, 60, 196, 0.25);
    }

    /* ---------- Recommendation Area ---------- */

    .recommendation-heading {
        font-size: 30px;
        font-weight: 750;
        color: #292b3a;
        margin-top: 35px;
        margin-bottom: 20px;
    }

    /* ---------- Course Card ---------- */

    .course-card {
        background: #ffffff;
        border: 1px solid #e5e7ee;
        border-radius: 18px;
        padding: 25px;
        margin-top: 20px;
        margin-bottom: 20px;
        box-shadow: 0px 5px 18px rgba(30, 35, 60, 0.07);
    }

    .course-card:hover {
        box-shadow: 0px 9px 28px rgba(30, 35, 60, 0.11);
    }

    .course-title {
        font-size: 25px;
        font-weight: 750;
        color: #2d3042;
        margin-bottom: 18px;
    }

    /* ---------- Rank Badge ---------- */

    .rank-badge {
        display: inline-block;
        background: #f0eaff;
        color: #5a3db5;
        padding: 7px 13px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
        margin-bottom: 10px;
    }

    /* ---------- Info Boxes ---------- */

    .info-box {
        background: #f7f8fb;
        border-radius: 12px;
        padding: 14px;
        min-height: 78px;
        border: 1px solid #eceef3;
    }

    .info-label {
        color: #727789;
        font-size: 13px;
        margin-bottom: 5px;
    }

    .info-value {
        color: #292b3a;
        font-size: 16px;
        font-weight: 650;
    }

    /* ---------- Detail Box ---------- */

    .detail-box {
        background: #fafbfc;
        border-radius: 12px;
        padding: 16px 18px;
        border: 1px solid #eceef3;
        margin-top: 15px;
        line-height: 1.7;
        color: #404558;
    }

    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #777c8c;
        font-size: 14px;
        margin-top: 45px;
        padding-top: 20px;
        border-top: 1px solid #eeeeee;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🎓 Course Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Select your information to get personalized course recommendations.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD DATASET
# =========================================================

FILE_NAME = "course_recommendation_dataset_1000.csv"

try:
    df = pd.read_csv(FILE_NAME)

except FileNotFoundError:

    st.error(f"❌ Dataset not found: {FILE_NAME}")

    st.info(
        "Make sure the CSV file is in the same folder as app.py."
    )

    st.stop()


# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)


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

    st.error(
        "❌ Required columns are missing from the dataset."
    )

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

    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )


# =========================================================
# USER INFORMATION SECTION
# =========================================================

st.markdown(
    '<div class="section-title">👤 Select Your Information</div>',
    unsafe_allow_html=True
)


# =========================================================
# CATEGORY + SKILL LEVEL
# =========================================================

col1, col2 = st.columns(2)

with col1:

    category = st.selectbox(
        "📚 Select Category",
        sorted(
            df["Category"].unique()
        ),
        key="category"
    )

with col2:

    skill_level = st.selectbox(
        "📊 Select Skill Level",
        sorted(
            df["Skill_Level"].unique()
        ),
        key="skill_level"
    )


# =========================================================
# INTEREST + EDUCATION
# =========================================================

col1, col2 = st.columns(2)

with col1:

    interest = st.selectbox(
        "💡 Select Interest",
        sorted(
            df["Interest"].unique()
        ),
        key="interest"
    )

with col2:

    education = st.selectbox(
        "🎓 Select Education",
        sorted(
            df["Education"].unique()
        ),
        key="education"
    )


# =========================================================
# CAREER GOAL
# =========================================================

career_goal = st.selectbox(
    "🎯 Select Career Goal",
    sorted(
        df["Career_Goal"].unique()
    ),
    key="career_goal"
)


# =========================================================
# BUTTON
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

get_recommendations = st.button(
    "🎯  Get Personalized Recommendations"
)


# =========================================================
# RECOMMENDATION LOGIC
# =========================================================

if get_recommendations:

    results = []

    # -----------------------------------------------------
    # SCORE EACH COURSE
    # -----------------------------------------------------

    for _, row in df.iterrows():

        score = 0

        # Category = 25
        if (
            row["Category"].lower()
            == category.lower()
        ):
            score += 25

        # Skill Level = 20
        if (
            row["Skill_Level"].lower()
            == skill_level.lower()
        ):
            score += 20

        # Interest = 20
        if (
            row["Interest"].lower()
            == interest.lower()
        ):
            score += 20

        # Education = 10
        if (
            row["Education"].lower()
            == education.lower()
        ):
            score += 10

        # Career Goal = 25
        if (
            row["Career_Goal"].lower()
            == career_goal.lower()
        ):
            score += 25

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
    # SORT RESULTS
    # =====================================================

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )


    # =====================================================
    # TOP 5 DIFFERENT COURSES
    # =====================================================

    recommendations = []

    used_courses = set()

    for item in results:

        course_name = str(
            item["course"]
        ).strip()

        if not course_name:
            continue

        if course_name.lower() not in used_courses:

            recommendations.append(item)

            used_courses.add(
                course_name.lower()
            )

        if len(recommendations) == 5:
            break


    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    if recommendations:

        st.success(
            f"🎉 We found {len(recommendations)} personalized "
            "course recommendations for you!"
        )

        st.markdown(
            '<div class="recommendation-heading">'
            '🏆 Your Personalized Course Recommendations'
            '</div>',
            unsafe_allow_html=True
        )


        # =================================================
        # COURSE CARDS
        # =================================================

        for i, course in enumerate(
            recommendations,
            1
        ):

            st.markdown(
                '<div class="course-card">',
                unsafe_allow_html=True
            )

            # ---------------------------------------------
            # Rank + Course Name
            # ---------------------------------------------

            st.markdown(
                f'<div class="rank-badge">🏆 Recommendation #{i}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="course-title">'
                f'🎓 {course["course"]}'
                f'</div>',
                unsafe_allow_html=True
            )


            # ---------------------------------------------
            # TOP INFORMATION
            # ---------------------------------------------

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.markdown(
                    f"""
                    <div class="info-box">
                        <div class="info-label">🎯 Match Score</div>
                        <div class="info-value">
                            {course["score"]}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c2:

                st.markdown(
                    f"""
                    <div class="info-box">
                        <div class="info-label">⭐ Rating</div>
                        <div class="info-value">
                            {course["rating"]}/5
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c3:

                st.markdown(
                    f"""
                    <div class="info-box">
                        <div class="info-label">⏱️ Duration</div>
                        <div class="info-value">
                            {course["duration"]} Months
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c4:

                st.markdown(
                    f"""
                    <div class="info-box">
                        <div class="info-label">💼 Job Role</div>
                        <div class="info-value">
                            {course["job"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # ---------------------------------------------
            # MATCH PROGRESS
            # ---------------------------------------------

            st.write("")

            st.progress(
                min(
                    course["score"] / 100,
                    1.0
                )
            )


            # ---------------------------------------------
            # COURSE DETAILS
            # ---------------------------------------------

            d1, d2 = st.columns(2)

            with d1:

                st.markdown(
                    f"""
                    <div class="detail-box">

                    📚 <b>Category:</b>
                    {course["category"]}<br>

                    💡 <b>Interest:</b>
                    {course["interest"]}<br>

                    🛠️ <b>Skills Covered:</b>
                    {course["skills"]}<br>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with d2:

                st.markdown(
                    f"""
                    <div class="detail-box">

                    🎯 <b>Career Goal:</b>
                    {course["career"]}<br>

                    🎓 <b>Education:</b>
                    {course["education"]}<br>

                    ⚡ <b>Difficulty:</b>
                    {course["difficulty"]}<br>

                    💰 <b>Salary Range:</b>
                    {course["salary"]}<br>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


    else:

        st.warning(
            "❌ No course recommendations found."
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        🎓 <b>Course Recommendation System</b>
        <br>
        Personalized learning recommendations using Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)
