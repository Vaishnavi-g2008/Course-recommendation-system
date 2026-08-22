import streamlit as st
import pandas as pd

# =========================================================
# PAGE CONFIGURATION
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

.stApp {
    background: #f5f7fb;
}

/* Main Hero */
.hero {
    background: linear-gradient(135deg, #667eea, #764ba2);
    padding: 35px;
    border-radius: 22px;
    text-align: center;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
}

.hero h1 {
    font-size: 42px;
    margin: 0;
    font-weight: 700;
}

.hero p {
    font-size: 18px;
    margin-top: 10px;
}


/* Preference Card */

.preference-card {
    background: white;
    padding: 18px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #e5e7eb;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.06);
}

.preference-title {
    color: #777;
    font-size: 13px;
}

.preference-value {
    color: #222;
    font-size: 17px;
    font-weight: 600;
}


/* Course Card */

.course-card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    margin-top: 15px;
    margin-bottom: 20px;
    border: 1px solid #e5e7eb;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.08);
}

.best-course {
    border: 3px solid #667eea;
    box-shadow: 0px 8px 30px rgba(102,126,234,0.20);
}

.course-title {
    font-size: 27px;
    font-weight: 700;
    color: #222;
    margin-top: 12px;
}

.badge {
    display: inline-block;
    background: #667eea;
    color: white;
    padding: 7px 15px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
}

.info-box {
    background: #f7f8fc;
    padding: 15px;
    border-radius: 12px;
    margin-top: 10px;
}

.footer {
    text-align: center;
    color: #777;
    padding: 25px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATASET
# =========================================================

try:
    df = pd.read_csv("courses.csv")

except FileNotFoundError:

    st.error(
        "❌ courses.csv file सापडली नाही. "
        "app.py आणि courses.csv एकाच folder मध्ये ठेवा."
    )

    st.stop()


# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

df.columns = (
    df.columns
    .str.strip()
)


# =========================================================
# REQUIRED COLUMNS
# =========================================================

required_columns = [
    "Course",
    "Category",
    "Skill Level",
    "Skills",
    "Interest",
    "Education",
    "Career Goal",
    "Difficulty",
    "Rating",
    "Job Role",
    "Duration"
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    st.error("❌ CSV मध्ये खालील columns missing आहेत:")

    for column in missing_columns:
        st.write("•", column)

    st.stop()


# =========================================================
# DATA CLEANING
# =========================================================

# Remove completely empty rows
df = df.dropna(how="all")


# Course name string मध्ये convert
df["Course"] = (
    df["Course"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# Empty course remove
df = df[df["Course"] != ""]


# Duplicate courses remove
df = df.drop_duplicates(
    subset=["Course"],
    keep="first"
)


# Rating numeric करा
df["Rating"] = pd.to_numeric(
    df["Rating"],
    errors="coerce"
).fillna(0)


# =========================================================
# HERO HEADER
# =========================================================

st.markdown("""
<div class="hero">

<h1>🎓 Smart Course Recommendation System</h1>

<p>
Find the perfect course based on your skills, interests & career goals 🚀
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## 🎯 Select Your Preferences")

st.sidebar.write(
    "Choose your preferences to get personalized course recommendations."
)


# =========================================================
# CATEGORY
# =========================================================

categories = sorted(
    df["Category"]
    .dropna()
    .astype(str)
    .str.strip()
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
    df["Skill Level"]
    .dropna()
    .astype(str)
    .str.strip()
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
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)

skill = st.sidebar.selectbox(
    "🛠️ Skill",
    skills
)


# =========================================================
# INTEREST
# =========================================================

interests = sorted(
    df["Interest"]
    .dropna()
    .astype(str)
    .str.strip()
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
    .dropna()
    .astype(str)
    .str.strip()
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
    df["Career Goal"]
    .dropna()
    .astype(str)
    .str.strip()
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
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)

difficulty = st.sidebar.selectbox(
    "⚡ Difficulty",
    difficulties
)


# =========================================================
# USER PREFERENCES DISPLAY
# =========================================================

st.subheader("🔎 Your Selected Preferences")

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        f"""
        <div class="preference-card">
            <div class="preference-title">📚 Category</div>
            <div class="preference-value">{category}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="preference-card">
            <div class="preference-title">🛠️ Skill</div>
            <div class="preference-value">{skill}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        f"""
        <div class="preference-card">
            <div class="preference-title">💡 Interest</div>
            <div class="preference-value">{interest}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        f"""
        <div class="preference-card">
            <div class="preference-title">🎯 Career Goal</div>
            <div class="preference-value">{career_goal}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")


# =========================================================
# RECOMMENDATION FUNCTION
# =========================================================

def calculate_match(row):

    score = 0

    # -----------------------------------------
    # CATEGORY - 25%
    # -----------------------------------------

    if (
        str(row["Category"]).strip().lower()
        ==
        str(category).strip().lower()
    ):
        score += 25


    # -----------------------------------------
    # SKILL LEVEL - 10%
    # -----------------------------------------

    if (
        str(row["Skill Level"]).strip().lower()
        ==
        str(skill_level).strip().lower()
    ):
        score += 10


    # -----------------------------------------
    # SKILL - 15%
    # -----------------------------------------

    selected_skill = (
        str(skill)
        .strip()
        .lower()
    )

    course_skills = (
        str(row["Skills"])
        .strip()
        .lower()
    )

    if selected_skill in course_skills:
        score += 15


    # -----------------------------------------
    # INTEREST - 20%
    # -----------------------------------------

    selected_interest = (
        str(interest)
        .strip()
        .lower()
    )

    course_interest = (
        str(row["Interest"])
        .strip()
        .lower()
    )

    if selected_interest in course_interest:
        score += 20


    # -----------------------------------------
    # EDUCATION - 10%
    # -----------------------------------------

    if (
        str(row["Education"]).strip().lower()
        ==
        str(education).strip().lower()
    ):
        score += 10


    # -----------------------------------------
    # CAREER GOAL - 15%
    # -----------------------------------------

    selected_goal = (
        str(career_goal)
        .strip()
        .lower()
    )

    course_goal = (
        str(row["Career Goal"])
        .strip()
        .lower()
    )

    if selected_goal in course_goal:
        score += 15


    # -----------------------------------------
    # DIFFICULTY - 5%
    # -----------------------------------------

    if (
        str(row["Difficulty"]).strip().lower()
        ==
        str(difficulty).strip().lower()
    ):
        score += 5


    return score


# =========================================================
# CALCULATE MATCH SCORE
# =========================================================

df["Match Score"] = df.apply(
    calculate_match,
    axis=1
)


# =========================================================
# SORT COURSES
# =========================================================

df = df.sort_values(
    by=["Match Score", "Rating"],
    ascending=[False, False]
)


# =========================================================
# GET DIFFERENT COURSES
# =========================================================

recommendations = []

used_courses = set()


for _, row in df.iterrows():

    course_name = (
        str(row["Course"])
        .strip()
        .lower()
    )


    # -----------------------------------------
    # Do not repeat same course
    # -----------------------------------------

    if course_name in used_courses:
        continue


    used_courses.add(course_name)

    recommendations.append(row)


    # -----------------------------------------
    # Only 5 different courses
    # -----------------------------------------

    if len(recommendations) == 5:
        break


# Convert recommendations to DataFrame

recommendations = pd.DataFrame(
    recommendations
)


# =========================================================
# RECOMMENDATION SECTION
# =========================================================

st.markdown("---")

st.subheader(
    "🏆 Your Personalized Recommendations"
)

st.write(
    "Top 5 different courses based on your selected preferences."
)


# =========================================================
# NO RECOMMENDATIONS
# =========================================================

if recommendations.empty:

    st.warning(
        "⚠️ No courses available. "
        "Please change your preferences."
    )

else:

    # =====================================================
    # BEST MATCH
    # =====================================================

    best_course = recommendations.iloc[0]


    st.markdown(
        f"""
        <div class="course-card best-course">

            <span class="badge">
                🥇 BEST MATCH
            </span>

            <div class="course-title">
                🎓 {best_course["Course"]}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # Best course metrics

    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "🎯 Course Match",
            f"{int(best_course['Match Score'])}%"
        )


    with c2:

        st.metric(
            "⭐ Rating",
            f"{best_course['Rating']}/5"
        )


    with c3:

        st.metric(
            "⏱️ Duration",
            str(best_course["Duration"])
        )


    with c4:

        st.metric(
            "💼 Job Role",
            str(best_course["Job Role"])
        )


    # Progress bar

    st.progress(
        min(
            int(best_course["Match Score"]),
            100
        ) / 100
    )


    st.write("")


    # Best course details

    d1, d2, d3 = st.columns(3)


    with d1:

        st.markdown(
            f"""
            <div class="info-box">
            📚 <b>Category</b><br>
            {best_course["Category"]}
            </div>
            """,
            unsafe_allow_html=True
        )


    with d2:

        st.markdown(
            f"""
            <div class="info-box">
            🛠️ <b>Skills</b><br>
            {best_course["Skills"]}
            </div>
            """,
            unsafe_allow_html=True
        )


    with d3:

        st.markdown(
            f"""
            <div class="info-box">
            ⚡ <b>Difficulty</b><br>
            {best_course["Difficulty"]}
            </div>
            """,
            unsafe_allow_html=True
        )


    st.write("")


    # =====================================================
    # OTHER 4 RECOMMENDATIONS
    # =====================================================

    st.subheader(
        "✨ More Recommended Courses"
    )


    for position in range(
        1,
        len(recommendations)
    ):

        course = recommendations.iloc[position]


        # ---------------------------------------------
        # Different badge
        # ---------------------------------------------

        if position == 1:

            badge_text = "🥈 #2 RECOMMENDATION"

        elif position == 2:

            badge_text = "🥉 #3 RECOMMENDATION"

        else:

            badge_text = (
                f"⭐ #{position + 1} RECOMMENDATION"
            )


        # ---------------------------------------------
        # Course card
        # ---------------------------------------------

        st.markdown(
            f"""
            <div class="course-card">

                <span class="badge">
                    {badge_text}
                </span>

                <div class="course-title">
                    🎓 {course["Course"]}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ---------------------------------------------
        # Course information
        # ---------------------------------------------

        c1, c2, c3, c4 = st.columns(4)


        with c1:

            st.metric(
                "🎯 Match",
                f"{int(course['Match Score'])}%"
            )


        with c2:

            st.metric(
                "⭐ Rating",
                f"{course['Rating']}/5"
            )


        with c3:

            st.metric(
                "⏱️ Duration",
                str(course["Duration"])
            )


        with c4:

            st.metric(
                "💼 Job Role",
                str(course["Job Role"])
            )


        # Match progress

        st.progress(
            min(
                int(course["Match Score"]),
                100
            ) / 100
        )


        st.write("")


        # Details

        d1, d2, d3 = st.columns(3)


        with d1:

            st.write(
                f"📚 **Category:** "
                f"{course['Category']}"
            )


        with d2:

            st.write(
                f"🛠️ **Skills:** "
                f"{course['Skills']}"
            )


        with d3:

            st.write(
                f"⚡ **Difficulty:** "
                f"{course['Difficulty']}"
            )


        st.write(
            f"💡 **Interest:** "
            f"{course['Interest']}"
        )


        st.write(
            f"🎯 **Career Goal:** "
            f"{course['Career Goal']}"
        )


        st.divider()


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

    🎓 <b>Smart Course Recommendation System</b>

    <br><br>

    🤖 Personalized Learning
    &nbsp; | &nbsp;
    🎯 Career Guidance
    &nbsp; | &nbsp;
    🚀 Smart Recommendations

    </div>
    """,
    unsafe_allow_html=True
)
