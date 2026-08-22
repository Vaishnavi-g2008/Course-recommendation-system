import os
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
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.header {
    background: linear-gradient(135deg, #667eea, #764ba2);
    padding: 35px;
    border-radius: 25px;
    text-align: center;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
}

.header h1 {
    font-size: 42px;
    margin-bottom: 10px;
}

.header p {
    font-size: 18px;
}

.pref-card {
    background: #eef4ff;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
}

.course-card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    margin: 20px 0;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.10);
    border-left: 6px solid #667eea;
}

.course-title {
    font-size: 28px;
    font-weight: bold;
    color: #222;
}

.badge {
    background: #667eea;
    color: white;
    padding: 8px 15px;
    border-radius: 20px;
    font-weight: bold;
}

.match {
    font-size: 30px;
    font-weight: bold;
    color: #667eea;
}

.info {
    font-size: 16px;
    margin: 8px 0;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
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


# =========================================================
# CSV NOT FOUND
# =========================================================

if csv_path is None:

    st.error("❌ Course dataset सापडला नाही!")

    st.write("app.py आणि CSV file एकाच folder मध्ये ठेवा.")

    st.write("Expected files:")

    for file_name in possible_files:
        st.write("•", file_name)

    st.write("Current folder:")

    st.code(BASE_DIR)

    st.stop()


# =========================================================
# LOAD DATA
# =========================================================

try:
    df = pd.read_csv(csv_path)

except Exception as e:

    st.error("❌ CSV file read करताना error आला.")
    st.code(str(e))
    st.stop()


# Remove extra spaces from column names
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

    st.error("❌ Dataset मध्ये काही required columns missing आहेत.")

    st.write("### Missing Columns")

    for col in missing_columns:
        st.write("•", col)

    st.write("### तुझ्या CSV मधील actual columns:")

    st.code(", ".join(df.columns.tolist()))

    st.stop()


# =========================================================
# CLEAN DATA
# =========================================================

for col in required_columns:

    df[col] = df[col].fillna("Unknown")

    df[col] = df[col].astype(str).str.strip()


# Rating numeric
df["Rating"] = pd.to_numeric(
    df["Rating"],
    errors="coerce"
).fillna(0)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="header">

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

with st.sidebar:

    st.markdown("## 🎯 Your Preferences")

    st.write("Select your preferences below.")


    # Category
    category_list = sorted(
        df["Category"].dropna().unique().tolist()
    )

    category = st.selectbox(
        "📚 Category",
        category_list
    )


    # Skill Level
    skill_level_list = sorted(
        df["Skill_Level"].dropna().unique().tolist()
    )

    skill_level = st.selectbox(
        "📈 Skill Level",
        skill_level_list
    )


    # Skills
    skills_list = sorted(
        df["Skills"].dropna().unique().tolist()
    )

    skills = st.selectbox(
        "🛠️ Skills",
        skills_list
    )


    # Interest
    interest_list = sorted(
        df["Interest"].dropna().unique().tolist()
    )

    interest = st.selectbox(
        "💡 Interest",
        interest_list
    )


    # Education
    education_list = sorted(
        df["Education"].dropna().unique().tolist()
    )

    education = st.selectbox(
        "🎓 Education",
        education_list
    )


    # Career Goal
    career_list = sorted(
        df["Career_Goal"].dropna().unique().tolist()
    )

    career_goal = st.selectbox(
        "🎯 Career Goal",
        career_list
    )


    st.write("")


    # =====================================================
    # RECOMMENDATION BUTTON
    # =====================================================

    recommend_button = st.button(
        "🎯 Get Recommendations"
    )


# =========================================================
# SELECTED PREFERENCES
# =========================================================

st.markdown("## 🔍 Your Selected Preferences")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="pref-card">
        📚 <b>Category</b><br><br>
        {category}
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="pref-card">
        🛠️ <b>Skill</b><br><br>
        {skills}
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="pref-card">
        💡 <b>Interest</b><br><br>
        {interest}
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""
        <div class="pref-card">
        🎯 <b>Career Goal</b><br><br>
        {career_goal}
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# RECOMMENDATION FUNCTION
# =========================================================

def get_recommendations():

    data = df.copy()

    data["Score"] = 0

    # -----------------------------------------------------
    # CATEGORY MATCH
    # -----------------------------------------------------

    data.loc[
        data["Category"].str.lower() == category.lower(),
        "Score"
    ] += 30


    # -----------------------------------------------------
    # SKILL LEVEL MATCH
    # -----------------------------------------------------

    data.loc[
        data["Skill_Level"].str.lower() == skill_level.lower(),
        "Score"
    ] += 20


    # -----------------------------------------------------
    # SKILLS MATCH
    # -----------------------------------------------------

    data.loc[
        data["Skills"].str.lower().str.contains(
            skills.lower(),
            na=False
        ),
        "Score"
    ] += 15


    # -----------------------------------------------------
    # INTEREST MATCH
    # -----------------------------------------------------

    data.loc[
        data["Interest"].str.lower() == interest.lower(),
        "Score"
    ] += 15


    # -----------------------------------------------------
    # EDUCATION MATCH
    # -----------------------------------------------------

    data.loc[
        data["Education"].str.lower() == education.lower(),
        "Score"
    ] += 10


    # -----------------------------------------------------
    # CAREER GOAL MATCH
    # -----------------------------------------------------

    data.loc[
        data["Career_Goal"].str.lower() == career_goal.lower(),
        "Score"
    ] += 10


    # -----------------------------------------------------
    # SORT
    # -----------------------------------------------------

    data = data.sort_values(
        by=["Score", "Rating"],
        ascending=False
    )


    # -----------------------------------------------------
    # REMOVE DUPLICATE COURSE NAMES
    # -----------------------------------------------------

    data = data.drop_duplicates(
        subset=["Course_Name"]
    )


    # -----------------------------------------------------
    # TOP 5 COURSES
    # -----------------------------------------------------

    return data.head(5)


# =========================================================
# SHOW RECOMMENDATIONS ONLY AFTER BUTTON CLICK
# =========================================================

if recommend_button:

    recommendations = get_recommendations()


    st.markdown("---")

    st.markdown(
        "## 🏆 Your Personalized Recommendations"
    )

    st.write(
        f"Top {len(recommendations)} different courses "
        "selected according to your preferences."
    )


    # =====================================================
    # COURSE CARDS
    # =====================================================

    for index, row in recommendations.reset_index(drop=True).iterrows():

        rank = index + 1

        match_score = int(row["Score"])


        # Make minimum visible score
        if match_score < 5:
            match_score = 5


        st.markdown(
            f"""
            <div class="course-card">

            <span class="badge">
            ⭐ #{rank} RECOMMENDATION
            </span>

            <br><br>

            <div class="course-title">
            🎓 {row["Course_Name"]}
            </div>

            <br>

            <div class="match">
            🎯 Match: {match_score}%
            </div>

            <hr>

            <div class="info">
            ⭐ <b>Rating:</b> {row["Rating"]}/5
            </div>

            <div class="info">
            ⏱️ <b>Duration:</b> {row["Duration_Months"]} Months
            </div>

            <div class="info">
            💼 <b>Job Role:</b> {row["Job_Role"]}
            </div>

            <div class="info">
            📚 <b>Category:</b> {row["Category"]}
            </div>

            <div class="info">
            🛠️ <b>Skills:</b> {row["Skills"]}
            </div>

            <div class="info">
            💡 <b>Interest:</b> {row["Interest"]}
            </div>

            <div class="info">
            🎯 <b>Career Goal:</b> {row["Career_Goal"]}
            </div>

            <div class="info">
            🎓 <b>Education:</b> {row["Education"]}
            </div>

            <div class="info">
            ⚡ <b>Difficulty:</b> {row["Difficulty"]}
            </div>

            <div class="info">
            💰 <b>Salary Range:</b> {row["Salary_Range"]}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # DATASET INFORMATION
    # =====================================================

    st.markdown("---")

    st.markdown("## 📊 Dataset Information")

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


else:

    # =====================================================
    # BEFORE BUTTON CLICK
    # =====================================================

    st.markdown("---")

    st.info(
        "👈 तुमच्या preferences select करा आणि "
        "🎯 Get Recommendations button वर click करा."
    )

    st.markdown("""
    ### 🚀 How it works

    1. 📚 Select Category  
    2. 📈 Select Skill Level  
    3. 🛠️ Select Skills  
    4. 💡 Select Interest  
    5. 🎓 Select Education  
    6. 🎯 Select Career Goal  
    7. 🏆 Click **Get Recommendations**
    """)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <center>
    <b>🎓 Smart Course Recommendation System</b><br>
    Personalized Learning • Career Guidance • AI Based Recommendation
    </center>
    """,
    unsafe_allow_html=True
)
