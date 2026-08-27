# ============================================================
# COLORFUL STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Course Recommendation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# COLORFUL THEME
# ------------------------------------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        135deg,
        #eef2ff,
        #fdf4ff,
        #ecfeff
    );
}

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #312e81,
        #4c1d95,
        #831843
    );
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.stButton > button {
    background: linear-gradient(
        90deg,
        #2563eb,
        #7c3aed,
        #db2777
    );
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: bold;
    padding: 12px;
}

.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #1d4ed8,
        #6d28d9,
        #be185d
    );
}

div[data-testid="stMetric"] {
    background: white;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.10);
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
}

h1 {
    color: #312e81;
}

h2 {
    color: #4c1d95;
}

h3 {
    color: #6d28d9;
}

div[data-testid="stProgress"] > div > div {
    background: linear-gradient(
        90deg,
        #06b6d4,
        #6366f1,
        #ec4899
    );
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎓 Course Finder")

    st.write(
        "✨ Enter your profile to discover personalized courses."
    )

    st.divider()

    interest = st.text_input(
        "💡 Your Interest",
        placeholder="Artificial Intelligence"
    )

    career_goal = st.text_input(
        "🎯 Career Goal",
        placeholder="AI Engineer"
    )

    education = st.text_input(
        "🎓 Education",
        placeholder="Diploma"
    )

    skills = st.text_input(
        "🛠️ Your Skills",
        placeholder="Python, SQL, Pandas"
    )

    skill_level = st.selectbox(
        "📊 Skill Level",
        [
            "Beginner",
            "Intermediate",
            "Advanced"
        ]
    )

    st.divider()

    recommend_button = st.button(
        "✨ Get AI Recommendations",
        use_container_width=True
    )

    st.success(
        f"📊 {len(df):,} records loaded"
    )


# ============================================================
# HOME PAGE
# ============================================================

if not recommend_button:

    st.title("🎓 Course Recommendation System")

    st.subheader(
        "🤖 AI-Powered Personalized Course Recommendations"
    )

    st.write(
        "Find the most suitable courses based on your "
        "interest, career goal, education, skills and skill level."
    )

    st.divider()

    # Metrics
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "📚 Dataset Records",
            f"{len(df):,}"
        )

    with c2:

        unique_courses = (
            df[COURSE_COL]
            .astype(str)
            .apply(normalize)
            .nunique()
        )

        st.metric(
            "🎓 Unique Courses",
            f"{unique_courses:,}"
        )

    with c3:
        st.metric(
            "🤖 Matching",
            "AI + NLP"
        )

    with c4:
        st.metric(
            "🏆 Results",
            "Top 5"
        )

    st.divider()

    # Features
    f1, f2, f3 = st.columns(3)

    with f1:
        st.info(
            "🧠 **Smart Matching**\n\n"
            "AI analyzes your profile and course information."
        )

    with f2:
        st.success(
            "🎯 **Personalized Results**\n\n"
            "Courses are selected according to your preferences."
        )

    with f3:
        st.warning(
            "📈 **AI Ranking**\n\n"
            "Courses are ranked according to their matching score."
        )

    st.stop()


# ============================================================
# VALIDATION
# ============================================================

if not interest.strip():
    st.warning("⚠️ Please enter your Interest.")
    st.stop()

if not career_goal.strip():
    st.warning("⚠️ Please enter your Career Goal.")
    st.stop()

if not education.strip():
    st.warning("⚠️ Please enter your Education.")
    st.stop()

if not skills.strip():
    st.warning("⚠️ Please enter your Skills.")
    st.stop()


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

with st.spinner(
    "🤖 AI is analyzing your profile..."
):

    recommendations = recommend_courses(
        interest,
        career_goal,
        education,
        skills,
        skill_level
    ).head(5)


# ============================================================
# RESULTS
# ============================================================

st.title("✨ Your Recommended Courses")

st.write(
    "Based on your profile, these are the most suitable courses:"
)

st.divider()


# ============================================================
# PROFILE SUMMARY
# ============================================================

st.subheader("👤 Your Profile")

p1, p2, p3, p4, p5 = st.columns(5)

with p1:
    st.info(f"💡 Interest\n\n{interest}")

with p2:
    st.info(f"🎯 Career\n\n{career_goal}")

with p3:
    st.info(f"🎓 Education\n\n{education}")

with p4:
    st.info(f"🛠️ Skills\n\n{skills}")

with p5:
    st.info(f"📊 Level\n\n{skill_level}")


st.divider()


# ============================================================
# BEST MATCH
# ============================================================

if not recommendations.empty:

    best_course = recommendations.iloc[0]

    best_name = row_value(
        best_course,
        COURSE_COL
    )

    best_score = float(
        best_course["AI_Score"]
    )

    st.subheader("🏆 Best Match")

    b1, b2 = st.columns([4, 1])

    with b1:
        st.success(
            f"🥇 **{best_name}**\n\n"
            "This course is your strongest match."
        )

    with b2:
        st.metric(
            "🎯 AI Match",
            f"{best_score:.0f}%"
        )

    st.progress(
        min(best_score / 100, 1.0)
    )


# ============================================================
# TOP 5
# ============================================================

st.divider()

st.subheader("📚 Top 5 Course Recommendations")


for index, (_, row) in enumerate(
    recommendations.iterrows()
):

    course_name = row_value(
        row,
        COURSE_COL
    )

    score = float(
        row["AI_Score"]
    )

    career_role = row_value(
        row,
        CAREER_COL
    )

    duration_value = row_value(
        row,
        DURATION_COL
    )

    rating_value = row_value(
        row,
        RATING_COL
    )

    course_skills = row_value(
        row,
        SKILLS_COL
    )

    # Course heading
    st.subheader(
        f"⭐ Rank {index + 1} — {course_name}"
    )

    # Match score
    c1, c2 = st.columns([5, 1])

    with c1:
        st.progress(
            min(score / 100, 1.0)
        )

    with c2:
        st.metric(
            "AI Match",
            f"{score:.0f}%"
        )

    # Information
    d1, d2, d3 = st.columns(3)

    with d1:
        st.info(
            f"💼 **Career Role**\n\n"
            f"{career_role if career_role else 'Not specified'}"
        )

    with d2:
        st.warning(
            f"⏱️ **Duration**\n\n"
            f"{duration_value if duration_value else 'Not specified'}"
        )

    with d3:
        st.success(
            f"⭐ **Rating**\n\n"
            f"{rating_value if rating_value else 'Not specified'}"
        )

    if course_skills:

        st.write(
            f"🛠️ **Skills:** {course_skills}"
        )

    st.divider()


# ============================================================
# SUMMARY
# ============================================================

st.subheader("📊 Recommendation Summary")

summary_data = []

for index, (_, row) in enumerate(
    recommendations.iterrows()
):

    summary_data.append({

        "Rank":
            index + 1,

        "Course":
            row_value(
                row,
                COURSE_COL
            ),

        "AI Match":
            f"{float(row['AI_Score']):.0f}%",

        "Career Role":
            row_value(
                row,
                CAREER_COL
            ),

        "Duration":
            row_value(
                row,
                DURATION_COL
            ),

        "Rating":
            row_value(
                row,
                RATING_COL
            )
    })


summary_df = pd.DataFrame(
    summary_data
)

st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# CHART
# ============================================================

st.subheader("📈 AI Match Comparison")

chart_data = pd.DataFrame({

    "Course": [
        row_value(
            row,
            COURSE_COL
        )
        for _, row
        in recommendations.iterrows()
    ],

    "AI Match": [
        round(
            float(row["AI_Score"]),
            1
        )
        for _, row
        in recommendations.iterrows()
    ]
})

st.bar_chart(
    chart_data.set_index("Course")
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.success(
    "🎓 Course Recommendation System | "
    "🤖 AI-Powered Personalized Recommendations"
)
