# curriculum.py
import streamlit as st
from io import StringIO

from helpers import to_dataframe, to_markdown
from data import CURRICULUM

st.set_page_config(
    page_title="GenAI Forge — Data • Science • Engineering",
    page_icon="📚",
    layout="wide",
)

# Sidebar
with st.sidebar:
    st.title("📚 GenAI Forge Planner")
    st.caption("Python is a prerequisite for this program.")
    org_name = st.text_input("Organization / Program Name", value="Data Academy")
    show_links = st.toggle("Show dataset links", value=True)

    weeks_all = [c["week"] for c in CURRICULUM]
    selected_weeks = st.multiselect("Filter by week(s)", options=weeks_all, default=weeks_all)

    phases = sorted(set(c["phase"] for c in CURRICULUM))
    phase_filter = st.multiselect("Filter by phase", options=phases, default=phases)

    st.divider()
    st.write("### Export")

    df_all = to_dataframe(CURRICULUM)
    csv_buf = StringIO()
    df_all.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Download CSV",
        csv_buf.getvalue(),
        file_name="genai_forge_curriculum.csv",
        mime="text/csv",
    )

    md = to_markdown(CURRICULUM, org_name)
    st.download_button(
        "⬇️ Download Markdown Syllabus",
        data=md,
        file_name="GenAI_Forge_12_week_curriculum.md",
        mime="text/markdown",
    )

# Main title
st.title("GenAI Forge: Data • Science • Engineering — 12-Week Program")
st.write(f"**Program:** GenAI Forge — Data • Science • Engineering  •  **Organization:** {org_name}")

# Summary
cols = st.columns(3)
with cols[0]:
    st.metric("Phases", len(set(c["phase"] for c in CURRICULUM)))
with cols[1]:
    st.metric("Weeks", len(CURRICULUM))
with cols[2]:
    st.metric("Projects", len([c["project"]["name"] for c in CURRICULUM]))

# Filter
filtered = [c for c in CURRICULUM if c["week"] in selected_weeks and c["phase"] in phase_filter]
filtered = sorted(filtered, key=lambda x: x["week"])

# Tabs
tab_overview, tab_weeks, tab_projects = st.tabs(["Overview", "Weekly Plan", "Projects"])

from helpers import render_week_card  # small UI helper

with tab_overview:
    st.subheader("Course Overview")
    st.write("This program assumes prior knowledge of Python and covers data engineering, data science, and AI.")
    st.write("Use the sidebar to filter weeks and export the plan.")
    st.dataframe(to_dataframe(filtered), use_container_width=True)

with tab_weeks:
    for c in filtered:
        render_week_card(c, show_links)

with tab_projects:
    st.subheader("All Projects")
    for c in filtered:
        st.markdown(f"### Week {c['week']}: {c['project']['name']}")
        st.markdown("**Tasks**")
        st.markdown("\n".join([f"- {t}" for t in c["project"]["tasks"]]))
        if show_links and c["project"].get("datasets"):
            st.markdown("**Datasets**")
            for ds in c["project"].get("datasets", []):
                label = ds.get("label", "Dataset")
                url = ds.get("url")
                st.markdown(f"- [{label}]({url})" if url else f"- {label}")
        st.divider()

# Footer (keep simple to avoid dedent issues)
st.caption("""
**Deploying**: Run locally with `streamlit run curriculum.py`.  
**Hosting**: Streamlit Community Cloud (share.streamlit.io) or any Python server.  
**Editing**: Modify `data.py` (CURRICULUM). Helpers live in `helpers.py`.
""")
