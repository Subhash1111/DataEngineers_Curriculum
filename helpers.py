# helpers.py
import pandas as pd
import streamlit as st
from textwrap import dedent


def to_dataframe(curriculum: list) -> pd.DataFrame:
    """Convert curriculum list of dicts into a clean Pandas DataFrame."""
    rows = []
    for c in curriculum:
        rows.append(
            {
                "Phase": c["phase"],
                "Week": c["week"],
                "Title": c["title"],
                "Topics": " • ".join(c["topics"]),
                "Goals": " • ".join(c["goals"]),
                "Project": c["project"]["name"],
                "Project Tasks": " • ".join(c["project"]["tasks"]),
                "Datasets": ", ".join([d["label"] for d in c["project"].get("datasets", [])]),
            }
        )
    return pd.DataFrame(rows).sort_values("Week")


def to_markdown(curriculum: list, org_name: str = "Your Organization") -> str:
    """Generate a Markdown syllabus from the curriculum."""
    header = dedent(f"""\
    # GenAI Forge: Data • Science • Engineering — 12-Week Program
    **Organization:** {org_name}

    """)
    lines = [header]
    phases = {}
    for c in curriculum:
        phases.setdefault(c["phase"], []).append(c)

    for phase, weeks in phases.items():
        lines.append(f"## {phase}\n")
        for c in sorted(weeks, key=lambda x: x["week"]):
            lines.append(f"### Week {c['week']}: {c['title']}\n")
            lines.append("**Topics**")
            lines.extend(f"- {t}" for t in c["topics"])
            lines.append("**Learning Goals**")
            lines.extend(f"- {g}" for g in c["goals"])
            p = c["project"]
            lines.append(f"**Sample Project: {p['name']}**")
            lines.append("Tasks:")
            lines.extend(f"- {task}" for task in p["tasks"])
            if p.get("datasets"):
                lines.append("Datasets:")
                for d in p["datasets"]:
                    label = d.get("label", "Dataset")
                    url = d.get("url")
                    lines.append(f"- [{label}]({url})" if url else f"- {label}")
            lines.append("")  # blank line
    return "\n".join(lines)


def render_week_card(c: dict, show_links: bool) -> None:
    """Render one week's curriculum details in Streamlit."""
    with st.expander(f"Week {c['week']}: {c['title']} — {c['phase']}"):
        left, right = st.columns([2, 1])

        # Left column: topics & goals
        with left:
            st.markdown("#### Topics")
            st.markdown("\n".join(f"- {t}" for t in c["topics"]))
            st.markdown("#### Learning Goals")
            st.markdown("\n".join(f"- {g}" for g in c["goals"]))

        # Right column: project summary & datasets
        with right:
            st.markdown("#### Week Summary")
            tasks_md = "\n".join([f"- {t}" for t in c["project"]["tasks"]])
            st.info(
                f"**Project:** {c['project']['name']}\n\n{tasks_md}"
            )

            if show_links and c["project"].get("datasets"):
                st.markdown("**Datasets**")
                for ds in c["project"].get("datasets", []):
                    label = ds.get("label", "Dataset")
                    url = ds.get("url")
                    st.markdown(f"- [{label}]({url})" if url else f"- {label}")
