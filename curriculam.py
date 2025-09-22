import sys
sys.path.append("/Users/subhasht/Desktop/VSCode Programs/Curriculam")


import streamlit as st
import pandas as pd
from io import StringIO
from textwrap import dedent

st.set_page_config(
    page_title="12‑Week Curriculum Planner — Data Eng + DS + AI",
    page_icon="📚",
    layout="wide",
)

# ----------------------------
# Curriculum Data (Edit as needed)
# ----------------------------
CURRICULUM = [
    {
        "phase": "Data Engineering Foundations",
        "week": 1,
        "title": "SQL & NoSQL Basics",
        "topics": [
            "Relational DB design (schema, normalization)",
            "SQL queries (joins, aggregates, window functions)",
            "Intro to NoSQL (MongoDB basics: documents, collections)",
        ],
        "goals": [
            "Understand different database systems",
            "Write complex queries to extract insights",
            "Compare SQL vs. NoSQL use cases",
        ],
        "project": {
            "name": "Library Management Database",
            "tasks": [
                "Design schema for books, members, loans",
                "Run analytical queries (e.g., most borrowed books)",
                "Extend with MongoDB to store JSON-based book reviews",
            ],
            "datasets": [
                {
                    "label": "Open Library API",
                    "url": "https://openlibrary.org/developers/api",
                }
            ],
        },
    },
    {
        "phase": "Data Engineering Foundations",
        "week": 2,
        "title": "Data Pipelines (ETL/ELT)",
        "topics": [
            "ETL vs. ELT concepts",
            "Data ingestion: APIs, CSV/JSON, streaming sources",
            "Batch processing with Apache Spark",
        ],
        "goals": [
            "Automate data ingestion",
            "Transform raw data into clean formats",
            "Handle large-scale batch processing",
        ],
        "project": {
            "name": "Sales Data ETL Pipeline",
            "tasks": [
                "Extract CSV/JSON sales data",
                "Clean missing values, convert data types",
                "Load into PostgreSQL or MySQL",
            ],
            "datasets": [
                {
                    "label": "Kaggle — Retail Sales Forecasting",
                    "url": "https://www.kaggle.com/c/demand-forecasting-kernels-only",
                }
            ],
        },
    },
    {
        "phase": "Data Engineering Foundations",
        "week": 3,
        "title": "Workflow Orchestration",
        "topics": [
            "Apache Airflow/Prefect basics",
            "DAGs (Directed Acyclic Graphs)",
            "Task scheduling, error handling",
        ],
        "goals": [
            "Automate multi-step data workflows",
            "Schedule recurring jobs",
            "Monitor pipeline execution",
        ],
        "project": {
            "name": "Weather Data Pipeline with Airflow",
            "tasks": [
                "Extract daily weather data via API",
                "Transform (clean, enrich)",
                "Load into warehouse (BigQuery/PostgreSQL)",
            ],
            "datasets": [
                {"label": "OpenWeatherMap API", "url": "https://openweathermap.org/api"},
            ],
        },
    },
    {
        "phase": "Data Engineering Foundations",
        "week": 4,
        "title": "Data Warehousing & Best Practices",
        "topics": [
            "Data Lakes vs. Data Warehouses",
            "Schema design (Star, Snowflake)",
            "Data governance, testing, CI/CD",
        ],
        "goals": [
            "Build scalable warehouse schemas",
            "Ensure data quality",
            "Optimize for analytics",
        ],
        "project": {
            "name": "Retail Analytics Warehouse",
            "tasks": [
                "Build star schema with sales fact + dimension tables",
                "Load ETL data from Week 2",
                "Write BI queries for insights (sales by region/churn)",
            ],
            "datasets": [
                {
                    "label": "Kaggle — Superstore Sales",
                    "url": "https://www.kaggle.com/datasets/vivek468/superstore-dataset-final",
                }
            ],
        },
    },
    {
        "phase": "Data Science Foundations",
        "week": 5,
        "title": "Exploratory Data Analysis (EDA)",
        "topics": [
            "Data cleaning with Pandas",
            "Visualization with Matplotlib/Seaborn",
            "Statistical summaries, correlation analysis",
        ],
        "goals": [
            "Explore patterns in datasets",
            "Identify data quality issues",
            "Build visual insights",
        ],
        "project": {
            "name": "EDA on Titanic Dataset",
            "tasks": [
                "Clean and preprocess",
                "Visualize survival rate by gender, age, class",
                "Generate insights",
            ],
            "datasets": [
                {"label": "Kaggle — Titanic", "url": "https://www.kaggle.com/c/titanic"},
            ],
        },
    },
    {
        "phase": "Data Science Foundations",
        "week": 6,
        "title": "Statistics & Probability",
        "topics": [
            "Probability distributions, sampling",
            "Hypothesis testing, confidence intervals",
            "A/B testing methodology",
        ],
        "goals": [
            "Apply inferential statistics",
            "Run and interpret hypothesis tests",
            "Use stats to validate assumptions",
        ],
        "project": {
            "name": "Website A/B Test Simulation",
            "tasks": [
                "Simulate conversion rates for control vs. variant",
                "Perform t-test/chi-square test",
                "Decide which design performs better",
            ],
            "datasets": [
                {
                    "label": "Kaggle — Online A/B Testing",
                    "url": "https://www.kaggle.com/datasets/zhangluyuan/ab-testing",
                }
            ],
        },
    },
    {
        "phase": "Data Science Foundations",
        "week": 7,
        "title": "Supervised Machine Learning",
        "topics": [
            "ML workflow (train/test split, cross-validation)",
            "Regression, classification algorithms",
            "Evaluation metrics (AUC, Precision, Recall, F1)",
        ],
        "goals": [
            "Train supervised ML models",
            "Compare algorithm performance",
            "Interpret model accuracy & limitations",
        ],
        "project": {
            "name": "Credit Card Fraud Detection",
            "tasks": [
                "Train Logistic Regression, Random Forest, Gradient Boosting",
                "Evaluate with ROC & F1",
                "Discuss class imbalance techniques",
            ],
            "datasets": [
                {
                    "label": "Kaggle — Credit Card Fraud",
                    "url": "https://www.kaggle.com/mlg-ulb/creditcardfraud",
                }
            ],
        },
    },
    {
        "phase": "Data Science Foundations",
        "week": 8,
        "title": "Unsupervised Learning & Feature Engineering",
        "topics": [
            "Clustering (K-Means, DBSCAN)",
            "Dimensionality reduction (PCA, t-SNE)",
            "Feature scaling & engineering",
        ],
        "goals": [
            "Apply clustering for segmentation",
            "Reduce dataset dimensions",
            "Engineer meaningful features",
        ],
        "project": {
            "name": "Customer Segmentation",
            "tasks": [
                "Cluster retail data",
                "Reduce dimensions with PCA",
                "Profile clusters for marketing insights",
            ],
            "datasets": [
                {
                    "label": "Kaggle — Mall Customers",
                    "url": "https://www.kaggle.com/datasets/shwetabh123/mall-customers",
                }
            ],
        },
    },
    {
        "phase": "AI & Advanced ML",
        "week": 9,
        "title": "Deep Learning Basics",
        "topics": [
            "Neural networks (perceptrons, layers, activations)",
            "Training (optimizers, loss, dropout)",
            "TensorFlow & PyTorch basics",
        ],
        "goals": [
            "Build neural networks from scratch",
            "Train and evaluate models",
            "Understand backpropagation and overfitting",
        ],
        "project": {
            "name": "Handwritten Digit Recognition (MNIST)",
            "tasks": [
                "Build a simple NN in PyTorch/TensorFlow",
                "Train, validate, test accuracy",
                "Experiment with architecture/depth",
            ],
            "datasets": [
                {"label": "MNIST", "url": "http://yann.lecun.com/exdb/mnist/"},
            ],
        },
    },
    {
        "phase": "AI & Advanced ML",
        "week": 10,
        "title": "Natural Language Processing (NLP)",
        "topics": [
            "Text preprocessing (tokenization, stemming, lemmatization)",
            "Embeddings (Word2Vec, GloVe, BERT)",
            "Applications: Sentiment analysis, chatbot basics",
        ],
        "goals": [
            "Process and represent text",
            "Train NLP models",
            "Apply embeddings in real tasks",
        ],
        "project": {
            "name": "Sentiment Analysis on Movie Reviews",
            "tasks": [
                "Preprocess reviews",
                "Train classifier with embeddings",
                "Evaluate and interpret errors",
            ],
            "datasets": [
                {
                    "label": "IMDB Movie Reviews",
                    "url": "https://ai.stanford.edu/~amaas/data/sentiment/",
                }
            ],
        },
    },
    {
        "phase": "AI & Advanced ML",
        "week": 11,
        "title": "Model Deployment & MLOps",
        "topics": [
            "Model serving (Flask, FastAPI)",
            "Containerization with Docker",
            "Model tracking & monitoring (MLflow)",
        ],
        "goals": [
            "Deploy ML models as APIs",
            "Use Docker for portability",
            "Track and version models",
        ],
        "project": {
            "name": "Deploy Predictive Model as API",
            "tasks": [
                "Take trained model from Week 7",
                "Expose API using FastAPI/Flask",
                "Containerize with Docker",
            ],
            "datasets": [
                {"label": "Uses Week 7 trained model", "url": ""},
            ],
        },
    },
    {
        "phase": "AI & Advanced ML",
        "week": 12,
        "title": "Capstone Projects",
        "topics": [
            "Integrate DE + DS + AI into production-like workflow",
            "Reporting and stakeholder presentation",
        ],
        "goals": [
            "Deliver end‑to‑end project",
            "Document and present outcomes",
        ],
        "project": {
            "name": "Choose 1–3 Capstones",
            "tasks": [
                "Data Eng: End‑to‑end ETL with Airflow + warehouse + dashboard",
                "Data Science: Predictive analytics (loan default/churn)",
                "AI: NLP chatbot or image classifier deployed as API",
            ],
            "datasets": [
                {"label": "NYC Taxi Data", "url": "https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page"},
                {"label": "Kaggle — Loan Default", "url": "https://www.kaggle.com/datasets/wordsforthewise/lending-club"},
                {"label": "Kaggle — Disaster Tweets", "url": "https://www.kaggle.com/c/nlp-getting-started"},
            ],
        },
    },
]

# ----------------------------
# Helpers
# ----------------------------

def to_dataframe(curriculum):
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
    df = pd.DataFrame(rows).sort_values("Week")
    return df


def to_markdown(curriculum, org_name: str = "Your Organization") -> str:
    lines = [f"# 12‑Week Training — Data Engineering, Data Science & AI\n", f"**Organization:** {org_name}\n", "\n"]
    phases = {}
    for c in curriculum:
        phases.setdefault(c["phase"], []).append(c)
    for phase, weeks in phases.items():
        lines.append(f"## {phase}\n")
        for c in sorted(weeks, key=lambda x: x["week"]):
            lines.append(f"### Week {c['week']}: {c['title']}\n")
            lines.append("**Topics**\n")
            for t in c["topics"]:
                lines.append(f"- {t}")
            lines.append("**Learning Goals**\n")
            for g in c["goals"]:
                lines.append(f"- {g}")
            p = c["project"]
            lines.append(f"**Sample Project: {p['name']}**\n")
            lines.append("Tasks:")
            for task in p["tasks"]:
                lines.append(f"- {task}")
            if p.get("datasets"):
                lines.append("Datasets:")
                for d in p["datasets"]:
                    if d.get("url"):
                        lines.append(f"- [{d['label']}]({d['url']})")
                    else:
                        lines.append(f"- {d['label']}")
            lines.append("")
    return "\n".join(lines)


# ----------------------------
# Sidebar Controls
# ----------------------------
with st.sidebar:
    st.title("📚 Curriculum Planner")
    st.caption("Python is a prerequisite for this course.")
    org_name = st.text_input("Organization / Program Name", value="Data Academy")
    show_links = st.toggle("Show dataset links", value=True)
    selected_weeks = st.multiselect(
        "Filter by week(s)", options=[c["week"] for c in CURRICULUM], default=[c["week"] for c in CURRICULUM]
    )
    phases = sorted(set([c["phase"] for c in CURRICULUM]))
    phase_filter = st.multiselect("Filter by phase", options=phases, default=phases)

    st.divider()
    st.write("### Export")
    df = to_dataframe(CURRICULUM)
    buf_csv = StringIO()
    df.to_csv(buf_csv, index=False)
    st.download_button("⬇️ Download CSV", buf_csv.getvalue(), file_name="12_week_curriculum.csv", mime="text/csv")

    md = to_markdown(CURRICULUM, org_name)
    st.download_button(
        "⬇️ Download Markdown Syllabus",
        data=md,
        file_name="12_week_curriculum.md",
        mime="text/markdown",
    )

# ----------------------------
# Main Layout
# ----------------------------
st.title("12‑Week Training — Data Engineering, Data Science & AI")
st.write(f"**Organization:** {org_name}")

# Summary cards
cols = st.columns(3)
with cols[0]:
    st.metric("Phases", len(set([c["phase"] for c in CURRICULUM])))
with cols[1]:
    st.metric("Weeks", len(CURRICULUM))
with cols[2]:
    all_projects = [c["project"]["name"] for c in CURRICULUM]
    st.metric("Projects", len(all_projects))

# Filtered view
filtered = [c for c in CURRICULUM if c["week"] in selected_weeks and c["phase"] in phase_filter]
filtered = sorted(filtered, key=lambda x: x["week"])

# Tabs for Overview / Weeks / Projects
tab_overview, tab_weeks, tab_projects = st.tabs(["Overview", "Weekly Plan", "Projects"])

with tab_overview:
    st.subheader("Course Overview")
    st.write(
        "This program assumes prior knowledge of Python and covers data engineering, data science, and AI."
    )
    st.write("Use the sidebar to filter weeks and export the plan.")
    st.dataframe(to_dataframe(filtered), use_container_width=True)

with tab_weeks:
    for c in filtered:
        with st.expander(f"Week {c['week']}: {c['title']} — {c['phase']}"):
            left, right = st.columns([2, 1])
            with left:
                st.markdown("#### Topics")
                st.markdown("\n".join([f"- {t}" for t in c["topics"]]))
                st.markdown("#### Learning Goals")
                st.markdown("\n".join([f"- {g}" for g in c["goals"]]))
            with right:
                st.markdown("#### Week Summary")
                st.info(
                    f"**Project:** {c['project']['name']}\n\n" +
                    "\n".join([f"• {t}" for t in c["project"]["tasks"]])
                )
                if show_links and c["project"].get("datasets"):
                    st.markdown("**Datasets**")
                    for ds in c["project"]["datasets"]:
                        label = ds.get("label", "Dataset")
                        url = ds.get("url")
                        if url:
                            st.markdown(f"- [{label}]({url})")
                        else:
                            st.markdown(f"- {label}")

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
                if url:
                    st.markdown(f"- [{label}]({url})")
                else:
                    st.markdown(f"- {label}")
        st.divider()

# ----------------------------
# Deployment Notes (display only)
# ----------------------------
st.caption(
    dedent(
        """
        **Deploying**: Save this file as `app.py`, then run locally with `streamlit run app.py`.\n
        For easy deployment, you can use Streamlit Community Cloud (share.streamlit.io) or deploy on any server with Python.\n
        **Editing the curriculum**: Modify the `CURRICULUM` list at the top of this file to add/remove weeks, update projects, or change dataset links.
        """
    )
)
