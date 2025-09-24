# data.py
# Full 12-week curriculum for GenAI Forge: Data • Science • Engineering

CURRICULUM = [
    # -------------------- WEEK 1 --------------------
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
                {"label": "Open Library API", "url": "https://openlibrary.org/developers/api"}
            ],
        },
    },

    # -------------------- WEEK 2 --------------------
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

    # -------------------- WEEK 3 --------------------
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
                {"label": "OpenWeatherMap API", "url": "https://openweathermap.org/api"}
            ],
        },
    },

    # -------------------- WEEK 4 --------------------
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

    # -------------------- WEEK 5 --------------------
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
                {"label": "Kaggle — Titanic", "url": "https://www.kaggle.com/c/titanic"}
            ],
        },
    },

    # -------------------- WEEK 6 --------------------
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

    # -------------------- WEEK 7 --------------------
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

    # -------------------- WEEK 8 --------------------
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

    # -------------------- WEEK 9 --------------------
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
                {"label": "MNIST", "url": "http://yann.lecun.com/exdb/mnist/"}
            ],
        },
    },

    # -------------------- WEEK 10 --------------------
    {
        "phase": "AI & Advanced ML",
        "week": 10,
        "title": "NLP & Generative AI",
        "topics": [
            "Text preprocessing (tokenization, stemming, lemmatization)",
            "Classical NLP: TF-IDF, n-grams, sentiment classification",
            "Embeddings (Word2Vec, GloVe, BERT family)",
            "Generative AI: LLM basics, prompt engineering, few/zero-shot",
            "Retrieval-Augmented Generation (RAG): indexing, retrieval, grounding",
            "Fine-tuning & LoRA basics, safety/guardrails, evaluation of LLMs",
        ],
        "goals": [
            "Process and represent text for downstream tasks",
            "Train/evaluate classic and transformer-based NLP models",
            "Build a small GenAI app with prompts or RAG",
        ],
        "project": {
            "name": "Sentiment + GenAI Assistant",
            "tasks": [
                "Baseline: train a sentiment classifier on movie reviews",
                "GenAI: build a simple Q&A or summarizer using an LLM API or local model",
                "(Optional) Add RAG over a small document set and evaluate responses",
            ],
            "datasets": [
                {
                    "label": "IMDB Movie Reviews",
                    "url": "https://ai.stanford.edu/~amaas/data/sentiment/",
                },
                {"label": "(Optional) Any small docs for RAG", "url": ""},
            ],
        },
    },

    # -------------------- WEEK 11 --------------------
    {
        "phase": "AI & Advanced ML",
        "week": 11,
        "title": "Model Deployment, MLOps & AI Governance",
        "topics": [
            "Model serving (Flask, FastAPI)",
            "Containerization with Docker",
            "Model tracking & monitoring (MLflow)",
            "AI governance: fairness, bias, accountability, transparency, privacy",
            "Explainability (SHAP/LIME), model cards, data/documentation lineage",
            "Compliance & regulations (GDPR, EU AI Act — overview), auditability",
        ],
        "goals": [
            "Deploy ML models as APIs",
            "Use Docker for portability",
            "Track, monitor, and document models responsibly",
            "Evaluate bias & fairness; produce a model card and monitoring plan",
        ],
        "project": {
            "name": "Deploy Model API + AI Governance Report",
            "tasks": [
                "Take trained model from Week 7",
                "Expose API using FastAPI/Flask; containerize with Docker",
                "Integrate MLflow for experiment tracking & model registry",
                "Run SHAP on validation set; analyze subgroup performance metrics",
                "Create a Model Card (intended use, risks, metrics, monitoring)",
            ],
            "datasets": [
                {
                    "label": "Google AI — Model Cards",
                    "url": "https://modelcards.withgoogle.com/",
                },
                {
                    "label": "Microsoft Responsible AI Practices",
                    "url": "https://learn.microsoft.com/en-us/azure/architecture/guide/responsible-ai/",
                },
                {
                    "label": "EU AI Act — Community Summary",
                    "url": "https://artificialintelligenceact.eu/",
                },
            ],
        },
    },

    # -------------------- WEEK 12 --------------------
    {
        "phase": "AI & Advanced ML",
        "week": 12,
        "title": "Capstone Projects",
        "topics": [
            "Integrate DE + DS + AI into production-like workflow",
            "Reporting and stakeholder presentation",
        ],
        "goals": [
            "Deliver end-to-end project",
            "Document and present outcomes",
        ],
        "project": {
            "name": "Choose 1–3 Capstones",
            "tasks": [
                "Data Eng: End-to-end ETL with Airflow + warehouse + dashboard",
                "Data Science: Predictive analytics (loan default/churn)",
                "AI: NLP chatbot or image classifier deployed as API",
            ],
            "datasets": [
                {
                    "label": "NYC Taxi Data",
                    "url": "https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page",
                },
                {
                    "label": "Kaggle — Loan Default",
                    "url": "https://www.kaggle.com/datasets/wordsforthewise/lending-club",
                },
                {
                    "label": "Kaggle — Disaster Tweets",
                    "url": "https://www.kaggle.com/c/nlp-getting-started",
                },
            ],
        },
    },
]
