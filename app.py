import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(page_title="Personalized News Finder", page_icon="", layout="wide")

st.markdown("""
<style>
/* ===== Global ===== */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    color: #f1f5f9;
}

/* Container spacing */
.block-container {
    padding-top: 1.5rem;
}

/* Headings */
h1, h2, h3 {
    color: #ffffff;
    font-weight: 700;
    letter-spacing: 0.3px;
}

/* Divider */
hr {
    border: 1px solid #1f2937;
}

/* ===== Cards ===== */
.card {
    background: rgba(30, 41, 59, 0.85);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    padding: 20px 22px;
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.15);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
    margin-bottom: 16px;
    transition: all 0.25s ease;
}
.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 35px rgba(59, 130, 246, 0.25);
}

.card-title {
    font-size: 0.9rem;
    color: #94a3b8;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.card-value {
    font-size: 1.5rem;
    color: #ffffff;
    font-weight: 700;
}

/* ===== Buttons ===== */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    color: #ffffff;
    border-radius: 12px;
    padding: 10px 18px;
    border: none;
    font-weight: 600;
    transition: all 0.25s ease;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(59, 130, 246, 0.4);
}

/* ===== Inputs ===== */
textarea, .stTextInput > div > div > input {
    background-color: #020617 !important;
    color: #f1f5f9 !important;
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
}

/* ===== Select & Radio ===== */
.stSelectbox, .stRadio {
    background: transparent;
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #0f172a 100%);
    border-right: 1px solid #334155;
    padding-top: 20px;
}

/* Sidebar title */
section[data-testid="stSidebar"] h1 {
    font-size: 20px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 20px;
}

/* Radio buttons (navigation items) */
section[data-testid="stSidebar"] .stRadio > div {
    gap: 10px;
}

section[data-testid="stSidebar"] label {
    background: rgba(30, 41, 59, 0.6);
    padding: 10px 12px;
    border-radius: 10px;
    border: 1px solid transparent;
    transition: all 0.2s ease;
    display: block;
}

/* Hover effect */
section[data-testid="stSidebar"] label:hover {
    border: 1px solid #3b82f6;
    background: rgba(59, 130, 246, 0.15);
}

/* Selected option */
section[data-testid="stSidebar"] input[type="radio"]:checked + div {
    border: 1px solid #6366f1 !important;
    background: rgba(99, 102, 241, 0.2) !important;
}

/* ===== Tables ===== */
[data-testid="stDataFrame"] {
    border: 1px solid #1f2937;
    border-radius: 12px;
    overflow: hidden;
}

/* ===== Metric (if used) ===== */
[data-testid="stMetric"] {
    background: rgba(17, 24, 39, 0.75);
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
for key in ['data_loaded', 'models_trained', 'df', 'models']:
    if key not in st.session_state:
        st.session_state[key] = False if 'loaded' in key or 'trained' in key else None if key == 'df' else {}

# Load dataset
def load_dataset():
    url = "https://raw.githubusercontent.com/suraj-deshmukh/BBC-Dataset-News-Classification/master/dataset/dataset.csv"
    try:
        df = pd.read_csv(url, encoding='ISO-8859-1')
        df.columns = df.columns.str.strip().str.replace('\ufeff', '').str.lower()  
        st.write("Columns in dataset:", df.columns.tolist())
        st.session_state.df = df
        st.session_state.data_loaded = True
        st.success("Dataset loaded successfully")
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")


# Train models
def train_models():
    df = st.session_state.df

    if 'news' not in df.columns or 'type' not in df.columns:
        st.error("Required columns 'news' and 'type' not found")
        st.write("Columns in dataset:", df.columns.tolist())
        return

    tfidf = TfidfVectorizer(stop_words='english', max_features=8000, ngram_range=(1,2))
    X = tfidf.fit_transform(df['news'])   
    y = df['type']                        

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)
    report_lr = classification_report(y_test, y_pred_lr, output_dict=True)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    report_knn = classification_report(y_test, y_pred_knn, output_dict=True)

    st.session_state.models_trained = True
    st.session_state.models = {
        'Logistic Regression': report_lr,
        'K-Nearest Neighbors': report_knn
    }
    st.success("Models trained successfully")

# Dataset overview
def show_dataset_overview():
    if not st.session_state.data_loaded:
        st.warning("Load the dataset first.")
        return
    df = st.session_state.df
    st.markdown("## Dataset Overview")
    st.markdown("Explore the dataset used for training the model.")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(df.head())
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Category Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, y='type', order=df['type'].value_counts().index, ax=ax)
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# Model performance
def show_model_performance():
    if not st.session_state.models_trained:
        st.warning("Train the models first.")
        return
    st.subheader("Model Performance")
    for name, report in st.session_state.models.items():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"#### {name}")
        st.dataframe(pd.DataFrame(report).transpose())
        st.markdown('</div>', unsafe_allow_html=True)
        st.subheader("Model Comparison: Precision, Recall & F1-Score")

    logreg_df = pd.DataFrame(st.session_state.models['Logistic Regression']).transpose()
    knn_df = pd.DataFrame(st.session_state.models['K-Nearest Neighbors']).transpose()

    label_rows = logreg_df.index.difference(['accuracy', 'macro avg', 'weighted avg'])
    metrics = ['precision', 'recall', 'f1-score']

    for metric in metrics:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(label_rows, logreg_df.loc[label_rows, metric], alpha=0.6, label='Logistic Regression')
        ax.bar(label_rows, knn_df.loc[label_rows, metric], alpha=0.6, label='KNN', bottom=logreg_df.loc[label_rows, metric] * 0)
        ax.set_title(f'{metric.capitalize()} Comparison by Class')
        ax.set_ylabel(metric.capitalize())
        ax.set_xlabel("Class")
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
def get_recommendations(user_article, df, tfidf_vectorizer, top_n=7):
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['news'])
    user_vec = tfidf_vectorizer.transform([user_article])
    cosine_similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()

    top_indices = cosine_similarities.argsort()[-(top_n + 1):][::-1]  
    return df.iloc[top_indices[1:]]  


def show_recommendations():
    df = st.session_state.df

    if 'news' not in df.columns or 'type' not in df.columns:
        st.error("Required columns ('news', 'type') not found")
        return

    st.subheader("Choose Recommendation Mode")
    mode = st.radio("How would you like to get recommendations?", ["By Category & Article", "By Custom Query"])

    tfidf = TfidfVectorizer(stop_words='english', max_features=8000, ngram_range=(1,2))

    if mode == "By Category & Article":
        all_categories = sorted(df['type'].unique())
        category = st.selectbox("Choose a news category:", all_categories)

        filtered_df = df[df['type'] == category]

        if filtered_df.empty:
            st.warning("No articles available in this category")
            return

        article_choices = {
            f"{i+1}. {row['news'][:80]}...": idx
            for i, (idx, row) in enumerate(filtered_df.iterrows())
        }

        selected_label = st.selectbox("Choose an article:", list(article_choices.keys()))
        selected_idx = article_choices[selected_label]
        selected_article = filtered_df.loc[selected_idx, 'news']

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Selected Article")
        st.write(selected_article)
        st.markdown('</div>', unsafe_allow_html=True)

        recommendations = get_recommendations(selected_article, df, tfidf, top_n=7)

        st.subheader("Recommended Articles")
        for i, rec in enumerate(recommendations['news']):
            st.markdown(f""" <div class="card"> <b>{i+1}.</b> {rec[:250]}...</div> """, unsafe_allow_html=True)

    elif mode == "By Custom Query":
        user_input = st.text_area("Enter your news interest or custom query")

        if user_input.strip():
            tfidf_matrix = tfidf.fit_transform(df['news'])
            user_vec = tfidf.transform([user_input])
            cosine_similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()

            top_indices = cosine_similarities.argsort()[::-1][:7]
            top_scores = cosine_similarities[top_indices]

            st.subheader("Top Matches Based on Your Input")
            for i, idx in enumerate(top_indices):
                score_percent = round(top_scores[i] * 100, 2)
                st.markdown(f"**{i+1}. ({score_percent}% match)** {df.iloc[idx]['news'][:250]}...")
        else:
            st.info("Enter a query above to get personalized article suggestions")


# Home
def show_home():
    st.title("Personalized News Finder")
    st.markdown("Machine Learning based News Classification and Recommendation System")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Step 1")
        st.markdown("Load dataset")
        if st.button("Load Dataset"):
            load_dataset()

    with col2:
        st.markdown("### Step 2")
        st.markdown("Train models")
        if st.button("Train Models"):
            if st.session_state.data_loaded:
                train_models()
            else:
                st.warning("Load dataset first")

    with col3:
        st.markdown("### Step 3")
        st.markdown("View performance")
        if st.button("Show Performance"):
            if st.session_state.models_trained:
                show_model_performance()
            else:
                st.warning("Train models first")

    st.markdown("---")

    st.markdown("### System Features")
    st.markdown("""
    - News classification using TF-IDF and Logistic Regression  
    - Model performance comparison  
    - Content-based recommendation system  
    """)

def show_prediction():
    if not st.session_state.models_trained:
        st.warning("Train the model first.")
        return

    st.markdown("## News Category Prediction")

    user_input = st.text_area(
        "Enter a news article",
        placeholder="Enter a news article here..."
    )

    if user_input.strip():
        df = st.session_state.df

        # TF-IDF with n-grams
        tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=8000,
            ngram_range=(1,2)
        )

        X = tfidf.fit_transform(df['news'])
        y = df['type']

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        user_vec = tfidf.transform([user_input])
        prediction = model.predict(user_vec)[0]
        probs = model.predict_proba(user_vec)[0]

        # 🔥 Result Card
        st.markdown(f"""
        <div class="card">
            <b>Predicted Category:</b> {prediction}<br><br>
            <b>Confidence:</b> {round(max(probs)*100,2)}%
        </div>
        """, unsafe_allow_html=True)

        # Probabilities
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Category Probabilities")
        for label, prob in zip(model.classes_, probs):
            st.write(f"{label}: {round(prob*100,2)}%")
        st.markdown('</div>', unsafe_allow_html=True)

        # Keywords
        feature_names = tfidf.get_feature_names_out()
        scores = user_vec.toarray()[0]
        top_idx = scores.argsort()[-5:][::-1]
        keywords = [feature_names[i] for i in top_idx]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Keywords")
        st.write(", ".join(keywords))
        st.markdown('</div>', unsafe_allow_html=True)


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Home", "Dataset Overview", "Model Performance", "News Prediction", "Personalized Recommendations"
    ])
    if page == "Home":
        show_home()
    elif page == "Dataset Overview":
        show_dataset_overview()
    elif page == "Model Performance":
        show_model_performance()
    elif page == "News Prediction":
      show_prediction()
    elif page == "Personalized Recommendations":
        show_recommendations()

if __name__ == "__main__":
    main()
