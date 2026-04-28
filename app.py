import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
import os
from io import BytesIO
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

# Constants
DEFAULT_URL = "https://raw.githubusercontent.com/suraj-deshmukh/BBC-Dataset-News-Classification/master/dataset/dataset.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Sample fake news dataset (placeholder)
FAKE_NEWS_SAMPLE = pd.DataFrame({
    'news': [
        "Breaking: Scientists discover new planet with life signs",
        "Government announces free healthcare for all citizens",
        "Celebrity spotted in secret wedding",
        "Stock market crashes due to unknown reasons",
        "New vaccine cures all diseases instantly",
        "Aliens land in major city, world leaders meet",
        "Economy grows by 100% in one month",
        "Famous actor dies in mysterious accident",
        "Technology allows time travel",
        "World peace declared after global summit"
    ],
    'type': ['real', 'fake', 'real', 'fake', 'fake', 'fake', 'fake', 'real', 'fake', 'fake']
})

# CSS Themes
DARK_THEME = """
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

/* ===== Tables ===== */
[data-testid="stDataFrame"] {
    border: 1px solid #1f2937;
    border-radius: 12px;
    overflow: hidden;
}

/* ===== Metric ===== */
[data-testid="stMetric"] {
    background: rgba(17, 24, 39, 0.75);
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 10px;
}

/* ===== Tabs ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(30, 41, 59, 0.5);
    border-radius: 12px;
    padding: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #94a3b8;
    padding: 8px 16px;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: rgba(59, 130, 246, 0.2);
    color: #ffffff;
}
</style>
"""

LIGHT_THEME = """
<style>
/* ===== Global ===== */
.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #f8fafc 100%);
    color: #1e293b;
}

/* Container spacing */
.block-container {
    padding-top: 1.5rem;
}

/* Headings */
h1, h2, h3 {
    color: #0f172a;
    font-weight: 700;
    letter-spacing: 0.3px;
}

/* Divider */
hr {
    border: 1px solid #cbd5e1;
}

/* ===== Cards ===== */
.card {
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    padding: 20px 22px;
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.3);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.1);
    margin-bottom: 16px;
    transition: all 0.25s ease;
}
.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 35px rgba(59, 130, 246, 0.15);
}

.card-title {
    font-size: 0.9rem;
    color: #64748b;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.card-value {
    font-size: 1.5rem;
    color: #0f172a;
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
    background-color: #ffffff !important;
    color: #1e293b !important;
    border-radius: 12px !important;
    border: 1px solid #cbd5e1 !important;
}

/* ===== Select & Radio ===== */
.stSelectbox, .stRadio {
    background: transparent;
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    border-right: 1px solid #cbd5e1;
    padding-top: 20px;
}

/* Sidebar title */
section[data-testid="stSidebar"] h1 {
    font-size: 20px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 20px;
}

/* ===== Tables ===== */
[data-testid="stDataFrame"] {
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    overflow: hidden;
}

/* ===== Metric ===== */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    padding: 10px;
}

/* ===== Tabs ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(241, 245, 249, 0.8);
    border-radius: 12px;
    padding: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #64748b;
    padding: 8px 16px;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: rgba(59, 130, 246, 0.2);
    color: #0f172a;
}
</style>
"""

# Helper Functions
@st.cache_data
def load_default_dataset():
    try:
        df = pd.read_csv(DEFAULT_URL, encoding='ISO-8859-1')
        df.columns = df.columns.str.strip().str.replace('\ufeff', '').str.lower()
        return df
    except Exception as e:
        st.error(f"Failed to load default dataset: {e}")
        return pd.DataFrame()

@st.cache_data
def load_uploaded_dataset(uploaded_file):
    try:
        name = uploaded_file.name.lower()
        if name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file type")
        df.columns = df.columns.str.strip().str.lower()
        return df
    except Exception as e:
        st.error(f"Failed to load uploaded dataset: {e}")
        return pd.DataFrame()

def clean_dataset(df, remove_nulls=False, remove_duplicates=False, lowercase_text=False):
    if remove_nulls:
        df = df.dropna()
    if remove_duplicates:
        df = df.drop_duplicates()
    if lowercase_text and 'news' in df.columns:
        df['news'] = df['news'].str.lower()
    return df

def auto_detect_columns(df):
    text_cols = ['news', 'article', 'content', 'headline', 'news_text', 'text']
    label_cols = ['type', 'label', 'category', 'class', 'target']
    
    text_col = next((c for c in df.columns if c in text_cols), df.columns[0])
    label_col = next((c for c in df.columns if c in label_cols), df.columns[-1] if len(df.columns) > 1 else df.columns[0])
    
    return text_col, label_col

def save_model(model, vectorizer, name):
    if not JOBLIB_AVAILABLE:
        st.warning("Joblib not available. Models will not be saved.")
        return
    joblib.dump(model, os.path.join(MODEL_DIR, f'{name}_model.pkl'))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, f'{name}_vectorizer.pkl'))

def load_model(name):
    if not JOBLIB_AVAILABLE:
        return None, None
    try:
        model = joblib.load(os.path.join(MODEL_DIR, f'{name}_model.pkl'))
        vectorizer = joblib.load(os.path.join(MODEL_DIR, f'{name}_vectorizer.pkl'))
        return model, vectorizer
    except:
        return None, None

def train_and_save_models(df):
    if 'news' not in df.columns or 'type' not in df.columns:
        raise ValueError("Required columns 'news' and 'type' not found")
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=8000, ngram_range=(1,2))
    X = tfidf.fit_transform(df['news'])
    y = df['type']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': MultinomialNB(),
        'Linear SVM': LinearSVC(max_iter=1000)
    }
    
    results = {}
    best_acc = 0
    best_model = None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (name, model) in enumerate(models.items()):
        status_text.text(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        cv_scores = cross_val_score(model, X, y, cv=5)
        
        results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model': model,
            'predictions': y_pred,
            'y_test': y_test
        }
        
        if acc > best_acc:
            best_acc = acc
            best_model = name
        
        progress_bar.progress((i + 1) / len(models))
    
    status_text.text("Saving best model...")
    save_model(results[best_model]['model'], tfidf, 'best_classification')
    results['best_model'] = best_model
    
    progress_bar.empty()
    status_text.empty()
    
    return results, tfidf

def train_fake_news_model(df):
    fake_df = df.copy()
    labels = fake_df['type'].astype(str).str.lower()
    fake_df['binary'] = labels.apply(lambda x: 0 if x in ['real', 'true', 'genuine'] else 1)
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(fake_df['news'])
    y = fake_df['binary']
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    save_model(model, tfidf, 'fake_news')
    
    return tfidf, model

def predict_category(text, model, vectorizer):
    vec = vectorizer.transform([text])

    pred = model.predict(vec)[0]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]

    elif hasattr(model, "decision_function"):
        scores = model.decision_function(vec)

        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        probs = probs[0]

    else:
        probs = np.array([1.0])

    return pred, probs

def get_recommendations(user_article, df, tfidf_vectorizer, top_n=7):
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['news'])
    user_vec = tfidf_vectorizer.transform([user_article])
    cosine_similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
    
    top_indices = cosine_similarities.argsort()[-(top_n + 1):][::-1]
    return df.iloc[top_indices[1:]], cosine_similarities[top_indices[1:]]

def create_word_cloud(text_data):
    if not WORDCLOUD_AVAILABLE:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "WordCloud not available.\nInstall with: pip install wordcloud", 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig
    text = ' '.join(text_data)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# Page Functions
def dashboard():
    st.header("📊 Analytics Dashboard")
    
    if not st.session_state.get('data_loaded', False):
        st.warning("Please load a dataset first.")
        return
    
    df = st.session_state.df
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Articles", len(df))
    
    with col2:
        st.metric("Categories", len(df['type'].unique()))
    
    with col3:
        if st.session_state.get('models_trained', False):
            best_acc = max([data['accuracy'] for data in st.session_state.models.values() if isinstance(data, dict)])
            st.metric("Best Model Accuracy", f"{best_acc:.2%}")
        else:
            st.metric("Best Model Accuracy", "N/A")
    
    with col4:
        if st.session_state.get('fake_trained', False):
            st.metric("Fake News Model", "Trained")
        else:
            st.metric("Fake News Model", "Not Trained")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Category Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(data=df, y='type', order=df['type'].value_counts().index, ax=ax, palette='viridis')
        ax.set_title("Articles per Category")
        st.pyplot(fig)
    
    with col2:
        if st.session_state.get('models_trained', False):
            st.subheader("Model Performance Comparison")
            models_data = st.session_state.models
            model_names = [name for name in models_data.keys() if name != 'best_model']
            accuracies = [models_data[name]['accuracy'] for name in model_names]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.bar(model_names, accuracies, color=['#3b82f6', '#6366f1', '#8b5cf6', '#06b6d4'])
            ax.set_ylabel('Accuracy')
            ax.set_title('Model Accuracy Comparison')
            ax.set_ylim(0, 1)
            
            for bar, acc in zip(bars, accuracies):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc:.2%}', 
                       ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.subheader("Word Cloud")
            if 'news' in df.columns:
                fig = create_word_cloud(df['news'].dropna())
                st.pyplot(fig)

def dataset_page():
    st.header("📁 Dataset Management")
    
    tab1, tab2 = st.tabs(["Load Dataset", "Dataset Overview"])
    
    with tab1:
        st.subheader("Load Dataset")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Use Default Dataset", use_container_width=True):
                with st.spinner("Loading default dataset..."):
                    df = load_default_dataset()
                    if not df.empty:
                        st.session_state.df = df
                        st.session_state.data_loaded = True
                        st.session_state.models_trained = False
                        st.session_state.fake_trained = False
                        st.success("Default dataset loaded successfully!")
        
        with col2:
            uploaded_file = st.file_uploader("Upload Custom Dataset", type=["csv", "xlsx"])
            if uploaded_file is not None:
                with st.spinner("Processing uploaded file..."):
                    df = load_uploaded_dataset(uploaded_file)
                    if not df.empty:
                        st.write("Dataset Preview:")
                        st.dataframe(df.head())
                        
                        text_col, label_col = auto_detect_columns(df)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            text_col = st.selectbox("Text Column", df.columns, index=list(df.columns).index(text_col))
                        with col2:
                            label_col = st.selectbox("Label Column", df.columns, index=list(df.columns).index(label_col))
                        
                        if st.button("Confirm and Load Dataset", use_container_width=True):
                            df_clean = df.rename(columns={text_col: 'news', label_col: 'type'})
                            df_clean = df_clean[['news', 'type']].dropna()
                            
                            st.session_state.df = df_clean
                            st.session_state.data_loaded = True
                            st.session_state.models_trained = False
                            st.session_state.fake_trained = False
                            st.success("Custom dataset loaded successfully!")
    
    with tab2:
        if not st.session_state.get('data_loaded', False):
            st.warning("Please load a dataset first.")
            return
        
        df = st.session_state.df
        
        st.subheader("Dataset Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Duplicates", df.duplicated().sum())
        
        st.subheader("Data Cleaning Options")
        col1, col2, col3 = st.columns(3)
        
        remove_nulls = col1.checkbox("Remove Null Values")
        remove_dups = col2.checkbox("Remove Duplicates")
        lowercase = col3.checkbox("Lowercase Text")
        
        if st.button("Apply Cleaning", use_container_width=True):
            df_clean = clean_dataset(df, remove_nulls, remove_dups, lowercase)
            st.session_state.df = df_clean
            st.success("Dataset cleaned successfully!")
            st.rerun()
        
        st.subheader("Category Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, x='type', ax=ax, palette='viridis')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        st.subheader("Data Preview")
        st.dataframe(df.head(10))
        
        # Export cleaned dataset
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Cleaned Dataset as CSV",
            data=csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )

def training_page():
    st.header("🎯 Model Training")
    
    if not st.session_state.get('data_loaded', False):
        st.warning("Please load a dataset first.")
        return
    
    df = st.session_state.df
    
    if st.button("Train Models", use_container_width=True):
        with st.spinner("Training models... This may take a few minutes."):
            try:
                results, tfidf = train_and_save_models(df)
                st.session_state.models = results
                st.session_state.tfidf = tfidf
                st.session_state.models_trained = True
                st.success("Models trained successfully!")
            except Exception as e:
                st.error(f"Training failed: {e}")
    
    if st.session_state.get('models_trained', False):
        st.subheader("Training Results")
        
        results = st.session_state.models
        best_model = results['best_model']
        
        st.info(f"🏆 Best Model: {best_model} (Accuracy: {results[best_model]['accuracy']:.2%})")
        
        # Model comparison table
        comparison_data = []
        for name, data in results.items():
            if name != 'best_model':
                comparison_data.append({
                    'Model': name,
                    'Accuracy': f"{data['accuracy']:.2%}",
                    'Precision': f"{data['precision']:.2%}",
                    'Recall': f"{data['recall']:.2%}",
                    'F1-Score': f"{data['f1']:.2%}",
                    'CV Mean': f"{data['cv_mean']:.2%}",
                    'CV Std': f"{data['cv_std']:.3f}"
                })
        
        st.dataframe(pd.DataFrame(comparison_data))
        
        # Confusion Matrix for best model
        st.subheader(f"Confusion Matrix - {best_model}")
        cm = confusion_matrix(results[best_model]['y_test'], results[best_model]['predictions'])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {best_model}')
        st.pyplot(fig)
        
        # Download report
        report_df = pd.DataFrame(comparison_data)
        csv = report_df.to_csv(index=False)
        st.download_button(
            label="Download Training Report",
            data=csv,
            file_name="training_report.csv",
            mime="text/csv",
            use_container_width=True
        )

def prediction_page():
    st.header("🔮 News Category Prediction")
    
    if not st.session_state.get('models_trained', False):
        st.warning("Please train models first.")
        return
    
    # Load best model
    model, vectorizer = load_model('best_classification')
    if model is None:
        st.error("Failed to load trained model. Please retrain.")
        return
    
    user_input = st.text_area("Enter a news article", height=150, placeholder="Paste your news article here...")
    
    if st.button("Predict Category", use_container_width=True) and user_input.strip():
        with st.spinner("Analyzing article..."):
            prediction, probs = predict_category(user_input, model, vectorizer)
            
            # Add to history
            if 'prediction_history' not in st.session_state:
                st.session_state.prediction_history = []
            st.session_state.prediction_history.append({
                'text': user_input[:100] + '...' if len(user_input) > 100 else user_input,
                'prediction': prediction,
                'confidence': max(probs)
            })
        
        # Result card
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Prediction Result</div>
            <div class="card-value">{prediction}</div>
            <p><strong>Confidence:</strong> {max(probs)*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Probability chart
        st.subheader("Category Probabilities")
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = model.classes_ if hasattr(model, "classes_") else ["Prediction"]
        ax.bar(categories, probs, color='skyblue')
        ax.set_ylabel('Probability')
        ax.set_title('Prediction Probabilities')
        plt.xticks(rotation=45)
        for i, v in enumerate(probs):
            ax.text(i, v + 0.01, f'{v*100:.1f}%', ha='center')
        st.pyplot(fig)
        
        # Top keywords
        feature_names = vectorizer.get_feature_names_out()
        vec = vectorizer.transform([user_input])
        scores = vec.toarray()[0]
        top_idx = scores.argsort()[-5:][::-1]
        keywords = [feature_names[i] for i in top_idx if scores[i] > 0]
        
        st.subheader("Top Keywords")
        st.write(", ".join(keywords))
    
    # Recent predictions
    if 'prediction_history' in st.session_state and st.session_state.prediction_history:
        st.subheader("Recent Predictions")
        history_df = pd.DataFrame(st.session_state.prediction_history[-5:][::-1])
        st.dataframe(history_df)

def recommendations_page():
    st.header("🎯 Personalized Recommendations")
    
    if not st.session_state.get('data_loaded', False):
        st.warning("Please load a dataset first.")
        return
    
    df = st.session_state.df
    
    tab1, tab2 = st.tabs(["By Category & Article", "By Custom Query"])
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=8000, ngram_range=(1,2))
    
    with tab1:
        st.subheader("Get Recommendations by Category & Article")
        
        categories = sorted(df['type'].unique())
        category = st.selectbox("Choose a news category", categories)
        
        filtered_df = df[df['type'] == category]
        
        if filtered_df.empty:
            st.warning("No articles in this category.")
            return
        
        # Article selection
        article_options = [f"{i+1}. {row['news'][:80]}..." for i, row in filtered_df.iterrows()]
        selected_option = st.selectbox("Choose an article", article_options)
        
        selected_idx = article_options.index(selected_option)
        selected_article = filtered_df.iloc[selected_idx]['news']
        
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Selected Article</div>
            <p>{selected_article}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Get Recommendations", use_container_width=True):
            with st.spinner("Finding recommendations..."):
                recommendations, scores = get_recommendations(selected_article, df, tfidf, top_n=7)
            
            st.subheader("Recommended Articles")
            for i, (idx, row) in enumerate(recommendations.iterrows()):
                score_percent = scores[i] * 100
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">{i+1}. {score_percent:.1f}% match</div>
                    <p><strong>Category:</strong> {row['type']}</p>
                    <p>{row['news'][:300]}...</p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Get Recommendations by Custom Query")
        
        user_query = st.text_area("Enter your news interest or custom query", height=100)
        
        if st.button("Find Matches", use_container_width=True) and user_query.strip():
            with st.spinner("Searching for matches..."):
                tfidf_matrix = tfidf.fit_transform(df['news'])
                user_vec = tfidf.transform([user_query])
                cosine_similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
                
                top_indices = cosine_similarities.argsort()[::-1][:7]
                top_scores = cosine_similarities[top_indices]
            
            st.subheader("Top Matches")
            for i, idx in enumerate(top_indices):
                score_percent = top_scores[i] * 100
                row = df.iloc[idx]
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">{i+1}. {score_percent:.1f}% match</div>
                    <p><strong>Category:</strong> {row['type']}</p>
                    <p>{row['news'][:300]}...</p>
                </div>
                """, unsafe_allow_html=True)

def fake_news_page():
    st.header("🕵️ Fake News Detection")
    
    if not st.session_state.get('data_loaded', False):
        st.warning("Please load a dataset first.")
        return
    
    df = st.session_state.df
    
    # Train fake news model if not trained
    if not st.session_state.get('fake_trained', False):
        if st.button("Train Fake News Model", use_container_width=True):
            with st.spinner("Training fake news detection model..."):
                try:
                    tfidf, model = train_fake_news_model(df)
                    st.session_state.fake_tfidf = tfidf
                    st.session_state.fake_model = model
                    st.session_state.fake_trained = True
                    st.success("Fake news model trained successfully!")
                except Exception as e:
                    st.error(f"Training failed: {e}")
        return
    
    model = st.session_state.fake_model
    vectorizer = st.session_state.fake_tfidf
    
    tab1, tab2 = st.tabs(["Single Article", "Bulk Detection"])
    
    with tab1:
        st.subheader("Detect Fake News in Single Article")
        
        user_input = st.text_area("Enter a news article", height=150, placeholder="Paste your news article here...")
        
        if st.button("Detect", use_container_width=True) and user_input.strip():
            with st.spinner("Analyzing article..."):
                vec = vectorizer.transform([user_input])
                pred = model.predict(vec)[0]
                probs = model.predict_proba(vec)[0]
                
                result = "Real News" if pred == 0 else "Fake News"
                confidence = max(probs) * 100
                
                # Add to history
                if 'fake_history' not in st.session_state:
                    st.session_state.fake_history = []
                st.session_state.fake_history.append({
                    'text': user_input[:100] + '...' if len(user_input) > 100 else user_input,
                    'prediction': result,
                    'confidence': confidence
                })
            
            # Result
            color = "green" if pred == 0 else "red"
            st.markdown(f"""
            <div class="card" style="border-left: 5px solid {color};">
                <div class="card-title">Detection Result</div>
                <div class="card-value" style="color: {color};">{result}</div>
                <p><strong>Confidence:</strong> {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability chart
            st.subheader("Real vs Fake Probability")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(['Real', 'Fake'], probs, color=['green', 'red'])
            ax.set_ylabel('Probability')
            ax.set_title('Detection Probabilities')
            for i, v in enumerate(probs):
                ax.text(i, v + 0.01, f'{v*100:.1f}%', ha='center')
            st.pyplot(fig)
            
            # Keywords
            feature_names = vectorizer.get_feature_names_out()
            scores = vec.toarray()[0]
            top_idx = scores.argsort()[-5:][::-1]
            keywords = [feature_names[i] for i in top_idx if scores[i] > 0]
            
            st.subheader("Top Keywords")
            st.write(", ".join(keywords))
    
    with tab2:
        st.subheader("Bulk Fake News Detection")
        
        uploaded_csv = st.file_uploader("Upload CSV with news articles", type=["csv"])
        
        if uploaded_csv is not None:
            bulk_df = pd.read_csv(uploaded_csv)
            
            if 'news' not in bulk_df.columns:
                st.error("CSV must have a 'news' column.")
                return
            
            st.write("Preview:")
            st.dataframe(bulk_df.head())
            
            if st.button("Detect Fake News in Bulk", use_container_width=True):
                with st.spinner("Processing bulk detection..."):
                    vecs = vectorizer.transform(bulk_df['news'])
                    preds = model.predict(vecs)
                    probs = model.predict_proba(vecs)
                    
                    results_df = bulk_df.copy()
                    results_df['prediction'] = ['Real' if p == 0 else 'Fake' for p in preds]
                    results_df['confidence'] = [max(prob) * 100 for prob in probs]
                    results_df['real_prob'] = probs[:, 0] * 100
                    results_df['fake_prob'] = probs[:, 1] * 100
                
                st.subheader("Detection Results")
                st.dataframe(results_df)
                
                # Summary
                real_count = (preds == 0).sum()
                fake_count = (preds == 1).sum()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Real News", real_count)
                with col2:
                    st.metric("Fake News", fake_count)
                
                # Download results
                csv_results = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Detection Results",
                    data=csv_results,
                    file_name="fake_news_detection_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    # Recent detections
    if 'fake_history' in st.session_state and st.session_state.fake_history:
        st.subheader("Recent Detections")
        history_df = pd.DataFrame(st.session_state.fake_history[-5:][::-1])
        st.dataframe(history_df)

# Main App
def main():
    st.set_page_config(
        page_title="AI News Analytics Platform",
        page_icon="📰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Theme toggle
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'
    
    # Sidebar
    with st.sidebar:
        st.title("📰 AI News Analytics")
        
        theme_toggle = st.toggle("Light Mode", value=(st.session_state.theme == 'light'))
        if theme_toggle != (st.session_state.theme == 'light'):
            st.session_state.theme = 'light' if theme_toggle else 'dark'
            st.rerun()
        
        st.markdown("---")
        
        # Search functionality
        if st.session_state.get('data_loaded', False):
            search_query = st.text_input("🔍 Search Articles", placeholder="Enter keywords...")
            if search_query:
                df = st.session_state.df
                mask = df['news'].str.contains(search_query, case=False, na=False)
                results = df[mask]
                if not results.empty:
                    st.write(f"Found {len(results)} articles:")
                    for _, row in results.head(5).iterrows():
                        st.write(f"- {row['news'][:100]}...")
                else:
                    st.write("No articles found.")
    
    # Apply theme
    if st.session_state.theme == 'dark':
        st.markdown(DARK_THEME, unsafe_allow_html=True)
    else:
        st.markdown(LIGHT_THEME, unsafe_allow_html=True)
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'fake_trained' not in st.session_state:
        st.session_state.fake_trained = False
    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame()
    if 'models' not in st.session_state:
        st.session_state.models = {}
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'fake_history' not in st.session_state:
        st.session_state.fake_history = []
    
    # Main content with tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard", "📁 Dataset", "🎯 Training", "🔮 Prediction", 
        "🎯 Recommendations", "🕵️ Fake News"
    ])
    
    with tab1:
        dashboard()
    
    with tab2:
        dataset_page()
    
    with tab3:
        training_page()
    
    with tab4:
        prediction_page()
    
    with tab5:
        recommendations_page()
    
    with tab6:
        fake_news_page()

if __name__ == "__main__":
    main()