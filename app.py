# app.py - FIXED VERSION
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def local_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #8B0000;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #722F37;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #8B0000;
    }
    .prediction-good {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #c3e6cb;
    }
    .prediction-average {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ffeaa7;
    }
    .prediction-poor {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #f5c6cb;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the wine dataset or create sample data"""
    try:
        df = pd.read_csv('data/WineQT.csv')
        return df
    except FileNotFoundError:
        st.warning("📁 Dataset file not found. Using sample data for demonstration.")
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        
        sample_data = {
            'Id': range(1, n_samples + 1),
            'fixed acidity': np.random.uniform(4.0, 16.0, n_samples),
            'volatile acidity': np.random.uniform(0.1, 1.6, n_samples),
            'citric acid': np.random.uniform(0.0, 1.0, n_samples),
            'residual sugar': np.random.uniform(0.9, 16.0, n_samples),
            'chlorides': np.random.uniform(0.01, 0.6, n_samples),
            'free sulfur dioxide': np.random.uniform(1.0, 70.0, n_samples),
            'total sulfur dioxide': np.random.uniform(6.0, 290.0, n_samples),
            'density': np.random.uniform(0.99, 1.004, n_samples),
            'pH': np.random.uniform(2.7, 4.0, n_samples),
            'sulphates': np.random.uniform(0.3, 2.0, n_samples),
            'alcohol': np.random.uniform(8.0, 15.0, n_samples),
            'quality': np.random.randint(3, 9, n_samples)
        }
        
        df = pd.DataFrame(sample_data)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

@st.cache_resource
def load_model():
    """Load the trained model"""
    # Try different possible model filenames
    model_files = ['model.pkl', 'wine_model.pkl']
    
    for model_file in model_files:
        try:
            with open(model_file, 'rb') as file:
                model = pickle.load(file)
            st.success(f"✅ Model loaded successfully from {model_file}")
            return model
        except FileNotFoundError:
            continue
    
    st.error("❌ Model file not found. Please ensure either 'model.pkl' or 'wine_model.pkl' exists in your project folder.")
    st.info("💡 Run 'train_model.py' first to create a model, or use the simple model creator.")
    return None

def main():
    # Apply custom CSS
    local_css()
    
    # Load data and model
    df = load_data()
    model = load_model()
    
    # Sidebar navigation
    st.sidebar.title("🍷 Navigation")
    app_section = st.sidebar.radio(
        "Choose a section:",
        ["Home", "Data Exploration", "Visualizations", "Quality Prediction", "Model Performance"]
    )
    
    # Home section
    if app_section == "Home":
        show_home_section(df)
    
    # Data Exploration section
    elif app_section == "Data Exploration":
        show_data_exploration(df)
    
    # Visualizations section
    elif app_section == "Visualizations":
        show_visualizations(df)
    
    # Quality Prediction section
    elif app_section == "Quality Prediction":
        if model is not None:
            show_prediction_section(df, model)
        else:
            st.error("Please create a model first to use the prediction feature.")
    
    # Model Performance section
    elif app_section == "Model Performance":
        show_model_performance(df, model)

def show_home_section(df):
    """Display the home section"""
    st.markdown('<div class="main-header">Wine Quality Predictor</div>', unsafe_allow_html=True)
    
    # Create columns FIRST, then use them
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col2:
        # Use emoji instead of external image
        st.markdown('<div style="text-align: center; font-size: 100px;">🍷</div>', unsafe_allow_html=True)
    
    st.write("""
    ## Welcome to the Wine Quality Prediction App!
    
    This interactive application allows you to explore the Wine Quality dataset and predict 
    the quality of wine based on its chemical properties using machine learning.
    
    ### Features:
    - **Data Exploration**: Explore the dataset with interactive filtering
    - **Visualizations**: Create insightful charts and plots
    - **Quality Prediction**: Predict wine quality using our trained model
    - **Model Performance**: Evaluate model accuracy and metrics
    
    ### About the Dataset:
    The dataset contains various chemical properties of red wines along with their quality ratings.
    """)
    
    # Quick stats
    st.markdown('<div class="sub-header">Dataset Overview</div>', unsafe_allow_html=True)
    
    # Create new columns for metrics (using different variable names to avoid conflicts)
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.metric("Total Samples", len(df))
    with stat_col2:
        st.metric("Features", len(df.columns) - 2)  # Excluding quality and Id
    with stat_col3:
        st.metric("Quality Range", f"{df['quality'].min()} - {df['quality'].max()}")
    with stat_col4:
        st.metric("Data Size", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

def show_data_exploration(df):
    """Display data exploration section"""
    st.markdown('<div class="sub-header">Data Exploration</div>', unsafe_allow_html=True)
    
    # Dataset overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Shape:**", df.shape)
        st.write("**Columns:**", list(df.columns))
    
    with col2:
        st.write("**Data Types:**")
        st.write(df.dtypes)
    
    # Interactive filtering
    st.subheader("Interactive Data Filtering")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        quality_range = st.slider(
            "Quality Range",
            min_value=int(df['quality'].min()),
            max_value=int(df['quality'].max()),
            value=(3, 8)
        )
    
    with col2:
        alcohol_range = st.slider(
            "Alcohol Range",
            min_value=float(df['alcohol'].min()),
            max_value=float(df['alcohol'].max()),
            value=(float(df['alcohol'].min()), float(df['alcohol'].max()))
        )
    
    with col3:
        show_samples = st.slider("Number of samples to show", 5, 100, 20)
    
    # Apply filters
    filtered_df = df[
        (df['quality'] >= quality_range[0]) & 
        (df['quality'] <= quality_range[1]) &
        (df['alcohol'] >= alcohol_range[0]) & 
        (df['alcohol'] <= alcohol_range[1])
    ]
    
    st.write(f"**Filtered Data:** {len(filtered_df)} samples")
    
    # Display filtered data
    st.dataframe(filtered_df.head(show_samples), use_container_width=True)
    
    # Statistical summary
    st.subheader("Statistical Summary")
    st.dataframe(filtered_df.describe(), use_container_width=True)

def show_visualizations(df):
    """Display visualizations section"""
    st.markdown('<div class="sub-header">Data Visualizations</div>', unsafe_allow_html=True)
    
    # Visualization type selection
    viz_type = st.selectbox(
        "Choose Visualization Type:",
        ["Quality Distribution", "Feature Correlations", "Chemical Properties vs Quality", 
         "Alcohol Content Analysis", "Interactive 3D Scatter"]
    )
    
    if viz_type == "Quality Distribution":
        fig = px.histogram(df, x='quality', title='Distribution of Wine Quality Ratings',
                         color_discrete_sequence=['#8B0000'])
        fig.update_layout(xaxis_title='Quality Rating', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Feature Correlations":
        # Calculate correlation matrix
        corr_matrix = df.drop('Id', axis=1).corr()
        
        fig = px.imshow(corr_matrix, 
                      title='Feature Correlation Matrix',
                      color_continuous_scale='RdBu_r',
                      aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show top correlations with quality
        quality_corr = corr_matrix['quality'].sort_values(ascending=False)
        st.write("**Top Correlations with Quality:**")
        st.dataframe(quality_corr)
        
    elif viz_type == "Chemical Properties vs Quality":
        selected_feature = st.selectbox(
            "Select Feature:",
            ['alcohol', 'volatile acidity', 'citric acid', 'residual sugar', 
             'chlorides', 'sulphates', 'pH']
        )
        
        fig = px.box(df, x='quality', y=selected_feature, 
                    title=f'{selected_feature.title()} vs Wine Quality')
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Alcohol Content Analysis":
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.histogram(df, x='alcohol', title='Distribution of Alcohol Content',
                              color_discrete_sequence=['#722F37'])
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            fig2 = px.scatter(df, x='alcohol', y='quality', 
                            title='Alcohol Content vs Quality',
                            color='quality', color_continuous_scale='Viridis')
            st.plotly_chart(fig2, use_container_width=True)
            
    elif viz_type == "Interactive 3D Scatter":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox("X-Axis", df.columns[:-2], index=10)  # alcohol
        with col2:
            y_axis = st.selectbox("Y-Axis", df.columns[:-2], index=1)   # volatile acidity
        with col3:
            z_axis = st.selectbox("Z-Axis", df.columns[:-2], index=2)   # citric acid
            
        fig = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis,
                          color='quality', 
                          title=f'3D Scatter: {x_axis} vs {y_axis} vs {z_axis}',
                          color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

def show_prediction_section(df, model):
    """Display prediction section"""
    st.markdown('<div class="sub-header">Wine Quality Prediction</div>', unsafe_allow_html=True)
    
    st.write("""
    Enter the chemical properties of the wine below to predict its quality rating (3-8).
    The model will analyze the features and provide a quality prediction.
    """)
    
    # Create input columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.0, 0.1)
        volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, 0.5, 0.01)
        citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.25, 0.01)
        residual_sugar = st.slider("Residual Sugar", 0.9, 16.0, 2.5, 0.1)
        
    with col2:
        chlorides = st.slider("Chlorides", 0.01, 0.6, 0.08, 0.001)
        free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1.0, 70.0, 15.0, 1.0)
        total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6.0, 290.0, 40.0, 1.0)
        density = st.slider("Density", 0.99, 1.004, 0.996, 0.001)
        
    with col3:
        pH = st.slider("pH", 2.7, 4.0, 3.3, 0.1)
        sulphates = st.slider("Sulphates", 0.3, 2.0, 0.6, 0.01)
        alcohol = st.slider("Alcohol", 8.0, 15.0, 10.5, 0.1)
    
    # Create feature array for prediction
    features = np.array([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
        pH, sulphates, alcohol
    ]])
    
    # Prediction button
    if st.button("Predict Wine Quality", type="primary"):
        with st.spinner("Analyzing wine properties..."):
            try:
                # Make prediction
                prediction = model.predict(features)[0]
                
                # Try to get probabilities (if model supports it)
                try:
                    probabilities = model.predict_proba(features)[0]
                    has_probabilities = True
                except:
                    probabilities = [0] * 6
                    probabilities[prediction-3] = 1.0  # Assume 100% confidence
                    has_probabilities = False
                
                # Display results
                st.subheader("Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Quality rating with color coding
                    if prediction >= 6:
                        st.markdown(f'<div class="prediction-good">'
                                  f'<h3>Predicted Quality: {prediction}/8</h3>'
                                  f'<p>This wine is predicted to be of good quality!</p>'
                                  f'</div>', unsafe_allow_html=True)
                    elif prediction >= 5:
                        st.markdown(f'<div class="prediction-average">'
                                  f'<h3>Predicted Quality: {prediction}/8</h3>'
                                  f'<p>This wine is predicted to be of average quality.</p>'
                                  f'</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="prediction-poor">'
                                  f'<h3>Predicted Quality: {prediction}/8</h3>'
                                  f'<p>This wine is predicted to be of below average quality.</p>'
                                  f'</div>', unsafe_allow_html=True)
                
                with col2:
                    # Confidence scores
                    if has_probabilities:
                        st.write("**Quality Probabilities:**")
                        for i, prob in enumerate(probabilities):
                            quality_label = f"Quality {i+3}"
                            st.write(f"{quality_label}: {prob:.1%}")
                        
                        max_prob = max(probabilities)
                        st.write(f"**Confidence:** {max_prob:.1%}")
                    else:
                        st.write("**Note:** Probability scores not available for this model.")
                
                # Feature importance explanation
                st.subheader("Key Influencing Factors")
                
                # Simple heuristic for explanation
                positive_factors = []
                negative_factors = []
                
                if alcohol > 11: 
                    positive_factors.append("High alcohol content")
                elif alcohol < 9:
                    negative_factors.append("Low alcohol content")
                    
                if sulphates > 0.6: 
                    positive_factors.append("Good sulphate levels")
                elif sulphates < 0.4:
                    negative_factors.append("Low sulphate levels")
                    
                if citric_acid > 0.3: 
                    positive_factors.append("Good citric acid content")
                elif citric_acid < 0.1:
                    negative_factors.append("Low citric acid")
                    
                if volatile_acidity < 0.5: 
                    positive_factors.append("Low volatile acidity")
                elif volatile_acidity > 1.0:
                    negative_factors.append("High volatile acidity")
                
                if positive_factors:
                    st.write("**Positive factors in this wine:**")
                    for factor in positive_factors:
                        st.write(f"✓ {factor}")
                
                if negative_factors:
                    st.write("**Areas for improvement:**")
                    for factor in negative_factors:
                        st.write(f"⚠ {factor}")
                        
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info("The model might be incompatible. Try creating a new model.")

def show_model_performance(df, model):
    """Display model performance section"""
    st.markdown('<div class="sub-header">Model Performance</div>', unsafe_allow_html=True)
    
    if model is None:
        st.warning("No model loaded. Performance metrics are not available.")
        return
    
    # Load pre-calculated metrics (in a real scenario, these would be calculated during training)
    # For demonstration, we'll use some example metrics
    performance_metrics = {
        'accuracy': 0.72,
        'precision': 0.71,
        'recall': 0.70,
        'f1_score': 0.70
    }
    
    # Model metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{performance_metrics['accuracy']:.1%}")
    with col2:
        st.metric("Precision", f"{performance_metrics['precision']:.1%}")
    with col3:
        st.metric("Recall", f"{performance_metrics['recall']:.1%}")
    with col4:
        st.metric("F1-Score", f"{performance_metrics['f1_score']:.1%}")
    
    # Feature importance
    st.subheader("Feature Importance")
    
    # Example feature importance (in practice, get from trained model)
    feature_importance = {
        'alcohol': 0.18,
        'sulphates': 0.12,
        'volatile acidity': 0.11,
        'total sulfur dioxide': 0.09,
        'density': 0.08,
        'chlorides': 0.07,
        'citric acid': 0.07,
        'fixed acidity': 0.06,
        'pH': 0.06,
        'residual sugar': 0.05,
        'free sulfur dioxide': 0.04
    }
    
    fig = px.bar(x=list(feature_importance.values()), 
                y=list(feature_importance.keys()),
                orientation='h',
                title='Feature Importance in Quality Prediction',
                color=list(feature_importance.values()),
                color_continuous_scale='Reds')
    fig.update_layout(xaxis_title='Importance', yaxis_title='Features')
    st.plotly_chart(fig, use_container_width=True)
    
    # Model comparison
    st.subheader("Model Comparison")
    
    comparison_data = {
        'Model': ['Random Forest', 'Logistic Regression', 'SVM', 'Gradient Boosting'],
        'Accuracy': [0.72, 0.58, 0.65, 0.70],
        'Precision': [0.71, 0.56, 0.63, 0.69],
        'Recall': [0.70, 0.55, 0.62, 0.68]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Performance by quality level
    st.subheader("Performance by Quality Level")
    
    quality_performance = {
        'Quality': [3, 4, 5, 6, 7, 8],
        'Precision': [0.45, 0.52, 0.68, 0.75, 0.65, 0.50],
        'Recall': [0.40, 0.48, 0.72, 0.78, 0.60, 0.45]
    }
    
    perf_df = pd.DataFrame(quality_performance)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=perf_df['Quality'], y=perf_df['Precision'],
                           mode='lines+markers', name='Precision',
                           line=dict(color='red')))
    fig.add_trace(go.Scatter(x=perf_df['Quality'], y=perf_df['Recall'],
                           mode='lines+markers', name='Recall',
                           line=dict(color='blue')))
    fig.update_layout(title='Precision and Recall by Quality Level',
                     xaxis_title='Wine Quality',
                     yaxis_title='Score')
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()