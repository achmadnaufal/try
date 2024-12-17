import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, precision_recall_curve, auc
import plotly.graph_objects as go
import plotly.express as px
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Model Analysis", page_icon="🤖", layout="wide")

@st.cache_data
def load_and_preprocess_data() -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Load and preprocess the dataset"""
    df = pd.read_csv('invehiclecouponrecommendation.csv')
    
    # Create dictionary to store label encoders
    encoders = {}
    
    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Encode categorical variables
    for column in categorical_columns:
        encoders[column] = LabelEncoder()
        df[column] = encoders[column].fit_transform(df[column].astype(str))
    
    return df, encoders

def create_feature_matrix(df: pd.DataFrame, target_coupon: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Create feature matrix and target variable"""
    # Define features to use
    features = [
        'destination', 'passanger', 'weather', 'temperature', 'time',
        'expiration', 'gender', 'age', 'maritalStatus', 'has_children',
        'education', 'occupation', 'income', 'Bar', 'CoffeeHouse',
        'CarryAway', 'RestaurantLessThan20', 'Restaurant20To50',
        'toCoupon_GEQ15min', 'toCoupon_GEQ25min', 'direction_same'
    ]
    
    X = df[features]
    
    if target_coupon:
        # Filter for specific coupon type
        mask = df['coupon'] == target_coupon
        X = X[mask]
        y = df[mask]['Y']
    else:
        y = df['Y']
    
    return X, y

def train_decision_tree(X: pd.DataFrame, y: pd.Series) -> Tuple[DecisionTreeClassifier, Dict[str, Any]]:
    """Train decision tree with hyperparameter tuning"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Define parameter grid
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Perform grid search
    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Calculate metrics
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # PR curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    metrics = {
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_data': (fpr, tpr, roc_auc),
        'pr_data': (precision, recall, pr_auc),
        'best_params': grid_search.best_params_,
        'X_test': X_test,
        'y_test': y_test
    }
    
    return best_model, metrics

def plot_confusion_matrix(conf_matrix: np.ndarray) -> go.Figure:
    """Create confusion matrix plot"""
    fig = px.imshow(
        conf_matrix,
        labels=dict(x="Predicted", y="Actual"),
        x=['Rejected', 'Accepted'],
        y=['Rejected', 'Accepted'],
        text=conf_matrix,
        color_continuous_scale='Blues'
    )
    fig.update_traces(texttemplate="%{z}")
    return fig

def plot_curve(data: Tuple, curve_type: str) -> go.Figure:
    """Create ROC or PR curve plot"""
    x, y, auc_score = data
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        name=f'{curve_type} curve (AUC = {auc_score:.3f})'
    ))
    
    if curve_type == 'ROC':
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash')
        ))
        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate'
        )
    else:
        fig.update_layout(
            xaxis_title='Recall',
            yaxis_title='Precision'
        )
    
    # Set axis ranges to be the same
    fig.update_layout(
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def main():
    st.title("🤖 Machine Learning Model Analysis")
    
    # Load data
    try:
        df, encoders = load_and_preprocess_data()
    except FileNotFoundError:
        st.error("Please upload the dataset file 'invehiclecouponrecommendation.csv'")
        return
    
    # Sidebar controls
    st.sidebar.header("Model Configuration")
    
    # Option to analyze specific coupon type or all
    coupon_options = ['All Coupons'] + list(encoders['coupon'].classes_)
    selected_coupon = st.sidebar.selectbox(
        "Select Coupon Type",
        options=coupon_options
    )
    
    # Prepare data
    if selected_coupon == 'All Coupons':
        X, y = create_feature_matrix(df)
        coupon_type = None
    else:
        coupon_idx = list(encoders['coupon'].classes_).index(selected_coupon)
        X, y = create_feature_matrix(df, coupon_idx)
        coupon_type = selected_coupon
    
    # Train model and get metrics
    with st.spinner("Training model and computing metrics..."):
        model, metrics = train_decision_tree(X, y)
    
    # Display results in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance")
        st.metric("F1 Score", f"{metrics['f1']:.3f}")
        
        st.subheader("Best Hyperparameters")
        st.json(metrics['best_params'])
        
        st.subheader("Confusion Matrix")
        conf_matrix_fig = plot_confusion_matrix(metrics['confusion_matrix'])
        st.plotly_chart(conf_matrix_fig, use_container_width=True)
    
    with col2:
        st.subheader("ROC Curve")
        roc_fig = plot_curve(metrics['roc_data'], 'ROC')
        st.plotly_chart(roc_fig, use_container_width=True)
        
        st.subheader("Precision-Recall Curve")
        pr_fig = plot_curve(metrics['pr_data'], 'PR')
        st.plotly_chart(pr_fig, use_container_width=True)
    
    # Decision Tree Visualization
    st.subheader("Decision Tree Visualization")
    
    # Create matplotlib figure for decision tree
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        model, 
        feature_names=X.columns,
        class_names=['Rejected', 'Accepted'],
        filled=True,
        rounded=True,
        ax=ax
    )
    st.pyplot(fig)
    
    # Feature Importance
    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance'
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()