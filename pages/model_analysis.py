import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (f1_score, confusion_matrix, roc_curve, precision_recall_curve,
                           auc, classification_report)
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
import matplotlib.pyplot as plt
import io
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ML Analysis", page_icon="🤖", layout="wide")

class MLAnalyzer:
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.feature_importance = {}
        self.metrics = {}
        self.X_test = {}
        self.y_test = {}
        
    def preprocess_data(self, df: pd.DataFrame, target: str = 'Y') -> tuple:
        """Preprocess the data for ML"""
        # Create copy to avoid modifying original data
        df_processed = df.copy()
        
        # Initialize encoders dictionary for this run
        self.encoders = {}
        
        # Encode categorical variables
        for column in df_processed.select_dtypes(include=['object']):
            self.encoders[column] = LabelEncoder()
            df_processed[column] = self.encoders[column].fit_transform(df_processed[column].fillna('missing'))
        
        # Fill numeric nulls with median
        numeric_columns = df_processed.select_dtypes(include=['int64', 'float64']).columns
        for column in numeric_columns:
            df_processed[column] = df_processed[column].fillna(df_processed[column].median())
        
        # Split features and target
        X = df_processed.drop(target, axis=1)
        y = df_processed[target]
        
        return X, y
        
    def train_model(self, X, y, coupon_type: str = 'all'):
        """Train decision tree model with hyperparameter tuning"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Store test data for later use
        self.X_test[coupon_type] = X_test
        self.y_test[coupon_type] = y_test
        
        # Define parameter grid
        param_grid = {
            'max_depth': [3, 4, 5, 6],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', None]
        }
        
        # Set up GridSearchCV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        
        # Fit model
        grid_search.fit(X_train, y_train)
        
        # Store best model
        self.models[coupon_type] = grid_search.best_estimator_
        
        # Calculate feature importance
        self.feature_importance[coupon_type] = pd.DataFrame({
            'feature': X.columns,
            'importance': grid_search.best_estimator_.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calculate metrics
        y_pred = grid_search.best_estimator_.predict(X_test)
        y_pred_proba = grid_search.best_estimator_.predict_proba(X_test)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        # PR curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
        pr_auc = auc(recall, precision)
        
        self.metrics[coupon_type] = {
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'test_f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'roc_data': {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc},
            'pr_data': {'precision': precision, 'recall': recall, 'auc': pr_auc}
        }

def plot_tree_rules(model, feature_names, figsize=(20, 10)):
    """Plot decision tree rules"""
    fig, ax = plt.subplots(figsize=figsize)
    plot_tree(model, feature_names=feature_names, 
             class_names=['Reject', 'Accept'], 
             filled=True, rounded=True, ax=ax)
    
    # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    
    return buf

def plot_curve_comparison(metrics, curve_type='roc'):
    """Plot ROC or PR curves with same axis range"""
    fig = go.Figure()
    
    for coupon_type, metric in metrics.items():
        if curve_type == 'roc':
            x_data = metric['roc_data']['fpr']
            y_data = metric['roc_data']['tpr']
            auc_score = metric['roc_data']['auc']
            title = 'ROC Curves Comparison'
            x_label = 'False Positive Rate'
            y_label = 'True Positive Rate'
        else:  # PR curve
            x_data = metric['pr_data']['recall']
            y_data = metric['pr_data']['precision']
            auc_score = metric['pr_data']['auc']
            title = 'Precision-Recall Curves Comparison'
            x_label = 'Recall'
            y_label = 'Precision'
            
        fig.add_trace(go.Scatter(
            x=x_data, y=y_data,
            name=f'{coupon_type} (AUC={auc_score:.3f})',
            mode='lines'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        width=800,
        height=500
    )
    
    return fig

def main():
    st.title("🤖 Machine Learning Analysis")
    
    # Load data
    try:
        df = pd.read_csv('invehiclecouponrecommendation.csv')
    except FileNotFoundError:
        st.error("Please upload the dataset file 'invehiclecouponrecommendation.csv'")
        return
    
    # Initialize analyzer
    analyzer = MLAnalyzer()
    
    # Sidebar controls
    st.sidebar.header("Analysis Options")
    
    analysis_type = st.sidebar.radio(
        "Select Analysis Type",
        ["All Coupons Combined", "Individual Coupon Types"]
    )
    
    # Main analysis
    if analysis_type == "All Coupons Combined":
        st.header("Analysis for All Coupons Combined")
        
        X, y = analyzer.preprocess_data(df)
        analyzer.train_model(X, y, 'all_coupons')
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        metrics = analyzer.metrics['all_coupons']
        
        with col1:
            st.metric("F1 Score", f"{metrics['test_f1']:.3f}")
        with col2:
            st.metric("ROC AUC", f"{metrics['roc_data']['auc']:.3f}")
        with col3:
            st.metric("PR AUC", f"{metrics['pr_data']['auc']:.3f}")
        
        # Display confusion matrix
        st.subheader("Confusion Matrix")
        cm = metrics['confusion_matrix']
        fig = ff.create_annotated_heatmap(
            z=cm, 
            x=['Predicted 0', 'Predicted 1'],
            y=['Actual 0', 'Actual 1'],
            colorscale='Viridis'
        )
        st.plotly_chart(fig)
        
        # Display decision tree
        st.subheader("Decision Tree Visualization")
        tree_buf = plot_tree_rules(
            analyzer.models['all_coupons'],
            analyzer.feature_importance['all_coupons']['feature'].tolist()
        )
        st.image(tree_buf)
        
        # Display feature importance
        st.subheader("Feature Importance")
        fig = px.bar(
            analyzer.feature_importance['all_coupons'].head(10),
            x='importance',
            y='feature',
            orientation='h'
        )
        st.plotly_chart(fig)
        
    else:
        st.header("Analysis by Coupon Type")
        
        # Train models for each coupon type
        coupon_types = df['coupon'].unique()
        
        for coupon_type in coupon_types:
            st.subheader(f"Analysis for {coupon_type}")
            
            # Filter data for this coupon type
            df_coupon = df[df['coupon'] == coupon_type]
            X, y = analyzer.preprocess_data(df_coupon)
            analyzer.train_model(X, y, coupon_type)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            metrics = analyzer.metrics[coupon_type]
            
            with col1:
                st.metric("F1 Score", f"{metrics['test_f1']:.3f}")
            with col2:
                st.metric("ROC AUC", f"{metrics['roc_data']['auc']:.3f}")
            with col3:
                st.metric("PR AUC", f"{metrics['pr_data']['auc']:.3f}")
            
            # Display confusion matrix
            st.write("Confusion Matrix")
            cm = metrics['confusion_matrix']
            fig = ff.create_annotated_heatmap(
                z=cm, 
                x=['Predicted 0', 'Predicted 1'],
                y=['Actual 0', 'Actual 1'],
                colorscale='Viridis'
            )
            st.plotly_chart(fig)
    
    # Compare ROC and PR curves
    st.header("Curve Comparisons")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ROC Curves")
        roc_fig = plot_curve_comparison(analyzer.metrics, 'roc')
        st.plotly_chart(roc_fig)
    
    with col2:
        st.subheader("Precision-Recall Curves")
        pr_fig = plot_curve_comparison(analyzer.metrics, 'pr')
        st.plotly_chart(pr_fig)

if __name__ == "__main__":
    main()
