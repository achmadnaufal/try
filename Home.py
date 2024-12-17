import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import numpy as np

# Set page config
st.set_page_config(
    page_title="Coupon Acceptance Analysis",
    page_icon="🎟️",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .stPlotlyChart {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and cache the dataset"""
    df = pd.read_csv('invehiclecouponrecommendation.csv')
    return df

def calculate_acceptance_rates(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Calculate acceptance rates for a given feature"""
    grouped = df.groupby(feature)['Y'].agg(['count', 'sum']).reset_index()
    grouped['acceptance_rate'] = (grouped['sum'] / grouped['count'] * 100).round(2)
    grouped = grouped.sort_values('acceptance_rate', ascending=False)
    return grouped

def create_bar_chart(data: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    """Create a bar chart using plotly"""
    fig = px.bar(
        data,
        x=x,
        y=y,
        title=title,
        text=y,
        color=y,
        color_continuous_scale='Viridis'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        height=500,
        showlegend=False,
        yaxis_title='Acceptance Rate (%)',
        xaxis_title=x.capitalize()
    )
    return fig

def analyze_combinations(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Analyze feature combinations"""
    grouped = df.groupby(features)['Y'].agg(['count', 'sum']).reset_index()
    grouped['acceptance_rate'] = (grouped['sum'] / grouped['count'] * 100).round(2)
    grouped = grouped[grouped['count'] >= 50]  # Filter for statistical significance
    grouped = grouped.sort_values('acceptance_rate', ascending=False)
    return grouped

def main():
    st.title("🎟️ Coupon Acceptance Analysis")
    st.write("Analyze and simulate in-vehicle coupon recommendation acceptance rates")

    # Load data
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("Please upload the dataset file 'invehiclecouponrecommendation.csv'")
        return

    # Sidebar filters
    st.header("Filters")
    selected_features = st.sidebar.multiselect(
        "Select features to analyze",
        options=['destination', 'coupon', 'time', 'weather', 'expiration', 'passanger'],
        default=['destination', 'coupon', 'time']
    )

    # Main content
    col1, col2 = st.columns(2)

    for idx, feature in enumerate(selected_features):
        if idx % 2 == 0:
            with col1:
                rates = calculate_acceptance_rates(df, feature)
                fig = create_bar_chart(
                    rates,
                    x=feature,
                    y='acceptance_rate',
                    title=f'Acceptance Rates by {feature.capitalize()}'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            with col2:
                rates = calculate_acceptance_rates(df, feature)
                fig = create_bar_chart(
                    rates,
                    x=feature,
                    y='acceptance_rate',
                    title=f'Acceptance Rates by {feature.capitalize()}'
                )
                st.plotly_chart(fig, use_container_width=True)

    # Combination Analysis
    st.header("Feature Combination Analysis")
    selected_combo_features = st.multiselect(
        "Select features for combination analysis",
        options=['destination', 'coupon', 'time', 'weather', 'expiration', 'passanger'],
        default=['coupon', 'time', 'destination']
    )

    if len(selected_combo_features) > 1:
        combo_rates = analyze_combinations(df, selected_combo_features)
        combo_rates['combination'] = combo_rates.apply(
            lambda x: ' | '.join([f"{f}: {x[f]}" for f in selected_combo_features]),
            axis=1
        )
        
        fig = px.bar(
            combo_rates.head(10),
            x='acceptance_rate',
            y='combination',
            orientation='h',
            title='Top 10 Feature Combinations',
            text='acceptance_rate'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=600, yaxis_title='')
        st.plotly_chart(fig, use_container_width=True)

        # Show detailed stats
        st.subheader("Detailed Statistics")
        stats_df = combo_rates.head(10)[['combination', 'count', 'sum', 'acceptance_rate']]
        stats_df.columns = ['Combination', 'Total Samples', 'Accepted', 'Acceptance Rate (%)']
        st.dataframe(stats_df)

if __name__ == "__main__":
    main()
