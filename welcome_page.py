import streamlit as st
from typing import Dict, List
import plotly.graph_objects as go

def set_page_config():
    st.set_page_config(
        page_title="Coupon Analysis Welcome",
        page_icon="🎯",
        layout="wide"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stHeader {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        .success-metric {
            padding: 1rem;
            background-color: #e8f0fe;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

def create_project_overview() -> None:
    """Display project overview section"""
    st.title("🎯 In-Vehicle Coupon Recommendation Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Project Overview
        This analysis aims to optimize in-vehicle coupon recommendations through a comprehensive 
        two-phase approach combining customer segmentation and contextual analysis.
        
        ### Key Dataset Information
        - Survey data from Amazon Mechanical Turk (95%+ rated Turkers)
        - 652 accepted surveys
        - 12,684 total data cases
        - Each user responded to 20 different driving scenarios
        """)
    
    with col2:
        # Create a simple metrics display
        metrics = {
            "Total Surveys": "652",
            "Data Cases": "12,684",
            "Scenarios per User": "20",
            "Turker Rating": "95%+"
        }
        
        for metric, value in metrics.items():
            st.markdown(f"""
                <div class='success-metric'>
                    <h3>{metric}</h3>
                    <h2>{value}</h2>
                </div>
            """, unsafe_allow_html=True)

def create_analysis_approach_diagram() -> None:
    """Create and display the analysis approach flow diagram"""
    fig = go.Figure()
    
    # Add nodes
    nodes = [
        ("Phase 1", 0, 1, "Customer Segmentation"),
        ("Phase 2", 1, 1, "Contextual Analysis"),
        ("Final", 2, 1, "Optimization & Recommendations")
    ]
    
    for i, (phase, x, y, text) in enumerate(nodes):
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            name=phase,
            text=[text],
            textposition='bottom center',
            marker=dict(size=40, symbol='circle', color='#1f77b4'),
            showlegend=False
        ))
    
    # Add arrows
    for i in range(len(nodes)-1):
        fig.add_annotation(
            x=nodes[i][1],
            y=nodes[i][2],
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            ax=nodes[i+1][1],
            ay=nodes[i+1][2],
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor="#1f77b4"
        )
    
    fig.update_layout(
        title="Analysis Approach Flow",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_analysis_details() -> None:
    """Display detailed analysis approaches"""
    st.markdown("## Analysis Approaches")
    
    tab1, tab2 = st.tabs(["Customer Segmentation", "Contextual Analysis"])
    
    with tab1:
        st.markdown("""
        ### Customer Segmentation Variables
        
        #### Core Variables:
        **Demographics:**
        - Age (ordinal: below21 to 50plus)
        - Income (ordinal: Less than $12500 to $100000 or More)
        - Has Children (binary: 0, 1)
        
        **Behavioral:**
        - Bar frequency
        - CoffeeHouse frequency
        - CarryAway frequency
        - RestaurantLessThan20 frequency
        - Restaurant20To50 frequency
        """)
        
    with tab2:
        st.markdown("""
        ### Contextual Analysis Variables
        
        #### Time-based factors:
        - Time of day (7AM to 10PM)
        - Expiration duration (2h, 1d)
        - Distance (toCoupon_GEQ15min, toCoupon_GEQ25min)
        
        #### Situational factors:
        - Destination (No Urgent Place, Home, Work)
        - Passenger (Alone, Friend(s), Kid(s), Partner)
        - Weather (Sunny, Rainy, Snowy)
        - Temperature (30, 55, 80)
        
        #### Coupon characteristics:
        - Venue type (Restaurant<$20, Coffee House, Carry out, Bar, Restaurant$20-$50)
        - Direction relative to destination
        """)

def display_expected_outputs() -> None:
    """Display expected outputs and next steps"""
    st.markdown("## Expected Outputs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Key Deliverables
        1. Customer segments with detailed profiles
        2. Scenario effectiveness patterns
        3. Optimized targeting recommendations
        
        ### Optimization Considerations
        - Customer segment size
        - Scenario frequency
        - Acceptance probability
        """)
    
    with col2:
        st.markdown("""
        ### Next Steps
        1. Implement customer clustering with reduced variable set
        2. Try both direct modeling and scenario clustering
        3. Compare results and refine approach
        4. Build final optimization model
        """)

def main():
    set_page_config()
    create_project_overview()
    create_analysis_approach_diagram()
    display_analysis_details()
    display_expected_outputs()
    
    # Navigation hint
    st.markdown("---")
    st.info("👈 Use the sidebar to navigate to detailed analysis pages")

if __name__ == "__main__":
    main()
