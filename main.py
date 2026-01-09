import streamlit as st
import pandas as pd
import plotly.express as px
from clustering_utils import NewsClusteringEngine
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page Configuration with wide layout and custom title
st.set_page_config(
    page_title="News Clustering System",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Engine
@st.cache_resource
def get_engine():
    return NewsClusteringEngine()

engine = get_engine()

# --- CUSTOM CSS & DESIGN ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0f172a 0%, #000000 100%);
        color: #e2e8f0;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #1e293b;
    }
    ::-webkit-scrollbar-thumb {
        background: #475569;
        border-radius: 5px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    
    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%) !important;
        border: none !important;
        color: white !important;
        padding: 0.75rem 1.5rem !important;
        font-size: 1.1rem !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.6);
    }
    
    /* Metric Cards */
    .metric-container {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-container:hover {
        transform: translateY(-5px);
        border-color: #6366f1;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(to right, #fff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 1rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Article Cards */
    .article-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        height: 100%;
    }
    .article-card:hover {
        border-color: #38bdf8;
        background: rgba(30, 41, 59, 0.8);
    }
    .article-title {
        color: #f8fafc;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-decoration: none;
    }
    .article-source {
        color: #38bdf8;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 1rem;
        display: inline-block;
        background: rgba(56, 189, 248, 0.1);
        padding: 2px 8px;
        border-radius: 4px;
    }
    .article-desc {
        color: #94a3b8;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function for Custom Metric
def custom_metric(label, value, icon):
    st.markdown(f"""
        <div class="metric-container">
            <div style="font-size: 2rem; margin-bottom: 10px;">{icon}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    env_api_key = os.getenv("NEWS_API_KEY")
    api_key = st.text_input("🔑 NewsAPI Key", value=env_api_key if env_api_key else "", type="password")
    if not api_key:
        st.warning("API Key required!")
    
    st.markdown("---")
    
    st.markdown("### 🎯 Search Parameters")
    search_query = st.text_input("Keywords", value="Artificial Intelligence")
    
    c1, c2 = st.columns(2)
    with c1:
        num_articles = st.number_input("Fetch Limit", min_value=10, max_value=100, value=100, step=10)
    with c2:
        n_clusters = st.number_input("Clusters (k)", min_value=2, max_value=10, value=4)
        
    st.markdown("---")
    st.info("💡 **Tip:** Increase 'k' for broader topics to find improved sub-groupings.")

# Header
st.markdown("# News Clustering System")
st.markdown("### Discover the hidden structure in global headlines using AI.")

# Fetch Logic
if st.button("🚀 Fetch & Analyze News", use_container_width=True):
    if not api_key or not search_query:
        st.toast("❌ Please check your API Key and Search Query!", icon="🛑")
    else:
        try:
            with st.status("🤖 AI Agent Working...", expanded=True) as status:
                st.write("📡 Connecting to NewsAPI...")
                articles = engine.fetch_news(api_key, search_query, num_articles)
                
                if not articles:
                    status.update(label="No articles found!", state="error")
                    st.error("No articles found to cluster. Try a different query.")
                else:
                    st.write(f"✅ Fetched {len(articles)} articles. Processing text...")
                    df = pd.DataFrame(articles)
                    
                    st.write("🧠 Vectorizing and Clustering...")
                    clustered_df, tfidf_matrix, _, sil_score = engine.perform_clustering(df, n_clusters)
                    
                    status.update(label="Processing Complete!", state="complete", expanded=False)
                    st.balloons()
            
            if clustered_df is not None:
                # Top Metrics
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1: custom_metric("Articles", len(clustered_df), "📚")
                with m2: custom_metric("Clusters", n_clusters, "🌀")
                with m3: custom_metric("Sources", len(clustered_df['source'].unique()), "🌍")
                with m4: custom_metric("Avg Len", f"{int(clustered_df['description'].str.len().mean())}c", "📏")
                with m5: custom_metric("Silhouette", f"{sil_score:.3f}", "📊")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Tabs for different views
                tab1, tab2, tab3 = st.tabs(["🗺️ Cluster Map", "📑 Article Feed", "🏷️ Topic Keywords"])
                
                with tab1:
                    st.markdown("### 🔍 Interactive Topic Map")
                    fig = px.scatter(
                        clustered_df,
                        x='x', y='y',
                        color=clustered_df['cluster'].astype(str),
                        hover_data={'title': True, 'source': True, 'x': False, 'y': False, 'cluster': False},
                        custom_data=['title', 'source', 'url'],
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig.update_traces(
                        marker=dict(size=12, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')),
                        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}"
                    )
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(showgrid=False, zeroline=False, visible=False),
                        yaxis=dict(showgrid=False, zeroline=False, visible=False),
                        showlegend=True,
                        legend=dict(
                            yanchor="top", y=0.99, xanchor="left", x=0.01,
                            bgcolor="rgba(0,0,0,0.5)"
                        ),
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                with tab2:
                    st.markdown("### 📰 Curated Article Feed")
                    
                    # Group by cluster for creating sections
                    clusters = sorted(clustered_df['cluster'].unique())
                    for c_id in clusters:
                        c_articles = clustered_df[clustered_df['cluster'] == c_id]
                        
                        with st.expander(f"📌 Cluster {c_id} - {len(c_articles)} Articles", expanded=True):
                            # Create a grid layout for cards
                            cols = st.columns(3)
                            for idx, (_, row) in enumerate(c_articles.iterrows()):
                                with cols[idx % 3]:
                                    st.markdown(f"""
                                    <div class="article-card">
                                        <div class="article-source">{row['source']}</div>
                                        <a href="{row['url']}" target="_blank" class="article-title">{row['title']}</a>
                                        <p class="article-desc">{row['description'][:150]}...</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                with tab3:
                    st.markdown("### 🔑 Top Keywords by Cluster")
                    keywords = engine.get_cluster_keywords(
                        tfidf_matrix, 
                        engine.vectorizer.get_feature_names_out(), 
                        clustered_df['cluster']
                    )
                    
                    k_cols = st.columns(len(keywords))
                    for idx, (cid, terms) in enumerate(keywords.items()):
                        with k_cols[idx % len(k_cols)]:
                            st.markdown(f"""
                                <div style="background:#1e293b; padding:15px; border-radius:10px; border-top: 3px solid #6366f1;">
                                    <h3 style="text-align:center; margin:0; font-size:1.5rem;">C{cid}</h3>
                                    <hr style="border-color:#334155;">
                                    <ul style="padding-left:20px; color:#cbd5e1;">
                                        {''.join([f'<li style="margin-bottom:5px;">{t}</li>' for t in terms])}
                                    </ul>
                                </div>
                            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.code(str(e))
else:
    # Landing State
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; opacity: 0.6;">
        <h2>👈 Ready to explore?</h2>
        <p>Enter your configuration in the sidebar and click Fetch.</p>
    </div>
    """, unsafe_allow_html=True)
