import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import Counter
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Initialize Supabase client
@st.cache_resource
def get_supabase_client():
    """Get cached Supabase client."""
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# Load conversation data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_conversation_data(start_date: Optional[datetime] = None, end_date: Optional[datetime] = None):
    """Load conversation data from Supabase."""
    try:
        supabase = get_supabase_client()
        query = supabase.table("conversation_log").select("*")
        
        if start_date:
            query = query.gte("created_at", start_date.isoformat())
        if end_date:
            query = query.lte("created_at", end_date.isoformat())
        
        response = query.order("created_at", desc=True).execute()
        return pd.DataFrame(response.data)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def process_conversation_data(df: pd.DataFrame):
    """Process and enrich the dataframe with derived metrics."""
    if df.empty:
        return df
    
    # Convert created_at to datetime
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    # Extract date components
    df['date'] = df['created_at'].dt.date
    df['hour'] = df['created_at'].dt.hour
    df['day_of_week'] = df['created_at'].dt.day_name()
    
    # Process list columns (stored as JSON strings or lists)
    def safe_eval(x):
        if isinstance(x, str):
            try:
                return eval(x) if x else []
            except:
                return []
        return x if x else []
    
    df['tools_called_list'] = df['tools_called'].apply(safe_eval)
    df['agents_used_list'] = df['agents_used'].apply(safe_eval)
    df['tools_result_dict'] = df['tools_result'].apply(
        lambda x: eval(x) if isinstance(x, str) and x else {} if isinstance(x, str) else (x if x else {})
    )
    
    # Calculate metrics
    df['has_tools'] = df['tools_called_list'].apply(lambda x: len(x) > 0)
    df['has_agents'] = df['agents_used_list'].apply(lambda x: len(x) > 0)
    df['num_tools'] = df['tools_called_list'].apply(len)
    df['num_agents'] = df['agents_used_list'].apply(len)
    df['message_length'] = df['human_message'].apply(lambda x: len(str(x)) if x else 0)
    df['ai_message_length'] = df['ai_message'].apply(lambda x: len(str(x)) if x else 0)
    
    # Risk categorization
    df['risk_category'] = pd.cut(
        df['risk_probability'],
        bins=[-0.01, 0.0, 0.3, 0.7, 1.01],
        labels=['None', 'Low', 'Medium', 'High']
    )
    
    return df

def get_conversation_stats(df: pd.DataFrame):
    """Calculate conversation-level statistics."""
    if df.empty:
        return {}
    
    stats = {}
    
    # Basic metrics
    stats['total_conversations'] = df['conversation_id'].nunique()
    stats['total_turns'] = len(df)
    stats['total_conversations_ended'] = df[df['conversation_ended'] == True]['conversation_id'].nunique()
    stats['avg_turns_per_conversation'] = df.groupby('conversation_id').size().mean() if stats['total_conversations'] > 0 else 0
    
    # Risk metrics
    stats['conversations_with_risk'] = df[df['risk_probability'] > 0.0]['conversation_id'].nunique()
    stats['high_risk_conversations'] = df[df['risk_probability'] > 0.7]['conversation_id'].nunique()
    stats['avg_risk_probability'] = df['risk_probability'].mean()
    stats['max_risk_probability'] = df['risk_probability'].max()
    
    # Tool metrics
    all_tools = []
    for tools_list in df['tools_called_list']:
        all_tools.extend(tools_list)
    stats['total_tool_calls'] = len(all_tools)
    stats['unique_tools'] = len(set(all_tools))
    stats['turns_with_tools'] = df['has_tools'].sum()
    stats['tool_usage_rate'] = (stats['turns_with_tools'] / stats['total_turns'] * 100) if stats['total_turns'] > 0 else 0

    # Agent metrics
    all_agents = []
    for agents_list in df['agents_used_list']:
        all_agents.extend(agents_list)
    stats['total_agent_uses'] = len(all_agents)
    stats['unique_agents'] = len(set(all_agents))
    stats['turns_with_agents'] = df['has_agents'].sum()
    
    # Time metrics
    if not df.empty:
        stats['first_conversation'] = df['created_at'].min()
        stats['last_conversation'] = df['created_at'].max()
        stats['date_range_days'] = (stats['last_conversation'] - stats['first_conversation']).days + 1
    
    return stats

# Streamlit App
def main():
    st.set_page_config(
        page_title="Chatbot Analytics Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 Mental Health Chatbot Analytics Dashboard")
    st.markdown("---")
    
    # Sidebar filters
    st.sidebar.header("📅 Date Range Filter")
    
    # Date range selector
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
    
    # Load and process data
    with st.spinner("Loading conversation data..."):
        df = load_conversation_data(
            start_date=datetime.combine(start_date, datetime.min.time()),
            end_date=datetime.combine(end_date, datetime.max.time())
        )
        
        if df.empty:
            st.warning("⚠️ No conversation data found for the selected date range.")
            return
        
        df = process_conversation_data(df)
    
    # Calculate statistics
    stats = get_conversation_stats(df)
    
    # ========== OVERVIEW METRICS ==========
    st.header("📈 Overview Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Conversations",
            value=f"{stats['total_conversations']:,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Total Conversation Turns",
            value=f"{stats['total_turns']:,}",
            delta=None
        )
    
    with col3:
        avg_turns = f"{stats['avg_turns_per_conversation']:.1f}"
        st.metric(
            label="Avg Turns/Conversation",
            value=avg_turns,
            delta=None
        )
    
    with col4:
        risk_rate = f"{(stats['high_risk_conversations'] / stats['total_conversations'] * 100):.1f}%" if stats['total_conversations'] > 0 else "0%"
        st.metric(
            label="High Risk Rate",
            value=risk_rate,
            delta=None
        )
    
    st.markdown("---")
    
    # ========== KEY PERFORMANCE INDICATORS ==========
    st.header("🎯 Key Performance Indicators")
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    
    with kpi_col1:
        st.metric(
            label="Tool Usage Rate",
            value=f"{stats['tool_usage_rate']:.1f}%",
            help="Percentage of turns that used tools"
        )
    
    with kpi_col2:
        st.metric(
            label="Unique Tools Used",
            value=f"{stats['unique_tools']}",
            delta=None
        )
    
    with kpi_col3:
        st.metric(
            label="Avg Risk Probability",
            value=f"{stats['avg_risk_probability']:.4f}",
            delta=None
        )
    
    with kpi_col4:
        st.metric(
            label="Conversations Ended",
            value=f"{stats['total_conversations_ended']:,}",
            delta=None
        )
    
    st.markdown("---")
    
    # ========== CONVERSATION ANALYTICS ==========
    st.header("💬 Conversation Analytics")
    
    conv_col1, conv_col2 = st.columns(2)
    
    with conv_col1:
        # Conversation length distribution
        conv_lengths = df.groupby('conversation_id').size()
        fig_length = px.histogram(
            x=conv_lengths.values,
            nbins=20,
            title="Conversation Length Distribution",
            labels={'x': 'Number of Turns', 'count': 'Number of Conversations'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_length.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_length, use_container_width=True)
    
    with conv_col2:
        # Conversations over time
        daily_conversations = df.groupby('date')['conversation_id'].nunique().reset_index()
        daily_conversations.columns = ['date', 'conversations']
        
        fig_timeline = px.line(
            daily_conversations,
            x='date',
            y='conversations',
            title="Daily Conversations Over Time",
            labels={'date': 'Date', 'conversations': 'Number of Conversations'},
            markers=True
        )
        fig_timeline.update_layout(height=400, xaxis_title="Date", yaxis_title="Conversations")
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    # ========== TOOL USAGE ANALYTICS ==========
    st.header("🔧 Tool Usage Analytics")
    
    tool_col1, tool_col2 = st.columns(2)
    
    with tool_col1:
        # Tool usage frequency
        all_tools = []
        for tools_list in df['tools_called_list']:
            all_tools.extend(tools_list)
        
        if all_tools:
            tool_counts = Counter(all_tools)
            tool_df = pd.DataFrame(list(tool_counts.items()), columns=['Tool', 'Count'])
            tool_df = tool_df.sort_values('Count', ascending=True)
            
            fig_tools = px.bar(
                tool_df,
                x='Count',
                y='Tool',
                orientation='h',
                title="Tool Usage Frequency",
                labels={'Count': 'Number of Calls', 'Tool': 'Tool Name'},
                color='Count',
                color_continuous_scale='Blues'
            )
            fig_tools.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_tools, use_container_width=True)
        else:
            st.info("No tools were used in the selected period.")
    
    with tool_col2:
        # Tool usage over time
        df['tools_called_str'] = df['tools_called_list'].apply(lambda x: ', '.join(x) if x else 'None')
        tool_timeline = df[df['has_tools']].groupby('date')['num_tools'].sum().reset_index()
        
        if not tool_timeline.empty:
            fig_tool_time = px.line(
                tool_timeline,
                x='date',
                y='num_tools',
                title="Tool Calls Over Time",
                labels={'date': 'Date', 'num_tools': 'Total Tool Calls'},
                markers=True
            )
            fig_tool_time.update_layout(height=400)
            st.plotly_chart(fig_tool_time, use_container_width=True)
        else:
            st.info("No tool usage data for the selected period.")
    
    # ========== AGENT USAGE ANALYTICS ==========
    st.header("🤖 Agent Usage Analytics")
    
    agent_col1, agent_col2 = st.columns(2)
    
    with agent_col1:
        # Agent usage frequency
        all_agents = []
        for agents_list in df['agents_used_list']:
            all_agents.extend(agents_list)
        
        if all_agents:
            agent_counts = Counter(all_agents)
            agent_df = pd.DataFrame(list(agent_counts.items()), columns=['Agent', 'Count'])
            agent_df = agent_df.sort_values('Count', ascending=True)
            
            fig_agents = px.bar(
                agent_df,
                x='Count',
                y='Agent',
                orientation='h',
                title="Agent Usage Frequency",
                labels={'Count': 'Number of Uses', 'Agent': 'Agent Name'},
                color='Count',
                color_continuous_scale='Greens'
            )
            fig_agents.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_agents, use_container_width=True)
        else:
            st.info("No agents were used in the selected period.")
    
    with agent_col2:
        # Agent-tool correlation (heatmap)
        # Create a matrix of agent-tool co-occurrence
        agent_tool_matrix = {}
        for _, row in df.iterrows():
            agents = row['agents_used_list']
            tools = row['tools_called_list']
            for agent in agents:
                if agent not in agent_tool_matrix:
                    agent_tool_matrix[agent] = Counter()
                for tool in tools:
                    agent_tool_matrix[agent][tool] += 1
        
        if agent_tool_matrix:
            # Create heatmap data
            all_agents_list = list(agent_tool_matrix.keys())
            all_tools_list = list(set([tool for sublist in df['tools_called_list'] for tool in sublist]))
            
            if all_tools_list:
                heatmap_data = []
                for agent in all_agents_list:
                    row = [agent_tool_matrix[agent].get(tool, 0) for tool in all_tools_list]
                    heatmap_data.append(row)
                
                fig_heatmap = px.imshow(
                    heatmap_data,
                    labels=dict(x="Tools", y="Agents", color="Frequency"),
                    x=all_tools_list,
                    y=all_agents_list,
                    title="Agent-Tool Usage Correlation",
                    aspect="auto",
                    color_continuous_scale='Viridis'
                )
                fig_heatmap.update_layout(height=400)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("No tool-agent correlation data available.")
        else:
            st.info("No agent usage data for heatmap.")
    
    # ========== RISK ANALYSIS ==========
    st.header("⚠️ Risk Analysis")
    
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        # Risk probability distribution
        risk_data = df[df['risk_probability'] > 0.0]
        if not risk_data.empty:
            fig_risk_dist = px.histogram(
                risk_data,
                x='risk_probability',
                nbins=20,
                title="Risk Probability Distribution",
                labels={'risk_probability': 'Risk Probability', 'count': 'Frequency'},
                color_discrete_sequence=['#ff6b6b']
            )
            fig_risk_dist.update_layout(height=400)
            st.plotly_chart(fig_risk_dist, use_container_width=True)
        else:
            st.info("No risk cases detected in the selected period.")
    
    with risk_col2:
        # Risk over time
        risk_timeline = df.groupby('date')['risk_probability'].mean().reset_index()
        risk_count_timeline = df[df['risk_probability'] > 0.0].groupby('date').size().reset_index()
        risk_count_timeline.columns = ['date', 'count']
        
        fig_risk_time = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_risk_time.add_trace(
            go.Scatter(
                x=risk_timeline['date'],
                y=risk_timeline['risk_probability'],
                name="Avg Risk Probability",
                mode='lines+markers',
                line=dict(color='red')
            ),
            secondary_y=False,
        )
        
        if not risk_count_timeline.empty:
            fig_risk_time.add_trace(
                go.Bar(
                    x=risk_count_timeline['date'],
                    y=risk_count_timeline['count'],
                    name="Risk Cases Count",
                    marker_color='orange',
                    opacity=0.6
                ),
                secondary_y=True,
            )
        
        fig_risk_time.update_xaxes(title_text="Date")
        fig_risk_time.update_yaxes(title_text="Avg Risk Probability", secondary_y=False)
        fig_risk_time.update_yaxes(title_text="Risk Cases Count", secondary_y=True)
        fig_risk_time.update_layout(title="Risk Analysis Over Time", height=400)
        st.plotly_chart(fig_risk_time, use_container_width=True)
    
    # Risk category breakdown
    risk_category_counts = df['risk_category'].value_counts()
    if not risk_category_counts.empty:
        fig_risk_pie = px.pie(
            values=risk_category_counts.values,
            names=risk_category_counts.index,
            title="Risk Category Distribution",
            color_discrete_map={
                'None': '#90EE90',
                'Low': '#FFE4B5',
                'Medium': '#FFA500',
                'High': '#FF6347'
            }
        )
        fig_risk_pie.update_layout(height=400)
        st.plotly_chart(fig_risk_pie, use_container_width=True)
    
    # ========== TIME-BASED ANALYTICS ==========
    st.header("⏰ Time-Based Analytics")
    
    time_col1, time_col2 = st.columns(2)
    
    with time_col1:
        # Hourly distribution
        hourly_counts = df.groupby('hour').size().reset_index()
        hourly_counts.columns = ['hour', 'count']
        
        fig_hourly = px.bar(
            hourly_counts,
            x='hour',
            y='count',
            title="Conversation Activity by Hour of Day",
            labels={'hour': 'Hour (24h format)', 'count': 'Number of Turns'},
            color='count',
            color_continuous_scale='Plasma'
        )
        fig_hourly.update_layout(height=400, xaxis=dict(tickmode='linear', tick0=0, dtick=2))
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    with time_col2:
        # Day of week distribution
        dow_counts = df['day_of_week'].value_counts().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ], fill_value=0)
        
        fig_dow = px.bar(
            x=dow_counts.index,
            y=dow_counts.values,
            title="Conversation Activity by Day of Week",
            labels={'x': 'Day of Week', 'y': 'Number of Turns'},
            color=dow_counts.values,
            color_continuous_scale='Cividis'
        )
        fig_dow.update_layout(height=400)
        st.plotly_chart(fig_dow, use_container_width=True)
    
    # ========== CONVERSATION EXPLORER ==========
    st.header("🔍 Conversation Explorer")
    
    # Search and filter options
    search_col1, search_col2 = st.columns(2)
    
    with search_col1:
        search_query = st.text_input("🔎 Search Conversations", placeholder="Enter conversation ID or keyword...")
    
    with search_col2:
        risk_filter = st.selectbox("Filter by Risk Level", ["All", "High", "Medium", "Low", "None"])
    
    # Filter dataframe
    filtered_df = df.copy()
    
    if search_query:
        filtered_df = filtered_df[
            filtered_df['conversation_id'].str.contains(search_query, case=False, na=False) |
            filtered_df['human_message'].str.contains(search_query, case=False, na=False) |
            filtered_df['ai_message'].str.contains(search_query, case=False, na=False)
        ]
    
    if risk_filter != "All":
        filtered_df = filtered_df[filtered_df['risk_category'] == risk_filter]
    
    # Display conversation list
    if not filtered_df.empty:
        st.subheader(f"Found {filtered_df['conversation_id'].nunique()} conversations")
        
        # Get unique conversations
        unique_conversations = filtered_df['conversation_id'].unique()
        
        selected_conv = st.selectbox(
            "Select a conversation to view details:",
            unique_conversations
        )
        
        if selected_conv:
            conv_data = filtered_df[filtered_df['conversation_id'] == selected_conv].sort_values('conversation_turn')
            
            st.subheader(f"Conversation Details: {selected_conv[:8]}...")
            
            # Conversation summary
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            
            with summary_col1:
                st.metric("Total Turns", len(conv_data))
            
            with summary_col2:
                max_risk = conv_data['risk_probability'].max()
                st.metric("Max Risk Probability", f"{max_risk:.4f}")
            
            with summary_col3:
                tools_used = set()
                for tools_list in conv_data['tools_called_list']:
                    tools_used.update(tools_list)
                st.metric("Tools Used", len(tools_used))
            
            with summary_col4:
                ended = "Yes" if conv_data['conversation_ended'].any() else "No"
                st.metric("Conversation Ended", ended)
            
            # Display conversation turns
            st.subheader("Conversation Turns")
            
            for idx, row in conv_data.iterrows():
                with st.expander(f"Turn {row['conversation_turn']} - {row['created_at'].strftime('%Y-%m-%d %H:%M:%S')}"):
                    col_turn1, col_turn2 = st.columns(2)
                    
                    with col_turn1:
                        st.markdown("**👤 Human Message:**")
                        st.text(row['human_message'] if row['human_message'] else "(Welcome message)")
                    
                    with col_turn2:
                        st.markdown("**🤖 AI Response:**")
                        st.text(row['ai_message'])
                    
                    # Tools and agents
                    if row['tools_called_list']:
                        st.markdown(f"**🔧 Tools Called:** {', '.join(row['tools_called_list'])}")
                    
                    if row['agents_used_list']:
                        st.markdown(f"**🤖 Agents Used:** {', '.join(row['agents_used_list'])}")
                    
                    if row['risk_probability'] > 0.0:
                        st.markdown(f"**⚠️ Risk Probability:** {row['risk_probability']:.4f}")
                    
                    if row['conversation_ended']:
                        st.markdown("**🏁 Conversation Ended**")
    else:
        st.info("No conversations found matching the search criteria.")
    
    # ========== EXPORT OPTIONS ==========
    st.markdown("---")
    st.header("📥 Export Data")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        if st.button("📊 Download Summary Report"):
            # Create summary report
            summary_data = {
                'Metric': [
                    'Total Conversations',
                    'Total Turns',
                    'Avg Turns per Conversation',
                    'High Risk Conversations',
                    'Tool Usage Rate (%)',
                    'Unique Tools',
                    'Unique Agents',
                    'Avg Risk Probability',
                    'Max Risk Probability'
                ],
                'Value': [
                    stats['total_conversations'],
                    stats['total_turns'],
                    f"{stats['avg_turns_per_conversation']:.2f}",
                    stats['high_risk_conversations'],
                    f"{stats['tool_usage_rate']:.2f}",
                    stats['unique_tools'],
                    stats['unique_agents'],
                    f"{stats['avg_risk_probability']:.4f}",
                    f"{stats['max_risk_probability']:.4f}"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"chatbot_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with export_col2:
        if st.button("📋 Download Full Data"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"chatbot_full_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()