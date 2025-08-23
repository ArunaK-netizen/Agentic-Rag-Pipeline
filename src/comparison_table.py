"""Generate comparison tables for different strategies and databases."""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
import time

class ComparisonAnalyzer:
    """Analyzes and compares different RAG strategies and databases."""
    
    def __init__(self):
        self.comparison_data = []
        self.performance_metrics = {}
    
    def add_experiment(self, experiment_data: Dict[str, Any]):
        """Add experiment results for comparison."""
        self.comparison_data.append(experiment_data)
    
    def generate_strategy_comparison_table(self) -> pd.DataFrame:
        """Generate comparison table for different strategies."""
        if not self.comparison_data:
            return pd.DataFrame()
        
        comparison_rows = []
        
        for exp in self.comparison_data:
            row = {
                "Vector Database": exp.get("vector_db_type", "Unknown"),
                "Chunking Strategy": exp.get("chunking_strategy", "Unknown"),
                "Search Strategy": exp.get("search_strategy", "Unknown"),
                "Processing Time (s)": exp.get("processing_time", 0),
                "Indexing Time (s)": exp.get("indexing_time", 0),
                "Search Time (s)": exp.get("search_time", 0),
                "Total Chunks": exp.get("num_chunks", 0),
                "Avg Chunk Length": exp.get("avg_chunk_length", 0),
                "Search Results": exp.get("num_results", 0),
                "Avg Score": exp.get("avg_score", 0),
                "Memory Usage": exp.get("memory_usage", "N/A")
            }
            comparison_rows.append(row)
        
        return pd.DataFrame(comparison_rows)
    
    def generate_database_comparison(self) -> pd.DataFrame:
        """Generate detailed comparison of vector databases."""
        db_comparisons = []
        
        db_features = {
            "Qdrant": {
                "Type": "Server",
                "Persistence": "Yes",
                "Scalability": "High",
                "Ease of Setup": "Medium",
                "Cost": "Free/Paid",
                "Vector Similarity": "Cosine, Dot, Euclidean",
                "Filtering": "Advanced",
                "Updates": "Yes"
            },
            "Pinecone": {
                "Type": "Cloud",
                "Persistence": "Yes",
                "Scalability": "Very High",
                "Ease of Setup": "High",
                "Cost": "Paid",
                "Vector Similarity": "Cosine, Euclidean, Dot",
                "Filtering": "Metadata",
                "Updates": "Yes"
            }
        }
        
        for db_name, features in db_features.items():
            row = {"Database": db_name, **features}
            db_comparisons.append(row)
        
        return pd.DataFrame(db_comparisons)
    
    def generate_chunking_comparison(self) -> pd.DataFrame:
        """Generate comparison of chunking strategies."""
        chunking_comparisons = [
            {
                "Strategy": "Fixed Size",
                "Description": "Split into equal-sized chunks",
                "Pros": "Simple, consistent chunk sizes",
                "Cons": "May break semantic boundaries",
                "Best For": "Uniform processing, simple documents",
                "Complexity": "Low"
            },
            {
                "Strategy": "Recursive Character",
                "Description": "Split by multiple separators recursively",
                "Pros": "Respects document structure",
                "Cons": "Variable chunk sizes",
                "Best For": "Structured documents",
                "Complexity": "Medium"
            },
            {
                "Strategy": "Semantic",
                "Description": "Split based on semantic similarity",
                "Pros": "Maintains semantic coherence",
                "Cons": "Computationally expensive",
                "Best For": "High-quality retrieval",
                "Complexity": "High"
            },
            {
                "Strategy": "Sentence-based",
                "Description": "Split by sentences with size limits",
                "Pros": "Natural language boundaries",
                "Cons": "May not respect document structure",
                "Best For": "Question-answering systems",
                "Complexity": "Medium"
            }
        ]
        
        return pd.DataFrame(chunking_comparisons)
    
    def generate_search_comparison(self) -> pd.DataFrame:
        """Generate comparison of search strategies."""
        search_comparisons = [
            {
                "Strategy": "Vector Search",
                "Description": "Pure semantic similarity search",
                "Pros": "Fast, good semantic understanding",
                "Cons": "May miss exact keyword matches",
                "Best For": "Semantic similarity queries",
                "Latency": "Low"
            },
            {
                "Strategy": "Semantic Search",
                "Description": "Enhanced vector search with re-ranking",
                "Pros": "Better relevance, handles context",
                "Cons": "Slower than pure vector search",
                "Best For": "Complex queries requiring nuance",
                "Latency": "Medium"
            },
            {
                "Strategy": "Hybrid Search",
                "Description": "Combines keyword and vector search",
                "Pros": "Best of both worlds, flexible",
                "Cons": "Complex tuning, slower",
                "Best For": "Mixed query types",
                "Latency": "High"
            }
        ]
        
        return pd.DataFrame(search_comparisons)
    
    def create_performance_chart(self) -> go.Figure:
        """Create performance comparison chart."""
        if not self.comparison_data:
            return go.Figure()
        
        df = self.generate_strategy_comparison_table()
        
        fig = go.Figure()
        
        # Processing time comparison
        fig.add_trace(go.Bar(
            name='Processing Time',
            x=df['Vector Database'],
            y=df['Processing Time (s)'],
            yaxis='y'
        ))
        
        # Indexing time comparison
        fig.add_trace(go.Bar(
            name='Indexing Time',
            x=df['Vector Database'],
            y=df['Indexing Time (s)'],
            yaxis='y'
        ))
        
        # Search time comparison
        fig.add_trace(go.Bar(
            name='Search Time',
            x=df['Vector Database'],
            y=df['Search Time (s)'],
            yaxis='y'
        ))
        
        fig.update_layout(
            title='Performance Comparison Across Vector Databases',
            xaxis_title='Vector Database',
            yaxis_title='Time (seconds)',
            barmode='group'
        )
        
        return fig
    
    def create_accuracy_chart(self) -> go.Figure:
        """Create accuracy comparison chart."""
        if not self.comparison_data:
            return go.Figure()
        
        df = self.generate_strategy_comparison_table()
        
        fig = px.scatter(
            df, 
            x='Search Strategy', 
            y='Avg Score',
            size='Search Results',
            color='Vector Database',
            title='Search Quality Comparison',
            hover_data=['Chunking Strategy']
        )
        
        return fig
    
    def export_comparison_report(self, filename: str = "rag_comparison_report"):
        """Export comprehensive comparison report."""
        try:
            with pd.ExcelWriter(f"{filename}.xlsx", engine='openpyxl') as writer:
                # Strategy comparison
                strategy_df = self.generate_strategy_comparison_table()
                strategy_df.to_excel(writer, sheet_name='Strategy_Comparison', index=False)
                
                # Database comparison
                db_df = self.generate_database_comparison()
                db_df.to_excel(writer, sheet_name='Database_Features', index=False)
                
                # Chunking comparison
                chunk_df = self.generate_chunking_comparison()
                chunk_df.to_excel(writer, sheet_name='Chunking_Strategies', index=False)
                
                # Search comparison
                search_df = self.generate_search_comparison()
                search_df.to_excel(writer, sheet_name='Search_Strategies', index=False)
            
            return f"Report exported to {filename}.xlsx"
        
        except Exception as e:
            return f"Error exporting report: {e}"