from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import asyncio
import json
import random
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from groq import Groq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseOutputParser
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Analytics Dashboard", 
    description="Advanced AI-Powered Analytics Platform",
    version="1.0.0",
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables with defaults
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "default")
HF_TOKEN = os.getenv("HF_TOKEN")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Check for required environment variables
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not set - AI features will be limited")

# Initialize AI components with error handling
llm = None
memory = ConversationBufferMemory(return_messages=True)

try:
    if GROQ_API_KEY:
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama3-70b-8192",
            temperature=0.3
        )
        logger.info("AI components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize AI components: {str(e)}")

# Pydantic models
class AnalyticsQuery(BaseModel):
    query: str
    dashboard_type: Optional[str] = "general"

class DataUpload(BaseModel):
    data: List[Dict[str, Any]]
    data_type: str

class InsightRequest(BaseModel):
    metric: str
    time_range: str
    filters: Optional[Dict[str, Any]] = {}

# Global data storage (in production, use a proper database)
dashboard_data = {
    "metrics": {
        "total_revenue": 1250000,
        "active_campaigns": 45,
        "conversion_rate": 3.4,
        "customer_acquisition_cost": 125,
        "lifetime_value": 2500,
        "roi": 340,
        "traffic": 125000,
        "bounce_rate": 2.3
    },
    "time_series": [],
    "campaigns": [],
    "user_segments": [],
    "predictions": {}
}

# Generate realistic sample data
def generate_sample_data():
    """Generate comprehensive sample data for the dashboard"""
    
    try:
        # Time series data for the last 30 days
        dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30, 0, -1)]
        
        time_series_data = []
        base_revenue = 40000
        base_traffic = 4000
        
        for i, date in enumerate(dates):
            # Add some realistic variance and trends
            revenue_trend = base_revenue + (i * 500) + random.randint(-5000, 8000)
            traffic_trend = base_traffic + (i * 100) + random.randint(-500, 1000)
            
            time_series_data.append({
                "date": date,
                "revenue": max(0, revenue_trend),
                "traffic": max(0, traffic_trend),
                "conversions": random.randint(50, 200),
                "impressions": random.randint(10000, 50000),
                "clicks": random.randint(500, 2000),
                "cost": random.randint(5000, 15000)
            })
        
        # Campaign data
        campaigns = [
            {
                "id": 1,
                "name": "Summer Sale 2024",
                "status": "active",
                "budget": 50000,
                "spent": 32000,
                "impressions": 1200000,
                "clicks": 24000,
                "conversions": 480,
                "ctr": 2.0,
                "conversion_rate": 2.0,
                "roi": 250
            },
            {
                "id": 2,
                "name": "Brand Awareness Q3",
                "status": "active",
                "budget": 75000,
                "spent": 68000,
                "impressions": 2500000,
                "clicks": 45000,
                "conversions": 900,
                "ctr": 1.8,
                "conversion_rate": 2.0,
                "roi": 180
            },
            {
                "id": 3,
                "name": "Holiday Special",
                "status": "paused",
                "budget": 30000,
                "spent": 25000,
                "impressions": 800000,
                "clicks": 16000,
                "conversions": 320,
                "ctr": 2.0,
                "conversion_rate": 2.0,
                "roi": 160
            }
        ]
        
        # User segments
        user_segments = [
            {"segment": "Premium Users", "count": 15000, "revenue": 750000, "avg_order": 250},
            {"segment": "Regular Users", "count": 35000, "revenue": 420000, "avg_order": 120},
            {"segment": "New Users", "count": 8000, "revenue": 80000, "avg_order": 100},
            {"segment": "Churned Users", "count": 2000, "revenue": 0, "avg_order": 0}
        ]
        
        dashboard_data["time_series"] = time_series_data
        dashboard_data["campaigns"] = campaigns
        dashboard_data["user_segments"] = user_segments
        
        logger.info("Sample data generated successfully")
    except Exception as e:
        logger.error(f"Failed to generate sample data: {str(e)}")

# AI Analysis Functions
class AnalyticsOutputParser(BaseOutputParser):
    def parse(self, text: str) -> Dict[str, Any]:
        try:
            # Try to extract JSON from the response
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            else:
                return {"analysis": text, "insights": [], "recommendations": []}
        except Exception as e:
            logger.error(f"Failed to parse AI response: {str(e)}")
            return {"analysis": text, "insights": [], "recommendations": []}

async def analyze_with_ai(query: str, data_context: Dict) -> Dict[str, Any]:
    """Use Groq LLaMA to analyze data and provide insights"""
    
    if not llm:
        return {
            "analysis": "AI analysis is currently unavailable. Please check your API configuration.",
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.0
        }
    
    # Create analysis prompt
    analysis_prompt = PromptTemplate(
        input_variables=["query", "data_context"],
        template="""
        You are an expert data analyst and business intelligence specialist. Analyze the following data and query:
        
        Query: {query}
        
        Data Context: {data_context}
        
        Provide a comprehensive analysis including:
        1. Key insights from the data
        2. Trends and patterns
        3. Actionable recommendations
        4. Potential risks or opportunities
        5. Performance metrics interpretation
        
        Format your response as a detailed analysis with specific insights and recommendations.
        Focus on actionable business intelligence that can drive decision-making.
        """
    )
    
    try:
        chain = LLMChain(llm=llm, prompt=analysis_prompt, memory=memory)
        response = await asyncio.to_thread(
            chain.run,
            query=query,
            data_context=json.dumps(data_context, indent=2)
        )
        
        return {
            "analysis": response,
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.85
        }
    except Exception as e:
        logger.error(f"AI analysis failed: {str(e)}")
        return {
            "analysis": f"AI analysis temporarily unavailable. Error: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.0
        }

async def generate_predictions(metric: str, historical_data: List[Dict]) -> Dict[str, Any]:
    """Generate AI-powered predictions"""
    
    if not llm:
        # Fallback to simple predictions without AI
        predictions = []
        last_value = historical_data[-1].get(metric, 0) if historical_data else 0
        
        for i in range(7):
            trend = random.uniform(0.95, 1.08)
            predicted_value = last_value * trend
            predictions.append({
                "date": (datetime.now() + timedelta(days=i+1)).strftime("%Y-%m-%d"),
                "predicted_value": round(predicted_value, 2),
                "confidence": random.uniform(0.7, 0.9)
            })
            last_value = predicted_value
        
        return {
            "predictions": predictions,
            "analysis": "Predictions generated using statistical models (AI unavailable)",
            "model_info": "Statistical Prediction Model"
        }
    
    prediction_prompt = PromptTemplate(
        input_variables=["metric", "historical_data"],
        template="""
        Based on the historical data for {metric}, provide predictions for the next 7 days.
        
        Historical Data: {historical_data}
        
        Analyze trends, seasonality, and patterns to make accurate predictions.
        Consider business factors that might affect future performance.
        
        Provide:
        1. 7-day forecast values
        2. Confidence intervals
        3. Key factors influencing the predictions
        4. Risk assessment
        """
    )
    
    try:
        chain = LLMChain(llm=llm, prompt=prediction_prompt)
        response = await asyncio.to_thread(
            chain.run,
            metric=metric,
            historical_data=json.dumps(historical_data[-14:], indent=2)  # Last 14 days
        )
        
        # Generate actual prediction values (simplified for demo)
        predictions = []
        last_value = historical_data[-1].get(metric, 0) if historical_data else 0
        
        for i in range(7):
            # Simple trend-based prediction with some randomness
            trend = random.uniform(0.95, 1.08)
            predicted_value = last_value * trend
            predictions.append({
                "date": (datetime.now() + timedelta(days=i+1)).strftime("%Y-%m-%d"),
                "predicted_value": round(predicted_value, 2),
                "confidence": random.uniform(0.7, 0.9)
            })
            last_value = predicted_value
        
        return {
            "predictions": predictions,
            "analysis": response,
            "model_info": "LLaMA3-70B Enhanced Prediction Model"
        }
    except Exception as e:
        logger.error(f"Prediction generation failed: {str(e)}")
        return {
            "predictions": [],
            "analysis": f"Prediction analysis temporarily unavailable: {str(e)}",
            "model_info": "Error in prediction model"
        }

# API Routes
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main dashboard"""
    try:
        # Try to read from templates directory first
        html_path = os.path.join("templates", "index.html")
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        
        # Fallback to current directory
        if os.path.exists("index.html"):
            with open("index.html", "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        
        # If no HTML file found, return a basic response
        return HTMLResponse(content="""
        <html>
            <head><title>AI Analytics Dashboard</title></head>
            <body>
                <h1>AI Analytics Dashboard</h1>
                <p>Dashboard is running! Please ensure index.html is in the templates/ directory.</p>
                <p><a href="/docs">View API Documentation</a></p>
            </body>
        </html>
        """)
    except Exception as e:
        logger.error(f"Failed to serve dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load dashboard: {str(e)}")

@app.get("/api/dashboard-data")
async def get_dashboard_data():
    """Get comprehensive dashboard data"""
    try:
        return {
            "status": "success",
            "data": dashboard_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")

@app.post("/api/ai-analysis")
async def ai_analysis(query: AnalyticsQuery):
    """Get AI-powered analysis of data"""
    try:
        analysis = await analyze_with_ai(query.query, dashboard_data)
        return {
            "status": "success",
            "analysis": analysis,
            "query": query.query
        }
    except Exception as e:
        logger.error(f"AI analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/predictions")
async def get_predictions(request: InsightRequest):
    """Generate AI predictions for specific metrics"""
    try:
        predictions = await generate_predictions(request.metric, dashboard_data["time_series"])
        dashboard_data["predictions"][request.metric] = predictions
        return {
            "status": "success",
            "predictions": predictions
        }
    except Exception as e:
        logger.error(f"Prediction generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/charts/revenue-trend")
async def revenue_trend_chart():
    """Generate revenue trend chart data"""
    try:
        df = pd.DataFrame(dashboard_data["time_series"])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['revenue'],
            mode='lines+markers',
            name='Revenue',
            line=dict(color='#00D2FF', width=3),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="Revenue Trend (Last 30 Days)",
            xaxis_title="Date",
            yaxis_title="Revenue ($)",
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
    except Exception as e:
        logger.error(f"Failed to generate revenue chart: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {str(e)}")

@app.get("/api/charts/campaign-performance")
async def campaign_performance_chart():
    """Generate campaign performance chart"""
    try:
        campaigns = dashboard_data["campaigns"]
        
        fig = go.Figure(data=[
            go.Bar(
                name='Budget',
                x=[c['name'] for c in campaigns],
                y=[c['budget'] for c in campaigns],
                marker_color='rgba(55, 128, 191, 0.7)'
            ),
            go.Bar(
                name='Spent',
                x=[c['name'] for c in campaigns],
                y=[c['spent'] for c in campaigns],
                marker_color='rgba(219, 64, 82, 0.7)'
            )
        ])
        
        fig.update_layout(
            title="Campaign Budget vs Spend",
            xaxis_title="Campaigns",
            yaxis_title="Amount ($)",
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
    except Exception as e:
        logger.error(f"Failed to generate campaign chart: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint for Render"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ai_model": "llama3-70b-8192" if llm else "unavailable",
        "version": "1.0.0",
        "environment": ENVIRONMENT
    }

@app.get("/api/health")
async def api_health_check():
    """API Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ai_model": "llama3-70b-8192" if llm else "unavailable",
        "version": "1.0.0"
    }

# Background tasks for real-time updates
async def update_metrics():
    """Simulate real-time metric updates"""
    while True:
        try:
            await asyncio.sleep(30)  # Update every 30 seconds
            
            # Update key metrics with slight variations
            dashboard_data["metrics"]["total_revenue"] += random.randint(-1000, 5000)
            dashboard_data["metrics"]["traffic"] += random.randint(-100, 500)
            dashboard_data["metrics"]["conversion_rate"] = round(
                dashboard_data["metrics"]["conversion_rate"] + random.uniform(-0.1, 0.1), 2
            )
            
            logger.info("Metrics updated successfully")
        except Exception as e:
            logger.error(f"Failed to update metrics: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting AI Analytics Dashboard...")
    generate_sample_data()
    if ENVIRONMENT != "production":
        asyncio.create_task(update_metrics())
    logger.info("AI Analytics Dashboard started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on application shutdown"""
    logger.info("Shutting down AI Analytics Dashboard...")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False  # Set to False for production
    )