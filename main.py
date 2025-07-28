from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import asyncio
import json
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from groq import Groq
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseOutputParser
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

# Initialize FastAPI app
app = FastAPI(title="AI Analytics Dashboard", description="Advanced AI-Powered Analytics Platform")

# Templates setup
templates = Jinja2Templates(directory="templates")

# Initialize Groq client and LangChain
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
# groq_client = Groq(api_key=GROQ_API_KEY)  # Disabled due to proxies argument error
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192",
    temperature=0.3
)

# Memory for conversation history
memory = ConversationBufferMemory(return_messages=True)

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

# Initialize sample data
generate_sample_data()

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
        except:
            return {"analysis": text, "insights": [], "recommendations": []}

async def analyze_with_ai(query: str, data_context: Dict) -> Dict[str, Any]:
    """Use Groq LLaMA to analyze data and provide insights"""
    
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
        return {
            "analysis": f"AI analysis temporarily unavailable. Error: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.0
        }

async def generate_predictions(metric: str, historical_data: List[Dict]) -> Dict[str, Any]:
    """Generate AI-powered predictions"""
    
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
        return {
            "predictions": [],
            "analysis": f"Prediction analysis temporarily unavailable: {str(e)}",
            "model_info": "Error in prediction model"
        }

# API Routes
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main dashboard"""
    with open("templates/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/dashboard-data")
async def get_dashboard_data():
    """Get comprehensive dashboard data"""
    return {
        "status": "success",
        "data": dashboard_data,
        "timestamp": datetime.now().isoformat()
    }

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
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/charts/revenue-trend")
async def revenue_trend_chart():
    """Generate revenue trend chart data"""
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

@app.get("/api/charts/campaign-performance")
async def campaign_performance_chart():
    """Generate campaign performance chart"""
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

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ai_model": "llama3-70b-8192",
        "version": "1.0.0"
    }

# Background tasks for real-time updates
async def update_metrics():
    """Simulate real-time metric updates"""
    while True:
        await asyncio.sleep(30)  # Update every 30 seconds
        
        # Update key metrics with slight variations
        dashboard_data["metrics"]["total_revenue"] += random.randint(-1000, 5000)
        dashboard_data["metrics"]["traffic"] += random.randint(-100, 500)
        dashboard_data["metrics"]["conversion_rate"] = round(
            dashboard_data["metrics"]["conversion_rate"] + random.uniform(-0.1, 0.1), 2
        )

@app.on_event("startup")
async def startup_event():
    """Initialize background tasks"""
    asyncio.create_task(update_metrics())

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )