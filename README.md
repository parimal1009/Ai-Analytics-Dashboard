![Dashboard Screenshot](https://github.com/parimal1009/Ai-Analytics-Dashboard/blob/main/images/Screenshot%202025-07-28%20163110.png?raw=true)

# AI Analytics Dashboard

![AI Chat Screenshot](https://github.com/parimal1009/Ai-Analytics-Dashboard/blob/main/images/Screenshot%202025-07-28%20163119.png?raw=true)

## Overview

**AI Analytics Dashboard** is a modern, AI-powered analytics platform that leverages LLaMA 3 and Groq for advanced business intelligence, campaign analysis, and predictive insights. The dashboard provides real-time metrics, interactive charts, campaign management, and an integrated AI assistant for natural language analytics.

## Features

- 📊 **Real-Time Metrics**: Visualize key business metrics like revenue, campaigns, conversion rate, and traffic.
- 📈 **Interactive Charts**: Revenue trends, user segments, and campaign performance with Plotly.js.
- 🤖 **AI Assistant**: Chat with an LLaMA 3-powered assistant for data analysis, insights, and recommendations.
- 🔮 **AI Predictions**: Generate and visualize 7-day forecasts for revenue, traffic, and campaign performance.
- 🗂️ **Campaign Management**: View, analyze, and optimize marketing campaigns.
- ✨ **Modern UI**: Responsive, glassmorphic design with smooth animations and dark mode.

## Screenshots

| Dashboard | AI Chat Assistant |
|-----------|------------------|
| ![Dashboard](https://github.com/parimal1009/Ai-Analytics-Dashboard/blob/main/images/Screenshot%202025-07-28%20163110.png?raw=true) | ![AI Chat](https://github.com/parimal1009/Ai-Analytics-Dashboard/blob/main/images/Screenshot%202025-07-28%20163119.png?raw=true) |

## Getting Started

### Prerequisites
- Python 3.10+
- [pip](https://pip.pypa.io/en/stable/)

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/parimal1009/Ai-Analytics-Dashboard.git
   cd Ai-Analytics-Dashboard
   ```
2. **Install dependencies:**
   ```bash
   pip install --force-reinstall --no-cache-dir -r requirements.txt
   ```
3. **Set your Groq API key:**
   - Open `main.py` and replace `your_groq_api_key_here` with your actual Groq API key.

### Running the App
```bash
python main.py
```
- The dashboard will be available at [http://localhost:8000](http://localhost:8000)

## Deployment on Render.com

You can easily deploy this project as a web service on [Render](https://render.com/):

### 1. Create a New Web Service
- Go to [Render Dashboard](https://dashboard.render.com/) and click **New + > Web Service**.
- Connect your GitHub repository.

### 2. Configure the Service
- **Environment:** Python 3.10 or higher
- **Build Command:**
  ```sh
  pip install --no-cache-dir -r requirements.txt
  ```
- **Start Command:**
  ```sh
  uvicorn main:app --host 0.0.0.0 --port 10000
  ```
  (Or use `python main.py` if you want auto-reload, but `uvicorn` is recommended for production.)
- **Instance Type:** Starter or higher

### 3. Environment Variables
- Add your Groq API key as an environment variable:
  - Key: `GROQ_API_KEY`
  - Value: `your_actual_groq_api_key`
- (Optional) Add any other secrets or config as needed.

### 4. Expose the Web Service
- Set the port to **10000** (or change the `uvicorn` command to use Render's default port `$PORT`):
  ```sh
  uvicorn main:app --host 0.0.0.0 --port $PORT
  ```
- Render will automatically assign a public URL to your app.

### 5. Deploy
- Click **Create Web Service** and wait for the build and deploy to finish.
- Visit your Render URL to access the dashboard.

---

## Usage
- View real-time business metrics and trends.
- Use the AI chat assistant to ask questions about your data, get insights, and generate predictions.
- Manage and analyze marketing campaigns.

## Tech Stack
- **Backend:** FastAPI, LangChain, Groq, Python
- **Frontend:** HTML, CSS (Glassmorphism), JavaScript, Plotly.js, Chart.js
- **AI/ML:** LLaMA 3, Groq, LangChain

## License
This project is licensed under the MIT License.
