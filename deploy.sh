#!/bin/bash

# AI Analytics Dashboard - Deployment Setup Script
echo "🚀 Setting up AI Analytics Dashboard for Render deployment..."

# Create project structure
echo "📁 Creating project structure..."
mkdir -p templates
mkdir -p static

# Copy HTML file to templates if it exists
if [ -f "paste.txt" ]; then
    echo "📄 Moving HTML file to templates directory..."
    cp paste.txt templates/index.html
    echo "✅ HTML file copied to templates/index.html"
elif [ -f "index.html" ]; then
    echo "📄 Moving HTML file to templates directory..."
    cp index.html templates/
    echo "✅ HTML file copied to templates/"
else
    echo "⚠️  No HTML file found (paste.txt or index.html)"
fi

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo "📝 Creating .gitignore..."
    cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment variables
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Database
*.db
*.sqlite3

# Temporary files
tmp/
temp/
EOF
    echo "✅ .gitignore created"
fi

# Create README if it doesn't exist
if [ ! -f "README.md" ]; then
    echo "📝 Creating README.md..."
    cat > README.md << EOF
# AI Analytics Dashboard

🚀 Advanced AI-Powered Analytics Platform built with FastAPI and LLaMA 3

## Features

- 📊 Real-time analytics dashboard
- 🤖 AI-powered insights using Groq LLaMA 3
- 📈 Interactive data visualizations
- 🔮 Predictive analytics
- 💬 AI chat assistant
- 📱 Responsive design

## Deployment

This project is ready for deployment on Render. See the deployment guide for detailed instructions.

### Environment Variables

- \`GROQ_API_KEY\` - Your Groq API key (required for AI features)
- \`ENVIRONMENT\` - Set to "production" for production deployment
- \`LANGSMITH_API_KEY\` - Optional, for LangSmith monitoring
- \`LANGSMITH_PROJECT\` - Project name for LangSmith

### Local Development

1. Install dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

2. Set environment variables:
   \`\`\`bash
   export GROQ_API_KEY="your_api_key_here"
   \`\`\`

3. Run the application:
   \`\`\`bash
   python main.py
   \`\`\`

4. Open http://localhost:8000 in your browser

## API Endpoints

- \`/\` - Main dashboard
- \`/api/dashboard-data\` - Get dashboard data
- \`/api/ai-analysis\` - AI analysis endpoint
- \`/api/predictions\` - Generate predictions
- \`/health\` - Health check
- \`/docs\` - API documentation (development only)

## License

MIT License - see LICENSE file for details
EOF
    echo "✅ README.md created"
fi

# Validate Python files
echo "🔍 Validating Python syntax..."
if python -m py_compile main.py; then
    echo "✅ main.py syntax is valid"
else
    echo "❌ main.py has syntax errors"
    exit 1
fi

# Check if all required files exist
echo "📋 Checking required files..."
files=("main.py" "requirements.txt" "runtime.txt")
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file is missing"
        exit 1
    fi
done

# Display next steps
echo ""
echo "🎉 Setup complete! Next steps:"
echo ""
echo "1. Push your code to GitHub:"
echo "   git add ."
echo "   git commit -m 'Initial commit for Render deployment'"
echo "   git push origin main"
echo ""
echo "2. Go to https://dashboard.render.com"
echo "3. Click 'New +' → 'Web Service'"
echo "4. Connect your GitHub repository"
echo "5. Configure the service:"
echo "   - Name: ai-analytics-dashboard"
echo "   - Environment: Python 3"
echo "   - Build Command: pip install -r requirements.txt"
echo "   - Start Command: uvicorn main:app --host 0.0.0.0 --port \$PORT --workers 1"
echo ""
echo "6. Set environment variables:"
echo "   - ENVIRONMENT=production"
echo "   - GROQ_API_KEY=your_actual_groq_api_key"
echo ""
echo "7. Deploy and enjoy your AI Analytics Dashboard! 🚀"
echo ""
echo "Need help? Check the deployment guide for detailed instructions."