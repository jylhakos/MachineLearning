#!/bin/bash

# Start guide for Apache Airflow ML pipeline
# ================================================

echo "🚀 Apache Airflow ML Pipeline - Quick Start"
echo "==========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "❌ Virtual environment not found. Please run setup.sh first:"
    echo "   ./setup.sh"
    exit 1
fi

echo "✅ Found virtual environment"

# Activate virtual environment
echo ""
echo "� Activating virtual environment..."
source venv/bin/activate

# Set Airflow home
export AIRFLOW_HOME=$(pwd)/airflow_home
echo "🏠 AIRFLOW_HOME set to: $AIRFLOW_HOME"

# Check if Airflow is configured
if [ ! -f "$AIRFLOW_HOME/airflow.cfg" ]; then
    echo ""
    echo "❌ Airflow not configured. Please run setup.sh first:"
    echo "   ./setup.sh"
    exit 1
fi

# Check if packages are installed
echo ""
echo "🔍 Checking if required packages are installed..."
python3 -c "
try:
    import pandas, sklearn, numpy, matplotlib, airflow
    print('✅ All packages are already installed')
    installed = True
except ImportError as e:
    print('📥 Some packages missing:', str(e))
    installed = False
" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "📥 Installing required packages..."
    pip install -q -r requirements.txt
    
    if [ $? -eq 0 ]; then
        echo "✅ Packages installed successfully"
    else
        echo "❌ Failed to install packages"
        exit 1
    fi
fi

echo ""
echo ""
echo "🎉 Setup complete! Choose what to run:"
echo ""
echo "1️⃣  Run enhanced fish weight prediction pipeline:"
echo "   python3 fish_supervised_pipeline.py"
echo ""
echo "2️⃣  Run original fish regression analysis:"
echo "   python3 fish_regression.py"
echo ""
echo "3️⃣  Run comprehensive fish analysis:"
echo "   python3 fish_analysis.py"
echo ""
echo "4️⃣  Start Airflow webserver:"
echo "   airflow webserver --port 8080 --daemon"
echo ""
echo "5️⃣  Start Airflow scheduler:"
echo "   airflow scheduler --daemon"
echo ""
echo "6️⃣  Start both Airflow services (webserver + scheduler):"
echo "   Start both in background for fish weight prediction pipeline"
echo ""
echo "7️⃣  Run with Docker:"
echo "   docker-compose up -d"
echo ""
echo "8️⃣  View Airflow Web UI:"
echo "   Open http://localhost:8080"
echo ""
echo "9️⃣  Monitor fish pipeline:"
echo "   python3 monitor_pipeline.py"
echo ""

# Interactive menu
echo "Enter your choice (1-9), or press Enter to see project info:"
read -r choice

case $choice in
    1)
        echo "🐟 Running enhanced fish weight prediction pipeline..."
        python3 fish_supervised_pipeline.py
        ;;
    2)
        echo "🎯 Running original fish regression analysis..."
        python3 fish_regression.py
        ;;
    3)
        echo "📊 Running comprehensive fish analysis..."
        python3 fish_analysis.py
        ;;
    4)
        echo "🌐 Starting Airflow webserver..."
        airflow webserver --port 8080 --daemon
        echo "✅ Airflow webserver started on http://localhost:8080"
        echo "   Username: admin"
        echo "   Password: admin"
        echo "   Fish weight prediction DAG: fish_weight_prediction_pipeline"
        ;;
    5)
        echo "⏰ Starting Airflow scheduler..."
        airflow scheduler --daemon
        echo "✅ Airflow scheduler started"
        ;;
    6)
        echo "🚀 Starting both Airflow webserver and scheduler for fish pipeline..."
        airflow webserver --port 8080 --daemon
        sleep 5
        airflow scheduler --daemon
        echo "✅ Airflow services started!"
        echo "🌐 Web UI: http://localhost:8080"
        echo "   Username: admin"
        echo "   Password: admin"
        echo "   Fish DAG: fish_weight_prediction_pipeline"
        ;;
    7)
        echo "🐳 Starting with Docker..."
        if command -v docker-compose &> /dev/null; then
            docker-compose up -d
            echo "✅ Docker services started!"
            echo "🌐 Web UI: http://localhost:8080"
            echo "   Username: admin"
            echo "   Password: admin"
        else
            echo "❌ Docker Compose not found. Please install Docker and Docker Compose."
        fi
        ;;
    8)
        echo "🌐 Opening Airflow Web UI..."
        if command -v xdg-open &> /dev/null; then
            xdg-open http://localhost:8080
        elif command -v open &> /dev/null; then
            open http://localhost:8080
        else
            echo "Please open http://localhost:8080 in your browser"
        fi
        ;;
    9)
        echo "📊 Running fish pipeline monitoring..."
        python3 monitor_pipeline.py
        ;;
    *)
        echo ""
        echo "📋 Fish Weight Prediction Pipeline Project Information:"
        echo "====================================================="
        echo "📁 Project structure:"
        find . -type f -name "*.py" -o -name "*.md" -o -name "*.txt" -o -name "*.csv" -o -name "*.sh" -o -name "*.yml" -o -name "Dockerfile" | grep -v "__pycache__" | grep -v ".git" | sort
        echo ""
        echo "🐟 Available fish analysis scripts:"
        echo "   - fish_supervised_pipeline.py: Enhanced fish weight prediction"
        echo "   - fish_regression.py: Original fish regression analysis"
        echo "   - fish_analysis.py: Comprehensive fish analysis"
        echo "   - fish_classification.py: Fish species classification"
        echo ""
        echo "📖 Read the documentation:"
        echo "   cat README.md"
        echo ""
        echo "🔄 To run this quick start again:"
        echo "   ./quickstart.sh"
        echo ""
        echo "🛑 To stop Airflow services:"
        echo "   pkill -f airflow"
        echo ""
        echo "🐳 To stop Docker services:"
        echo "   docker-compose down"
        ;;
esac

echo ""
echo "💡 Useful fish analysis commands:"
echo "   - Check fish dataset: python3 verify_dataset.py"
echo "   - Check Airflow processes: ps aux | grep airflow"
echo "   - Stop Airflow: pkill -f airflow"
echo "   - View logs: tail -f $AIRFLOW_HOME/logs/scheduler/latest/*.log"
echo "   - Monitor fish pipeline: python3 monitor_pipeline.py"
echo "📚 For detailed instructions, see README.md"

# Interactive menu
echo "Enter your choice (1-6), or press Enter to see project info:"
read -r choice

case $choice in
    1)
        echo "🎯 Running supervised learning pipeline..."
        python3 supervised_regression_pipeline.py
        ;;
    2)
        echo "🌐 Starting Airflow webserver..."
        airflow webserver --port 8080 --daemon
        echo "✅ Airflow webserver started on http://localhost:8080"
        echo "   Username: admin"
        echo "   Password: admin"
        ;;
    3)
        echo "⏰ Starting Airflow scheduler..."
        airflow scheduler --daemon
        echo "✅ Airflow scheduler started"
        ;;
    4)
        echo "� Starting both Airflow webserver and scheduler..."
        airflow webserver --port 8080 --daemon
        sleep 5
        airflow scheduler --daemon
        echo "✅ Airflow services started!"
        echo "🌐 Web UI: http://localhost:8080"
        echo "   Username: admin"
        echo "   Password: admin"
        ;;
    5)
        echo "� Starting with Docker..."
        if command -v docker-compose &> /dev/null; then
            docker-compose up -d
            echo "✅ Docker services started!"
            echo "🌐 Web UI: http://localhost:8080"
            echo "   Username: admin"
            echo "   Password: admin"
        else
            echo "❌ Docker Compose not found. Please install Docker and Docker Compose."
        fi
        ;;
    6)
        echo "🌐 Opening Airflow Web UI..."
        if command -v xdg-open &> /dev/null; then
            xdg-open http://localhost:8080
        elif command -v open &> /dev/null; then
            open http://localhost:8080
        else
            echo "Please open http://localhost:8080 in your browser"
        fi
        ;;
    *)
        echo ""
        echo "📋 Apache Airflow ML Pipeline Project Information:"
        echo "================================================="
        echo "📁 Project structure:"
        find . -type f -name "*.py" -o -name "*.md" -o -name "*.txt" -o -name "*.csv" -o -name "*.sh" -o -name "*.yml" -o -name "Dockerfile" | grep -v "__pycache__" | grep -v ".git" | sort
        echo ""
        echo "📖 Read the documentation:"
        echo "   cat README.md"
        echo ""
        echo "🔄 To run this quick start again:"
        echo "   ./quickstart.sh"
        echo ""
        echo "🛑 To stop Airflow services:"
        echo "   pkill -f airflow"
        echo ""
        echo "🐳 To stop Docker services:"
        echo "   docker-compose down"
        ;;
esac

echo ""
echo "💡 Useful commands:"
echo "   - Check Airflow processes: ps aux | grep airflow"
echo "   - Stop Airflow: pkill -f airflow"
echo "   - View logs: tail -f $AIRFLOW_HOME/logs/scheduler/latest/*.log"
echo "📚 For detailed instructions, see README.md"
