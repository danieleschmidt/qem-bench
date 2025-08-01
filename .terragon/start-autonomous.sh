#!/bin/bash
# Terragon Autonomous SDLC Startup Script
# Launches perpetual value discovery and execution for QEM-Bench

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🤖 Starting Terragon Autonomous SDLC for QEM-Bench"
echo "📁 Repository: $REPO_ROOT"
echo "⚙️ Configuration: $SCRIPT_DIR/value-config.yaml"

# Ensure required tools are installed
check_dependencies() {
    echo "🔍 Checking dependencies..."
    
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python 3 is required but not installed"
        exit 1
    fi
    
    if ! command -v git &> /dev/null; then
        echo "❌ Git is required but not installed"
        exit 1
    fi
    
    # Check for optional tools
    if ! command -v claude &> /dev/null; then
        echo "⚠️ Claude CLI not found, installing..."
        npm i -g @anthropic-ai/claude-code || echo "Could not install Claude CLI"
    fi
    
    if ! command -v claude-flow &> /dev/null; then
        echo "⚠️ Claude Flow not found, installing..." 
        npm i -g claude-flow@alpha || echo "Could not install Claude Flow"
    fi
    
    echo "✅ Dependencies checked"
}

# Setup Python environment
setup_environment() {
    echo "🐍 Setting up Python environment..."
    
    cd "$REPO_ROOT"
    
    # Install package in development mode if not already installed
    if ! python3 -c "import qem_bench" 2>/dev/null; then
        echo "📦 Installing qem-bench in development mode..."
        pip install -e ".[dev]" || {
            echo "❌ Failed to install package"
            exit 1
        }
    fi
    
    # Install additional autonomous tools
    echo "🔧 Installing autonomous execution tools..."
    pip install pyyaml asyncio-mqtt structlog prometheus-client || true
    
    echo "✅ Python environment ready"
}

# Initialize value tracking
initialize_tracking() {
    echo "📊 Initializing value tracking..."
    
    # Create metrics file if it doesn't exist
    METRICS_FILE="$SCRIPT_DIR/value-metrics.json"
    if [[ ! -f "$METRICS_FILE" ]]; then
        cat > "$METRICS_FILE" << EOF
{
  "initialized_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "repository": "qem-bench",
  "maturity_level": "maturing",
  "autonomous_execution": {
    "enabled": true,
    "last_run": null,
    "success_rate": 0.0,
    "items_completed": 0,
    "items_discovered": 0
  },
  "execution_history": [],
  "learning_metrics": {
    "prediction_accuracy": 0.0,
    "effort_estimation_error": 0.0,
    "value_realization_rate": 0.0
  }
}
EOF
        echo "📝 Created initial metrics file"
    fi
    
    echo "✅ Value tracking initialized"
}

# Start autonomous executor
start_autonomous_execution() {
    echo "🚀 Starting autonomous execution..."
    
    cd "$SCRIPT_DIR"
    
    # Make executor executable
    chmod +x autonomous-executor.py
    
    # Start with nohup for background execution
    nohup python3 autonomous-executor.py > autonomous.log 2>&1 &
    EXECUTOR_PID=$!
    
    echo "$EXECUTOR_PID" > autonomous.pid
    echo "🎯 Autonomous executor started with PID: $EXECUTOR_PID"
    echo "📄 Logs: $SCRIPT_DIR/autonomous.log"
    echo "🛑 Stop with: kill $EXECUTOR_PID"
}

# Setup cron jobs for scheduled execution
setup_scheduled_execution() {
    echo "⏰ Setting up scheduled execution..."
    
    # Create cron entries (commented out - would need user permission)
    cat > "$SCRIPT_DIR/cron-jobs.txt" << EOF
# Terragon Autonomous SDLC Scheduled Jobs
# Add these to your crontab with: crontab -e

# Hourly security scan
0 * * * * cd $REPO_ROOT && python3 $SCRIPT_DIR/autonomous-executor.py --mode=security-scan

# Daily comprehensive analysis
0 2 * * * cd $REPO_ROOT && python3 $SCRIPT_DIR/autonomous-executor.py --mode=full-analysis

# Weekly deep SDLC assessment
0 3 * * 1 cd $REPO_ROOT && python3 $SCRIPT_DIR/autonomous-executor.py --mode=strategic-review

# Monthly value model recalibration
0 4 1 * * cd $REPO_ROOT && python3 $SCRIPT_DIR/autonomous-executor.py --mode=learning-update
EOF
    
    echo "📅 Scheduled jobs configured in: $SCRIPT_DIR/cron-jobs.txt"
    echo "ℹ️ Add to crontab manually with: crontab -e"
}

# Create status monitoring script
create_status_script() {
    cat > "$SCRIPT_DIR/status.sh" << 'EOF'
#!/bin/bash
# Check status of Terragon Autonomous SDLC

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🤖 Terragon Autonomous SDLC Status"
echo "=================================="

# Check if executor is running
if [[ -f "$SCRIPT_DIR/autonomous.pid" ]]; then
    PID=$(cat "$SCRIPT_DIR/autonomous.pid")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✅ Autonomous executor is running (PID: $PID)"
    else
        echo "❌ Autonomous executor is not running (stale PID file)"
        rm -f "$SCRIPT_DIR/autonomous.pid"
    fi
else
    echo "❌ Autonomous executor is not running"
fi

# Show recent metrics
if [[ -f "$SCRIPT_DIR/value-metrics.json" ]]; then
    echo ""
    echo "📊 Recent Metrics:"
    python3 -c "
import json
with open('$SCRIPT_DIR/value-metrics.json') as f:
    metrics = json.load(f)
    exec_metrics = metrics.get('autonomous_execution', {})
    print(f'  Success Rate: {exec_metrics.get(\"success_rate\", 0)*100:.1f}%')
    print(f'  Items Completed: {exec_metrics.get(\"items_completed\", 0)}')
    print(f'  Items Discovered: {exec_metrics.get(\"items_discovered\", 0)}')
    print(f'  Last Run: {exec_metrics.get(\"last_run\", \"Never\")}')
"
fi

# Show recent logs
if [[ -f "$SCRIPT_DIR/autonomous.log" ]]; then
    echo ""
    echo "📄 Recent Log Entries:"
    tail -5 "$SCRIPT_DIR/autonomous.log"
fi
EOF
    
    chmod +x "$SCRIPT_DIR/status.sh"
    echo "📊 Status script created: $SCRIPT_DIR/status.sh"
}

# Main execution flow
main() {
    echo "🌟 Terragon Autonomous SDLC Initialization"
    echo "=========================================="
    
    check_dependencies
    setup_environment
    initialize_tracking
    setup_scheduled_execution
    create_status_script
    
    echo ""
    echo "🎉 Terragon Autonomous SDLC Setup Complete!"
    echo ""
    echo "Next Steps:"
    echo "1. Review configuration: $SCRIPT_DIR/value-config.yaml"
    echo "2. Check backlog: $REPO_ROOT/BACKLOG.md"
    echo "3. Start autonomous execution: $SCRIPT_DIR/start-autonomous.sh --start"
    echo "4. Monitor status: $SCRIPT_DIR/status.sh"
    echo ""
    echo "🤖 The system will continuously:"
    echo "   • Discover high-value work items"
    echo "   • Score and prioritize using WSJF + ICE + Technical Debt"
    echo "   • Execute autonomous improvements"
    echo "   • Learn and adapt from outcomes"
    echo ""
    echo "🔒 Safety Features:"
    echo "   • All changes require PR approval"
    echo "   • Automatic rollback on failures"
    echo "   • Comprehensive testing validation"
    echo "   • Conservative risk assessment"
    
    # Ask if user wants to start immediately
    if [[ "${1:-}" == "--start" ]]; then
        echo ""
        echo "🚀 Starting autonomous execution now..."
        start_autonomous_execution
    else
        echo ""
        echo "Run with --start to begin autonomous execution immediately"
    fi
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi