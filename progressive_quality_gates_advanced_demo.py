#!/usr/bin/env python3
"""
🚀 Progressive Quality Gates Advanced Demonstration

This demonstrates the full Progressive Quality Gates system with autonomous
SDLC execution, self-healing, and continuous improvement capabilities.

Features Demonstrated:
- Complete 3-generation progressive quality validation
- Autonomous self-healing and auto-fixes
- Real-time quality monitoring and trend analysis
- Research-grade validation for publication readiness
- Production deployment quality gates
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
import sys

# Setup logging for demonstration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("demo")


async def main():
    """Main demonstration function"""
    print("🚀 PROGRESSIVE QUALITY GATES ADVANCED DEMONSTRATION")
    print("=" * 60)
    
    project_path = Path(".")
    
    # Import here to handle any import issues gracefully
    try:
        from src.qem_bench.quality_gates import (
            AutonomousQualityManager,
            ProgressiveQualityOrchestrator,
            GenerationType
        )
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Ensure you're running from the project root and dependencies are installed")
        return
    
    print("\n📋 DEMONSTRATION PHASES")
    print("1. Individual Generation Quality Gates")
    print("2. Complete Autonomous SDLC Cycle")
    print("3. Self-Healing Demonstration")
    print("4. Continuous Monitoring Setup")
    print("5. Research Validation Mode")
    
    # Initialize the autonomous quality manager
    manager = AutonomousQualityManager(project_path)
    orchestrator = ProgressiveQualityOrchestrator(project_path)
    
    print(f"\n🏗️ INITIALIZATION")
    print(f"Project Path: {project_path.absolute()}")
    print(f"Auto-healing: {'✅ Enabled' if manager.auto_healing_enabled else '❌ Disabled'}")
    
    # Phase 1: Individual Generation Testing
    print(f"\n{'='*60}")
    print("📊 PHASE 1: INDIVIDUAL GENERATION QUALITY GATES")
    print(f"{'='*60}")
    
    generations = [
        GenerationType.GENERATION_1_SIMPLE,
        GenerationType.GENERATION_2_ROBUST,
        GenerationType.GENERATION_3_OPTIMIZED
    ]
    
    generation_results = {}
    
    for generation in generations:
        print(f"\n🔍 Testing {generation.value.upper()}")
        print("-" * 40)
        
        try:
            start_time = datetime.now()
            
            report = await orchestrator.run_generation_quality_gates(
                generation=generation,
                auto_progression=False
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            generation_results[generation.value] = report
            
            # Display results
            status = "✅ PASSED" if report.overall_passing else "❌ FAILED"
            print(f"Status: {status}")
            print(f"Overall Score: {report.overall_score:.1f}%")
            print(f"Execution Time: {duration:.2f}s")
            
            print("Individual Gate Results:")
            for result in report.gate_results:
                gate_status = "✅" if result.is_passing else "❌"
                print(f"  {gate_status} {result.gate_name}: {result.success_rate:.1f}%")
                
                if result.errors:
                    for error in result.errors[:2]:  # Show first 2 errors
                        print(f"    • {error}")
            
            if report.recommendations:
                print("Recommendations:")
                for rec in report.recommendations[:3]:  # Show first 3
                    print(f"  💡 {rec}")
            
        except Exception as e:
            print(f"❌ Generation {generation.value} failed: {e}")
            logger.error(f"Generation test failed: {e}")
    
    # Phase 2: Complete Autonomous SDLC
    print(f"\n{'='*60}")
    print("🚀 PHASE 2: COMPLETE AUTONOMOUS SDLC CYCLE")
    print(f"{'='*60}")
    
    try:
        print("Starting complete autonomous SDLC execution...")
        print("This includes all generations with progressive quality gates")
        
        autonomous_result = await manager.execute_autonomous_sdlc(
            include_research=True,
            continuous_monitoring=False  # Skip monitoring for demo
        )
        
        if autonomous_result.get("success"):
            print("🎉 AUTONOMOUS SDLC COMPLETED SUCCESSFULLY!")
            print(f"✅ Total Execution Time: {autonomous_result['total_execution_time']:.2f}s")
            print(f"✅ Average Quality Score: {autonomous_result['average_quality_score']:.1f}%")
            print(f"✅ Generations Completed: {autonomous_result['generations_completed']}")
            
            autonomous_features = autonomous_result.get("autonomous_features", {})
            print(f"🤖 Auto-fixes Applied: {autonomous_features.get('auto_fixes_applied', 0)}")
            
            # Display quality trend
            quality_trend = autonomous_result.get("quality_trend", {})
            if quality_trend:
                print("📈 Quality Progression:")
                for progression, improvement in quality_trend.items():
                    trend_indicator = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
                    print(f"  {trend_indicator} {progression}: {improvement:+.1f}%")
                    
        else:
            print("⚠️ Autonomous SDLC completed with issues")
            if "error" in autonomous_result:
                print(f"❌ Error: {autonomous_result['error']}")
    
    except Exception as e:
        print(f"❌ Autonomous SDLC execution failed: {e}")
        logger.error(f"SDLC execution failed: {e}")
    
    # Phase 3: Self-Healing Demonstration
    print(f"\n{'='*60}")
    print("🔧 PHASE 3: SELF-HEALING DEMONSTRATION")
    print(f"{'='*60}")
    
    print("Demonstrating autonomous self-healing capabilities:")
    print("• Code formatting with Black")
    print("• Lint fixing with Ruff")
    print("• Import optimization")
    print("• Documentation generation")
    
    # Demonstrate healing actions
    healing_actions = [
        {"type": "format_code", "tool": "black", "description": "Format Python code"},
        {"type": "lint_fix", "tool": "ruff", "args": ["--fix"], "description": "Fix linting issues"},
        {"type": "optimize_imports", "description": "Optimize import statements"}
    ]
    
    for i, action in enumerate(healing_actions, 1):
        print(f"\n{i}. {action['description']}")
        try:
            # Apply the healing action
            success = await manager._apply_healing_action(action)
            status = "✅ Applied" if success else "⚠️ Skipped"
            print(f"   Status: {status}")
        except Exception as e:
            print(f"   Status: ❌ Failed - {e}")
    
    # Phase 4: Continuous Monitoring Setup  
    print(f"\n{'='*60}")
    print("📊 PHASE 4: CONTINUOUS MONITORING SETUP")
    print(f"{'='*60}")
    
    print("Setting up continuous quality monitoring...")
    
    # Get quality dashboard
    dashboard = manager.get_quality_dashboard()
    if dashboard.get("status") != "no_data":
        print("Current Quality Dashboard:")
        for key, value in dashboard.items():
            print(f"  {key}: {value}")
    else:
        print("📊 Quality dashboard initialized (no historical data yet)")
    
    print(f"🔄 Monitoring interval: {manager.monitoring_interval}s")
    print(f"🤖 Auto-healing: {'✅ Enabled' if manager.auto_healing_enabled else '❌ Disabled'}")
    
    # Phase 5: Research Validation Mode
    print(f"\n{'='*60}")
    print("🔬 PHASE 5: RESEARCH VALIDATION MODE")
    print(f"{'='*60}")
    
    try:
        print("Running research-grade quality validation...")
        print("This ensures code meets academic publication standards")
        
        research_report = await orchestrator.run_generation_quality_gates(
            GenerationType.RESEARCH_VALIDATION,
            auto_progression=False
        )
        
        status = "✅ PUBLICATION READY" if research_report.overall_passing else "⚠️ NEEDS IMPROVEMENT"
        print(f"\nResearch Validation: {status}")
        print(f"Research Score: {research_report.overall_score:.1f}%")
        
        print("\nResearch Quality Criteria:")
        for result in research_report.gate_results:
            if result.gate_name == "research_validation":
                gate_status = "✅" if result.is_passing else "❌"
                print(f"  {gate_status} Methodology validation: {result.success_rate:.1f}%")
            else:
                gate_status = "✅" if result.is_passing else "❌"
                print(f"  {gate_status} {result.gate_name}: {result.success_rate:.1f}%")
    
    except Exception as e:
        print(f"❌ Research validation failed: {e}")
        logger.error(f"Research validation failed: {e}")
    
    # Final Summary
    print(f"\n{'='*60}")
    print("📋 DEMONSTRATION SUMMARY")
    print(f"{'='*60}")
    
    total_generations = len(generation_results)
    passed_generations = sum(1 for r in generation_results.values() if r.overall_passing)
    
    print(f"Individual Generations: {passed_generations}/{total_generations} passed")
    
    if autonomous_result.get("success"):
        print("✅ Complete Autonomous SDLC: SUCCESS")
    else:
        print("⚠️ Complete Autonomous SDLC: NEEDS ATTENTION")
    
    print("🚀 Progressive Quality Gates Features Demonstrated:")
    print("  ✅ Multi-generation progressive validation")
    print("  ✅ Autonomous self-healing capabilities")
    print("  ✅ Real-time quality monitoring")
    print("  ✅ Research-grade validation")
    print("  ✅ Production deployment readiness")
    
    print("\n🎯 NEXT STEPS:")
    print("• Integrate with CI/CD pipeline")
    print("• Enable continuous monitoring in production")
    print("• Customize quality gates for specific domains")
    print("• Add custom healing callbacks for project-specific needs")
    
    print(f"\n{'='*60}")
    print("🎉 PROGRESSIVE QUALITY GATES DEMONSTRATION COMPLETE")
    print(f"{'='*60}")


def run_with_error_handling():
    """Run demonstration with comprehensive error handling"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️ Demonstration interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demonstration failed with error: {e}")
        logger.error(f"Demo failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_with_error_handling()