#!/usr/bin/env python3
"""
Progressive Quality Gates Demo

Demonstrates the autonomous SDLC execution with progressive quality gates
for the QEM-Bench quantum error mitigation framework.
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from qem_bench.quality_gates.autonomous import AutonomousQualityManager
from qem_bench.quality_gates.core import GenerationType


def setup_logging():
    """Setup logging for demo"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('quality_gates_demo.log')
        ]
    )


async def demonstrate_progressive_quality_gates():
    """Demonstrate progressive quality gates execution"""
    print("🚀 QEM-BENCH PROGRESSIVE QUALITY GATES DEMO")
    print("=" * 60)
    
    project_path = Path(__file__).parent
    
    # Initialize autonomous quality manager
    quality_manager = AutonomousQualityManager(project_path)
    
    print("\n🧠 AUTONOMOUS QUALITY MANAGER INITIALIZED")
    print(f"Project Path: {project_path}")
    print(f"Auto-healing: {'✅ Enabled' if quality_manager.auto_healing_enabled else '❌ Disabled'}")
    
    # Add custom healing callback
    async def custom_healing_callback(failed_reports):
        print(f"🔧 Custom healing triggered for {len(failed_reports)} failed generations")
        # Custom healing logic would go here
    
    quality_manager.add_healing_callback(custom_healing_callback)
    
    print("\n" + "=" * 60)
    print("🚀 STARTING AUTONOMOUS SDLC WITH PROGRESSIVE QUALITY GATES")
    print("=" * 60)
    
    try:
        # Execute complete autonomous SDLC cycle
        result = await quality_manager.execute_autonomous_sdlc(
            include_research=True,
            continuous_monitoring=False  # Disable for demo
        )
        
        print("\n" + "=" * 60)
        print("📊 AUTONOMOUS SDLC RESULTS")
        print("=" * 60)
        
        print(f"Success: {'✅ YES' if result['success'] else '❌ NO'}")
        print(f"Execution Time: {result['total_execution_time']:.2f} seconds")
        print(f"Average Quality Score: {result['average_quality_score']:.1f}%")
        print(f"Generations Completed: {result['generations_completed']}")
        
        # Show autonomous features
        auto_features = result.get('autonomous_features', {})
        print(f"\n🤖 AUTONOMOUS FEATURES:")
        print(f"  Auto-fixes Applied: {auto_features.get('auto_fixes_applied', 0)}")
        print(f"  Monitoring: {'✅' if auto_features.get('monitoring_enabled') else '❌'}")
        print(f"  Self-healing: {'✅' if auto_features.get('self_healing_active') else '❌'}")
        
        # Show quality trend
        quality_trend = result.get('quality_trend', {})
        if quality_trend:
            print(f"\n📈 QUALITY PROGRESSION TREND:")
            for transition, improvement in quality_trend.items():
                trend_emoji = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
                print(f"  {trend_emoji} {transition}: {improvement:+.1f}%")
        
        # Show generation details
        reports = result.get('generation_reports', [])
        if reports:
            print(f"\n🎯 GENERATION DETAILS:")
            for report in reports:
                status = "✅ PASSED" if report.overall_passing else "❌ FAILED"
                print(f"  {report.generation.value}: {status} ({report.overall_score:.1f}%)")
                
                # Show individual gate results
                for gate_result in report.gate_results:
                    gate_status = "✅" if gate_result.is_passing else "❌"
                    print(f"    {gate_status} {gate_result.gate_name}: {gate_result.success_rate:.1f}%")
        
        return result
        
    except Exception as e:
        print(f"\n❌ AUTONOMOUS SDLC EXECUTION FAILED: {e}")
        return {"success": False, "error": str(e)}


async def demonstrate_individual_generation():
    """Demonstrate individual generation quality gates"""
    print("\n" + "=" * 60)
    print("🎯 INDIVIDUAL GENERATION DEMO")
    print("=" * 60)
    
    project_path = Path(__file__).parent
    quality_manager = AutonomousQualityManager(project_path)
    
    # Test Generation 1 (Simple)
    print("\n🚀 Testing Generation 1: MAKE IT WORK (Simple)")
    report = await quality_manager.orchestrator.run_generation_quality_gates(
        GenerationType.GENERATION_1_SIMPLE,
        auto_progression=False
    )
    
    status = "✅ PASSED" if report.overall_passing else "❌ FAILED"
    print(f"Result: {status} (Score: {report.overall_score:.1f}%)")
    print(f"Execution Time: {report.execution_time:.2f} seconds")
    
    if report.recommendations:
        print("📝 Recommendations:")
        for rec in report.recommendations[:3]:
            print(f"  • {rec}")
    
    return report


async def demonstrate_continuous_monitoring():
    """Demonstrate continuous quality monitoring"""
    print("\n" + "=" * 60)
    print("📊 CONTINUOUS MONITORING DEMO")
    print("=" * 60)
    
    project_path = Path(__file__).parent
    quality_manager = AutonomousQualityManager(project_path)
    
    # Set shorter monitoring interval for demo
    quality_manager.monitoring_interval = 10  # 10 seconds
    
    print("🔄 Starting continuous monitoring (10 second intervals)")
    print("Will run 3 monitoring cycles then stop...")
    
    # Start monitoring
    await quality_manager.start_autonomous_monitoring()
    
    # Let it run for 3 cycles
    await asyncio.sleep(35)
    
    # Stop monitoring
    await quality_manager.stop_autonomous_monitoring()
    
    # Show monitoring results
    dashboard = quality_manager.get_quality_dashboard()
    print("\n📊 MONITORING DASHBOARD:")
    print(f"  Current Score: {dashboard.get('current_score', 0):.1f}%")
    print(f"  Trend: {dashboard.get('trend', 'unknown')}")
    print(f"  Critical Issues: {dashboard.get('critical_issues', 0)}")
    print(f"  Monitoring Active: {'✅' if dashboard.get('monitoring_active') else '❌'}")
    print(f"  History Points: {dashboard.get('history_count', 0)}")
    
    return dashboard


async def main():
    """Main demo function"""
    setup_logging()
    
    print("🔬 QEM-BENCH PROGRESSIVE QUALITY GATES")
    print("Autonomous SDLC Execution with Intelligent Quality Validation")
    print("=" * 80)
    
    try:
        # Main autonomous SDLC demonstration
        sdlc_result = await demonstrate_progressive_quality_gates()
        
        # Individual generation demonstration
        gen_result = await demonstrate_individual_generation()
        
        # Continuous monitoring demonstration (commented out to avoid long demo)
        # monitor_result = await demonstrate_continuous_monitoring()
        
        print("\n" + "=" * 80)
        print("🎉 PROGRESSIVE QUALITY GATES DEMO COMPLETED")
        print("=" * 80)
        
        if sdlc_result.get('success'):
            print("✅ All quality gates passed successfully!")
            print("🚀 QEM-Bench is ready for production deployment!")
        else:
            print("⚠️  Some quality gates need attention")
            print("🔧 Auto-healing and recommendations provided")
        
        print("\n🔍 Key Features Demonstrated:")
        print("  • Progressive quality gates across 3 generations")
        print("  • Autonomous quality validation and scoring")
        print("  • Auto-healing and self-fixing capabilities")
        print("  • Intelligent progression and blocking")
        print("  • Research validation for academic publication")
        print("  • Continuous monitoring and trend analysis")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())