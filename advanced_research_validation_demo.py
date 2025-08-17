#!/usr/bin/env python3
"""
Advanced Research Validation Demo

Demonstration of the comprehensive research validation framework for novel
quantum error mitigation techniques implemented in QEM-Bench.

This demo showcases:
1. Quantum-Enhanced Error Syndrome Correlation Learning
2. Cross-Platform Error Model Transfer Learning  
3. Real-Time Adaptive QEM with Causal Inference
4. Integrated Statistical Validation Framework
"""

import sys
import time
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from qem_bench.research.integrated_validation import run_comprehensive_research_validation
    from qem_bench.research import (
        run_research_validation,
        run_cross_platform_validation, 
        run_causal_adaptive_validation
    )
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure QEM-Bench is properly installed and all dependencies are available.")
    sys.exit(1)


def print_banner():
    """Print demo banner"""
    print("🚀" + "=" * 78 + "🚀")
    print("🎯 QEM-BENCH ADVANCED RESEARCH VALIDATION DEMO")
    print("🔬 Novel Quantum Error Mitigation Research Validation")
    print("=" * 80)
    print()
    print("This demo validates three groundbreaking research contributions:")
    print("1. 🧠 Quantum-Enhanced Error Syndrome Correlation Learning")
    print("2. 🔄 Cross-Platform Error Model Transfer Learning")
    print("3. ⚡ Real-Time Adaptive QEM with Causal Inference")
    print()
    print("Each method represents a novel advancement in quantum error mitigation")
    print("with comprehensive experimental validation and statistical significance testing.")
    print("=" * 80)


def run_individual_validations():
    """Run individual research validation demos"""
    
    print("\n📊 INDIVIDUAL RESEARCH VALIDATIONS")
    print("=" * 50)
    
    # 1. Quantum Syndrome Learning Validation
    print("\n🧠 Validating Quantum-Enhanced Error Syndrome Correlation Learning...")
    print("-" * 60)
    
    try:
        syndrome_start = time.time()
        syndrome_results = run_research_validation()
        syndrome_time = time.time() - syndrome_start
        
        print(f"✅ Validation completed in {syndrome_time:.2f} seconds")
        print(f"   Quantum vs Classical Improvement: {syndrome_results.get('improvement_percentage', 0):.1f}%")
        print(f"   Hypothesis Validated: {'✅ Yes' if syndrome_results.get('hypothesis_validated') else '❌ No'}")
        print(f"   Statistical Significance (p-value): {syndrome_results.get('statistical_significance', 1.0):.4f}")
        
    except Exception as e:
        print(f"❌ Syndrome learning validation failed: {e}")
    
    # 2. Cross-Platform Transfer Learning Validation
    print("\n🔄 Validating Cross-Platform Error Model Transfer Learning...")
    print("-" * 60)
    
    try:
        transfer_start = time.time()
        transfer_results = run_cross_platform_validation()
        transfer_time = time.time() - transfer_start
        
        print(f"✅ Validation completed in {transfer_time:.2f} seconds")
        print(f"   Calibration Time Reduction: {transfer_results.get('time_savings_percentage', 0):.1f}%")
        print(f"   Hypothesis Validated: {'✅ Yes' if transfer_results.get('hypothesis_validated') else '❌ No'}")
        print(f"   Target Accuracy: {transfer_results.get('avg_target_accuracy', 0):.3f}")
        
    except Exception as e:
        print(f"❌ Cross-platform transfer validation failed: {e}")
    
    # 3. Causal Adaptive QEM Validation
    print("\n⚡ Validating Real-Time Adaptive QEM with Causal Inference...")
    print("-" * 60)
    
    try:
        causal_start = time.time()
        causal_results = run_causal_adaptive_validation()
        causal_time = time.time() - causal_start
        
        print(f"✅ Validation completed in {causal_time:.2f} seconds")
        print(f"   Error Propagation Reduction: {causal_results.get('propagation_reduction_percentage', 0):.1f}%")
        print(f"   Hypothesis Validated: {'✅ Yes' if causal_results.get('hypothesis_validated') else '❌ No'}")
        print(f"   Prediction Accuracy: {causal_results.get('prediction_accuracy', 0):.3f}")
        
    except Exception as e:
        print(f"❌ Causal adaptive validation failed: {e}")


def run_integrated_validation():
    """Run comprehensive integrated validation"""
    
    print("\n🔬 COMPREHENSIVE INTEGRATED VALIDATION")
    print("=" * 50)
    print("Running comprehensive research validation framework...")
    print("This integrates all novel methods with statistical analysis and publication-ready results.")
    print()
    
    try:
        integrated_start = time.time()
        results = run_comprehensive_research_validation()
        integrated_time = time.time() - integrated_start
        
        print(f"\n✅ Comprehensive validation completed in {integrated_time:.2f} seconds")
        
        # Display key results
        print("\n📈 KEY VALIDATION RESULTS:")
        print("-" * 30)
        
        # Overall metrics
        success_rate = results.integrated_metrics.get('overall_validation_success_rate', 0)
        innovation_score = results.integrated_metrics.get('innovation_impact_score', 0)
        quality_score = results.integrated_metrics.get('integrated_research_quality', 0)
        
        print(f"Overall Success Rate: {success_rate:.1%}")
        print(f"Innovation Impact Score: {innovation_score:.3f}")
        print(f"Research Quality Score: {quality_score:.3f}")
        
        # Individual validations
        print(f"\nIndividual Research Validations:")
        print(f"  Syndrome Learning: {'✅' if results.syndrome_learning_results.get('hypothesis_validated') else '❌'}")
        print(f"  Cross-Platform Transfer: {'✅' if results.cross_platform_results.get('hypothesis_validated') else '❌'}")
        print(f"  Causal Adaptive QEM: {'✅' if results.causal_adaptive_results.get('hypothesis_validated') else '❌'}")
        
        # Statistical significance
        all_significant = results.statistical_significance.get('all_hypotheses_significant', False)
        combined_p = results.statistical_significance.get('combined_p_value', 1.0)
        
        print(f"\nStatistical Significance:")
        print(f"  All Hypotheses Significant: {'✅ Yes' if all_significant else '❌ No'}")
        print(f"  Combined P-Value: {combined_p:.6f}")
        
        # Publication readiness
        pub_ready = results.integrated_metrics.get('publication_readiness', 0) > 0.8
        print(f"  Publication Ready: {'✅ Yes' if pub_ready else '❌ No'}")
        
        return results
        
    except Exception as e:
        print(f"❌ Integrated validation failed: {e}")
        return None


def display_research_summary(results):
    """Display comprehensive research summary"""
    
    if results is None:
        print("⚠️  No results to display")
        return
    
    print("\n📋 RESEARCH SUMMARY")
    print("=" * 50)
    
    # Extract executive summary
    if 'executive_summary' in results.publication_ready_data:
        print("EXECUTIVE SUMMARY:")
        print(results.publication_ready_data['executive_summary'])
    
    print("\n🎯 RESEARCH CONTRIBUTIONS:")
    print("-" * 30)
    
    contributions = results.publication_ready_data.get('research_contributions', [])
    for i, contribution in enumerate(contributions[:5], 1):  # Show top 5
        print(f"{i}. {contribution}")
    
    print("\n📊 PERFORMANCE COMPARISON:")
    print("-" * 30)
    
    # Performance table
    performance_table = results.publication_ready_data.get('performance_tables', {}).get('method_comparison', {})
    if 'headers' in performance_table and 'rows' in performance_table:
        headers = performance_table['headers']
        print("  ".join(f"{h:20s}" for h in headers))
        print("  ".join("-" * 20 for _ in headers))
        
        for row in performance_table['rows']:
            print("  ".join(f"{str(cell):20s}" for cell in row))
    
    print("\n🔮 FUTURE DIRECTIONS:")
    print("-" * 30)
    
    future_directions = results.publication_ready_data.get('future_directions', [])
    for i, direction in enumerate(future_directions[:3], 1):  # Show top 3
        print(f"{i}. {direction}")


def main():
    """Main demo function"""
    
    print_banner()
    
    # Option to run different validation modes
    print("Choose validation mode:")
    print("1. Individual research validations only")
    print("2. Comprehensive integrated validation only") 
    print("3. Complete validation suite (recommended)")
    print()
    
    try:
        choice = input("Enter choice (1-3) or press Enter for complete suite: ").strip()
        if not choice:
            choice = "3"
    except KeyboardInterrupt:
        print("\n👋 Demo cancelled by user")
        return
    
    print(f"\nStarting validation mode {choice}...")
    
    demo_start_time = time.time()
    results = None
    
    if choice in ["1", "3"]:
        run_individual_validations()
    
    if choice in ["2", "3"]:
        results = run_integrated_validation()
    
    if choice == "3" and results:
        display_research_summary(results)
    
    total_time = time.time() - demo_start_time
    
    print(f"\n🏁 DEMO COMPLETED")
    print("=" * 50)
    print(f"Total execution time: {total_time:.2f} seconds")
    
    if results:
        success_rate = results.integrated_metrics.get('overall_validation_success_rate', 0)
        if success_rate >= 1.0:
            print("🎉 ALL RESEARCH HYPOTHESES VALIDATED SUCCESSFULLY!")
            print("✨ Novel quantum error mitigation approaches demonstrate significant improvements")
            print("📚 Results are publication-ready with statistical significance")
        elif success_rate >= 0.67:
            print("🎯 MAJORITY OF RESEARCH HYPOTHESES VALIDATED")
            print("📈 Strong evidence for novel QEM approach effectiveness")
        else:
            print("⚠️  MIXED VALIDATION RESULTS")
            print("🔍 Further investigation recommended for some approaches")
    
    print("\n📖 For detailed results and analysis, see the QEM-Bench research module documentation.")
    print("🔗 Access full implementation in src/qem_bench/research/")


if __name__ == "__main__":
    main()