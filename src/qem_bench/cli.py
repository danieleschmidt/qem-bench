"""Command line interface for QEM-Bench"""

import argparse
import sys
from typing import Optional, List

def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog="qem-bench",
        description="Quantum Error Mitigation Benchmarking Suite"
    )
    
    parser.add_argument(
        "--version", 
        action="version",
        version=f"%(prog)s {get_version()}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_parser.add_argument("--method", choices=["zne", "pec", "vd", "cdr"], 
                             help="Mitigation method to benchmark")
    bench_parser.add_argument("--backend", help="Quantum backend to use")
    
    # Generate command  
    gen_parser = subparsers.add_parser("generate", help="Generate benchmark circuits")
    gen_parser.add_argument("--type", help="Circuit type to generate")
    gen_parser.add_argument("--qubits", type=int, help="Number of qubits")
    
    args = parser.parse_args(argv)
    
    if args.command is None:
        parser.print_help()
        return 1
        
    return handle_command(args)

def get_version() -> str:
    """Get package version"""
    try:
        from ._version import version
        return version
    except ImportError:
        return "unknown"

def handle_command(args: argparse.Namespace) -> int:
    """Handle CLI commands"""
    if args.command == "benchmark":
        print(f"Running benchmark with method: {args.method}")
        return 0
    elif args.command == "generate":
        print(f"Generating {args.type} circuits with {args.qubits} qubits")
        return 0
    else:
        print(f"Unknown command: {args.command}")
        return 1

if __name__ == "__main__":
    sys.exit(main())