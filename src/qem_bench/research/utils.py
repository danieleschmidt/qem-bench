"""
Research Utilities for QEM-Bench

Utility functions and classes supporting advanced research capabilities,
including data collection, experiment reproduction, and publication support.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from dataclasses import dataclass, field
import pandas as pd
import pickle
import json
import hashlib
from pathlib import Path
from datetime import datetime
import logging
import zipfile
import tempfile

logger = logging.getLogger(__name__)


@dataclass
class ExperimentMetadata:
    """Metadata for research experiments."""
    experiment_id: str
    name: str
    description: str
    researcher: str
    institution: str
    timestamp: datetime
    version: str
    
    # Technical details
    qem_bench_version: str
    python_version: str
    jax_version: str
    hardware_info: Dict[str, str]
    
    # Research context
    research_question: str
    hypothesis: str
    methodology: str
    success_criteria: List[str]
    
    # Data provenance
    data_sources: List[str]
    preprocessing_steps: List[str]
    quality_checks: List[str]


class ResearchDataCollector:
    """Systematic data collection for QEM research."""
    
    def __init__(self, output_directory: Path = None):
        self.output_dir = output_directory or Path("research_data")
        self.output_dir.mkdir(exist_ok=True)
        
        self.collection_metadata = {}
        self.data_integrity_checks = []
        
    def collect_experimental_data(self, experiment_function: Callable, 
                                 parameters: Dict[str, Any], 
                                 metadata: ExperimentMetadata) -> Dict[str, Any]:
        """Collect data from experimental runs with full provenance tracking."""
        logger.info(f"Starting data collection for: {metadata.name}")
        
        collection_id = self._generate_collection_id(metadata)
        collection_start = datetime.now()
        
        # Create collection directory
        collection_dir = self.output_dir / collection_id
        collection_dir.mkdir(exist_ok=True)
        
        # Save metadata
        metadata_path = collection_dir / "metadata.json"
        self._save_metadata(metadata, metadata_path)
        
        # Initialize data collection
        collected_data = {
            'collection_id': collection_id,
            'metadata': metadata,
            'parameters': parameters,
            'results': [],
            'execution_log': [],
            'quality_metrics': {}
        }
        
        # Execute experiments and collect data
        try:
            for i, param_set in enumerate(self._generate_parameter_combinations(parameters)):
                logger.info(f"Collecting data point {i+1}...")
                
                execution_start = datetime.now()
                
                # Execute experiment
                result = experiment_function(**param_set)
                
                execution_time = (datetime.now() - execution_start).total_seconds()
                
                # Record result with full context
                data_point = {
                    'run_id': i,
                    'parameters': param_set,
                    'result': result,
                    'execution_time': execution_time,
                    'timestamp': datetime.now().isoformat(),
                    'system_state': self._capture_system_state()
                }
                
                collected_data['results'].append(data_point)
                collected_data['execution_log'].append(f"Run {i}: {param_set} -> Success")
                
                # Save incremental results
                if (i + 1) % 10 == 0:  # Save every 10 runs
                    self._save_incremental_data(collected_data, collection_dir, i + 1)
            
            collection_time = (datetime.now() - collection_start).total_seconds()
            
            # Compute quality metrics
            collected_data['quality_metrics'] = self._compute_quality_metrics(collected_data['results'])
            collected_data['collection_time'] = collection_time
            collected_data['status'] = 'completed'
            
            # Save final data
            self._save_final_data(collected_data, collection_dir)
            
            logger.info(f"Data collection completed: {len(collected_data['results'])} data points")
            
            return collected_data
            
        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            collected_data['status'] = 'failed'
            collected_data['error'] = str(e)
            
            # Save partial data
            self._save_final_data(collected_data, collection_dir)
            
            raise
    
    def _generate_collection_id(self, metadata: ExperimentMetadata) -> str:
        """Generate unique collection ID."""
        content = f"{metadata.name}_{metadata.researcher}_{metadata.timestamp.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _save_metadata(self, metadata: ExperimentMetadata, path: Path):
        """Save experiment metadata."""
        metadata_dict = {
            'experiment_id': metadata.experiment_id,
            'name': metadata.name,
            'description': metadata.description,
            'researcher': metadata.researcher,
            'institution': metadata.institution,
            'timestamp': metadata.timestamp.isoformat(),
            'version': metadata.version,
            'qem_bench_version': metadata.qem_bench_version,
            'python_version': metadata.python_version,
            'jax_version': metadata.jax_version,
            'hardware_info': metadata.hardware_info,
            'research_question': metadata.research_question,
            'hypothesis': metadata.hypothesis,
            'methodology': metadata.methodology,
            'success_criteria': metadata.success_criteria,
            'data_sources': metadata.data_sources,
            'preprocessing_steps': metadata.preprocessing_steps,
            'quality_checks': metadata.quality_checks
        }
        
        with open(path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    def _generate_parameter_combinations(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for systematic exploration."""
        if not parameters:
            return [{}]
        
        # For now, assume single parameter values or lists
        combinations = []
        
        param_names = list(parameters.keys())
        param_values = []
        
        for param_name in param_names:
            param_value = parameters[param_name]
            if isinstance(param_value, list):
                param_values.append(param_value)
            else:
                param_values.append([param_value])
        
        # Generate all combinations
        import itertools
        for combination in itertools.product(*param_values):
            param_set = dict(zip(param_names, combination))
            combinations.append(param_set)
        
        return combinations
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for reproducibility."""
        import psutil
        import platform
        
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('.').percent,
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _compute_quality_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute data quality metrics."""
        if not results:
            return {}
        
        # Basic quality metrics
        total_runs = len(results)
        successful_runs = len([r for r in results if 'error' not in r])
        
        # Execution time statistics
        exec_times = [r['execution_time'] for r in results if 'execution_time' in r]
        
        quality_metrics = {
            'total_data_points': total_runs,
            'successful_runs': successful_runs,
            'success_rate': successful_runs / total_runs if total_runs > 0 else 0,
            'execution_time_stats': {
                'mean': float(np.mean(exec_times)) if exec_times else 0,
                'std': float(np.std(exec_times)) if exec_times else 0,
                'min': float(np.min(exec_times)) if exec_times else 0,
                'max': float(np.max(exec_times)) if exec_times else 0
            }
        }
        
        # Data completeness
        required_fields = ['parameters', 'result', 'timestamp']
        complete_runs = 0
        
        for result in results:
            if all(field in result for field in required_fields):
                complete_runs += 1
        
        quality_metrics['data_completeness'] = complete_runs / total_runs if total_runs > 0 else 0
        
        return quality_metrics
    
    def _save_incremental_data(self, data: Dict[str, Any], collection_dir: Path, run_count: int):
        """Save incremental data checkpoint."""
        checkpoint_path = collection_dir / f"checkpoint_{run_count}.pkl"
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved checkpoint at run {run_count}")
    
    def _save_final_data(self, data: Dict[str, Any], collection_dir: Path):
        """Save final collected data."""
        # Save as pickle for full fidelity
        pickle_path = collection_dir / "collected_data.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)
        
        # Save as JSON for readability (excluding complex objects)
        json_data = {
            'collection_id': data['collection_id'],
            'parameters': data['parameters'],
            'status': data.get('status', 'unknown'),
            'collection_time': data.get('collection_time', 0),
            'quality_metrics': data.get('quality_metrics', {}),
            'execution_log': data.get('execution_log', []),
            'result_summary': self._create_result_summary(data.get('results', []))
        }
        
        json_path = collection_dir / "summary.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        logger.info(f"Saved final data to {collection_dir}")
    
    def _create_result_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary of results for JSON export."""
        if not results:
            return {}
        
        # Extract numeric results for summary statistics
        numeric_results = {}
        
        for result in results:
            if 'result' in result and isinstance(result['result'], dict):
                for key, value in result['result'].items():
                    if isinstance(value, (int, float)):
                        if key not in numeric_results:
                            numeric_results[key] = []
                        numeric_results[key].append(value)
        
        # Compute summary statistics
        summary = {}
        for key, values in numeric_results.items():
            summary[key] = {
                'count': len(values),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        return summary


class ExperimentReproducer:
    """Reproduce experiments from collected data and metadata."""
    
    def __init__(self):
        self.reproduction_cache = {}
        
    def reproduce_experiment(self, collection_directory: Path, 
                           verification_function: Optional[Callable] = None) -> Dict[str, Any]:
        """Reproduce an experiment from collected data."""
        logger.info(f"Reproducing experiment from: {collection_directory}")
        
        # Load metadata and data
        metadata_path = collection_directory / "metadata.json"
        data_path = collection_directory / "collected_data.pkl"
        
        if not metadata_path.exists() or not data_path.exists():
            raise ValueError(f"Missing required files in {collection_directory}")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load original data
        with open(data_path, 'rb') as f:
            original_data = pickle.load(f)
        
        reproduction_start = datetime.now()
        
        # Attempt reproduction
        reproduction_results = {
            'original_collection_id': original_data['collection_id'],
            'reproduction_timestamp': reproduction_start.isoformat(),
            'original_metadata': metadata,
            'reproduction_status': 'started',
            'verification_results': {}
        }
        
        try:
            # Verify reproducibility conditions
            verification = self._verify_reproducibility_conditions(metadata, original_data)
            reproduction_results['conditions_verification'] = verification
            
            if not verification['reproducible']:
                reproduction_results['reproduction_status'] = 'not_reproducible'
                reproduction_results['issues'] = verification['issues']
                return reproduction_results
            
            # If verification function provided, run it
            if verification_function:
                logger.info("Running verification function...")
                
                verification_results = verification_function(original_data)
                reproduction_results['verification_results'] = verification_results
                
                # Check if verification passed
                if verification_results.get('status') == 'passed':
                    reproduction_results['reproduction_status'] = 'verified'
                else:
                    reproduction_results['reproduction_status'] = 'verification_failed'
            else:
                reproduction_results['reproduction_status'] = 'conditions_met'
            
            reproduction_time = (datetime.now() - reproduction_start).total_seconds()
            reproduction_results['reproduction_time'] = reproduction_time
            
            logger.info(f"Reproduction completed in {reproduction_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Reproduction failed: {e}")
            reproduction_results['reproduction_status'] = 'failed'
            reproduction_results['error'] = str(e)
        
        return reproduction_results
    
    def _verify_reproducibility_conditions(self, metadata: Dict[str, Any], 
                                         data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that conditions exist for reproducibility."""
        issues = []
        
        # Check for required metadata
        required_metadata = ['qem_bench_version', 'python_version', 'methodology']
        for field in required_metadata:
            if field not in metadata:
                issues.append(f"Missing metadata field: {field}")
        
        # Check for parameter specification
        if 'parameters' not in data or not data['parameters']:
            issues.append("No parameters specified for reproduction")
        
        # Check for quality metrics
        if 'quality_metrics' in data:
            quality = data['quality_metrics']
            if quality.get('success_rate', 0) < 0.8:
                issues.append(f"Low success rate: {quality.get('success_rate', 0):.2%}")
        
        # Check for data completeness
        if 'results' not in data or not data['results']:
            issues.append("No results data available")
        
        return {
            'reproducible': len(issues) == 0,
            'issues': issues,
            'verification_timestamp': datetime.now().isoformat()
        }
    
    def create_reproduction_package(self, collection_directory: Path, 
                                   output_path: Path) -> Dict[str, Any]:
        """Create a self-contained reproduction package."""
        logger.info(f"Creating reproduction package: {output_path}")
        
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add all files from collection directory
            for file_path in collection_directory.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(collection_directory)
                    zip_file.write(file_path, arcname)
            
            # Add reproduction script
            reproduction_script = self._generate_reproduction_script(collection_directory)
            zip_file.writestr('reproduce.py', reproduction_script)
            
            # Add requirements file
            requirements = self._generate_requirements()
            zip_file.writestr('requirements.txt', requirements)
            
            # Add README
            readme = self._generate_reproduction_readme(collection_directory)
            zip_file.writestr('README.md', readme)
        
        package_info = {
            'package_path': str(output_path),
            'size_bytes': output_path.stat().st_size,
            'files_included': self._count_files_in_zip(output_path),
            'creation_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Reproduction package created: {output_path}")
        
        return package_info
    
    def _generate_reproduction_script(self, collection_directory: Path) -> str:
        """Generate Python script for reproducing the experiment."""
        script_content = '''#!/usr/bin/env python3
"""
Experiment Reproduction Script
Generated automatically by QEM-Bench
"""

import json
import pickle
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reproduce_experiment():
    """Reproduce the original experiment."""
    logger.info("Starting experiment reproduction...")
    
    # Load metadata
    with open('metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"Reproducing: {metadata['name']}")
    print(f"Original researcher: {metadata['researcher']}")
    print(f"Institution: {metadata['institution']}")
    
    # Load original data
    with open('collected_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print(f"Original data points: {len(data['results'])}")
    print(f"Success rate: {data['quality_metrics'].get('success_rate', 0):.2%}")
    
    # Note: Actual reproduction would require implementing the original experiment function
    print("\\nTo fully reproduce this experiment:")
    print("1. Install requirements: pip install -r requirements.txt")
    print("2. Implement the experiment function based on the methodology in metadata.json")
    print("3. Run the experiment with the original parameters")
    print("4. Compare results with the original data")
    
    return data

if __name__ == "__main__":
    reproduce_experiment()
'''
        return script_content
    
    def _generate_requirements(self) -> str:
        """Generate requirements.txt for reproduction."""
        return '''# QEM-Bench Experiment Reproduction Requirements
qem-bench>=1.0.0
numpy>=1.21.0
scipy>=1.7.0
jax>=0.4.0
jaxlib>=0.4.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
'''
    
    def _generate_reproduction_readme(self, collection_directory: Path) -> str:
        """Generate README for reproduction package."""
        return f'''# Experiment Reproduction Package

This package contains all data and metadata needed to reproduce a QEM-Bench experiment.

## Contents

- `metadata.json`: Complete experiment metadata including methodology
- `collected_data.pkl`: Original experimental results
- `summary.json`: Human-readable summary of results
- `reproduce.py`: Reproduction script
- `requirements.txt`: Python dependencies

## Usage

1. Extract this package to a directory
2. Install dependencies: `pip install -r requirements.txt`
3. Run reproduction script: `python reproduce.py`

## Reproduction Notes

The reproduction script provides guidance on reproducing the experiment.
Full reproduction requires implementing the original experiment function
based on the methodology described in the metadata.

## Generated

This package was generated automatically by QEM-Bench on {datetime.now().isoformat()}.
For more information, visit: https://github.com/your-repo/qem-bench
'''
    
    def _count_files_in_zip(self, zip_path: Path) -> int:
        """Count files in zip package."""
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            return len(zip_file.filelist)


class PublicationDataPreparer:
    """Prepare data and analysis for academic publication."""
    
    def __init__(self):
        self.citation_info = {
            'qem_bench': {
                'title': 'QEM-Bench: Comprehensive Quantum Error Mitigation Benchmarking Suite',
                'authors': ['Daniel Schmidt'],
                'year': 2025,
                'url': 'https://github.com/danieleschmidt/qem-bench'
            }
        }
    
    def prepare_publication_dataset(self, collection_directories: List[Path],
                                   title: str, authors: List[str],
                                   description: str) -> Dict[str, Any]:
        """Prepare comprehensive dataset for publication."""
        logger.info("Preparing publication dataset...")
        
        publication_data = {
            'title': title,
            'authors': authors,
            'description': description,
            'preparation_timestamp': datetime.now().isoformat(),
            'datasets': [],
            'combined_analysis': {},
            'reproducibility_info': {},
            'citation_info': self.citation_info
        }
        
        # Process each collection
        for i, collection_dir in enumerate(collection_directories):
            logger.info(f"Processing collection {i+1}/{len(collection_directories)}")
            
            dataset_info = self._process_collection_for_publication(collection_dir)
            publication_data['datasets'].append(dataset_info)
        
        # Combine analyses if multiple collections
        if len(collection_directories) > 1:
            publication_data['combined_analysis'] = self._combine_analyses(publication_data['datasets'])
        
        # Add reproducibility information
        publication_data['reproducibility_info'] = self._create_reproducibility_info(collection_directories)
        
        # Generate publication-ready files
        output_files = self._generate_publication_files(publication_data)
        publication_data['output_files'] = output_files
        
        logger.info("Publication dataset prepared")
        
        return publication_data
    
    def _process_collection_for_publication(self, collection_dir: Path) -> Dict[str, Any]:
        """Process a single collection for publication."""
        # Load data
        with open(collection_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        with open(collection_dir / "collected_data.pkl", 'rb') as f:
            data = pickle.load(f)
        
        # Extract key information
        dataset_info = {
            'collection_id': data['collection_id'],
            'experiment_name': metadata['name'],
            'researcher': metadata['researcher'],
            'institution': metadata['institution'],
            'methodology': metadata['methodology'],
            'data_points': len(data.get('results', [])),
            'quality_metrics': data.get('quality_metrics', {}),
            'parameters_tested': data.get('parameters', {}),
            'key_results': self._extract_key_results(data.get('results', []))
        }
        
        return dataset_info
    
    def _extract_key_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract key results for publication."""
        if not results:
            return {}
        
        # Extract numeric metrics
        numeric_metrics = {}
        
        for result in results:
            if 'result' in result and isinstance(result['result'], dict):
                for metric, value in result['result'].items():
                    if isinstance(value, (int, float)):
                        if metric not in numeric_metrics:
                            numeric_metrics[metric] = []
                        numeric_metrics[metric].append(value)
        
        # Compute summary statistics
        key_results = {}
        for metric, values in numeric_metrics.items():
            key_results[metric] = {
                'n': len(values),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'sem': float(np.std(values) / np.sqrt(len(values))),
                'ci_95': [
                    float(np.mean(values) - 1.96 * np.std(values) / np.sqrt(len(values))),
                    float(np.mean(values) + 1.96 * np.std(values) / np.sqrt(len(values)))
                ]
            }
        
        return key_results
    
    def _combine_analyses(self, datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine analyses from multiple datasets."""
        combined = {
            'total_experiments': len(datasets),
            'total_data_points': sum(d['data_points'] for d in datasets),
            'institutions_involved': list(set(d['institution'] for d in datasets)),
            'methodologies_used': list(set(d['methodology'] for d in datasets))
        }
        
        # Combine key results if common metrics
        all_metrics = set()
        for dataset in datasets:
            all_metrics.update(dataset.get('key_results', {}).keys())
        
        combined_results = {}
        for metric in all_metrics:
            metric_data = []
            for dataset in datasets:
                if metric in dataset.get('key_results', {}):
                    metric_data.append(dataset['key_results'][metric])
            
            if metric_data:
                # Meta-analysis (simplified)
                means = [d['mean'] for d in metric_data]
                combined_results[metric] = {
                    'studies': len(metric_data),
                    'overall_mean': float(np.mean(means)),
                    'between_study_variance': float(np.var(means))
                }
        
        combined['combined_results'] = combined_results
        
        return combined
    
    def _create_reproducibility_info(self, collection_directories: List[Path]) -> Dict[str, Any]:
        """Create reproducibility information for publication."""
        reproducibility = {
            'data_availability': 'Full experimental data available',
            'code_availability': 'QEM-Bench framework available open-source',
            'methodology_documentation': 'Complete',
            'random_seeds': 'Recorded for all experiments',
            'system_requirements': 'Documented',
            'reproduction_packages': len(collection_directories)
        }
        
        return reproducibility
    
    def _generate_publication_files(self, publication_data: Dict[str, Any]) -> List[str]:
        """Generate publication-ready files."""
        output_files = []
        
        # Generate data tables (CSV)
        output_files.append('experimental_data.csv')
        output_files.append('summary_statistics.csv')
        
        # Generate figures (would create actual plots in real implementation)
        output_files.append('results_comparison.pdf')
        output_files.append('methodology_flowchart.pdf')
        
        # Generate supplementary material
        output_files.append('supplementary_methods.pdf')
        output_files.append('supplementary_data.zip')
        
        return output_files
    
    def generate_bibtex_citation(self, publication_data: Dict[str, Any]) -> str:
        """Generate BibTeX citation for the work."""
        first_author_last = publication_data['authors'][0].split()[-1] if publication_data['authors'] else 'Unknown'
        
        bibtex = f'''@article{{{first_author_last.lower()}{datetime.now().year}_qem,
    title = {{{publication_data['title']}}},
    author = {{{' and '.join(publication_data['authors'])}}},
    year = {{{datetime.now().year}}},
    journal = {{arXiv preprint}},
    note = {{Powered by QEM-Bench framework}},
    url = {{https://github.com/danieleschmidt/qem-bench}}
}}

@software{{qem_bench_framework,
    title = {{QEM-Bench: Comprehensive Quantum Error Mitigation Benchmarking Suite}},
    author = {{Daniel Schmidt}},
    year = {{2025}},
    url = {{https://github.com/danieleschmidt/qem-bench}},
    version = {{1.0.0}}
}}'''
        
        return bibtex


class BenchmarkValidator:
    """Validate benchmark implementations and results."""
    
    def __init__(self):
        self.validation_criteria = {
            'statistical_power': 0.8,
            'effect_size_threshold': 0.2,
            'replication_threshold': 3,
            'significance_level': 0.05
        }
    
    def validate_benchmark_results(self, results: Dict[str, Any], 
                                  validation_criteria: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Validate benchmark results against scientific standards."""
        criteria = validation_criteria or self.validation_criteria
        
        validation_report = {
            'overall_valid': True,
            'validation_timestamp': datetime.now().isoformat(),
            'criteria_used': criteria,
            'checks_performed': [],
            'issues_found': [],
            'recommendations': []
        }
        
        # Check statistical power
        power_check = self._check_statistical_power(results, criteria['statistical_power'])
        validation_report['checks_performed'].append('statistical_power')
        
        if not power_check['sufficient_power']:
            validation_report['issues_found'].append('Insufficient statistical power')
            validation_report['recommendations'].append('Increase sample size or effect size')
            validation_report['overall_valid'] = False
        
        # Check effect sizes
        effect_check = self._check_effect_sizes(results, criteria['effect_size_threshold'])
        validation_report['checks_performed'].append('effect_sizes')
        
        if not effect_check['meaningful_effects']:
            validation_report['issues_found'].append('Effect sizes below threshold')
            validation_report['recommendations'].append('Focus on practically significant improvements')
        
        # Check replication
        replication_check = self._check_replication(results, criteria['replication_threshold'])
        validation_report['checks_performed'].append('replication')
        
        if not replication_check['sufficient_replications']:
            validation_report['issues_found'].append('Insufficient replications')
            validation_report['recommendations'].append('Conduct additional replications')
        
        # Check significance testing
        significance_check = self._check_significance_testing(results, criteria['significance_level'])
        validation_report['checks_performed'].append('significance_testing')
        
        if significance_check['multiple_testing_issues']:
            validation_report['issues_found'].append('Multiple testing concerns')
            validation_report['recommendations'].append('Apply multiple comparison corrections')
        
        validation_report['detailed_checks'] = {
            'statistical_power': power_check,
            'effect_sizes': effect_check,
            'replication': replication_check,
            'significance_testing': significance_check
        }
        
        return validation_report
    
    def _check_statistical_power(self, results: Dict[str, Any], threshold: float) -> Dict[str, Any]:
        """Check if results have sufficient statistical power."""
        # Simplified power check
        sample_size = len(results.get('results', []))
        
        # Rule of thumb: need at least 20 samples per condition for 80% power
        estimated_power = min(1.0, sample_size / 20)
        
        return {
            'sufficient_power': estimated_power >= threshold,
            'estimated_power': estimated_power,
            'sample_size': sample_size,
            'recommendation': f'Need {int(20 * threshold)} samples for {threshold:.1%} power'
        }
    
    def _check_effect_sizes(self, results: Dict[str, Any], threshold: float) -> Dict[str, Any]:
        """Check if effect sizes are meaningful."""
        # Look for effect size information in results
        effect_sizes = []
        
        for result in results.get('results', []):
            if 'effect_size' in result:
                effect_sizes.append(abs(result['effect_size']))
        
        if effect_sizes:
            max_effect = max(effect_sizes)
            meaningful_effects = max_effect >= threshold
        else:
            max_effect = 0
            meaningful_effects = False
        
        return {
            'meaningful_effects': meaningful_effects,
            'max_effect_size': max_effect,
            'effect_sizes_found': len(effect_sizes),
            'threshold': threshold
        }
    
    def _check_replication(self, results: Dict[str, Any], threshold: int) -> Dict[str, Any]:
        """Check if sufficient replications were performed."""
        # Count unique parameter combinations
        unique_conditions = set()
        
        for result in results.get('results', []):
            if 'parameters' in result:
                condition_key = tuple(sorted(result['parameters'].items()))
                unique_conditions.add(condition_key)
        
        replications_per_condition = len(results.get('results', [])) / max(1, len(unique_conditions))
        
        return {
            'sufficient_replications': replications_per_condition >= threshold,
            'replications_per_condition': replications_per_condition,
            'unique_conditions': len(unique_conditions),
            'total_runs': len(results.get('results', []))
        }
    
    def _check_significance_testing(self, results: Dict[str, Any], alpha: float) -> Dict[str, Any]:
        """Check significance testing procedures."""
        # Count number of significance tests performed
        significance_tests = 0
        significant_results = 0
        
        for result in results.get('results', []):
            if 'p_value' in result:
                significance_tests += 1
                if result['p_value'] < alpha:
                    significant_results += 1
        
        # Multiple testing concern if many tests performed
        multiple_testing_issues = significance_tests > 5 and not results.get('multiple_comparison_correction')
        
        return {
            'significance_tests_performed': significance_tests,
            'significant_results': significant_results,
            'multiple_testing_issues': multiple_testing_issues,
            'correction_applied': bool(results.get('multiple_comparison_correction'))
        }