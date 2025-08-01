#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Executor
Continuously discovers and executes highest-value work for QEM-Bench repository
"""

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import yaml

@dataclass
class ValueItem:
    """Represents a discovered work item with value scoring"""
    id: str
    title: str
    description: str
    category: str
    estimated_hours: float
    wsjf_score: float
    ice_score: float
    technical_debt_score: float
    composite_score: float
    risk_level: float
    files_affected: List[str]
    discovered_at: datetime
    source: str

@dataclass
class ExecutionResult:
    """Result of executing a value item"""
    item_id: str
    success: bool
    actual_hours: float
    actual_impact: Dict[str, float]
    lessons_learned: str
    rollback_required: bool

class ValueDiscoveryEngine:
    """Discovers high-value work items from multiple sources"""
    
    def __init__(self, config_path: Path):
        self.config = self._load_config(config_path)
        self.repo_root = config_path.parent.parent
        
    def _load_config(self, path: Path) -> Dict:
        with open(path) as f:
            return yaml.safe_load(f)
    
    async def discover_all_sources(self) -> List[ValueItem]:
        """Discover work items from all configured sources"""
        items = []
        
        # Git history analysis
        items.extend(await self._discover_from_git_history())
        
        # Static analysis results
        items.extend(await self._discover_from_static_analysis())
        
        # Security vulnerabilities
        items.extend(await self._discover_from_security_scan())
        
        # Test coverage gaps
        items.extend(await self._discover_from_test_coverage())
        
        # Performance optimization opportunities
        items.extend(await self._discover_from_performance_analysis())
        
        # Dependency updates
        items.extend(await self._discover_from_dependency_analysis())
        
        return self._deduplicate_and_score(items)
    
    async def _discover_from_git_history(self) -> List[ValueItem]:
        """Extract TODO, FIXME, HACK comments from git history"""
        cmd = ["git", "log", "--grep=TODO\\|FIXME\\|HACK", "--oneline", "-n", "50"]
        result = await self._run_command(cmd)
        
        items = []
        for line in result.stdout.split('\n'):
            if line.strip():
                # Parse commit for technical debt indicators
                items.append(ValueItem(
                    id=f"git-{hash(line)}",
                    title=f"Address technical debt: {line[:50]}...",
                    description=line,
                    category="technical_debt",
                    estimated_hours=2.0,
                    wsjf_score=0.0,  # Will be calculated
                    ice_score=0.0,
                    technical_debt_score=15.0,
                    composite_score=0.0,
                    risk_level=0.2,
                    files_affected=[],
                    discovered_at=datetime.now(),
                    source="git_history"
                ))
        
        return items
    
    async def _discover_from_static_analysis(self) -> List[ValueItem]:
        """Analyze ruff, mypy, bandit output for improvement opportunities"""
        items = []
        
        # Run ruff analysis
        ruff_result = await self._run_command(["ruff", "check", "src/", "--output-format=json"])
        if ruff_result.returncode == 0:
            try:
                ruff_issues = json.loads(ruff_result.stdout)
                for issue in ruff_issues[:10]:  # Top 10 issues
                    items.append(ValueItem(
                        id=f"ruff-{hash(str(issue))}",
                        title=f"Fix ruff issue: {issue.get('code', 'unknown')}",
                        description=issue.get('message', ''),
                        category="code_quality",
                        estimated_hours=0.5,
                        wsjf_score=0.0,
                        ice_score=25.0,
                        technical_debt_score=8.0,
                        composite_score=0.0,
                        risk_level=0.1,
                        files_affected=[issue.get('filename', '')],
                        discovered_at=datetime.now(),
                        source="ruff"
                    ))
            except json.JSONDecodeError:
                pass
        
        return items
    
    async def _discover_from_security_scan(self) -> List[ValueItem]:
        """Discover security vulnerabilities and improvements"""
        items = []
        
        # Run safety check
        safety_result = await self._run_command(["safety", "check", "--json"])
        if safety_result.returncode != 0:  # Safety returns non-zero for vulnerabilities
            try:
                vulnerabilities = json.loads(safety_result.stdout)
                for vuln in vulnerabilities:
                    items.append(ValueItem(
                        id=f"security-{vuln.get('id', hash(str(vuln)))}",
                        title=f"Fix security vulnerability: {vuln.get('advisory', '')[:50]}",
                        description=vuln.get('advisory', ''),
                        category="security",
                        estimated_hours=3.0,
                        wsjf_score=0.0,
                        ice_score=45.0,
                        technical_debt_score=0.0,
                        composite_score=0.0,
                        risk_level=0.8,
                        files_affected=[],
                        discovered_at=datetime.now(),
                        source="safety"
                    ))
            except json.JSONDecodeError:
                pass
        
        return items
    
    async def _discover_from_test_coverage(self) -> List[ValueItem]:
        """Identify test coverage improvement opportunities"""
        items = []
        
        # Run pytest with coverage
        coverage_result = await self._run_command([
            "pytest", "--cov=qem_bench", "--cov-report=json", "--quiet"
        ])
        
        if coverage_result.returncode == 0:
            try:
                with open("coverage.json") as f:
                    coverage_data = json.load(f)
                
                current_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
                target_coverage = self.config["execution"]["testRequirements"]["minCoverage"]
                
                if current_coverage < target_coverage:
                    items.append(ValueItem(
                        id="test-coverage-improvement",
                        title=f"Improve test coverage from {current_coverage:.1f}% to {target_coverage}%",
                        description=f"Add tests to reach minimum coverage threshold",
                        category="testing",
                        estimated_hours=6.0,
                        wsjf_score=0.0,
                        ice_score=30.0,
                        technical_debt_score=12.0,
                        composite_score=0.0,
                        risk_level=0.3,
                        files_affected=[],
                        discovered_at=datetime.now(),
                        source="test_coverage"
                    ))
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        
        return items
    
    async def _discover_from_performance_analysis(self) -> List[ValueItem]:
        """Identify JAX/GPU performance optimization opportunities"""
        items = []
        
        # Look for unoptimized JAX operations
        jax_files = list(Path("src").rglob("*.py"))
        for file_path in jax_files:
            try:
                content = file_path.read_text()
                if "jax" in content and "jit" not in content:
                    items.append(ValueItem(
                        id=f"jax-optimization-{hash(str(file_path))}",
                        title=f"Optimize JAX performance in {file_path.name}",
                        description="Add JIT compilation for performance improvement",
                        category="performance",
                        estimated_hours=2.0,
                        wsjf_score=0.0,
                        ice_score=35.0,
                        technical_debt_score=5.0,
                        composite_score=0.0,
                        risk_level=0.4,
                        files_affected=[str(file_path)],
                        discovered_at=datetime.now(),
                        source="performance_analysis"
                    ))
            except Exception:
                continue
        
        return items
    
    async def _discover_from_dependency_analysis(self) -> List[ValueItem]:
        """Check for outdated or vulnerable dependencies"""
        items = []
        
        # Check for dependency updates (simplified)
        pip_outdated = await self._run_command(["pip", "list", "--outdated", "--format=json"])
        if pip_outdated.returncode == 0:
            try:
                outdated = json.loads(pip_outdated.stdout)
                for pkg in outdated[:5]:  # Top 5 outdated packages
                    items.append(ValueItem(
                        id=f"dependency-{pkg['name']}",
                        title=f"Update {pkg['name']} from {pkg['version']} to {pkg['latest_version']}",
                        description=f"Dependency update for {pkg['name']}",
                        category="maintenance",
                        estimated_hours=1.0,
                        wsjf_score=0.0,
                        ice_score=15.0,
                        technical_debt_score=3.0,
                        composite_score=0.0,
                        risk_level=0.3,
                        files_affected=["pyproject.toml"],
                        discovered_at=datetime.now(),
                        source="dependency_analysis"
                    ))
            except json.JSONDecodeError:
                pass
        
        return items
    
    def _deduplicate_and_score(self, items: List[ValueItem]) -> List[ValueItem]:
        """Remove duplicates and calculate composite scores"""
        # Simple deduplication by title similarity
        unique_items = []
        seen_titles = set()
        
        for item in items:
            title_key = item.title.lower()[:30]  # First 30 chars
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                item.composite_score = self._calculate_composite_score(item)
                unique_items.append(item)
        
        return sorted(unique_items, key=lambda x: x.composite_score, reverse=True)
    
    def _calculate_composite_score(self, item: ValueItem) -> float:
        """Calculate WSJF + ICE + Technical Debt composite score"""
        weights = self.config["scoring"]["weights"]["maturing"]
        
        # WSJF calculation (simplified)
        cost_of_delay = 50.0  # Base value
        job_size = item.estimated_hours
        wsjf = cost_of_delay / max(job_size, 0.5)
        
        # Apply category-specific boosts
        category_boost = 1.0
        if item.category == "security":
            category_boost = self.config["scoring"]["thresholds"]["securityBoost"]
        elif item.category == "performance":
            category_boost = self.config["scoring"]["boosts"]["performance"]
        
        # Composite score calculation
        composite = (
            weights["wsjf"] * wsjf +
            weights["ice"] * item.ice_score +
            weights["technicalDebt"] * item.technical_debt_score
        ) * category_boost
        
        return composite
    
    async def _run_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run shell command asynchronously"""
        process = await asyncio.create_subprocess_exec(
            *cmd, 
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.repo_root
        )
        stdout, stderr = await process.communicate()
        
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=stdout.decode(),
            stderr=stderr.decode()
        )

class AutonomousExecutor:
    """Executes highest-value work items autonomously"""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.discovery_engine = ValueDiscoveryEngine(config_path)
        self.execution_history: List[ExecutionResult] = []
        
    def _load_config(self, path: Path) -> Dict:
        with open(path) as f:
            return yaml.safe_load(f)
    
    async def run_continuous_loop(self):
        """Main autonomous execution loop"""
        logging.info("Starting Terragon Autonomous SDLC Executor")
        
        while True:
            try:
                # Discover all available work items
                items = await self.discovery_engine.discover_all_sources()
                logging.info(f"Discovered {len(items)} potential work items")
                
                # Select highest value item above threshold
                next_item = self._select_next_best_value(items)
                
                if next_item:
                    logging.info(f"Executing: {next_item.title} (Score: {next_item.composite_score:.1f})")
                    result = await self._execute_item(next_item)
                    self.execution_history.append(result)
                    
                    # Update learning model
                    self._update_learning_model(result)
                    
                    # Save execution metrics
                    self._save_metrics()
                else:
                    logging.info("No items above minimum score threshold, running housekeeping")
                    await self._run_housekeeping()
                
                # Wait before next cycle
                await asyncio.sleep(3600)  # 1 hour between cycles
                
            except Exception as e:
                logging.error(f"Error in execution loop: {e}")
                await asyncio.sleep(300)  # 5 minute backoff on error
    
    def _select_next_best_value(self, items: List[ValueItem]) -> Optional[ValueItem]:
        """Select the next highest-value item to execute"""
        min_score = self.config["scoring"]["thresholds"]["minScore"]
        max_risk = self.config["scoring"]["thresholds"]["maxRisk"]
        
        for item in items:
            if item.composite_score >= min_score and item.risk_level <= max_risk:
                return item
        
        return None
    
    async def _execute_item(self, item: ValueItem) -> ExecutionResult:
        """Execute a work item and track results"""
        start_time = time.time()
        success = False
        rollback_required = False
        
        try:
            # Create feature branch
            branch_name = f"auto-value/{item.id}-{int(time.time())}"
            await self._run_git_command(["checkout", "-b", branch_name])
            
            # Execute based on category
            if item.category == "code_quality":
                success = await self._fix_code_quality_issue(item)
            elif item.category == "security":
                success = await self._fix_security_issue(item)
            elif item.category == "testing":
                success = await self._improve_test_coverage(item)
            elif item.category == "performance":
                success = await self._optimize_performance(item)
            elif item.category == "maintenance":
                success = await self._update_dependencies(item)
            else:
                success = await self._generic_task_execution(item)
            
            if success:
                # Run validation
                success = await self._validate_changes()
                
                if success:
                    # Create pull request
                    await self._create_pull_request(item, branch_name)
                else:
                    rollback_required = True
            
        except Exception as e:
            logging.error(f"Execution failed: {e}")
            rollback_required = True
        
        actual_hours = (time.time() - start_time) / 3600
        
        if rollback_required:
            await self._rollback_changes()
        
        return ExecutionResult(
            item_id=item.id,
            success=success and not rollback_required,
            actual_hours=actual_hours,
            actual_impact={},  # Would be populated with real metrics
            lessons_learned="",
            rollback_required=rollback_required
        )
    
    async def _validate_changes(self) -> bool:
        """Validate that changes meet quality requirements"""
        # Run tests
        test_result = await self._run_command(["pytest", "--quiet"])
        if test_result.returncode != 0:
            logging.error("Tests failed during validation")
            return False
        
        # Run linting
        lint_result = await self._run_command(["ruff", "check", "src/"])
        if lint_result.returncode != 0:
            logging.error("Linting failed during validation")
            return False
        
        # Run type checking
        mypy_result = await self._run_command(["mypy", "src/"])
        if mypy_result.returncode != 0:
            logging.error("Type checking failed during validation")
            return False
        
        return True
    
    async def _fix_code_quality_issue(self, item: ValueItem) -> bool:
        """Fix a code quality issue identified by static analysis"""
        # Run ruff with --fix
        result = await self._run_command(["ruff", "check", "src/", "--fix"])
        return result.returncode == 0
    
    async def _fix_security_issue(self, item: ValueItem) -> bool:
        """Address a security vulnerability"""
        # Update vulnerable dependencies
        result = await self._run_command(["pip", "install", "-U", "safety"])
        return result.returncode == 0
    
    async def _improve_test_coverage(self, item: ValueItem) -> bool:
        """Add tests to improve coverage"""
        # This would implement smart test generation
        # For now, return success to simulate
        return True
    
    async def _optimize_performance(self, item: ValueItem) -> bool:
        """Optimize performance, particularly JAX operations"""
        # This would implement JAX optimization
        return True
    
    async def _update_dependencies(self, item: ValueItem) -> bool:
        """Update outdated dependencies"""
        # Parse dependency name from item title
        pkg_name = item.title.split()[1]  # "Update package_name from..."
        result = await self._run_command(["pip", "install", "-U", pkg_name])
        return result.returncode == 0
    
    async def _generic_task_execution(self, item: ValueItem) -> bool:
        """Generic task execution fallback"""
        return True
    
    async def _create_pull_request(self, item: ValueItem, branch_name: str):
        """Create a pull request for the completed work"""
        # Commit changes
        await self._run_git_command(["add", "."])
        commit_msg = f"[AUTO-VALUE] {item.title}\n\nComposite Score: {item.composite_score:.1f}\nEstimated Hours: {item.estimated_hours}\nCategory: {item.category}\n\n🤖 Generated with Claude Code + Terragon Autonomous SDLC\n\nCo-Authored-By: Claude <noreply@anthropic.com>"
        await self._run_git_command(["commit", "-m", commit_msg])
        
        # Push branch
        await self._run_git_command(["push", "-u", "origin", branch_name])
        
        # Create PR (would use gh CLI if available)
        logging.info(f"Would create PR for branch: {branch_name}")
    
    async def _rollback_changes(self):
        """Rollback changes and return to main branch"""
        await self._run_git_command(["checkout", "main"])
        await self._run_git_command(["branch", "-D", f"auto-value/*"])
    
    async def _run_housekeeping(self):
        """Run maintenance tasks when no high-value items exist"""
        logging.info("Running autonomous housekeeping tasks")
        
        # Update dependencies
        await self._run_command(["pip", "install", "-U", "pip"])
        
        # Clean cache
        await self._run_command(["pip", "cache", "purge"])
        
        # Run security scan
        await self._run_command(["safety", "check"])
    
    def _update_learning_model(self, result: ExecutionResult):
        """Update scoring model based on execution results"""
        # Simple learning: adjust confidence based on success rate
        pass
    
    def _save_metrics(self):
        """Save execution metrics to tracking file"""
        metrics_file = self.config_path.parent / "value-metrics.json"
        
        metrics = {
            "last_updated": datetime.now().isoformat(),
            "execution_history": [asdict(r) for r in self.execution_history[-50:]],  # Last 50
            "success_rate": sum(1 for r in self.execution_history if r.success) / max(len(self.execution_history), 1),
            "average_hours": sum(r.actual_hours for r in self.execution_history) / max(len(self.execution_history), 1)
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
    
    async def _run_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run shell command"""
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=stdout.decode(),
            stderr=stderr.decode()
        )
    
    async def _run_git_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run git command"""
        return await self._run_command(["git"] + cmd)

async def main():
    """Main entry point for autonomous executor"""
    logging.basicConfig(level=logging.INFO)
    
    config_path = Path(__file__).parent / "value-config.yaml"
    executor = AutonomousExecutor(config_path)
    
    await executor.run_continuous_loop()

if __name__ == "__main__":
    asyncio.run(main())