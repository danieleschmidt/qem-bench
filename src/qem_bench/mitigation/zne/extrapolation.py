"""Extrapolation methods for Zero-Noise Extrapolation."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Optional
from scipy.optimize import curve_fit, minimize
import warnings


class Extrapolator(ABC):
    """Abstract base class for extrapolation methods."""
    
    @abstractmethod
    def extrapolate(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Extrapolate to zero noise (x=0).
        
        Args:
            x: Noise factors (≥1)
            y: Corresponding expectation values
            
        Returns:
            Tuple of (extrapolated_value, fit_metadata)
        """
        pass


class RichardsonExtrapolator(Extrapolator):
    """
    Richardson extrapolation for ZNE.
    
    Uses Richardson extrapolation assuming polynomial error model:
    E(λ) = E₀ + c₁λ + c₂λ² + ...
    
    This is the most commonly used extrapolation method for ZNE.
    """
    
    def __init__(self, order: int = 1):
        """
        Initialize Richardson extrapolator.
        
        Args:
            order: Order of Richardson extrapolation (1 or 2)
        """
        if order not in [1, 2]:
            raise ValueError("Richardson extrapolation order must be 1 or 2")
        self.order = order
    
    def extrapolate(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Perform Richardson extrapolation."""
        if len(x) < self.order + 1:
            raise ValueError(f"Need at least {self.order + 1} points for order {self.order}")
        
        if self.order == 1:
            return self._linear_richardson(x, y)
        elif self.order == 2:
            return self._quadratic_richardson(x, y)
    
    def _linear_richardson(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Linear Richardson extrapolation using two points."""
        # Use first two points for linear extrapolation
        x1, x2 = x[0], x[1]
        y1, y2 = y[0], y[1]
        
        # Linear extrapolation: E(0) = (x2*y1 - x1*y2) / (x2 - x1)
        extrapolated = (x2 * y1 - x1 * y2) / (x2 - x1)
        
        # Calculate R² and other metrics
        r_squared = self._calculate_r_squared_linear(x[:2], y[:2], extrapolated)
        
        fit_data = {
            "method": "richardson_linear",
            "order": 1,
            "points_used": 2,
            "r_squared": r_squared,
            "coefficients": {"slope": (y2 - y1) / (x2 - x1), "intercept": extrapolated},
            "extrapolation_error": None  # Cannot estimate without ideal value
        }
        
        return extrapolated, fit_data
    
    def _quadratic_richardson(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Quadratic Richardson extrapolation using three points."""
        if len(x) < 3:
            # Fall back to linear if insufficient points
            warnings.warn("Insufficient points for quadratic Richardson, using linear")
            return self._linear_richardson(x, y)
        
        # Use first three points for quadratic extrapolation
        x1, x2, x3 = x[0], x[1], x[2]
        y1, y2, y3 = y[0], y[1], y[2]
        
        # Solve for quadratic coefficients: y = a + b*x + c*x²
        # At x=0: E(0) = a
        A = np.array([
            [1, x1, x1**2],
            [1, x2, x2**2], 
            [1, x3, x3**2]
        ])
        b = np.array([y1, y2, y3])
        
        try:
            coeffs = np.linalg.solve(A, b)
            extrapolated = coeffs[0]  # a coefficient is E(0)
            
            # Calculate R²
            y_pred = A @ coeffs
            r_squared = self._calculate_r_squared(y[:3], y_pred)
            
            fit_data = {
                "method": "richardson_quadratic",
                "order": 2,
                "points_used": 3,
                "r_squared": r_squared,
                "coefficients": {"a": coeffs[0], "b": coeffs[1], "c": coeffs[2]},
                "extrapolation_error": None
            }
            
            return extrapolated, fit_data
            
        except np.linalg.LinAlgError:
            warnings.warn("Quadratic Richardson failed, falling back to linear")
            return self._linear_richardson(x, y)
    
    def _calculate_r_squared_linear(self, x: np.ndarray, y: np.ndarray, intercept: float) -> float:
        """Calculate R² for linear fit."""
        slope = (y[1] - y[0]) / (x[1] - x[0])
        y_pred = slope * x + intercept
        return self._calculate_r_squared(y, y_pred)
    
    def _calculate_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate coefficient of determination (R²)."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return 1 - (ss_res / ss_tot)


class ExponentialExtrapolator(Extrapolator):
    """
    Exponential extrapolation for ZNE.
    
    Fits exponential decay model: E(λ) = A + B * exp(-C * λ)
    and extrapolates to λ=0 giving E(0) = A + B.
    """
    
    def __init__(self, max_iterations: int = 1000):
        """
        Initialize exponential extrapolator.
        
        Args:
            max_iterations: Maximum optimization iterations
        """
        self.max_iterations = max_iterations
    
    def extrapolate(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Perform exponential extrapolation."""
        if len(x) < 3:
            raise ValueError("Need at least 3 points for exponential extrapolation")
        
        # Define exponential model
        def exp_model(lambda_val, A, B, C):
            return A + B * np.exp(-C * lambda_val)
        
        # Initial parameter guess
        A_init = np.min(y)  # Asymptotic value
        B_init = np.max(y) - A_init  # Amplitude
        C_init = 1.0  # Decay rate
        
        try:
            # Fit exponential model
            popt, pcov = curve_fit(
                exp_model, x, y, 
                p0=[A_init, B_init, C_init],
                maxfev=self.max_iterations,
                bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf])
            )
            
            A, B, C = popt
            extrapolated = A + B  # E(0) = A + B*exp(0) = A + B
            
            # Calculate fit quality
            y_pred = exp_model(x, A, B, C)
            r_squared = self._calculate_r_squared(y, y_pred)
            
            # Calculate parameter uncertainties
            param_std = np.sqrt(np.diag(pcov))
            
            fit_data = {
                "method": "exponential",
                "r_squared": r_squared,
                "coefficients": {"A": A, "B": B, "C": C},
                "parameter_std": {"A_std": param_std[0], "B_std": param_std[1], "C_std": param_std[2]},
                "extrapolation_std": np.sqrt(param_std[0]**2 + param_std[1]**2),  # Error propagation
                "fitted_curve": lambda lam: exp_model(lam, A, B, C)
            }
            
            return extrapolated, fit_data
            
        except (RuntimeError, ValueError) as e:
            # Fallback to linear extrapolation
            warnings.warn(f"Exponential fit failed ({e}), falling back to linear")
            linear_extrap = RichardsonExtrapolator(order=1)
            return linear_extrap.extrapolate(x, y)
    
    def _calculate_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate coefficient of determination (R²)."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return 1 - (ss_res / ss_tot)


class PolynomialExtrapolator(Extrapolator):
    """
    Polynomial extrapolation for ZNE.
    
    Fits polynomial model: E(λ) = c₀ + c₁λ + c₂λ² + ... + cₙλⁿ
    and extrapolates to λ=0 giving E(0) = c₀.
    """
    
    def __init__(self, degree: Optional[int] = None, max_degree: int = 5):
        """
        Initialize polynomial extrapolator.
        
        Args:
            degree: Fixed polynomial degree (if None, auto-select)
            max_degree: Maximum degree for auto-selection
        """
        self.degree = degree
        self.max_degree = max_degree
    
    def extrapolate(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Perform polynomial extrapolation."""
        if self.degree is not None:
            return self._fit_polynomial(x, y, self.degree)
        else:
            return self._auto_select_degree(x, y)
    
    def _auto_select_degree(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Automatically select polynomial degree using cross-validation."""
        max_testable_degree = min(self.max_degree, len(x) - 1)
        
        best_degree = 1
        best_score = -np.inf
        best_result = None
        
        for degree in range(1, max_testable_degree + 1):
            try:
                score = self._cross_validate_polynomial(x, y, degree)
                if score > best_score:
                    best_score = score
                    best_degree = degree
                    best_result = self._fit_polynomial(x, y, degree)
            except:
                continue
        
        if best_result is None:
            # Fallback to linear
            best_result = self._fit_polynomial(x, y, 1)
        
        # Add auto-selection info to fit data
        extrapolated, fit_data = best_result
        fit_data["auto_selected_degree"] = best_degree
        fit_data["degree_selection_score"] = best_score
        
        return extrapolated, fit_data
    
    def _cross_validate_polynomial(self, x: np.ndarray, y: np.ndarray, degree: int) -> float:
        """Cross-validate polynomial fit using leave-one-out."""
        if len(x) <= degree:
            return -np.inf
        
        errors = []
        for i in range(len(x)):
            # Leave out point i
            x_train = np.concatenate([x[:i], x[i+1:]])
            y_train = np.concatenate([y[:i], y[i+1:]])
            x_test, y_test = x[i], y[i]
            
            try:
                # Fit polynomial
                coeffs = np.polyfit(x_train, y_train, degree)
                y_pred = np.polyval(coeffs, x_test)
                errors.append((y_test - y_pred)**2)
            except:
                return -np.inf
        
        return -np.mean(errors)  # Return negative MSE (higher is better)
    
    def _fit_polynomial(self, x: np.ndarray, y: np.ndarray, degree: int) -> Tuple[float, Dict[str, Any]]:
        """Fit polynomial of specified degree."""
        if len(x) <= degree:
            raise ValueError(f"Need at least {degree + 1} points for degree {degree} polynomial")
        
        try:
            # Fit polynomial
            coeffs = np.polyfit(x, y, degree)
            extrapolated = coeffs[-1]  # Constant term is E(0)
            
            # Calculate fit quality
            y_pred = np.polyval(coeffs, x)
            r_squared = self._calculate_r_squared(y, y_pred)
            
            # Calculate residuals
            residuals = y - y_pred
            rmse = np.sqrt(np.mean(residuals**2))
            
            fit_data = {
                "method": "polynomial",
                "degree": degree,
                "r_squared": r_squared,
                "rmse": rmse,
                "coefficients": coeffs.tolist(),
                "residuals": residuals.tolist(),
                "fitted_curve": lambda lam: np.polyval(coeffs, lam)
            }
            
            return extrapolated, fit_data
            
        except np.linalg.LinAlgError:
            raise ValueError(f"Polynomial fit failed for degree {degree}")
    
    def _calculate_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate coefficient of determination (R²)."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return 1 - (ss_res / ss_tot)


class AdaptiveExtrapolator(Extrapolator):
    """
    Adaptive extrapolator that selects the best method automatically.
    
    Tries multiple extrapolation methods and selects the one with
    the best fit quality based on statistical criteria.
    """
    
    def __init__(self, methods: Optional[List[str]] = None):
        """
        Initialize adaptive extrapolator.
        
        Args:
            methods: List of methods to try (if None, use default set)
        """
        if methods is None:
            methods = ["richardson", "exponential", "polynomial"]
        
        self.methods = methods
        self.extrapolators = {
            "richardson": RichardsonExtrapolator(order=1),
            "richardson_quad": RichardsonExtrapolator(order=2),
            "exponential": ExponentialExtrapolator(),
            "polynomial": PolynomialExtrapolator()
        }
    
    def extrapolate(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Select best extrapolation method automatically."""
        results = {}
        
        # Try each method
        for method in self.methods:
            if method == "richardson":
                for order in [1, 2]:
                    method_name = f"richardson_order_{order}"
                    try:
                        extrap = RichardsonExtrapolator(order=order)
                        result = extrap.extrapolate(x, y)
                        results[method_name] = result
                    except Exception as e:
                        results[method_name] = None
            else:
                try:
                    extrap = self.extrapolators[method]
                    result = extrap.extrapolate(x, y)
                    results[method] = result
                except Exception as e:
                    results[method] = None
        
        # Select best method based on fit quality
        best_method, best_result = self._select_best_method(results)
        
        # Add selection metadata
        extrapolated, fit_data = best_result
        fit_data["adaptive_selection"] = {
            "selected_method": best_method,
            "all_results": {k: v[1] if v else None for k, v in results.items()},
            "selection_criteria": "r_squared"
        }
        
        return extrapolated, fit_data
    
    def _select_best_method(self, results: Dict[str, Any]) -> Tuple[str, Tuple[float, Dict[str, Any]]]:
        """Select the best extrapolation method based on fit quality."""
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            raise ValueError("All extrapolation methods failed")
        
        # Select based on R² score
        best_score = -np.inf
        best_method = None
        best_result = None
        
        for method, (value, fit_data) in valid_results.items():
            score = fit_data.get("r_squared", -np.inf)
            if score > best_score:
                best_score = score
                best_method = method
                best_result = (value, fit_data)
        
        return best_method, best_result


# Utility function to create extrapolator from string
def create_extrapolator(method: str, **kwargs) -> Extrapolator:
    """
    Create an extrapolator from string specification.
    
    Args:
        method: Extrapolation method ("richardson", "exponential", "polynomial", "adaptive")
        **kwargs: Additional arguments for extrapolator initialization
        
    Returns:
        Configured extrapolator
    """
    extrapolators = {
        "richardson": RichardsonExtrapolator,
        "exponential": ExponentialExtrapolator,
        "polynomial": PolynomialExtrapolator,
        "adaptive": AdaptiveExtrapolator
    }
    
    if method not in extrapolators:
        available = ", ".join(extrapolators.keys())
        raise ValueError(f"Unknown extrapolator '{method}'. Available: {available}")
    
    return extrapolators[method](**kwargs)