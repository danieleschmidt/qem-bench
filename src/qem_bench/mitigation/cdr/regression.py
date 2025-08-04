"""Regression models for Clifford Data Regression."""

import jax
import jax.numpy as jnp
from jax import random, vmap, jit, grad
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from abc import ABC, abstractmethod
import time
import warnings

# Optional scikit-learn import for comparison
try:
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available, using JAX implementations only")


class BaseRegressor(ABC):
    """Abstract base class for regression models used in CDR."""
    
    @abstractmethod
    def fit(
        self, 
        X: np.ndarray, 
        y_ideal: np.ndarray, 
        y_noisy: np.ndarray
    ) -> None:
        """
        Fit the regression model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y_ideal: Ideal expectation values
            y_noisy: Noisy expectation values
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> Union[float, np.ndarray]:
        """
        Predict ideal values from features.
        
        Args:
            X: Feature matrix or single feature vector
            
        Returns:
            Predicted ideal values
        """
        pass
    
    @abstractmethod
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score of the model.
        
        Args:
            X: Feature matrix
            y: True values
            
        Returns:
            R² score
        """
        pass


class RidgeRegressor(BaseRegressor):
    """
    Ridge regression for CDR using JAX.
    
    Implements L2-regularized linear regression to predict ideal expectation
    values from circuit features and noisy measurements.
    """
    
    def __init__(
        self, 
        alpha: float = 1.0, 
        use_sklearn: bool = False,
        standardize: bool = True
    ):
        """
        Initialize Ridge regressor.
        
        Args:
            alpha: Regularization strength
            use_sklearn: Use scikit-learn implementation if available
            standardize: Whether to standardize features
        """
        self.alpha = alpha
        self.use_sklearn = use_sklearn and SKLEARN_AVAILABLE
        self.standardize = standardize
        
        # Model parameters
        self.weights = None
        self.bias = None
        self.scaler = None
        self.is_fitted = False
        
        if self.use_sklearn:
            self.sklearn_model = Ridge(alpha=alpha)
            if standardize:
                self.scaler = StandardScaler()
        else:
            # JAX implementation
            self._fit_jit = jit(self._fit_ridge_jax)
            self._predict_jit = jit(self._predict_jax)
    
    def fit(
        self, 
        X: np.ndarray, 
        y_ideal: np.ndarray, 
        y_noisy: np.ndarray
    ) -> None:
        """Fit Ridge regression model."""
        X = np.asarray(X)
        y_ideal = np.asarray(y_ideal)
        
        if self.use_sklearn:
            self._fit_sklearn(X, y_ideal)
        else:
            self._fit_jax(X, y_ideal)
        
        self.is_fitted = True
    
    def _fit_sklearn(self, X: np.ndarray, y_ideal: np.ndarray) -> None:
        """Fit using scikit-learn."""
        if self.standardize:
            X = self.scaler.fit_transform(X)
        
        self.sklearn_model.fit(X, y_ideal)
    
    def _fit_jax(self, X: np.ndarray, y_ideal: np.ndarray) -> None:
        """Fit using JAX implementation."""
        X_jax = jnp.array(X)
        y_jax = jnp.array(y_ideal)
        
        # Standardize features if requested
        if self.standardize:
            X_mean = jnp.mean(X_jax, axis=0)
            X_std = jnp.std(X_jax, axis=0) + 1e-8  # Avoid division by zero
            X_jax = (X_jax - X_mean) / X_std
            self.feature_mean = X_mean
            self.feature_std = X_std
        
        # Fit ridge regression
        weights, bias = self._fit_ridge_jax(X_jax, y_jax, self.alpha)
        self.weights = weights
        self.bias = bias
    
    def _fit_ridge_jax(
        self, 
        X: jnp.ndarray, 
        y: jnp.ndarray, 
        alpha: float
    ) -> Tuple[jnp.ndarray, float]:
        """JAX implementation of ridge regression."""
        n_samples, n_features = X.shape
        
        # Add intercept term
        X_with_intercept = jnp.column_stack([jnp.ones(n_samples), X])
        
        # Ridge regression: (X^T X + α I) w = X^T y
        XTX = X_with_intercept.T @ X_with_intercept
        XTy = X_with_intercept.T @ y
        
        # Add regularization (don't regularize intercept)
        regularization = alpha * jnp.eye(n_features + 1)
        regularization = regularization.at[0, 0].set(0)  # No regularization for intercept
        
        # Solve normal equations
        w = jnp.linalg.solve(XTX + regularization, XTy)
        
        bias = w[0]
        weights = w[1:]
        
        return weights, bias
    
    def predict(self, X: np.ndarray) -> Union[float, np.ndarray]:
        """Predict using fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if self.use_sklearn:
            if self.standardize:
                X = self.scaler.transform(X)
            predictions = self.sklearn_model.predict(X)
        else:
            X_jax = jnp.array(X)
            
            # Standardize if needed
            if self.standardize:
                X_jax = (X_jax - self.feature_mean) / self.feature_std
            
            predictions = self._predict_jax(X_jax, self.weights, self.bias)
            predictions = np.array(predictions)
        
        return predictions[0] if len(predictions) == 1 else predictions
    
    def _predict_jax(
        self, 
        X: jnp.ndarray, 
        weights: jnp.ndarray, 
        bias: float
    ) -> jnp.ndarray:
        """JAX prediction function."""
        return X @ weights + bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score."""
        y_pred = self.predict(X)
        if isinstance(y_pred, float):
            y_pred = np.array([y_pred])
        
        return r2_score(y, y_pred) if SKLEARN_AVAILABLE else self._r2_score_jax(y, y_pred)
    
    def _r2_score_jax(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """JAX implementation of R² score."""
        y_true = jnp.array(y_true)
        y_pred = jnp.array(y_pred)
        
        ss_res = jnp.sum((y_true - y_pred) ** 2)
        ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
        
        return float(1 - (ss_res / ss_tot)) if ss_tot > 1e-10 else 0.0


class LassoRegressor(BaseRegressor):
    """
    Lasso regression for CDR using JAX.
    
    Implements L1-regularized linear regression with feature selection
    capabilities for sparse circuit representations.
    """
    
    def __init__(
        self, 
        alpha: float = 1.0, 
        max_iter: int = 1000,
        tolerance: float = 1e-6,
        use_sklearn: bool = False,
        standardize: bool = True
    ):
        """
        Initialize Lasso regressor.
        
        Args:
            alpha: Regularization strength
            max_iter: Maximum optimization iterations
            tolerance: Convergence tolerance
            use_sklearn: Use scikit-learn implementation if available
            standardize: Whether to standardize features
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.use_sklearn = use_sklearn and SKLEARN_AVAILABLE
        self.standardize = standardize
        
        # Model parameters
        self.weights = None
        self.bias = None
        self.scaler = None
        self.is_fitted = False
        
        if self.use_sklearn:
            self.sklearn_model = Lasso(alpha=alpha, max_iter=max_iter, tol=tolerance)
            if standardize:
                self.scaler = StandardScaler()
        else:
            # JAX implementation using coordinate descent
            self._coordinate_descent_step = jit(self._coordinate_descent_update)
    
    def fit(
        self, 
        X: np.ndarray, 
        y_ideal: np.ndarray, 
        y_noisy: np.ndarray
    ) -> None:
        """Fit Lasso regression model."""
        X = np.asarray(X)
        y_ideal = np.asarray(y_ideal)
        
        if self.use_sklearn:
            self._fit_sklearn(X, y_ideal)
        else:
            self._fit_jax(X, y_ideal)
        
        self.is_fitted = True
    
    def _fit_sklearn(self, X: np.ndarray, y_ideal: np.ndarray) -> None:
        """Fit using scikit-learn."""
        if self.standardize:
            X = self.scaler.fit_transform(X)
        
        self.sklearn_model.fit(X, y_ideal)
    
    def _fit_jax(self, X: np.ndarray, y_ideal: np.ndarray) -> None:
        """Fit using JAX coordinate descent."""
        X_jax = jnp.array(X)
        y_jax = jnp.array(y_ideal)
        
        # Standardize features if requested
        if self.standardize:
            X_mean = jnp.mean(X_jax, axis=0)
            X_std = jnp.std(X_jax, axis=0) + 1e-8
            X_jax = (X_jax - X_mean) / X_std
            self.feature_mean = X_mean
            self.feature_std = X_std
        
        # Center target
        y_mean = jnp.mean(y_jax)
        y_centered = y_jax - y_mean
        self.target_mean = y_mean
        
        # Initialize weights
        n_features = X_jax.shape[1]
        weights = jnp.zeros(n_features)
        
        # Coordinate descent optimization
        for iteration in range(self.max_iter):
            weights_old = weights.copy()
            
            for j in range(n_features):
                # Compute residual without j-th feature
                residual = y_centered - X_jax @ weights + weights[j] * X_jax[:, j]
                
                # Update j-th weight using soft thresholding
                rho = jnp.dot(X_jax[:, j], residual)
                weights = weights.at[j].set(self._soft_threshold(rho, self.alpha))
            
            # Check convergence
            if jnp.max(jnp.abs(weights - weights_old)) < self.tolerance:
                break
        
        self.weights = weights
        self.bias = float(y_mean)
    
    def _soft_threshold(self, x: float, threshold: float) -> float:
        """Soft thresholding operator for Lasso."""
        return jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0.0)
    
    def _coordinate_descent_update(
        self, 
        X: jnp.ndarray, 
        y: jnp.ndarray, 
        weights: jnp.ndarray,
        j: int,
        alpha: float
    ) -> float:
        """Single coordinate descent update."""
        # Compute residual
        residual = y - X @ weights + weights[j] * X[:, j]
        rho = jnp.dot(X[:, j], residual)
        
        return self._soft_threshold(rho, alpha)
    
    def predict(self, X: np.ndarray) -> Union[float, np.ndarray]:
        """Predict using fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if self.use_sklearn:
            if self.standardize:
                X = self.scaler.transform(X)
            predictions = self.sklearn_model.predict(X)
        else:
            X_jax = jnp.array(X)
            
            # Standardize if needed
            if self.standardize:
                X_jax = (X_jax - self.feature_mean) / self.feature_std
            
            predictions = X_jax @ self.weights + self.bias
            predictions = np.array(predictions)
        
        return predictions[0] if len(predictions) == 1 else predictions
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score."""
        y_pred = self.predict(X)
        if isinstance(y_pred, float):
            y_pred = np.array([y_pred])
        
        return r2_score(y, y_pred) if SKLEARN_AVAILABLE else self._r2_score_jax(y, y_pred)
    
    def _r2_score_jax(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """JAX implementation of R² score."""
        y_true = jnp.array(y_true)
        y_pred = jnp.array(y_pred)
        
        ss_res = jnp.sum((y_true - y_pred) ** 2)
        ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
        
        return float(1 - (ss_res / ss_tot)) if ss_tot > 1e-10 else 0.0
    
    def get_selected_features(self) -> np.ndarray:
        """Get indices of features selected by Lasso (non-zero weights)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if self.use_sklearn:
            weights = self.sklearn_model.coef_
        else:
            weights = np.array(self.weights)
        
        return np.where(np.abs(weights) > 1e-10)[0]


class NeuralNetworkRegressor(BaseRegressor):
    """
    Neural network regressor for CDR using JAX.
    
    Implements a multi-layer perceptron for nonlinear regression
    when linear models are insufficient.
    """
    
    def __init__(
        self,
        layers: List[int] = [64, 32, 16],
        activation: str = "relu",
        learning_rate: float = 0.001,
        max_epochs: int = 1000,
        max_training_time: float = 60.0,  # seconds
        batch_size: int = 32,
        early_stopping_patience: int = 50,
        dropout_rate: float = 0.1,
        seed: int = 42
    ):
        """
        Initialize neural network regressor.
        
        Args:
            layers: Hidden layer sizes
            activation: Activation function ("relu", "tanh", "sigmoid")
            learning_rate: Learning rate for optimization
            max_epochs: Maximum training epochs
            max_training_time: Maximum training time in seconds
            batch_size: Mini-batch size
            early_stopping_patience: Early stopping patience
            dropout_rate: Dropout rate for regularization
            seed: Random seed
        """
        self.layers = layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.max_training_time = max_training_time
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.dropout_rate = dropout_rate
        
        # JAX setup
        self.key = random.PRNGKey(seed)
        
        # Model parameters
        self.params = None
        self.is_fitted = False
        self.training_history = []
        
        # Feature normalization
        self.feature_mean = None
        self.feature_std = None
        self.target_mean = None
        self.target_std = None
        
        # Compile JAX functions
        self._forward_pass = jit(self._network_forward)
        self._loss_fn = jit(self._compute_loss)
        self._update_step = jit(self._optimizer_step)
    
    def _init_network_params(self, input_dim: int) -> Dict[str, Any]:
        """Initialize network parameters."""
        params = {}
        layer_sizes = [input_dim] + self.layers + [1]
        
        for i in range(len(layer_sizes) - 1):
            key, subkey = random.split(self.key)
            self.key = key
            
            # Xavier initialization
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            bound = jnp.sqrt(6.0 / (fan_in + fan_out))
            
            params[f'W{i}'] = random.uniform(
                subkey, (fan_in, fan_out), minval=-bound, maxval=bound
            )
            params[f'b{i}'] = jnp.zeros(fan_out)
        
        return params
    
    def _activation_fn(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply activation function."""
        if self.activation == "relu":
            return jnp.maximum(0, x)
        elif self.activation == "tanh":
            return jnp.tanh(x)
        elif self.activation == "sigmoid":
            return 1.0 / (1.0 + jnp.exp(-x))
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _network_forward(
        self, 
        params: Dict[str, jnp.ndarray], 
        x: jnp.ndarray,
        training: bool = False
    ) -> jnp.ndarray:
        """Forward pass through network."""
        h = x
        
        # Hidden layers
        for i in range(len(self.layers)):
            h = h @ params[f'W{i}'] + params[f'b{i}']
            h = self._activation_fn(h)
            
            # Apply dropout during training
            if training and self.dropout_rate > 0:
                key, subkey = random.split(self.key)
                self.key = key
                keep_prob = 1.0 - self.dropout_rate
                mask = random.bernoulli(subkey, keep_prob, h.shape)
                h = h * mask / keep_prob
        
        # Output layer (no activation)
        output_idx = len(self.layers)
        output = h @ params[f'W{output_idx}'] + params[f'b{output_idx}']
        
        return output.squeeze()
    
    def _compute_loss(
        self, 
        params: Dict[str, jnp.ndarray], 
        X: jnp.ndarray, 
        y: jnp.ndarray,
        training: bool = False
    ) -> float:
        """Compute mean squared error loss."""
        predictions = self._network_forward(params, X, training)
        return jnp.mean((predictions - y) ** 2)
    
    def _optimizer_step(
        self, 
        params: Dict[str, jnp.ndarray], 
        X: jnp.ndarray, 
        y: jnp.ndarray,
        learning_rate: float
    ) -> Dict[str, jnp.ndarray]:
        """Single optimizer step using gradient descent."""
        grads = grad(self._compute_loss)(params, X, y, training=True)
        
        updated_params = {}
        for key in params:
            updated_params[key] = params[key] - learning_rate * grads[key]
        
        return updated_params
    
    def fit(
        self, 
        X: np.ndarray, 
        y_ideal: np.ndarray, 
        y_noisy: np.ndarray
    ) -> None:
        """Fit neural network model."""
        X = np.asarray(X, dtype=np.float32)
        y_ideal = np.asarray(y_ideal, dtype=np.float32)
        
        # Normalize features and targets
        self.feature_mean = jnp.mean(X, axis=0)
        self.feature_std = jnp.std(X, axis=0) + 1e-8
        X_norm = (X - self.feature_mean) / self.feature_std
        
        self.target_mean = jnp.mean(y_ideal)
        self.target_std = jnp.std(y_ideal) + 1e-8
        y_norm = (y_ideal - self.target_mean) / self.target_std
        
        # Convert to JAX arrays
        X_jax = jnp.array(X_norm)
        y_jax = jnp.array(y_norm)
        
        # Initialize parameters
        self.params = self._init_network_params(X.shape[1])
        
        # Training loop
        n_samples = X.shape[0]
        n_batches = max(1, n_samples // self.batch_size)
        
        best_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        self.training_history = []
        
        for epoch in range(self.max_epochs):
            # Check time limit
            if time.time() - start_time > self.max_training_time:
                print(f"Training stopped due to time limit at epoch {epoch}")
                break
            
            # Shuffle data
            key, subkey = random.split(self.key)
            self.key = key
            indices = random.permutation(subkey, n_samples)
            X_shuffled = X_jax[indices]
            y_shuffled = y_jax[indices]
            
            # Mini-batch training
            epoch_loss = 0.0
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Update parameters
                self.params = self._update_step(
                    self.params, X_batch, y_batch, self.learning_rate
                )
                
                # Accumulate loss
                batch_loss = self._loss_fn(self.params, X_batch, y_batch)
                epoch_loss += batch_loss
            
            epoch_loss /= n_batches
            self.training_history.append(float(epoch_loss))
            
            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # Print progress occasionally
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.6f}")
        
        self.is_fitted = True
        print(f"Training completed. Final loss: {self.training_history[-1]:.6f}")
    
    def predict(self, X: np.ndarray) -> Union[float, np.ndarray]:
        """Predict using fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Normalize features
        X_norm = (X - self.feature_mean) / self.feature_std
        X_jax = jnp.array(X_norm)
        
        # Make prediction
        y_pred_norm = self._forward_pass(self.params, X_jax, training=False)
        
        # Denormalize
        y_pred = y_pred_norm * self.target_std + self.target_mean
        y_pred = np.array(y_pred)
        
        return y_pred[0] if len(y_pred) == 1 else y_pred
    
    def predict_with_uncertainty(
        self, 
        X: np.ndarray, 
        n_samples: int = 100
    ) -> np.ndarray:
        """Predict with uncertainty estimation using dropout."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Normalize features
        X_norm = (X - self.feature_mean) / self.feature_std
        X_jax = jnp.array(X_norm)
        
        # Generate predictions with dropout
        predictions = []
        for _ in range(n_samples):
            y_pred_norm = self._forward_pass(self.params, X_jax, training=True)
            y_pred = y_pred_norm * self.target_std + self.target_mean
            predictions.append(np.array(y_pred))
        
        return np.array(predictions)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score."""
        y_pred = self.predict(X)
        if isinstance(y_pred, float):
            y_pred = np.array([y_pred])
        
        return r2_score(y, y_pred) if SKLEARN_AVAILABLE else self._r2_score_jax(y, y_pred)
    
    def _r2_score_jax(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """JAX implementation of R² score."""
        y_true = jnp.array(y_true)
        y_pred = jnp.array(y_pred)
        
        ss_res = jnp.sum((y_true - y_pred) ** 2)
        ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
        
        return float(1 - (ss_res / ss_tot)) if ss_tot > 1e-10 else 0.0
    
    def get_training_history(self) -> List[float]:
        """Get training loss history."""
        return self.training_history.copy()


# Utility function to create regressor from string
def create_regressor(method: str, **kwargs) -> BaseRegressor:
    """
    Create a regressor from string specification.
    
    Args:
        method: Regression method ("ridge", "lasso", "neural")
        **kwargs: Additional arguments for regressor initialization
        
    Returns:
        Configured regressor
    """
    regressors = {
        "ridge": RidgeRegressor,
        "lasso": LassoRegressor,
        "neural": NeuralNetworkRegressor
    }
    
    if method not in regressors:
        available = ", ".join(regressors.keys())
        raise ValueError(f"Unknown regression method '{method}'. Available: {available}")
    
    return regressors[method](**kwargs)