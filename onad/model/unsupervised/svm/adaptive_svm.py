import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque
from onad.base.model import BaseModel


class IncrementalOneClassSVMAdaptiveKernel(BaseModel):
    """
    Incremental One-Class SVM with Adaptive Kernel Tuning.
    
    This implementation uses:
    - RBF kernel with adaptive gamma parameter
    - Incremental learning without full retraining
    - Automatic kernel parameter tuning based on data characteristics
    - Support vector budget management
    """
    
    def __init__(
        self,
        nu: float = 0.1,
        initial_gamma: float = 1.0,
        gamma_bounds: Tuple[float, float] = (0.001, 100.0),
        adaptation_rate: float = 0.1,
        buffer_size: int = 200,
        sv_budget: int = 100,
        tolerance: float = 1e-6
    ):
        """
        Initialize Incremental One-Class SVM with Adaptive Kernel.
        
        Args:
            nu: Upper bound on fraction of outliers (0 < nu <= 1)
            initial_gamma: Initial RBF kernel parameter
            gamma_bounds: Min/max bounds for gamma adaptation
            adaptation_rate: Rate of gamma adaptation (0 < rate < 1)
            buffer_size: Size of data buffer for statistics
            sv_budget: Maximum number of support vectors to maintain
            tolerance: Numerical tolerance for computations
        """
        self.nu = nu
        self.gamma = initial_gamma
        self.gamma_min, self.gamma_max = gamma_bounds
        self.adaptation_rate = adaptation_rate
        self.buffer_size = buffer_size
        self.sv_budget = sv_budget
        self.tolerance = tolerance
        
        # Model parameters
        self.support_vectors: List[np.ndarray] = []
        self.alpha: List[float] = []  # Lagrange multipliers for support vectors
        self.rho: float = 0.0  # Decision boundary offset
        
        # Adaptation mechanism
        self.data_buffer = deque(maxlen=buffer_size)
        self.distance_stats = deque(maxlen=50)  # Recent distance statistics
        self.n_samples = 0
        
        # Precomputed kernel matrix 
        self.K_sv: Optional[np.ndarray] = None  # Kernel matrix between SVs
        
        # Feature handling
        self.feature_order: Optional[Tuple[str, ...]] = None
        
    def _get_feature_vector(self, x: Dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to numpy array with consistent ordering."""
        if self.feature_order is None:
            self.feature_order = tuple(sorted(x.keys()))
        
        if tuple(sorted(x.keys())) != self.feature_order:
            raise ValueError("Inconsistent feature keys")
        
        return np.fromiter((x[k] for k in self.feature_order), dtype=np.float64)
    
    def _rbf_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute RBF kernel value between two vectors."""
        squared_distance = np.sum((x1 - x2) ** 2)
        return np.exp(-self.gamma * squared_distance)
    
    def _compute_kernel_row(self, x: np.ndarray) -> np.ndarray:
        """Compute kernel values between x and all support vectors."""
        if not self.support_vectors:
            return np.array([])
        return np.array([self._rbf_kernel(x, sv) for sv in self.support_vectors])
    
    def _update_kernel_matrix(self):
        """Update the kernel matrix between support vectors."""
        n_sv = len(self.support_vectors)
        if n_sv == 0:
            self.K_sv = None
            return
            
        self.K_sv = np.zeros((n_sv, n_sv))
        for i in range(n_sv):
            for j in range(i, n_sv):
                k_val = self._rbf_kernel(self.support_vectors[i], self.support_vectors[j])
                self.K_sv[i, j] = k_val
                self.K_sv[j, i] = k_val
    
    def _estimate_optimal_gamma(self) -> float:
        """Estimate optimal gamma based on data distribution."""
        if len(self.data_buffer) < 10:
            return self.gamma
            
        # Sample random pairs to estimate pairwise distances
        data_array = np.array(list(self.data_buffer))
        n_samples = min(50, len(data_array))
        
        if n_samples < 2:
            return self.gamma
            
        # Random sampling to avoid computational overhead
        indices = np.random.choice(len(data_array), size=n_samples, replace=False)
        sampled_data = data_array[indices]
        
        # Compute pairwise distances
        distances = []
        for i in range(len(sampled_data)):
            for j in range(i + 1, min(i + 10, len(sampled_data))):  # Limit pairs
                dist = np.linalg.norm(sampled_data[i] - sampled_data[j])
                if dist > 1e-10:  # Avoid division by zero
                    distances.append(dist)
        
        if not distances:
            return self.gamma
            
        # Use median distance for robust estimation
        median_distance = np.median(distances)
        
        # Heuristic: gamma = 1 / (2 * median_distance^2)
        optimal_gamma = 1.0 / (2.0 * median_distance ** 2)
        
        return np.clip(optimal_gamma, self.gamma_min, self.gamma_max)
    
    def _adapt_gamma(self):
        """Adapt gamma parameter based on recent data."""
        if self.n_samples % 20 != 0:  # Adapt every 20 samples
            return
            
        target_gamma = self._estimate_optimal_gamma()
        
        # Smooth adaptation
        gamma_diff = target_gamma - self.gamma
        self.gamma += self.adaptation_rate * gamma_diff
        self.gamma = np.clip(self.gamma, self.gamma_min, self.gamma_max)
        
        # Update kernel matrix if gamma changed significantly
        if abs(gamma_diff) > 0.01:
            self._update_kernel_matrix()
    
    def _manage_support_vectors(self, x: np.ndarray, alpha_new: float):
        """Add new support vector and manage budget."""
        self.support_vectors.append(x.copy())
        self.alpha.append(alpha_new)
        
        # Enforce budget constraint
        if len(self.support_vectors) > self.sv_budget:
            # Remove support vector with smallest |alpha|
            min_idx = np.argmin(np.abs(self.alpha))
            self.support_vectors.pop(min_idx)
            self.alpha.pop(min_idx)
        
        self._update_kernel_matrix()
    
    def _decision_function(self, x: np.ndarray) -> float:
        """Compute decision function value."""
        if not self.support_vectors:
            return -self.rho
            
        kernel_values = self._compute_kernel_row(x)
        decision_value = np.dot(self.alpha, kernel_values) - self.rho
        
        return decision_value
    
    def learn_one(self, x: Dict[str, float]):
        """
        Incrementally learn from one sample.
        
        Args:
            x: Input sample as dictionary {feature_name: value}
        """
        x_vec = self._get_feature_vector(x)
        self.n_samples += 1
        
        # Store in buffer for adaptation
        self.data_buffer.append(x_vec.copy())
        
        # Adapt gamma parameter
        self._adapt_gamma()
        
        # Compute current decision value
        decision_value = self._decision_function(x_vec)
        
        # Check if sample violates current decision boundary
        margin_violation = decision_value < 0
        
        if margin_violation or len(self.support_vectors) == 0:
            # If margin is violated or no support vectors yet, update model
            
            if len(self.support_vectors) == 0:
                # First sample - initialize
                self._manage_support_vectors(x_vec, 1.0 / self.nu)
                self.rho = self._rbf_kernel(x_vec, x_vec) / self.nu
            else:
                # Update alpha based on decision value                
                alpha_new = max(0, -decision_value + self.tolerance)
                alpha_new = min(alpha_new, 1.0 / self.nu)  # Upper bound
                
                if alpha_new > self.tolerance:
                    self._manage_support_vectors(x_vec, alpha_new)
                    
                    # Update rho
                    kernel_values = self._compute_kernel_row(x_vec)
                    self.rho += alpha_new * np.mean(kernel_values) if len(kernel_values) > 0 else 0
    
    def predict_one(self, x: Dict[str, float]) -> int:
        """
        Predict if sample is normal (1) or anomaly (-1).
        
        Args:
            x: Input sample as dictionary {feature_name: value}
            
        Returns:
            1 for normal, -1 for anomaly
        """
        if self.feature_order is None:
            return 1  # Default to normal if no training data
            
        x_vec = self._get_feature_vector(x)
        decision_value = self._decision_function(x_vec)
        return 1 if decision_value >= 0 else -1
    
    def score_one(self, x: Dict[str, float]) -> float:
        """
        Compute anomaly score for one sample.
        
        Args:
            x: Input sample as dictionary {feature_name: value}
            
        Returns:
            Anomaly score (higher = more anomalous)
        """
        if self.feature_order is None:
            return 0.0  # No training data, return neutral score
            
        x_vec = self._get_feature_vector(x)
        decision_value = self._decision_function(x_vec)
        # Return negative decision value 
        return -decision_value
    
    def get_model_info(self) -> Dict:
        """Get current model information."""
        return {
            'n_support_vectors': len(self.support_vectors),
            'gamma': self.gamma,
            'rho': self.rho,
            'n_samples_processed': self.n_samples,
            'buffer_size': len(self.data_buffer)
        }


