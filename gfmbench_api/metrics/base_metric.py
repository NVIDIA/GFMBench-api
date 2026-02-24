import logging
from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """
    Base class for all metrics.
    
    Usage pattern:
    1. Initialize the metric: metric = SomeMetric()
    2. Process batches: call metric.calc(batch_data) for each batch
    3. Get final result: score = metric.get_final_results()
    4. Reset for new run: metric.reset()
    
    The calc() method performs on-the-fly calculations per batch and stores only
    the minimal required intermediate results, making it memory-efficient.
    The get_final_results() method aggregates these intermediate results to compute
    the final metric score.
    
    Note: calc() automatically handles None inputs by printing a one-time warning
    and skipping that batch. Metrics with no valid data return None from get_final_results().
    """
    
    def __init__(self):
        """Initialize the metric. Subclasses can override to set up storage."""
        self.reset()
    
    def reset(self):
        """
        Reset the metric's internal state for a new evaluation run.
        
        Subclasses must override this to initialize their storage lists and must
        call super().reset() to properly initialize base class state (_none_warned flag).
        """
        self._none_warned = False
    
    def calc(self, *args, **kwargs):
        """
        Template method: checks for None inputs and delegates to _calc_impl().
        
        This method handles None input checking and one-time warning, then
        delegates to the concrete implementation in _calc_impl().
        
        Args:
            *args: Variable positional arguments specific to each metric
            **kwargs: Variable keyword arguments specific to each metric
        """
        # Check if any required positional arguments are None
        if any(arg is None for arg in args):
            if not self._none_warned:
                logging.warning(f"Model did not provide required inputs for '{self.name}' metric. This metric will return None.")
                self._none_warned = True
            return
        
        # Delegate to concrete implementation
        self._calc_impl(*args, **kwargs)
    
    @abstractmethod
    def _calc_impl(self, *args, **kwargs):
        """
        Concrete implementation of calc logic (implemented by subclasses).
        
        This method performs on-the-fly calculations per batch, converting inputs
        to numpy and computing intermediate values (e.g., predictions, scores).
        Only the minimal required information is stored for final aggregation.
        
        All inputs are expected to be numpy arrays (or convertible to numpy).
        All calculations are performed on CPU.
        Guaranteed to receive no None values (handled by base class).
        
        Args:
            *args: Variable positional arguments specific to each metric
            **kwargs: Variable keyword arguments specific to each metric
        """
        pass
    
    @abstractmethod
    def get_final_results(self):
        """
        Compute the final metric score from stored intermediate values.
        
        Must be called after calc() has been called at least once.
        Aggregates the intermediate values stored during calc() calls
        and computes the final metric.
        
        Returns:
            float or dict: The calculated metric score(s)
        """
        pass
    
    @property
    @abstractmethod
    def name(self):
        """
        Return the metric name used as the key in results dictionaries.
        
        This allows tasks to dynamically construct result dictionaries
        without hardcoding key names.
        
        Returns:
            str: The key name for this metric in the results dictionary
                 (e.g., 'accuracy', 'mcc', 'auroc_all_probs_llr')
        """
        pass

