import os
import pandas as pd
from typing import Dict, Optional


class BenchmarkReport:
    """
    Aggregates test results from multiple tasks and models into a DataFrame.
    
    The DataFrame structure:
        - 'task': Task name (e.g., 'gue_promoter_all')
        - 'metric': Metric name (e.g., 'classification_accuracy')
        - Additional columns: One per model, containing scores or NO_RESULTS
    """
    
    NO_RESULTS = "NO_RESULTS"
    
    def __init__(self, csv_path: str) -> None:
        """
        Initialize a benchmark report.
        
        If csv_path exists, loads existing data from CSV.
        Otherwise, creates an empty report.
        
        Args:
            csv_path: Path to CSV file. If exists, loads data from it.
                      This path is also used for save_csv().
        """
        self._csv_path = csv_path
        
        if os.path.exists(csv_path):
            self._df = pd.read_csv(csv_path)
        else:
            self._df = pd.DataFrame(columns=['task', 'metric'])
    
    def add_scores(self, task_name: str, model_name: str, results: Dict[str, Optional[float]]) -> None:
        """
        Add scores for a task/model combination to the report.
        
        Args:
            task_name: Name of the task (e.g., 'gue_promoter_all')
            model_name: Name of the model (e.g., 'dna_bert')
            results: Dictionary mapping metric names to scores (from task.test())
        """
        # Add model column if it doesn't exist, filling existing rows with NO_RESULTS
        if model_name not in self._df.columns:
            self._df[model_name] = self.NO_RESULTS
        
        # Process each metric in the results
        for metric_name, score in results.items():
            # Check if this task/metric combination already exists
            mask = (self._df['task'] == task_name) & (self._df['metric'] == metric_name)
            
            if mask.any():
                # Update existing row
                self._df.loc[mask, model_name] = score
            else:
                # Create new row with NO_RESULTS for all existing models
                new_row = {'task': task_name, 'metric': metric_name}
                for col in self._df.columns:
                    if col not in ['task', 'metric']:
                        new_row[col] = self.NO_RESULTS
                # Set the score for the current model
                new_row[model_name] = score
                
                self._df = pd.concat([self._df, pd.DataFrame([new_row])], ignore_index=True)
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Get a copy of the results DataFrame.
        
        Returns:
            DataFrame with task, metric, and model columns
        """
        return self._df.copy()
    
    def save_csv(self) -> None:
        """Save the report to the CSV file specified during initialization."""
        self._df.to_csv(self._csv_path, index=False)
    
    def __repr__(self) -> str:
        """Return string representation of the report."""
        return f"BenchmarkReport(\n{self._df.to_string()}\n)"

