"""
MLflow integration utilities for experiment tracking.
"""

import logging
from typing import Optional, Any
from pathlib import Path

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not installed. MLflow logging will be disabled.")


def setup_mlflow(config: dict) -> bool:
    """
    Setup MLflow tracking.
    
    Args:
        config: Configuration dictionary with 'mlflow' section
    
    Returns:
        True if MLflow was successfully setup, False otherwise
    """
    if not MLFLOW_AVAILABLE:
        return False
    
    if 'mlflow' not in config:
        return False
    
    mlflow_config = config['mlflow']
    
    # Set tracking URI
    if 'tracking_uri' in mlflow_config:
        mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
    
    # Set experiment
    if 'experiment_name' in mlflow_config:
        mlflow.set_experiment(mlflow_config['experiment_name'])
    
    # Handle credentials if provided
    if 'credentials' in mlflow_config:
        credentials = mlflow_config['credentials']
        if 'username' in credentials and 'password' in credentials:
            import os
            os.environ['MLFLOW_TRACKING_USERNAME'] = credentials['username']
            os.environ['MLFLOW_TRACKING_PASSWORD'] = credentials['password']
    
    return True


def start_run(run_name: Optional[str] = None, tags: Optional[dict] = None):
    """
    Start MLflow run.
    
    Args:
        run_name: Name for this run (optional)
        tags: Tags to add to this run (optional)
    
    Returns:
        MLflow run object, or None if MLflow not available
    """
    if not MLFLOW_AVAILABLE:
        return None
    
    return mlflow.start_run(run_name=run_name, tags=tags)


def end_run():
    """End MLflow run."""
    if MLFLOW_AVAILABLE:
        mlflow.end_run()


def log_param(key: str, value: Any):
    """
    Log a single parameter.
    
    Args:
        key: Parameter name
        value: Parameter value
    """
    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_param(key, value)
        except Exception as e:
            logging.warning(f"Failed to log MLflow param {key}: {e}")


def log_params(params: dict):
    """
    Log multiple parameters.
    
    Args:
        params: Dictionary of parameter names and values
    """
    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_params(params)
        except Exception as e:
            logging.warning(f"Failed to log MLflow params: {e}")


def log_metric(key: str, value: float, step: Optional[int] = None):
    """
    Log a single metric.
    
    Args:
        key: Metric name
        value: Metric value
        step: Step/epoch number (optional)
    """
    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logging.warning(f"Failed to log MLflow metric {key}: {e}")


def log_metrics(metrics: dict, step: Optional[int] = None):
    """
    Log multiple metrics.
    
    Args:
        metrics: Dictionary of metric names and values
        step: Step/epoch number (optional)
    """
    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logging.warning(f"Failed to log MLflow metrics: {e}")


def log_artifact(local_path: str, artifact_path: Optional[str] = None):
    """
    Log a file or directory as an artifact.
    
    Args:
        local_path: Path to local file or directory
        artifact_path: Path in MLflow artifact store (optional)
    """
    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_artifact(local_path, artifact_path)
        except Exception as e:
            logging.warning(f"Failed to log MLflow artifact {local_path}: {e}")


def log_config_as_artifact(config: dict, filename: str = "config.yaml"):
    """
    Log configuration as YAML artifact.
    
    Args:
        config: Configuration dictionary
        filename: Filename for the artifact (default: config.yaml)
    """
    if not MLFLOW_AVAILABLE:
        return
    
    try:
        import yaml
        import tempfile
        
        # Save config to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
            temp_path = f.name
        
        # Log as artifact
        mlflow.log_artifact(temp_path, artifact_path=filename)
        
        # Clean up temporary file
        Path(temp_path).unlink()
        
    except Exception as e:
        logging.warning(f"Failed to log config as MLflow artifact: {e}")


def log_model_artifact(model_path: str, artifact_path: str = "model"):
    """
    Log model checkpoint as artifact.
    
    Args:
        model_path: Path to model checkpoint file
        artifact_path: Path in MLflow artifact store (default: "model")
    """
    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_artifact(model_path, artifact_path)
        except Exception as e:
            logging.warning(f"Failed to log model as MLflow artifact: {e}")


def log_dict(dictionary: dict, artifact_file: str):
    """
    Log dictionary as JSON artifact.
    
    Args:
        dictionary: Dictionary to log
        artifact_file: Filename for the artifact (e.g., "metrics.json")
    """
    if not MLFLOW_AVAILABLE:
        return
    
    try:
        mlflow.log_dict(dictionary, artifact_file)
    except Exception as e:
        logging.warning(f"Failed to log dict as MLflow artifact: {e}")


def set_tag(key: str, value: Any):
    """
    Set a tag for the current run.
    
    Args:
        key: Tag name
        value: Tag value
    """
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tag(key, value)
        except Exception as e:
            logging.warning(f"Failed to set MLflow tag {key}: {e}")


def set_tags(tags: dict):
    """
    Set multiple tags for the current run.
    
    Args:
        tags: Dictionary of tag names and values
    """
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tags(tags)
        except Exception as e:
            logging.warning(f"Failed to set MLflow tags: {e}")


def log_hyperparameters(config: dict):
    """
    Log hyperparameters from config to MLflow.
    
    Extracts relevant hyperparameters from config and logs them.
    
    Args:
        config: Configuration dictionary
    """
    if not MLFLOW_AVAILABLE:
        return
    
    try:
        # Experiment info
        if 'experiment' in config:
            log_params({
                'experiment_name': config['experiment'].get('name'),
                'seed': config['experiment'].get('seed'),
                'device': config['experiment'].get('device')
            })
        
        # Model params
        if 'model' in config:
            model_params = {
                f"model_{k}": v 
                for k, v in config['model'].get('params', {}).items()
            }
            log_params(model_params)
        
        # Training params
        if 'train' in config:
            train = config['train']
            train_params = {
                'epochs': train.get('epochs'),
                'batch_size': train.get('loader_params', {}).get('batch_size')
            }
            
            # Optimizer params
            if 'optimizer_config' in train:
                opt_params = {
                    f"optimizer_{k}": v 
                    for k, v in train['optimizer_config'].items()
                }
                train_params.update(opt_params)
            
            # Loss params
            if 'loss' in train:
                loss_params = {
                    f"loss_{k}": v 
                    for k, v in train['loss'].get('params', {}).items()
                }
                train_params.update(loss_params)
            
            log_params(train_params)
        
    except Exception as e:
        logging.warning(f"Failed to log hyperparameters: {e}")


class MLflowLogger:
    """
    Context manager for MLflow logging.
    
    Usage:
        with MLflowLogger(config) as mlf:
            mlf.log_metric('loss', 0.5, step=1)
    """
    
    def __init__(self, config: dict, run_name: Optional[str] = None):
        self.config = config
        self.run_name = run_name
        self.enabled = MLFLOW_AVAILABLE and 'mlflow' in config
        self.run = None
    
    def __enter__(self):
        if self.enabled:
            setup_mlflow(self.config)
            self.run = start_run(run_name=self.run_name)
            
            # Log config and hyperparameters
            log_config_as_artifact(self.config)
            log_hyperparameters(self.config)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            end_run()
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a metric."""
        log_metric(key, value, step)
    
    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """Log multiple metrics."""
        log_metrics(metrics, step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact."""
        log_artifact(local_path, artifact_path)