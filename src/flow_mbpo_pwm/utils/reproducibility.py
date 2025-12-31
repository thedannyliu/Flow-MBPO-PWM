"""
Data consistency verification for reproducible experiments.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import numpy as np


class DatasetVerifier:
    """
    Verify dataset consistency across experiments.
    """
    
    def __init__(self, dataset_root: str):
        """
        Args:
            dataset_root: Root directory containing datasets
        """
        self.dataset_root = Path(dataset_root)
        self.manifest_file = self.dataset_root / "dataset_manifest.json"
        self.manifest = self._load_or_create_manifest()
    
    def _load_or_create_manifest(self) -> Dict[str, Any]:
        """Load existing manifest or create new one."""
        if self.manifest_file.exists():
            with open(self.manifest_file, 'r') as f:
                return json.load(f)
        else:
            return {}
    
    def _save_manifest(self):
        """Save manifest to file."""
        self.manifest_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_file, 'w') as f:
            json.dump(self.manifest, f, indent=2)
    
    @staticmethod
    def compute_tensor_hash(tensor: torch.Tensor) -> str:
        """
        Compute hash of a tensor.
        
        Args:
            tensor: PyTorch tensor
        
        Returns:
            SHA256 hash string
        """
        # Convert to numpy and then to bytes
        arr = tensor.cpu().numpy()
        arr_bytes = arr.tobytes()
        return hashlib.sha256(arr_bytes).hexdigest()
    
    @staticmethod
    def compute_file_hash(filepath: str) -> str:
        """
        Compute hash of a file.
        
        Args:
            filepath: Path to file
        
        Returns:
            SHA256 hash string
        """
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def register_dataset(
        self,
        dataset_name: str,
        dataset_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register a dataset and compute its hash.
        
        Args:
            dataset_name: Name of the dataset
            dataset_path: Path to dataset file
            metadata: Optional metadata dictionary
        
        Returns:
            Dataset hash
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # Compute hash
        dataset_hash = self.compute_file_hash(str(dataset_path))
        
        # Register in manifest
        self.manifest[dataset_name] = {
            'path': str(dataset_path),
            'hash': dataset_hash,
            'size': dataset_path.stat().st_size,
            'metadata': metadata or {},
        }
        
        self._save_manifest()
        
        print(f"Registered dataset '{dataset_name}'")
        print(f"  Path: {dataset_path}")
        print(f"  Hash: {dataset_hash[:16]}...")
        print(f"  Size: {dataset_path.stat().st_size / (1024**2):.2f} MB")
        
        return dataset_hash
    
    def verify_dataset(self, dataset_name: str, dataset_path: str) -> bool:
        """
        Verify that a dataset matches the registered hash.
        
        Args:
            dataset_name: Name of the dataset
            dataset_path: Path to dataset file to verify
        
        Returns:
            True if hash matches, False otherwise
        """
        if dataset_name not in self.manifest:
            print(f"WARNING: Dataset '{dataset_name}' not registered in manifest")
            return False
        
        expected_hash = self.manifest[dataset_name]['hash']
        actual_hash = self.compute_file_hash(dataset_path)
        
        if expected_hash == actual_hash:
            print(f"✓ Dataset '{dataset_name}' verified successfully")
            return True
        else:
            print(f"✗ Dataset '{dataset_name}' HASH MISMATCH!")
            print(f"  Expected: {expected_hash[:16]}...")
            print(f"  Actual:   {actual_hash[:16]}...")
            return False
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a registered dataset.
        
        Args:
            dataset_name: Name of the dataset
        
        Returns:
            Dataset information dictionary or None if not found
        """
        return self.manifest.get(dataset_name)
    
    def list_datasets(self):
        """List all registered datasets."""
        print("\n" + "="*60)
        print("Registered Datasets")
        print("="*60)
        
        if not self.manifest:
            print("No datasets registered yet.")
        else:
            for name, info in self.manifest.items():
                print(f"\nDataset: {name}")
                print(f"  Path: {info['path']}")
                print(f"  Hash: {info['hash'][:16]}...")
                print(f"  Size: {info['size'] / (1024**2):.2f} MB")
                if info.get('metadata'):
                    print(f"  Metadata: {info['metadata']}")
        
        print("="*60 + "\n")


class ExperimentConfig:
    """
    Manages experiment configuration with hashing for reproducibility.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.config_hash = self._compute_config_hash()
    
    def _compute_config_hash(self) -> str:
        """Compute hash of configuration."""
        # Convert config to sorted JSON string for consistent hashing
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def save(self, filepath: str):
        """
        Save configuration to file.
        
        Args:
            filepath: Path to save configuration
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        config_with_hash = {
            'config': self.config,
            'config_hash': self.config_hash,
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_with_hash, f, indent=2)
        
        print(f"Saved configuration to {filepath}")
        print(f"Config hash: {self.config_hash[:16]}...")
    
    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """
        Load configuration from file.
        
        Args:
            filepath: Path to configuration file
        
        Returns:
            ExperimentConfig instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        config = cls(data['config'])
        
        # Verify hash
        saved_hash = data.get('config_hash')
        if saved_hash != config.config_hash:
            print("WARNING: Config hash mismatch!")
            print(f"  Saved:    {saved_hash[:16]}...")
            print(f"  Computed: {config.config_hash[:16]}...")
        
        return config
    
    def diff(self, other: 'ExperimentConfig') -> Dict[str, Any]:
        """
        Compare with another configuration.
        
        Args:
            other: Another ExperimentConfig instance
        
        Returns:
            Dictionary of differences
        """
        diffs = {}
        
        def recursive_diff(d1, d2, path=""):
            for key in set(list(d1.keys()) + list(d2.keys())):
                current_path = f"{path}.{key}" if path else key
                
                if key not in d1:
                    diffs[current_path] = {'self': None, 'other': d2[key]}
                elif key not in d2:
                    diffs[current_path] = {'self': d1[key], 'other': None}
                elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    recursive_diff(d1[key], d2[key], current_path)
                elif d1[key] != d2[key]:
                    diffs[current_path] = {'self': d1[key], 'other': d2[key]}
        
        recursive_diff(self.config, other.config)
        
        return diffs


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Make CUDA operations deterministic (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Set random seed to {seed}")


def verify_experiment_reproducibility(
    config_path: str,
    dataset_paths: Dict[str, str],
    dataset_verifier: DatasetVerifier,
) -> bool:
    """
    Verify that experiment can be reproduced.
    
    Args:
        config_path: Path to configuration file
        dataset_paths: Dictionary of dataset name -> path
        dataset_verifier: DatasetVerifier instance
    
    Returns:
        True if all checks pass
    """
    print("\n" + "="*60)
    print("Verifying Experiment Reproducibility")
    print("="*60)
    
    all_checks_passed = True
    
    # Check config
    print("\n[1/2] Verifying configuration...")
    config_path = Path(config_path)
    if config_path.exists():
        config = ExperimentConfig.load(str(config_path))
        print(f"✓ Configuration loaded: {config_path}")
    else:
        print(f"✗ Configuration not found: {config_path}")
        all_checks_passed = False
    
    # Check datasets
    print("\n[2/2] Verifying datasets...")
    for dataset_name, dataset_path in dataset_paths.items():
        if not dataset_verifier.verify_dataset(dataset_name, dataset_path):
            all_checks_passed = False
    
    print("\n" + "="*60)
    if all_checks_passed:
        print("✓ All reproducibility checks PASSED")
    else:
        print("✗ Some reproducibility checks FAILED")
    print("="*60 + "\n")
    
    return all_checks_passed
