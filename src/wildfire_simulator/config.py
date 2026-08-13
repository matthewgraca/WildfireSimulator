import yaml
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


def load_config(path=None):
    """Load configuration from YAML file.
    
    Args:
        path: Path to config file. Defaults to project root config.yaml.
    
    Returns:
        dict with parsed config values. Fraction strings (e.g. "1/48") 
        are evaluated to floats.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Evaluate fraction strings to floats
    if isinstance(config.get('dt'), str):
        config['dt'] = _eval_fraction(config['dt'])

    return config


def _eval_fraction(s):
    """Safely evaluate a fraction string like '1/48' to a float."""
    parts = s.strip().split('/')
    if len(parts) == 2:
        return float(parts[0]) / float(parts[1])
    return float(s)
