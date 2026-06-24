"""
Improved parameter detection for data_analysis.py

Enhanced functions for automatically detecting varying parameters across multiple
simulation jobs with comprehensive recursive search through nested structures.
"""

import numpy as np


def _explore_nested_dict(d, prefix='', max_depth=5):
    """
    Recursively explore nested dictionary structure and return all leaf paths.

    Args:
        d: Dictionary to explore
        prefix: Current path prefix (for recursion)
        max_depth: Maximum recursion depth to prevent infinite loops

    Returns:
        list: List of (path, value_type) tuples for all leaf nodes
    """
    if max_depth == 0:
        return []

    paths = []

    for key, value in d.items():
        current_path = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            paths.extend(_explore_nested_dict(value, current_path, max_depth - 1))
        elif isinstance(value, (int, float, complex, str, bool, type(None))):
            paths.append((current_path, type(value).__name__))
        elif isinstance(value, np.ndarray):
            if value.size == 1:
                paths.append((current_path, 'scalar_array'))

    return paths


def _values_are_equal(values, rtol=1e-10, atol=1e-14):
    """
    Check if all values in list are equal, handling different types appropriately.

    Args:
        values: List of values to compare
        rtol: Relative tolerance for numerical comparison
        atol: Absolute tolerance for numerical comparison

    Returns:
        bool: True if all values are equal (within tolerance for numbers)
    """
    if len(values) <= 1:
        return True

    first = values[0]

    if first is None:
        return all(v is None for v in values)

    if isinstance(first, (int, float, complex)):
        try:
            return np.allclose(values, first, rtol=rtol, atol=atol)
        except (TypeError, ValueError):
            return all(v == first for v in values)

    return all(v == first for v in values)


def _find_varying_parameters_recursive(all_kwargs, max_params=3, priority_prefixes=None):
    """
    Recursively identify parameters that vary across multiple jobs.

    Automatically explores the entire nested structure of input_kwargs and identifies
    all parameters that differ between jobs.

    Args:
        all_kwargs: List of input_kwargs dicts from multiple jobs
        max_params: Maximum number of varying parameters to return
        priority_prefixes: List of parameter path prefixes to prioritize

    Returns:
        list: List of (param_path, values) tuples for varying parameters
    """
    if len(all_kwargs) <= 1:
        return []

    if priority_prefixes is None:
        priority_prefixes = [
            'field_params',
            'system_parameters',
            'grid_parameters',
            'num_timesteps',
        ]

    all_paths = _explore_nested_dict(all_kwargs[0])
    varying_params = []

    for param_path, value_type in all_paths:
        try:
            values = []
            for kwargs in all_kwargs:
                keys = param_path.split('.')
                value = kwargs
                for key in keys:
                    value = value[key]
                values.append(value)

            if not _values_are_equal(values):
                varying_params.append((param_path, values))

        except (KeyError, TypeError, AttributeError):
            continue

    if not varying_params:
        return []

    def get_priority(param_tuple):
        param_path, values = param_tuple
        for idx, prefix in enumerate(priority_prefixes):
            if param_path.startswith(prefix):
                return (0, idx, param_path)
        return (1, 0, param_path)

    varying_params.sort(key=get_priority)
    return varying_params[:max_params]


def _create_label_from_params_enhanced(job_idx, varying_params, values_for_job):
    """
    Create a label string from varying parameters with enhanced formatting.

    Args:
        job_idx: Job index
        varying_params: List of (param_path, all_values) tuples
        values_for_job: List of values for this specific job

    Returns:
        str: Formatted label string
    """
    if not varying_params:
        return f'Job {job_idx}'

    label_parts = []

    for (param_path, all_values), value in zip(varying_params, values_for_job):
        param_name = param_path.split('.')[-1]

        if value is None:
            formatted_value = 'None'
        elif isinstance(value, bool):
            formatted_value = str(value)
        elif isinstance(value, complex):
            if abs(value.imag) < 1e-12:
                formatted_value = f'{value.real:.4g}'
            else:
                formatted_value = f'{value:.4g}'
        elif isinstance(value, float):
            if 'temperature' in param_path.lower() or 'critical' in param_path.lower():
                formatted_value = f'{value:.3f}'
            elif 'eta' in param_path.lower() or 'broadening' in param_path.lower():
                formatted_value = f'{value:.3f}'
            elif 'amplitude' in param_path.lower():
                formatted_value = f'{value:.2f}'
            elif 'frequency' in param_path.lower():
                formatted_value = f'{value:.3f}'
            elif 'fwhm' in param_path.lower() or 'duration' in param_path.lower():
                formatted_value = f'{value:.3f}'
            elif abs(value) < 1e-3 or abs(value) > 1e4:
                formatted_value = f'{value:.2e}'
            else:
                formatted_value = f'{value:.4g}'
        elif isinstance(value, int):
            formatted_value = f'{value}'
        elif isinstance(value, str):
            formatted_value = value
        else:
            formatted_value = str(value)

        label_parts.append(f'{param_name}={formatted_value}')

    return ', '.join(label_parts)


def print_all_varying_parameters(all_kwargs, max_display=10):
    """
    Print all varying parameters found across jobs (for debugging).

    Args:
        all_kwargs: List of input_kwargs dicts
        max_display: Maximum number of parameters to display
    """
    varying_params = _find_varying_parameters_recursive(all_kwargs, max_params=100)

    print("\n" + "="*70)
    print(f"VARYING PARAMETERS ACROSS {len(all_kwargs)} JOBS")
    print("="*70)

    if not varying_params:
        print("No varying parameters found - all jobs have identical parameters")
        print("="*70 + "\n")
        return

    print(f"\nFound {len(varying_params)} varying parameter(s):\n")

    for idx, (param_path, values) in enumerate(varying_params[:max_display]):
        print(f"{idx+1}. {param_path}")
        print(f"   Values: {values}")

        if all(isinstance(v, (int, float)) for v in values):
            print(f"   Range: [{min(values):.6g}, {max(values):.6g}]")

        print()

    if len(varying_params) > max_display:
        print(f"... and {len(varying_params) - max_display} more")

    print("="*70 + "\n")
