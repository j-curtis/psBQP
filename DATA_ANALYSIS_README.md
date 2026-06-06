# Data Analysis Module - Implementation Summary

## Overview

Created a clean, minimal data analysis module (`data_analysis.py`) for inspecting psBQP-Keldysh simulation results saved by the demler_tools framework.

## Files Created

### 1. `data_analysis.py` (392 lines)

Clean analysis module with four main functions:

#### `load_timestamp_data(timestamp, result_index=0)`
Load saved simulation results from demler_tools data directory.

**Returns:**
```python
{
    'state': StateObject,
    'gaps': array,
    'currents': array,
    'vector_potentials': array,
    'timestamp': timestamp,
    'result_index': result_index
}
```

#### `check_normalizations(timestamp, result_index=0, plot_data=True, save_dir='analysis_plots')`
Check gr and gk normalization relations.

**Returns:**
```python
{
    'gr_max_error': float,
    'gk_max_error': float,
    'gr_totals': array,
    'gk_totals': array
}
```

**Plots (if plot_data=True):**
- `norm_gr_{timestamp}.png` - g^R normalization check
- `norm_gk_{timestamp}.png` - g^K normalization check

#### `check_fdt(timestamp, result_index=0, plot_data=True, save_dir='analysis_plots')`
Check FDT relation: g^K = g^R @ f - f @ g^A.

**Returns:**
```python
{
    'max_error': float,
    'gk_actual': NambuKeldyshTensor,
    'gk_fdt': NambuKeldyshTensor,
    'error': NambuKeldyshTensor
}
```

**Plots (if plot_data=True):**
- `fdt_{timestamp}.png` - FDT comparison

#### `check_time_translational_invariance(timestamp, result_index=0, num_rows=10, plot_data=True, save_dir='analysis_plots')`
Check time-translation invariance for equilibrium state.

**Returns:**
```python
{
    'max_diff_gr': float,
    'max_diff_gk': float,
    'passed': bool,
    'threshold': float
}
```

**Plots (if plot_data=True):**
- `tti_{timestamp}.png` - Time-translation invariance check

### 2. `example_data_analysis.py` (67 lines)

Example script demonstrating how to use the module:

```python
from data_analysis import (
    load_timestamp_data,
    check_normalizations,
    check_fdt,
    check_time_translational_invariance
)

# Load data
data = load_timestamp_data(timestamp=1776335408, result_index=0)

# Check normalizations with plots
norm_results = check_normalizations(timestamp=1776335408, plot_data=True)

# Check FDT without plots
fdt_results = check_fdt(timestamp=1776335408, plot_data=False)

# Check time-translation invariance
tti_results = check_time_translational_invariance(timestamp=1776335408, num_rows=5)
```

## Files Modified

### `general_comparison_file.py`

**Removed verbose print statements from `load_state()` function:**

- ✓ State loaded confirmation
- Shape print statements
- Gap print statement
- Grid/system parameter prints
- Generation status messages
- Save confirmation print

**Kept only:**
- Error messages (file not found, loading errors)
- Warnings (save failures)

**Result:** Silent loading function that only prints errors.

## Key Features

### 1. Clean API Design
- Functions are silent except for errors
- All results returned as structured dicts
- Conditional plotting via `plot_data` parameter
- Consistent naming: `{test_type}_{timestamp}.png`

### 2. No Verbosity
- No decorative print headers (`===`)
- No intermediate status messages
- No "Computing..." or "Loading..." messages
- Functions return data, don't print it

### 3. Reusable Components
- Internal plotting functions prefixed with `_`
- Helper functions for gr, gk, FDT, time-translation plots
- All plots automatically closed (no interactive display)
- Automatic directory creation for plots

### 4. Compatible with demler_tools
- Uses `path_management.initialize()` for data directories
- Loads from timestamped result files (`sr_{index}`)
- Extracts all metadata (gaps, currents, vector potentials)

## Usage Examples

### Basic Analysis Workflow

```python
import sys
sys.path.append('/path/to/psBQP-keldysh')
from data_analysis import *

# 1. Load data
data = load_timestamp_data(timestamp=1776335408)
print(f"Loaded {len(data['gaps'])} timesteps")

# 2. Quick check without plots
norm = check_normalizations(1776335408, plot_data=False)
print(f"gr error: {norm['gr_max_error']:.2e}")

# 3. Full analysis with plots
fdt = check_fdt(1776335408, plot_data=True, save_dir='results')
tti = check_time_translational_invariance(1776335408, plot_data=True)

# 4. Inspect results
if tti['passed']:
    print("Equilibrium achieved!")
```

### Batch Analysis

```python
timestamps = [1776335408, 1776335500, 1776335600]

for ts in timestamps:
    try:
        # Quick validation (no plots)
        norm = check_normalizations(ts, plot_data=False)
        fdt = check_fdt(ts, plot_data=False)
        tti = check_time_translational_invariance(ts, plot_data=False)

        print(f"{ts}: gr={norm['gr_max_error']:.2e}, "
              f"fdt={fdt['max_error']:.2e}, "
              f"tti={tti['passed']}")
    except Exception as e:
        print(f"{ts}: Error - {e}")
```

### Selective Plotting

```python
# Load once, run multiple tests
data = load_timestamp_data(1776335408)

# Test without plots first
norm = check_normalizations(1776335408, plot_data=False)

# If issues detected, generate plots for debugging
if norm['gr_max_error'] > 1e-6:
    check_normalizations(1776335408, plot_data=True, save_dir='debug_plots')
```

## Design Philosophy

Following the project's coding standards:

1. **Performance-oriented** - Minimal overhead, direct data access
2. **Readable code** - Self-documenting structure, clear function names
3. **Minimal comments** - Code structure is self-evident
4. **Testing in notebooks** - Use Jupyter for interactive exploration
5. **No file bloat** - Single consolidated module, not many small files

## Comparison with Static_tests.py

| Feature | Static_tests.py | data_analysis.py |
|---------|-----------------|------------------|
| Verbosity | High (decorative prints) | Minimal (errors only) |
| Data loading | From local pickle | From demler_tools timestamps |
| Return values | Dicts | Dicts |
| Plotting | Always generated | Optional (plot_data flag) |
| Plot naming | Fixed names | Timestamp-based |
| Use case | Standalone testing | Pipeline integration |

## Integration with Existing Workflow

The new module complements existing tools:

- **Static_tests.py** - Verbose standalone testing with local files
- **general_comparison_file.py** - Tensor comparison utilities (now silent)
- **data_analysis.py** - Clean pipeline analysis for timestamped data

Use `data_analysis.py` when:
- Processing multiple simulation runs
- Building analysis pipelines
- Generating reports programmatically
- Need structured data returns for downstream processing

Use `Static_tests.py` when:
- Interactive debugging
- Need detailed progress feedback
- Working with ad-hoc pickle files
- Comprehensive test suite execution

## File Statistics

- `data_analysis.py`: 392 lines (4 main functions + 4 plotting helpers)
- `example_data_analysis.py`: 67 lines
- Modified `general_comparison_file.py`: Reduced from 276 to 254 lines
- Total new code: ~460 lines
- Lines removed: ~22 lines

## Next Steps

1. Test with actual timestamp data from cluster runs
2. Add to Jupyter notebooks for interactive analysis
3. Consider adding batch analysis helpers if needed
4. Document any edge cases discovered during usage
