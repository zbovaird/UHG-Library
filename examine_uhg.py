import inspect
import uhg
from uhg.projective import ProjectiveUHG

# List of functions to examine
functions = [
    'quadrance',
    'spread',
    'cross_ratio',
    'hyperbolic_dot',
    'is_null_point',
    'is_null_line',
    'join',
    'meet',
    'normalize_points',
    'triple_quad_formula',
    'triple_spread_formula',
    'pythagoras',
    'dual_pythagoras',
    'cross_law'
]

print(f"UHG Version: {uhg.__version__}\n")

# Get the source code for each function
for func_name in functions:
    print(f"{'=' * 40}")
    print(f"{func_name.upper()} IMPLEMENTATION:")
    print(f"{'=' * 40}")
    try:
        func = getattr(ProjectiveUHG, func_name)
        print(inspect.getsource(func))
    except (AttributeError, TypeError) as e:
        print(f"Error: {e}")
    print("\n") 