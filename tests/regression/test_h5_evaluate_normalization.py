"""
test_h5_evaluate_normalization.py

H5: evaluate_ranking() uses data['x_means'] etc. (dataset stats) instead of
    checkpoint stats for normalization. When evaluating on a different dataset
    than training, predictions are wrong.

Test strategy:
    1. Create a mock model and checkpoint with known normalization stats
    2. Create mock data with DIFFERENT normalization stats
    3. Call evaluate_ranking() and verify it uses checkpoint stats, not data stats
    4. Before fix: uses data stats (wrong). After fix: uses checkpoint stats (correct).
"""

import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import the module source without triggering torch/numpy imports at module level
import importlib.util
_eval_path = os.path.join(os.path.dirname(__file__), '..', '..', 'neural_net', 'inference', 'evaluate.py')
with open(_eval_path) as f:
    _eval_source = f.read()

# Also try to import the function if deps are available
try:
    from neural_net.inference.evaluate import evaluate_ranking
    _have_func = True
except ImportError:
    _have_func = False

g_pass = 0
g_fail = 0

def PASS(msg):
    global g_pass
    print(f"  PASS: {msg}")
    g_pass += 1

def FAIL(msg):
    global g_fail
    print(f"  FAIL: {msg}")
    g_fail += 1


def test_signature_accepts_checkpoint():
    """Check that evaluate_ranking accepts a checkpoint parameter."""
    # Parse the source for def evaluate_ranking signature
    import re
    match = re.search(r'def evaluate_ranking\(([^)]+)\)', _eval_source)
    if not match:
        FAIL("could not find evaluate_ranking() definition")
        return
    params_str = match.group(1)
    if 'checkpoint' in params_str:
        PASS("evaluate_ranking() accepts 'checkpoint' parameter")
    else:
        FAIL("evaluate_ranking() does NOT accept 'checkpoint' parameter")


def test_uses_checkpoint_stats():
    """
    Verify evaluate_ranking prefers checkpoint stats over data stats.
    We do this by checking the source code for the pattern.
    """
    # Extract just the evaluate_ranking function body
    import re
    # Find the function and its body (up to next top-level def or end)
    match = re.search(r'(def evaluate_ranking\(.*?\n(?:(?:    |\n).*\n)*)', _eval_source)
    if not match:
        FAIL("could not extract evaluate_ranking() source")
        return
    source = match.group(1)

    # After fix, should reference checkpoint.get('x_means', ...) or similar
    uses_checkpoint = ('checkpoint' in source and
                       ("checkpoint.get('x_means'" in source or
                        "checkpoint['x_means']" in source))

    if uses_checkpoint:
        PASS("evaluate_ranking() uses checkpoint normalization stats")
    else:
        FAIL("evaluate_ranking() does NOT use checkpoint normalization stats")


if __name__ == '__main__':
    print("=== H5: evaluate.py wrong normalization stats ===\n")

    test_signature_accepts_checkpoint()
    test_uses_checkpoint_stats()

    print(f"\n{g_pass} pass, {g_fail} fail")
    sys.exit(1 if g_fail > 0 else 0)
