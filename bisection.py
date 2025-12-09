"""
Bisection Method backend

Features:
 - Intelligent tokenizer/preprocessor to convert user input into valid Python expressions
 - Safe parsing/evaluation of user function strings (restricted eval with math functions)
 - Workability checks:
     * continuity/definedness
     * endpoint root detection
     * sign change test
     * single-root estimate
 - Compute required N using formula:
     N > (log10(b-a) - log10(0.5 * 10^-d)) / log10(2) - 1
    (N = ceil(RHS), N >= 0)
 - Run exactly N iterations to generate the iterations table
"""

from typing import Callable, Dict, List, Optional, Any, Union
import math
import re

# ---------------- SAFE MATH NAMESPACE ----------------
_SAFE_MATH: Dict[str, Any] = {
    name: getattr(math, name) for name in dir(math) if not name.startswith("_")
}
_SAFE_MATH.update({"pi": math.pi, "e": math.e, "abs": abs, "ln": math.log})
# note: 'ln' alias -> natural log (math.log)


# ---------------- EXPRESSION PREPROCESSOR ----------------
from typing import Any, Dict, Optional, Callable, List, Union

# ... keep your _SAFE_MATH etc ...

def preprocess_expression(expr: str) -> str:
    """
    Convert user-friendly input into a valid Python expression.
    Handles cases like:
      xsinx   -> x*sin(x)
      sinx    -> sin(x)
      3x      -> 3*x
      4(x+1)  -> 4*(x+1)
      x(x+1)  -> x*(x+1)
    This version first inserts obvious '*' where needed, then converts
    function-name+arg shorthand (e.g. sinx -> sin(x)).
    """
    s = expr.replace("^", "**")
    s = re.sub(r"\s+", "", s)  # remove spaces for simplicity

    # known functions (longer names first to avoid partial matches)
    funcs = [
        "asin", "acos", "atan", "asinh", "acosh", "atanh",
        "sinh", "cosh", "tanh",
        "sin", "cos", "tan",
        "log", "ln", "exp", "sqrt"
    ]

    # 1) Insert multiplication where it's safe & needed:
    #    - between digit and '('   : 3(  -> 3*(
    #    - between digit and letter: 3x  -> 3*x
    #    - between variable or ')' and letter or '(':
    #         xsin -> x*sin   , x( -> x*(
    # Use lookarounds to avoid touching function names like 'sin('.
    s = re.sub(r"(?<=[0-9x\)])(?=[A-Za-z\(])", "*", s)

    # 2) Convert shorthand function+arg like sinx or sin2 -> sin(x) or sin(2)
    # Only match when fn is NOT already followed by '('
    for fn in sorted(funcs, key=len, reverse=True):
        # match fn followed by x or a number (integer or decimal), but NOT already '('
        pattern = rf"(?<![A-Za-z0-9_]){fn}(?!\()(?P<arg>x|\d+(?:\.\d+)?)"
        s = re.sub(pattern, rf"{fn}(\g<arg>)", s)

    # collapse accidental multiple stars (defensive)
    s = re.sub(r"\*{2,}", "**", s)
    s = s.strip()
    return s



# ---------------- MAKE FUNCTION (using preprocessor) ----------------
def make_function(expr: str) -> Callable[[float], float]:
    """
    Build a callable f(x) from the user expression string.
    Uses preprocess_expression then restricted eval with _SAFE_MATH.
    Raises ValueError if compilation or evaluation fails.
    """
    processed = preprocess_expression(expr)
    #processed = expr
    try:
        code = compile(processed, "<user_function>", "eval")
    except Exception as exc:
        raise ValueError(f"Failed to compile expression. Preprocessed: {processed!r}. Error: {exc}")

    def f(x: float) -> float:
        try:
            raw = eval(code, {"__builtins__": {}}, {**_SAFE_MATH, "x": x})
        except Exception as exc:
            raise ValueError(f"Function evaluation error at x={x}: {exc}")

        if isinstance(raw, complex):
            raise ValueError(f"Function returned complex value at x={x}: {raw}")

        try:
            val: float = float(raw)  # type: ignore[arg-type]
        except Exception as exc:
            raise ValueError(f"Cannot convert function result to float at x={x}: {exc}")

        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"Function produced non-finite value at x={x}: {val}")

        return val

    return f


# ---------------- SAMPLING: definedness & continuity (typed) ----------------
def sampling_defined_and_continuous(f: Callable[[float], float], a: float, b: float, samples: int = 200) -> bool:
    """
    Sample f at `samples` points; return True if all finite and no extreme jumps detected.
    """
    if a >= b:
        return False
    if samples < 2:
        samples = 2

    step: float = (b - a) / (samples - 1)
    prev_val: Optional[float] = None

    for i in range(samples):
        x: float = a + i * step
        try:
            v: float = float(f(x))
        except Exception:
            return False

        if math.isnan(v) or math.isinf(v):
            return False

        if prev_val is not None:
            denom: float = max(1.0, abs(prev_val))
            if abs(v - prev_val) / denom > 1e6:
                return False
        prev_val = v

    return True


# ---------------- SAMPLING: single-root heuristic (typed) ----------------
def estimate_single_root(f: Callable[[float], float], a: float, b: float, samples: int = 500) -> bool:
    """
    Heuristic: count sign changes between consecutive samples (ignore near-zero).
    Return True if exactly one sign change is found.
    """
    if samples < 2:
        samples = 2

    step: float = (b - a) / (samples - 1)
    prev_val: Optional[float] = None
    sign_changes: int = 0

    for i in range(samples):
        x: float = a + i * step
        try:
            v: float = float(f(x))
        except Exception:
            return False

        if abs(v) < 1e-12:
            prev_val = None
            continue

        if prev_val is not None:
            if prev_val * v < 0:
                sign_changes += 1
                prev_val = None
                continue

        prev_val = v

    return sign_changes == 1


# ---------------- REQUIRED ITERATIONS (base-10) ----------------
def compute_required_N(a: float, b: float, d: int) -> int:
    """
    Compute required N using base-10 logarithm formula and return ceil(RHS), non-negative.
    """
    if b <= a:
        raise ValueError("b must be > a")
    if d < 0:
        raise ValueError("d must be >= 0")

    width: float = b - a
    numerator: float = math.log10(width) - math.log10(0.5 * 10 ** (-d))
    denom: float = math.log10(2)
    rhs: float = numerator / denom - 1.0
    N = math.ceil(rhs)
    return max(N, 0)


# ---------------- BISECTION (main) ----------------
def bisection(func_str: str, a: float, b: float, d: int) -> Dict[str, Any]:
    """
    Run bisection according to project requirements.

    Returns a dict containing:
      - 'root', 'iterations', 'N_required', 'tol', 'converged_by_tol', 'workability', 'convergence_text', 'rate_text'
    On failure returns {'error': <message>}.
    """
    # parse function
    try:
        f: Callable[[float], float] = make_function(func_str)
    except ValueError as exc:
        return {"error": f"parse_error: {exc}"}

    # sampling-definedness
    try:
        if not sampling_defined_and_continuous(f, a, b):
            return {"error": "Function is not defined/continuous on [a,b] (sampling failure)."}
    except Exception as exc:
        return {"error": f"sampling_error: {exc}"}

    # evaluate endpoints
    try:
        fa: float = float(f(a))
    except Exception as exc:
        return {"error": f"evaluation_error_at_a: {exc}"}

    try:
        fb: float = float(f(b))
    except Exception as exc:
        return {"error": f"evaluation_error_at_b: {exc}"}

    # endpoint root
    if abs(fa) <= 1e-15:
        return {
            "root": float(a),
            "iterations": [],
            "N_required": 0,
            "tol": 10 ** (-d),
            "converged_by_tol": True,
            "workability": "Exact root at left endpoint (a).",
            "convergence_text": "If f(a)==0 then a is a root; bisection is not required.",
            "rate_text": ""
        }

    if abs(fb) <= 1e-15:
        return {
            "root": float(b),
            "iterations": [],
            "N_required": 0,
            "tol": 10 ** (-d),
            "converged_by_tol": True,
            "workability": "Exact root at right endpoint (b).",
            "convergence_text": "If f(b)==0 then b is a root; bisection is not required.",
            "rate_text": ""
        }

    # sign change requirement
    if fa * fb > 0:
        return {"error": "No sign change at endpoints: f(a) * f(b) > 0. Bisection cannot be applied."}

    # single-root heuristic
    try:
        if not estimate_single_root(f, a, b):
            return {"error": "Interval does not appear to contain a single root (sampling estimate)."}
    except Exception as exc:
        return {"error": f"root_estimate_error: {exc}"}

    # compute N
    try:
        N_required: int = compute_required_N(a, b, d)
    except Exception as exc:
        return {"error": f"N_calculation_failed: {exc}"}

    tol: float = 10 ** (-d)
    iterations: List[Dict[str, Union[int, float, None]]] = []

    # pre-initialize p to satisfy linters
    p: Optional[float] = None
    prev_p: Optional[float] = None
    left: float = a
    right: float = b
    converged_by_tol: bool = False

    if N_required == 0:
        return {
            "root": None,
            "iterations": iterations,
            "N_required": 0,
            "tol": tol,
            "converged_by_tol": False,
            "workability": "N_required == 0 (no iterations performed).",
            "convergence_text": "",
            "rate_text": ""
        }

    # run exactly N_required iterations
    for n in range(1, N_required + 1):
        p = 0.5 * (left + right)
        try:
            fp: float = float(f(p))
        except Exception as exc:
            return {"error": f"evaluation_error_at_iteration_p: {exc}"}

        err: Optional[float] = None if prev_p is None else abs(p - prev_p)

        iterations.append({
            "n": n,
            "a": float(left),
            "b": float(right),
            "p": float(p),
            "f(p)": float(fp),
            "error": err
        })

        # re-evaluate fa (safe and keeps logic similar to your original)
        try:
            fa = float(f(left))
        except Exception as exc:
            return {"error": f"evaluation_error_at_left_during_iteration: {exc}"}

        if fa * fp < 0:
            right = p
            fb = fp
        else:
            left = p
            fa = fp

        if err is not None and err < tol:
            converged_by_tol = True

        prev_p = p

    final_root: Optional[float] = float(p) if p is not None else None

    return {
        "root": final_root,
        "iterations": iterations,
        "N_required": N_required,
        "tol": tol,
        "converged_by_tol": converged_by_tol,
        "workability": "Valid: continuous (sampled), sign change, single-root estimate passed.",
        "convergence_text": (
            "Convergence: If f is continuous on [a,b] and f(a)*f(b)<0, the Intermediate Value Theorem "
            "guarantees at least one root. Bisection halves the bracket each iteration and therefore "
            "converges to a root."
        ),
        "rate_text": (
            "Rate: linear convergence. Number of iterations is computed using the base-10 formula:\n"
            "N > (log10(b-a) - log10(0.5 * 10^-d)) / log10(2) - 1, N = ceil(RHS)."
        )
    }


# ---------------- self-test ----------------
if __name__ == "__main__":
    from pprint import pprint
    demo = bisection("xsinx-1", 0.0, 2.0, 8)
    pprint(demo)

