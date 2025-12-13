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

from typing import Callable, Dict, List, Optional, Any, Union, Tuple
import math
import re
import sympy as sp


# ---------------- SAFE MATH NAMESPACE ----------------
_SAFE_MATH: Dict[str, Any] = {
    name: getattr(math, name) for name in dir(math) if not name.startswith("_")
}
_SAFE_MATH.update({"pi": math.pi, "e": math.e, "abs": abs, "ln": math.log})


# ---------------- EXPRESSION PREPROCESSOR ----------------
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


# ---------------- find denominator expressions ----------------
def find_denominator_expressions(processed_expr: str) -> List[str]:
    """
    Return list of denominator sub-expressions found after '/', e.g. for '1/(x-2)' returns ['(x-2)'].
    The regex handles both '/(...)' and '/token' forms (like '/x', '/(x+1)', '/sin(x)').
    """
    denoms: List[str] = []
    # match / ( ... )
    for m in re.finditer(r"/\s*(\([^\)]+\))", processed_expr):
        denoms.append(m.group(1))
    # match /token (where token is functioncall or name/number)
    for m in re.finditer(r"/\s*([A-Za-z0-9_\.]+(?:\([^\)]*\))?)", processed_expr):
        token = m.group(1)
        # avoid double-adding tokens already matched as parenthesized
        if token.startswith("(") and token.endswith(")"):
            continue
        denoms.append(token)
    # unique
    uniq = []
    for d in denoms:
        if d not in uniq:
            uniq.append(d)
    return uniq


# ---------------- SymPy-based denominator zero detection ----------------
def _sympy_locals_for_sympify() -> Dict[str, Any]:
    """
    Map names used in preprocess_expression into SymPy equivalents for sympify.
    Keep this minimal and safe.
    """
    return {
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
        "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
        "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt,
        "pi": sp.pi, "E": sp.E, "e": sp.E
    }

def check_denominator_with_sympy(processed_expr: str, a: float, b: float) -> Tuple[bool, str]:
    """
    Analyze processed_expr symbolically to detect denominator zeros inside (a,b).
    Returns (True, "") if no denominator-zero found inside (a,b),
    otherwise (False, message).

    Strategy:
      1. sympify(processed_expr)
      2. get denominator via as_numer_denom()[1]
      3. if denominator == 1 -> safe
      4. solve denom == 0 over Reals with solveset:
         - if finite set: check numeric membership in (a,b)
         - if empty set: safe
         - otherwise (ImageSet/ConditionSet/Infinite): fallback numeric scan of denom only
    """
    x = sp.symbols("x", real=True)
    local_map = _sympy_locals_for_sympify()
    # try symbolic conversion
    try:
        sym_expr = sp.sympify(processed_expr, locals={**local_map, "x": x})
    except Exception as exc:
        # cannot sympify: fall back to numeric; return False with descriptive message
        return False, f"sympy: Cannot sympify expression: {exc}"

    # get denominator
    try:
        _, denom = sp.together(sym_expr).as_numer_denom()
    except Exception as exc:
        return False, f"sympy: Failed to extract denominator symbolically: {exc}"

    # if denom is 1 -> no denominator
    if denom == 1 or denom == sp.Integer(1):
        return True, ""

    # simplify/factor denom for nicer solveset
    try:
        denom_simpl = sp.factor(denom)
    except Exception:
        denom_simpl = denom

    # try algebraic solve for denom == 0 over reals
    try:
        sol = sp.solveset(sp.Eq(denom_simpl, 0), x, domain=sp.S.Reals)
    except Exception:
        sol = None

    # Case: solveset returned a FiniteSet -> check each element
    if isinstance(sol, sp.FiniteSet):
        for root in list(sol):
            try:
                rval = float(sp.N(root))
            except Exception:
                # symbolic root we cannot numeric evaluate reliably: be conservative
                return False, f"sympy: Denominator root {root!s} could not be evaluated numerically; possible discontinuity."
            # check open interval (a,b)
            if a < rval < b:
                return False, f"sympy: Denominator becomes zero at x={rval} (root of denominator) inside interval."
        # no root in (a,b)
        return True, ""

    # Case: empty
    if sol == sp.S.EmptySet:
        return True, ""

    # Case: solveset returned an ImageSet / ConditionSet / infinite set (transcendental)
    # We perform a targeted numeric scan **only on the denominator** (cheap).
    # Use a moderate number of samples, and look for sign changes or extremely large values.
    try:
        denom_lambda = sp.lambdify(x, denom_simpl, modules=["math"])
    except Exception:
        # lambdify may fail for some exotic pieces - fall back to numeric but mark symbolic fallback
        return False, "sympy: Could not lambdify denominator; falling back to numeric scan."

    # numeric scan parameters
    samples = 800  # dense but cheap for denominator only
    step = (b - a) / (samples - 1)
    prev_val = None
    for i in range(samples):
        xv = a + i * step
        try:
            dv = float(denom_lambda(xv))
        except Exception:
            return False, f"sympy_numeric: Denominator not defined at x={xv} (numeric check) -> discontinuity."
        if math.isnan(dv) or math.isinf(dv):
            return False, f"sympy_numeric: Denominator produced NaN/Inf at x={xv} -> discontinuity."
        # huge magnitude implies possible pole
        if abs(dv) > 1e8:
            return False, f"sympy_numeric: Denominator magnitude too large at x={xv} (possible pole) -> discontinuity."
        if prev_val is not None:
            if prev_val * dv < 0:
                # sign change in denom -> root in (prev_x, xv)
                return False, f"sympy_numeric: Denominator changes sign between x={xv-step} and x={xv} -> zero exists -> discontinuity."
        prev_val = dv

    # no evidence of denom zero found numerically
    return True, ""


# ---------------- MAKE FUNCTION (using preprocessor) ----------------
def make_function_from_processed(processed: str) -> Callable[[float], float]:
    """
    Build a callable f(x) from a processed expression.
    Raises ValueError if compilation/evaluation fails.
    """
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
            val: float = float(raw)
        except Exception as exc:
            raise ValueError(f"Cannot convert function result to float at x={x}: {exc}")
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"Function produced non-finite value at x={x}: {val}")
        return val

    return f


def make_function(expr: str) -> Callable[[float], float]:
    processed = preprocess_expression(expr)
    return make_function_from_processed(processed)


# ---------------- SAMPLING: definedness & continuity ----------------
def sampling_defined_and_continuous(
    f: Callable[[float], float],
    a: float,
    b: float,
    samples: int = 300,
    max_abs_threshold: float = 1e8
) -> bool:
    """
    Sample f at 'samples' points in [a,b].
    Return False when:
     - evaluation raises exception at a sample
     - sample is NaN/Inf
     - any sample abs(value) > max_abs_threshold (likely pole)
     - relative jump between consecutive samples is enormous (heuristic discontinuity)
    """
    if a >= b:
        return False
    if samples < 2:
        samples = 2

    step = (b - a) / (samples - 1)
    prev_val: Optional[float] = None

    for i in range(samples):
        x = a + i * step
        try:
            v = float(f(x))
        except Exception:
            return False

        if math.isnan(v) or math.isinf(v):
            return False

        if abs(v) > max_abs_threshold:
            # treat very large magnitude as sign of singularity/asymptote in interval
            return False

        if prev_val is not None:
            denom = max(1.0, abs(prev_val))
            if abs(v - prev_val) / denom > 1e6:
                return False
        prev_val = v

    return True


# ---------------- SAMPLING: single-root heuristic ----------------
def estimate_single_root(f: Callable[[float], float], a: float, b: float, samples: int = 500) -> Optional[bool]:
    """
    Heuristic: count sign changes between consecutive samples (ignore near-zero).
    Returns:
      - True  => exactly one sign change seen (likely single root)
      - False => multiple/no sign changes seen (ambiguous)
      - None  => evaluation error during sampling (treat as ambiguous)
    NOTE: This function is only a heuristic; bisection will proceed when endpoints have sign change.
    """
    if samples < 2:
        samples = 2

    step = (b - a) / (samples - 1)
    prev_val: Optional[float] = None
    sign_changes = 0
    try:
        for i in range(samples):
            x = a + i * step
            try:
                v = float(f(x))
            except Exception:
                return None  # sampling failed -> ambiguous
            # do not aggressively ignore small values; treat near-zero as exact zero only when very small
            if abs(v) < 1e-18:
                # exact sample root detected -> count as sign-change boundary
                prev_val = None
                sign_changes += 1
                continue
            if prev_val is not None:
                if prev_val * v < 0:
                    sign_changes += 1
                    prev_val = None
                    continue
            prev_val = v
    except Exception:
        return None

    return sign_changes == 1


# ---------------- REQUIRED ITERATIONS (base-10) ----------------
def compute_required_N(a: float, b: float, d: int) -> int:
    if b <= a:
        raise ValueError("b must be > a")
    if d < 0:
        raise ValueError("d must be >= 0")
    width = b - a
    numerator = math.log10(width) - math.log10(0.5 * 10 ** (-d))
    denom = math.log10(2)
    rhs = numerator / denom - 1.0
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
    analysis_logs: List[str] = []  # collect diagnostic messages for GUI/CLI

    # preprocess and inspect denominators first (so we can report discontinuities earlier)
    processed = preprocess_expression(func_str)

    # --- NEW: Try SymPy first for exact denominator zero detection ---
    try:
        sym_ok, sym_msg = check_denominator_with_sympy(processed, a, b)
    except Exception as exc:
        # SymPy crashed unexpectedly; fallback to numeric but record message
        sym_ok, sym_msg = False, f"sympy: unexpected error during denominator check: {exc}"
    if sym_ok:
        # SymPy says denominator safe (no zeros in (a,b))
        analysis_logs.append("sympy: no denominator-zero detected.")
        # continue to numeric checks below as well (we keep numeric fallback as extra safety)
    else:
        # if sym_msg is empty it's surprising; else inspect message
        analysis_logs.append(f"sympy: {sym_msg}")
        # If sym_msg explicitly points to a root/discontinuity found, return immediately
        # (sympy returns messages that include 'Denominator becomes zero' or 'changes sign' or 'is zero')
        lower_msg = sym_msg.lower()
        if ("becomes zero" in lower_msg) or ("is zero at x" in lower_msg) or ("changes sign" in lower_msg) or ("denominator root" in lower_msg):
            # definite discontinuity found symbolically -> abort
            return {"error": sym_msg, "analysis": analysis_logs}
        # Otherwise sympy couldn't fully decide (cannot sympify, lambdify failed, fallback indication)
        # We'll fall back to the original numeric denominator sampling below (but record message).
        analysis_logs.append("sympy: fallback to numeric denominator scan.")

    # find denominators (strings) and run numeric analysis 
    denom_strs = find_denominator_expressions(processed)
    if denom_strs:
        # for each denominator substring, build a function and test if it has a zero in [a,b]
        for ds in denom_strs:
            ds_clean = ds
            try:
                g = make_function_from_processed(ds_clean.strip("()"))
            except Exception:
                msg = f"Denominator expression '{ds}' could not be analyzed (potential discontinuity)."
                analysis_logs.append(msg)
                return {"error": msg, "analysis": analysis_logs}

            try:
                # Densify numeric scan for better reliability on denominators
                step = (b - a) / 800.0
                prev_val: Optional[float] = None
                for i in range(801):
                    x = a + i * step
                    try:
                        gv = g(x)
                    except Exception:
                        msg = f"Denominator '{ds}' not defined at x={x} inside interval."
                        analysis_logs.append(msg)
                        return {"error": msg, "analysis": analysis_logs}
                    if abs(gv) < 1e-12:
                        msg = f"Denominator '{ds}' is zero at x={x} inside interval -> discontinuity."
                        analysis_logs.append(msg)
                        return {"error": msg, "analysis": analysis_logs}
                    if prev_val is not None and prev_val * gv < 0:
                        msg = f"Denominator '{ds}' changes sign inside interval -> zero exists -> discontinuity."
                        analysis_logs.append(msg)
                        return {"error": msg, "analysis": analysis_logs}
                    prev_val = gv
            except Exception as exc:
                msg = f"Error analyzing denominator '{ds}': {exc}"
                analysis_logs.append(msg)
                return {"error": msg, "analysis": analysis_logs}

    # now build main function
    try:
        f = make_function_from_processed(processed)
    except ValueError as exc:
        msg = f"parse_error: {exc}"
        analysis_logs.append(msg)
        return {"error": msg, "analysis": analysis_logs}

    # sampling-definedness with threshold
    try:
        if not sampling_defined_and_continuous(f, a, b, samples=300, max_abs_threshold=1e8):
            msg = "Function is not defined/continuous on [a,b] (sampling/discontinuity detected)."
            analysis_logs.append(msg)
            return {"error": msg, "analysis": analysis_logs}
    except Exception as exc:
        msg = f"sampling_error: {exc}"
        analysis_logs.append(msg)
        return {"error": msg, "analysis": analysis_logs}

    # evaluate endpoints
    try:
        fa = float(f(a))
    except Exception as exc:
        msg = f"evaluation_error_at_a: {exc}"
        analysis_logs.append(msg)
        return {"error": msg, "analysis": analysis_logs}
    try:
        fb = float(f(b))
    except Exception as exc:
        msg = f"evaluation_error_at_b: {exc}"
        analysis_logs.append(msg)
        return {"error": msg, "analysis": analysis_logs}

    # endpoint roots
    if abs(fa) <= 1e-15:
        return {"root": float(a), "iterations": [], "N_required": 0, "tol": 10 ** (-d),
                "converged_by_tol": True, "workability": "Exact root at left endpoint (a).",
                "convergence_text": "If f(a)==0 then a is a root; bisection is not required.", "rate_text": ""}

    if abs(fb) <= 1e-15:
        return {"root": float(b), "iterations": [], "N_required": 0, "tol": 10 ** (-d),
                "converged_by_tol": True, "workability": "Exact root at right endpoint (b).",
                "convergence_text": "If f(b)==0 then b is a root; bisection is not required.", "rate_text": ""}

    # sign change test
    if fa * fb > 0:
        msg = "No sign change at endpoints: f(a) * f(b) > 0. Bisection cannot be applied."
        analysis_logs.append(msg)
        return {"error": msg, "analysis": analysis_logs}

    # single-root heuristic
    single_root_info = ""
    try:
        sr = estimate_single_root(f, a, b)
        if sr is True:
            single_root_info = "Single-root sampling estimate passed."
        elif sr is False:
            # multiple or zero sign changes by sampling, but endpoints have sign change -> proceed with caution
            single_root_info = "Warning: sampling suggests multiple/ambiguous roots inside interval. Proceeding because f(a)*f(b)<0."
        else:
            single_root_info = "Warning: sampling for single-root estimate was unreliable."
    except Exception as exc:
        single_root_info = f"Warning: root estimation error ignored: {exc}"

    # compute N
    try:
        N_required = compute_required_N(a, b, d)
    except Exception as exc:
        msg = f"N_calculation_failed: {exc}"
        analysis_logs.append(msg)
        return {"error": msg, "analysis": analysis_logs}

    tol: float = 10 ** (-d)
    iterations: List[Dict[str, Union[int, float, None]]] = []
    p: Optional[float] = None
    prev_p: Optional[float] = None
    left = a
    right = b
    converged_by_tol = False

    if N_required == 0:
        return {
            "root": None,
            "iterations": iterations,
            "N_required": 0,
            "tol": tol,
            "converged_by_tol": False,
            "workability": f"N_required == 0 (no iterations performed). {single_root_info}",
            "convergence_text": "",
            "rate_text": ""
        }

    # run exactly N_required iterations
    for n in range(1, N_required + 1):
        p = 0.5 * (left + right)
        try:
            fp = float(f(p))
        except Exception as exc:
            msg = f"evaluation_error_at_iteration_p: {exc}"
            analysis_logs.append(msg)
            return {"error": msg, "analysis": analysis_logs}
        err = None if prev_p is None else abs(p - prev_p)
        iterations.append({"n": n, "a": float(left), "b": float(right), "p": float(p), "f(p)": float(fp), "error": err})
        try:
            fa = float(f(left))
        except Exception as exc:
            msg = f"evaluation_error_at_left_during_iteration: {exc}"
            analysis_logs.append(msg)
            return {"error": msg, "analysis": analysis_logs}
        if fa * fp < 0:
            right = p
            fb = fp
        else:
            left = p
            fa = fp
        if err is not None and err < tol:
            converged_by_tol = True
        prev_p = p

    final_root = float(p) if p is not None else None

    result = {
        "root": final_root,
        "iterations": iterations,
        "N_required": N_required,
        "tol": tol,
        "converged_by_tol": converged_by_tol,
        "workability": f"Valid: continuous (sampled), sign change passed. {single_root_info}",
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

    # only attach analysis logs when non-empty (helps GUI/CLI show reasoning)
    if analysis_logs:
        result["analysis"] = analysis_logs

    return result


# ---------------- self-test ----------------
if __name__ == "__main__":
    from pprint import pprint
    # discontinuous example:
    pprint(bisection("1/(x-2)", 1.0, 7.0, 6))
    # continuous example:
    pprint(bisection("xsinx - 1", 0.0, 2.0, 8))
    # multiple conversions
    pprint(bisection("3x^2 + sinx - ln(x+1)", 0.1, 2, 6))
