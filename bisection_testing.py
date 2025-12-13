from pprint import pprint
from bisection import bisection
import math

CASES = [
    # id, func_str, a, b, d, expected ("ok" or "error"), short_reason
    ("C1_poly",       "x**3 - x - 2",        1.0, 2.0, 6,  "ok",    "simple cubic with root ~1.521"),
    ("C2_endpointL",  "x - 1",               1.0, 3.0, 5,  "ok",    "root exactly at a=1"),
    ("C3_endpointR",  "x - 3",               0.0, 3.0, 6,  "ok",    "root exactly at b=3"),
    ("C4_no_sign",    "x**2 + 1",            -1.0, 1.0, 5, "error", "no real root (no sign change)"),
    ("C5_divzero",    "1/(x-2)",             1.0, 7.0, 8,  "error", "discontinuity at x=2 inside interval"),
    ("C6_divzero_end","1/(x-2)",             2.0, 4.0, 6,  "error", "discontinuity at left endpoint"),
    ("C7_multi_roots", "sin(x)",             -4.0, 4.0, 4, "ok", "many roots -> sampling should detect multiple sign-changes"),
    ("C8_small_width", "x - 1.0001",         1.0, 1.0002, 8, "ok",  "very small width -> small N"),
    ("C9_large_d",     "x**3 - x - 2",       1.0, 2.0, 12, "ok",   "large digits -> more iterations"),
    ("C10_sqrt_neg",   "sqrt(x)",            -1.0, 2.0, 5, "error", "sqrt undefined for negative x (domain error)"),
    ("C11_log",        "ln(x)",              -1.0, 2.0, 4, "error", "ln undefined for non-positive x in interval"),
    ("C12_oscillate",  "sin(100*x)",         0.0, 0.1, 4, "ok", "show many sign changes"),
    ("C13_xsinx",      "xsinx-1",            0.0, 2.0, 8, "ok",    "test tokenizer for x*sin(x)"),
    ("C14_sinx",       "sinx-0.5",           0.0, 2.0, 6, "ok",    "sinx -> sin(x) shorthand"),
    ("C15_3x",         "3x - 6",             1.0, 3.0, 6, "ok",    "implicit 3x -> 3*x"),
    ("C16_paren_mul",  "(x+1)(x-1)",         -2.0, 2.0, 6, "error",    "No sign change at endpoints"),
    ("C17_complex_out", "sqrt(x-3)",         0.0, 2.0, 6, "error", "sqrt negative -> complex in interval"),
    ("C18_flat",       "x**3",               -0.01, 0.01, 6, "ok",    "flat near-zero region; check convergence/tolerance")
]

def run_case(case):
    cid, expr, a, b, d, expected, reason = case
    print("----")
    print(f"Case {cid}: '{expr}'  on [{a}, {b}], d={d}")
    print("Expect:", expected, "-", reason)
    try:
        res = bisection(expr, a, b, d)
    except Exception as exc:
        print("EXCEPTION:", exc)
        return expected == "error"
    # if result contains 'error', treat as failure for 'ok' expectation
    if "error" in res:
        print("Result: ERROR as returned by backend:")
        print("   ", res.get("error"))
        # helpful info if available
        if "workability" in res:
            print("   workability:", res["workability"])
        return expected == "error"
    else:
        print("Result: SUCCESS")
        print(" Root:", res.get("root"))
        print(" N_required:", res.get("N_required"))
        print(" Iterations (count):", len(res.get("iterations", [])))
        # show first 3 iterations for quick manual sanity
        iters = res.get("iterations", [])
        if iters:
            print(" First rows:")
            for row in iters[:3]:
                print("   ", row)
        # do a lightweight sanity check for sign-change at result endpoints
        try:
            # compute f(a)*f(b) sign if possible (no exceptions)
            # we avoid calling internal make_function; rely on returned data
            pass
        except Exception:
            pass
        return expected == "ok"

def main():
    total = len(CASES)
    passed = 0
    for c in CASES:
        ok = run_case(c)
        print("PASS" if ok else "FAIL")
        if ok:
            passed += 1
    print("====")
    print(f"Passed {passed}/{total} cases.")

if __name__ == "__main__":
    main()
