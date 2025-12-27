# Bisection Method Project

A robust and educational implementation of the **Bisection Method** for finding roots of nonlinear equations, with strong emphasis on **correctness**, **workability checks**, and **clear diagnostics**.

This project goes beyond a basic numerical method implementation by carefully validating mathematical assumptions such as **continuity**, **definedness**, and **denominator safety**, using both **symbolic** and **numeric** techniques.

---

## ✨ Features

- Intelligent **expression preprocessing**
  - Converts user-friendly math input into valid Python expressions  
  - Examples:
    - `xsinx` → `x*sin(x)`
    - `sinx` → `sin(x)`
    - `3x` → `3*x`
    - `4(x+1)` → `4*(x+1)`

- **Safe function evaluation**
  - Restricted evaluation environment
  - Only math-safe functions are allowed
  - Prevents arbitrary code execution

- **Advanced workability checks**
  - Symbolic denominator zero detection (using SymPy)
  - Numeric fallback sampling for difficult cases
  - Continuity and definedness verification
  - Endpoint root detection
  - Sign-change test
  - Single-root heuristic (sampling-based)

- **Guaranteed iteration count**
  - Computes the required number of iterations using:
    ```
    N > (log10(b-a) - log10(0.5 * 10^-d)) / log10(2) - 1
    ```

- **GUI Interface**
  - Built using `tkinter`
  - Displays:
    - Root approximation
    - Iteration table
    - Convergence explanation
    - Diagnostic messages
  - Scrollable summary and results table

---

## 📁 Repository Structure
├── bisection.py
├── bisection_testing.py
├── gui.py
├── main.py
├── utils.py
└── README.md



---

## 📄 File Descriptions

### `bisection.py`
The **core backend** of the project.

Contains:
- Expression preprocessing
- Safe function construction
- Symbolic denominator analysis using SymPy
- Numeric sampling checks
- Root estimation heuristics
- Required iteration computation
- Main `bisection()` implementation

This file enforces **all mathematical assumptions** required by the bisection method.

---

### `bisection_testing.py`
Used for **testing and validation**.

Includes:
- Discontinuous function tests
- Continuous function tests
- Edge cases (endpoint roots, invalid intervals)
- Debug-oriented outputs

Helps verify correctness independently of the GUI.

---

### `gui.py`
Graphical User Interface for the project.

Features:
- Function input
- Interval input
- Precision (digits) input
- Sample function selector
- Iteration table (TreeView)
- Scrollable summary box
- CSV export of iterations

The GUI communicates directly with the backend and displays **human-readable diagnostics**.

---

### `main.py`
Entry point of the project.

Typically responsible for:
- Launching the GUI
- Or running backend logic directly (depending on configuration)

---

### `utils.py`
Utility functions shared across the project.

Includes:
- CSV export helpers
- Directory handling
- Number formatting utilities

Keeps the core logic clean and modular.

---

## 🔍 Why Symbolic + Numeric Checks?

### Why not sampling only?
- Sampling can **miss exact roots** if they lie between sample points.
- Sampling cannot prove absence of roots.
- Sampling is heuristic by nature.

### Why use SymPy?
- Detects **exact denominator zeros**
- Handles algebraic expressions safely
- Provides mathematical guarantees when possible

### Why keep numeric fallback?
- Some expressions (e.g., `sin(x)`, `tan(x)`) produce infinite or implicit root sets
- Symbolic solvers may return:
  - `ImageSet`
  - `ConditionSet`
- Numeric scanning provides a safe fallback

➡️ **Result**: correctness first, performance second.

---

## 🧠 Mathematical Guarantees

The implementation ensures:

- The function is defined on `[a, b]`
- The function is continuous on `[a, b]`
- `f(a) * f(b) < 0`
- Required assumptions of the **Intermediate Value Theorem** are satisfied

Only then does the algorithm proceed.

---

## 🎓 Educational Focus

This project is designed for:

- Numerical Analysis courses
- Algorithm correctness discussions
- Demonstrating the gap between theory and implementation
- Understanding **why checks matter before algorithms**

