# Bisection-Method-Numerical-Computing-Project

This project implements the Bisection Method for finding roots of nonlinear equations, following the strict requirements of the Numerical Computing course project.

🚀 Project Idea:
The Bisection Method is a numerical technique for finding a root of an equation:
f(x)=0 
within an interval [a,b],
provided that:f(a)⋅f(b)<0

This project simulates the full mathematical procedure exactly as taught in lectures — not just the algorithm, but also:

✔ Workability Conditions

Before running Bisection, the system checks:

Function is continuous on [a,b]

Function is defined (no division by zero, imaginary values, infinity, etc.)

There is exactly one root (based on dense sampling)

Sign change: f(a) and f(b) must have opposite signs

Endpoints are checked for exact roots

These checks match academic requirements for Bisection applicability.
