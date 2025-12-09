"""
Command-line interface for the Bisection Project.
"""
from bisection import bisection

def print_iterations_table(iter_list):
    if not iter_list:
        print("\n(No iterations – root was exactly at endpoint.)")
        return

    # table header
    print("\n" + "=" * 80)
    print(f"{'n':<5} {'a':<15} {'b':<15} {'p':<15} {'f(p)':<15} {'error':<15}")
    print("-" * 80)

    # table rows
    for row in iter_list:
        n = row['n']
        a = row['a']
        b = row['b']
        p = row['p']
        fp = row['f(p)']
        err = row['error']

        err_str = f"{err:.8e}" if err is not None else "----"

        print(f"{n:<5} {a:<15.8f} {b:<15.8f} {p:<15.8f} {fp:<15.8f} {err_str:<15}")

    print("=" * 80)



def main():
    """Interactive test function"""
    print("Bisection Method - Numerical Project")
    func = input("Enter f(x): ")
    a = float(input("Enter a: "))
    b = float(input("Enter b: "))
    d = int(input("Enter d (digits): "))

    result = bisection(func, a, b, d)

    if "error" in result:
        print(f"\n❌ Error: {result['error']}\n")
    else:
        print(f"\n✅ Root approximation: {result['root']}")
        print(f"Required N = {result['N_required']}")
        print(f"Tolerance = {result['tol']}")
        print(f"Converged by tol: {result['converged_by_tol']}")
        print("\n📌 Workability:", result["workability"])
        print("\n📌 Convergence explanation:")
        print(result["convergence_text"])
        print("\n📌 Rate of convergence:")
        print(result["rate_text"])

        # print the table
        print_iterations_table(result["iterations"])


if __name__ == "__main__":
    main()
