from z3 import *

def smt_to_readable(smt_text):
    # Parse the SMT-LIB2 text
    solver = Solver()
    solver.from_string(smt_text)
    formulas = solver.assertions()

    readable = []
    for f in formulas:
        readable.append(pretty_formula(f))
    return "\nAND\n".join(readable)

def pretty_formula(expr):
    """
    Recursively converts a Z3 expression into a readable string.
    Simplifies (<= (- a) (* (-1) x)) etc.
    """
    if is_const(expr) and not expr.children():
        return str(expr)

    if expr.decl().kind() == Z3_OP_AND:
        return "(" + " ∧ ".join(pretty_formula(c) for c in expr.children()) + ")"

    if expr.decl().kind() == Z3_OP_OR:
        return "(" + " ∨ ".join(pretty_formula(c) for c in expr.children()) + ")"

    if expr.decl().kind() == Z3_OP_NOT:
        return "¬" + pretty_formula(expr.children()[0])

    if expr.decl().kind() == Z3_OP_LE:
        a, b = expr.children()
        return f"({pretty_formula(a)} ≤ {pretty_formula(b)})"

    if expr.decl().kind() == Z3_OP_LT:
        a, b = expr.children()
        return f"({pretty_formula(a)} < {pretty_formula(b)})"

    if expr.decl().kind() == Z3_OP_EQ:
        a, b = expr.children()
        return f"({pretty_formula(a)} = {pretty_formula(b)})"

    if expr.decl().kind() == Z3_OP_ADD:
        return "(" + " + ".join(pretty_formula(c) for c in expr.children()) + ")"

    if expr.decl().kind() == Z3_OP_MUL:
        parts = [pretty_formula(c) for c in expr.children()]
        return "(" + " * ".join(parts) + ")"

    if expr.decl().kind() == Z3_OP_UMINUS:
        return f"-{pretty_formula(expr.children()[0])}"

    return str(expr)


# Example usage:
if __name__ == "__main__":
    psi_file = "../psi_c0.smt2"
    # smt_text = """
    # (set-logic QF_LRA)
    # (declare-fun x1 () Real)
    # (declare-fun x2 () Real)
    # (assert (and (<= 0 x1) (<= x1 4) (or (<= x1 x2) (= x1 x2))))
    # (check-sat)
    # """
    with open(psi_file, "r") as f:
        smt_text = f.read()
    print(smt_to_readable(smt_text))
