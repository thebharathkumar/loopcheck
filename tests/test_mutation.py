import ast

from loopcheck.mutation import generate_mutants

SRC = '''
def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

def is_adult(age):
    return age >= 18

LIMIT = 10
ENABLED = True
'''


def _ops(mutants):
    return {m.operator for m in mutants}


def test_generates_mutants_for_each_operator():
    mutants = generate_mutants(SRC, max_mutants=100)
    assert {"flip_comparison", "off_by_one", "negate_condition",
            "delete_branch", "flip_bool"} <= _ops(mutants)


def test_each_mutant_differs_and_compiles():
    for m in generate_mutants(SRC, max_mutants=100):
        assert m.source != SRC
        ast.parse(m.source)  # must be valid python


def test_flip_comparison_actually_flips():
    mutants = [m for m in generate_mutants(SRC, max_mutants=100) if m.operator == "flip_comparison"]
    assert any("x <= lo" in m.source for m in mutants)


def test_max_mutants_cap():
    assert len(generate_mutants(SRC, max_mutants=3)) == 3


def test_swap_operands():
    src = "def sub(a, b):\n    return a - b\n"
    mutants = [m for m in generate_mutants(src) if m.operator == "swap_operands"]
    assert len(mutants) == 1 and "b - a" in mutants[0].source


def test_no_mutable_sites():
    assert generate_mutants("x = 'hello'\n") == []
