import ast
import copy
from dataclasses import dataclass

_CMP_SWAP: dict[type, type] = {
    ast.Lt: ast.LtE, ast.LtE: ast.Lt,
    ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
}
_SWAPPABLE_BINOPS = (ast.Sub, ast.Div, ast.FloorDiv, ast.Mod)

OPERATORS = [
    "flip_comparison", "off_by_one", "negate_condition",
    "swap_operands", "delete_branch", "flip_bool",
]


@dataclass
class Mutant:
    operator: str
    description: str
    source: str


class _Mutator(ast.NodeTransformer):
    """Applies exactly one mutation: the target_idx-th applicable site for `op`."""

    def __init__(self, op: str, target_idx: int) -> None:
        self.op = op
        self.target_idx = target_idx
        self.count = 0
        self.applied_at: int | None = None  # line number, None if not applied

    def _hit(self, node: ast.AST) -> bool:
        hit = self.count == self.target_idx
        self.count += 1
        if hit:
            self.applied_at = getattr(node, "lineno", 0)
        return hit

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if self.op == "flip_comparison":
            for i, cmp_op in enumerate(node.ops):
                if type(cmp_op) in _CMP_SWAP and self._hit(node):
                    node.ops[i] = _CMP_SWAP[type(cmp_op)]()
                    break
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if self.op == "off_by_one" and type(node.value) is int and self._hit(node):
            return ast.copy_location(ast.Constant(node.value + 1), node)
        if self.op == "flip_bool" and type(node.value) is bool and self._hit(node):
            return ast.copy_location(ast.Constant(not node.value), node)
        return node

    def visit_If(self, node: ast.If) -> ast.AST:
        self.generic_visit(node)
        if self.op == "negate_condition" and self._hit(node):
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
        elif self.op == "delete_branch" and self._hit(node):
            node.body = [ast.Pass()]
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if (
            self.op == "swap_operands"
            and isinstance(node.op, _SWAPPABLE_BINOPS)
            and self._hit(node)
        ):
            node.left, node.right = node.right, node.left
        return node


def _mutants_for_op(tree: ast.AST, source: str, op: str) -> list[Mutant]:
    mutants = []
    idx = 0
    while True:
        mutator = _Mutator(op, idx)
        mutated = mutator.visit(copy.deepcopy(tree))
        if mutator.applied_at is None:
            break
        new_source = ast.unparse(ast.fix_missing_locations(mutated))
        if new_source != ast.unparse(tree):
            mutants.append(
                Mutant(op, f"{op} at line {mutator.applied_at}", new_source)
            )
        idx += 1
    return mutants


def generate_mutants(source: str, max_mutants: int = 20) -> list[Mutant]:
    tree = ast.parse(source)
    per_op = [_mutants_for_op(tree, source, op) for op in OPERATORS]
    interleaved: list[Mutant] = []
    i = 0
    while any(per_op) and len(interleaved) < max_mutants:
        for lst in per_op:
            if i < len(lst) and len(interleaved) < max_mutants:
                interleaved.append(lst[i])
        i += 1
        if all(i >= len(lst) for lst in per_op):
            break
    return interleaved
