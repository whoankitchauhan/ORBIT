"""Computation tools for the Analysis Agent.

`calculate` evaluates arithmetic by walking a parsed AST and rejecting any node
type that is not arithmetic. It never calls `eval`, so there is no path from a
model-generated string to attribute access, imports or function calls.
"""

from __future__ import annotations

import ast
import math
import operator
import statistics
from typing import Any

from app.tools.registry import registry

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_FUNCTIONS: dict[str, Any] = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp,
    "floor": math.floor, "ceil": math.ceil, "pow": math.pow,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}

MAX_EXPONENT = 1000  # stops 9**9**9 from wedging the process


def _evaluate(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINARY_OPS:
            raise ValueError(f"unsupported operator: {op_type.__name__}")
        left, right = _evaluate(node.left), _evaluate(node.right)
        if op_type is ast.Pow and abs(right) > MAX_EXPONENT:
            raise ValueError("exponent too large")
        return _BINARY_OPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"unsupported unary operator: {op_type.__name__}")
        return _UNARY_OPS[op_type](_evaluate(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ValueError(f"unknown name: {node.id}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise ValueError("only whitelisted math functions may be called")
        if node.keywords:
            raise ValueError("keyword arguments are not supported")
        return _FUNCTIONS[node.func.id](*[_evaluate(a) for a in node.args])
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_evaluate(e) for e in node.elts]
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


@registry.tool(
    name="calculate",
    description="Evaluate an arithmetic expression, e.g. '(1240 - 980) / 980 * 100'.",
    risk="low",
    allowed_agents=("analysis", "research"),
    parameters={"expression": "str"},
    required=("expression",),
)
def calculate(expression: str) -> dict[str, Any]:
    try:
        tree = ast.parse(expression, mode="eval")
        value = _evaluate(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError) as exc:
        return {"expression": expression, "error": f"{type(exc).__name__}: {exc}"}
    return {"expression": expression, "result": value}


@registry.tool(
    name="summarise_numbers",
    description="Descriptive statistics for a list of numbers: mean, median, spread, range.",
    risk="low",
    allowed_agents=("analysis",),
    parameters={"values": "list[float]", "label": "str = ''"},
    required=("values",),
)
def summarise_numbers(values: list[float], label: str = "") -> dict[str, Any]:
    try:
        numbers = [float(v) for v in values]
    except (TypeError, ValueError) as exc:
        return {"error": f"values must be numeric: {exc}"}
    if not numbers:
        return {"error": "no values supplied"}

    return {
        "label": label or "series",
        "count": len(numbers),
        "sum": round(sum(numbers), 4),
        "mean": round(statistics.fmean(numbers), 4),
        "median": round(statistics.median(numbers), 4),
        "min": min(numbers),
        "max": max(numbers),
        # Standard deviation needs two points; one point has no spread.
        "stdev": round(statistics.stdev(numbers), 4) if len(numbers) > 1 else 0.0,
    }


@registry.tool(
    name="compare_options",
    description="Score labelled options on weighted criteria and rank them.",
    risk="low",
    allowed_agents=("analysis",),
    parameters={"options": "dict[str, dict[str, float]]", "weights": "dict[str, float] = {}"},
    required=("options",),
)
def compare_options(
    options: dict[str, dict[str, float]], weights: dict[str, float] | None = None
) -> dict[str, Any]:
    """Weighted-sum ranking, used when an objective asks ORBIT to choose."""
    if not options:
        return {"error": "no options supplied"}

    criteria = sorted({c for scores in options.values() for c in scores})
    weights = weights or {c: 1.0 for c in criteria}
    total_weight = sum(weights.get(c, 0.0) for c in criteria) or 1.0

    ranked: list[dict[str, Any]] = []
    for name, scores in options.items():
        total = sum(float(scores.get(c, 0.0)) * float(weights.get(c, 0.0)) for c in criteria)
        ranked.append(
            {"option": name, "score": round(total / total_weight, 4), "breakdown": scores}
        )
    ranked.sort(key=lambda r: r["score"], reverse=True)

    margin = (
        round(ranked[0]["score"] - ranked[1]["score"], 4) if len(ranked) > 1 else None
    )
    return {"criteria": criteria, "ranking": ranked, "winner": ranked[0]["option"], "margin": margin}
