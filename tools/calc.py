"""Safe arithmetic for the chat router. No eval, no names, no calls."""

from __future__ import annotations

import ast
from decimal import Decimal, DivisionByZero, InvalidOperation, getcontext
from typing import Optional

getcontext().prec = 28

_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}
_ALLOWED_UNARY = {
    ast.UAdd: lambda a: a,
    ast.USub: lambda a: -a,
}


class UnsafeExpression(ValueError):
    """Expression used a name, call, or other disallowed node."""


def _to_decimal(value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise UnsafeExpression("only numbers are allowed")
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _eval_node(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        return _to_decimal(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        try:
            return _ALLOWED_BINOPS[type(node.op)](left, right)
        except (DivisionByZero, InvalidOperation, ZeroDivisionError) as exc:
            raise ZeroDivisionError("division by zero") from exc
    raise UnsafeExpression(f"disallowed expression node: {type(node).__name__}")


def _prepare(text: str) -> str:
    s = " ".join((text or "").split())
    if not s:
        return ""
    s = s.replace("^", "**")
    if s.endswith("?") and not s.startswith("?"):
        s = s[:-1].rstrip()
    return s


def try_calc(text: str) -> Optional[str]:
    """Return a decimal string if *text* is a safe arithmetic expression.

    Returns None for empty input, prose, or anything with names/calls.
    """
    source = _prepare(text)
    if not source:
        return None
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError:
        return None
    try:
        value = _eval_node(tree)
    except (UnsafeExpression, ZeroDivisionError, InvalidOperation, OverflowError, ValueError):
        return None
    text_out = format(value, "f")
    if "." in text_out:
        text_out = text_out.rstrip("0").rstrip(".")
    return text_out or "0"
