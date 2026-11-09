"""Tools of the fictional Novin Kala store, used from session 12 onwards.

Every tool is a plain Python function with type hints and a docstring.
`tool_schema` turns such a function into the JSON tool definition of the
OpenAI-compatible API, and `run_tool` executes a tool call safely.
"""
import ast
import inspect
import json
import operator
import random
import re
import sqlite3
from typing import Optional

import pandas as pd
from pydantic import ValidationError, create_model

from llm_utils import download_data

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩يك", "01234567890123456789یک")
FAST_CITIES = {"تهران", "کرج", "اصفهان", "شیراز", "مشهد"}

products = pd.read_csv(download_data("products.csv"))
db = sqlite3.connect(":memory:", check_same_thread=False)
products.to_sql("products", db, index=False)
db.execute("CREATE TABLE orders (order_id TEXT, product_id TEXT, quantity INTEGER, color TEXT, "
           "city TEXT, total INTEGER, status TEXT)")
db.execute("INSERT INTO orders VALUES ('NK482913', 'P1003', 1, 'آبی', 'آمل', 21400000, 'ارسال شده')")
db.execute("INSERT INTO orders VALUES ('NK519027', 'P5001', 2, 'مشکی', 'تهران', 43800000, 'در حال پردازش')")


def _norm(text):
    return re.sub(r"\s+", " ", str(text).translate(_DIGITS).replace("‌", " ").lower()).strip()


def _product_dict(row):
    return {"product_id": row["product_id"], "name": row["name"], "category": row["category"],
            "price_toman": int(row["price_toman"]), "stock": int(row["stock"]),
            "colors": row["colors"].split("|") if row["colors"] != "—" else []}


# ------------------------------------------------------------------ tools
def search_products(query: str, category: Optional[str] = None, max_price: Optional[int] = None) -> list:
    """Search the product catalogue by name or brand.

    Args:
        query: words to search for, e.g. "سامسونگ" or "لپ‌تاپ ایسوس".
        category: optional category filter: گوشی موبایل، لپ‌تاپ، تلویزیون، لوازم خانگی بزرگ، لوازم جانبی.
        max_price: optional maximum price in toman.
    """
    words = [w for w in _norm(query).split() if len(w) > 1]
    results = []
    for _, row in products.iterrows():
        text = _norm(f"{row['name']} {row['brand']} {row['category']}")
        hits = sum(w in text for w in words)
        if words and hits == 0:
            continue
        if category and _norm(category) not in _norm(row["category"]):
            continue
        if max_price is not None and row["price_toman"] > max_price:
            continue
        results.append((hits, _product_dict(row)))
    results.sort(key=lambda x: (-x[0], x[1]["price_toman"]))
    return [r for _, r in results[:5]]


def get_product(product_id: str) -> dict:
    """Return the full details of one product.

    Args:
        product_id: the product id, e.g. "P1001".
    """
    row = products[products.product_id == product_id.strip().upper()]
    if row.empty:
        raise ValueError(f"no product with id {product_id}")
    return _product_dict(row.iloc[0])


def shipping_cost(order_total: int, method: str, city: str) -> dict:
    """Shipping cost in toman according to the Novin Kala shipping policy.

    Args:
        order_total: total price of the items in toman.
        method: "standard" (post, 3-5 working days) or "express" (courier, same day).
        city: destination city in Persian.
    """
    if method == "standard":
        return {"method": method, "cost": 0 if order_total > 1_000_000 else 45_000, "days": "3-5"}
    if method == "express":
        if _norm(city) not in {_norm(c) for c in FAST_CITIES}:
            return {"method": method, "available": False,
                    "note": "express delivery is only available in " + "، ".join(sorted(FAST_CITIES))}
        return {"method": method, "cost": 85_000, "days": "same day if ordered before 14:00"}
    raise ValueError("method must be 'standard' or 'express'")


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def calculator(expression: str) -> float:
    """Evaluate an arithmetic expression with numbers and + - * / ** and parentheses.

    Args:
        expression: for example "28900000 * 2 * 0.9 + 85000".
    """
    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](evaluate(node.operand))
        raise ValueError(f"unsupported expression: {expression!r}")
    return evaluate(ast.parse(expression.translate(_DIGITS).replace(",", ""), mode="eval").body)


def track_order(order_id: str) -> dict:
    """Return the status of an order.

    Args:
        order_id: the tracking code, eight characters starting with NK, e.g. "NK482913".
    """
    row = db.execute("SELECT order_id, product_id, quantity, city, total, status FROM orders "
                     "WHERE order_id = ?", (order_id.strip().upper(),)).fetchone()
    if row is None:
        raise ValueError(f"no order with code {order_id}")
    return dict(zip(["order_id", "product_id", "quantity", "city", "total", "status"], row))


def place_order(product_id: str, quantity: int, city: str, color: Optional[str] = None) -> dict:
    """Place an order. This changes data: call it only after the customer has confirmed the order.

    Args:
        product_id: the product id, e.g. "P1001".
        quantity: number of items (1 to 5).
        city: destination city in Persian.
        color: one of the colours available for the product, if it has colours.
    """
    product = get_product(product_id)
    if not 1 <= quantity <= 5:
        raise ValueError("quantity must be between 1 and 5")
    if product["stock"] < quantity:
        raise ValueError(f"only {product['stock']} items in stock")
    if product["colors"] and color not in product["colors"]:
        raise ValueError(f"colour must be one of {product['colors']}")
    order_id = f"NK{random.randint(100000, 999999)}"
    total = product["price_toman"] * quantity
    db.execute("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?)",
               (order_id, product["product_id"], quantity, color, city, total, "در انتظار پرداخت"))
    return {"order_id": order_id, "total": total, "status": "در انتظار پرداخت"}


STORE_TOOLS = [search_products, get_product, shipping_cost, calculator, track_order]


# ------------------------------------------------------------------ helpers
def _arg_docs(fn):
    docs, in_args = {}, False
    for line in (inspect.getdoc(fn) or "").splitlines():
        if line.strip() == "Args:":
            in_args = True
        elif in_args and ":" in line:
            name, text = line.strip().split(":", 1)
            docs[name.strip()] = text.strip()
    return docs


def tool_schema(fn):
    """Build an OpenAI-style tool definition from a function's signature and docstring."""
    docs = _arg_docs(fn)
    fields = {}
    for name, p in inspect.signature(fn).parameters.items():
        default = ... if p.default is inspect.Parameter.empty else p.default
        fields[name] = (p.annotation, default)
    schema = create_model(fn.__name__, **fields).model_json_schema()
    schema.pop("title", None)
    for name, prop in schema.get("properties", {}).items():
        prop.pop("title", None)
        if name in docs:
            prop["description"] = docs[name]
    description = (inspect.getdoc(fn) or "").split("\n\n")[0]
    return {"type": "function", "function": {"name": fn.__name__, "description": description,
                                             "parameters": schema}}


def run_tool(name, arguments, tools):
    """Execute one tool call and return its result as a JSON string (errors included)."""
    registry = {fn.__name__: fn for fn in tools}
    if name not in registry:
        return json.dumps({"error": f"unknown tool {name}"})
    fn = registry[name]
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
        fields = {k: (p.annotation, ... if p.default is inspect.Parameter.empty else p.default)
                  for k, p in inspect.signature(fn).parameters.items()}
        validated = create_model(name, **fields)(**args)
        result = fn(**validated.model_dump())
        return json.dumps(result, ensure_ascii=False)
    except (ValidationError, ValueError, TypeError, json.JSONDecodeError, SyntaxError, ZeroDivisionError) as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
