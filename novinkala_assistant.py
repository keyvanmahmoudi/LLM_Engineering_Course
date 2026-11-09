"""The complete Novin Kala assistant (session 16).

It combines the components built during the course:

* `DocsRAG`     - retrieval-augmented answers about store policies (sessions 6-9)
* `Agent`       - a ReAct agent with the store tools (sessions 12-13)
* `OrderBot`    - a task-oriented dialogue system for placing orders (session 15)
* `route`       - an LLM router that sends each message to the right component
* `NovinAssistant` - the conversation manager used by the Gradio demo
"""
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel

from llm_utils import download_data
from store_tools import STORE_TOOLS, get_product, place_order, run_tool, search_products, tool_schema

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def _structured(client, model, system, prompt, schema):
    response = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}})
    return schema.model_validate_json(response.choices[0].message.content)


# ------------------------------------------------------------------ RAG
class DocsRAG:
    REFUSAL = "متأسفانه در اسناد نوین‌کالا پاسخی برای این پرسش پیدا نکردم."
    SYSTEM = f"""You are the customer-support assistant of Novin Kala, an Iranian online store.
Answer the customer's question in Persian, using ONLY the numbered sources provided.
- End every factual sentence with its source number in brackets, e.g. [2].
- If the sources do not contain the answer, reply exactly: «{REFUSAL}»
- Do not guess numbers, prices or dates. Mention conditions and exceptions. Be concise and polite."""

    def __init__(self, client, model, embedder_name="intfloat/multilingual-e5-base"):
        from hazm import Normalizer
        from sentence_transformers import SentenceTransformer
        self.client, self.model = client, model
        self.normalizer = Normalizer(persian_numbers=False)
        with open(download_data("novinkala_chunks.jsonl"), encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]
        self.embedder = SentenceTransformer(embedder_name)
        self.vectors = self.embedder.encode([f"passage: {c['text']}" for c in self.chunks], normalize_embeddings=True)

    def search(self, query, k=4):
        q = self.embedder.encode([f"query: {self.normalizer.normalize(query).translate(_DIGITS)}"],
                                 normalize_embeddings=True)[0]
        top = np.argsort(-(self.vectors @ q))[:k]
        return [self.chunks[i] for i in top]

    def answer(self, question, k=4):
        hits = self.search(question, k)
        sources = "\n\n".join(f'<source id="{i}">\n{h["text"]}\n</source>' for i, h in enumerate(hits, 1))
        response = self.client.chat.completions.create(
            model=self.model, temperature=0,
            messages=[{"role": "system", "content": self.SYSTEM},
                      {"role": "user", "content": f"{sources}\n\nCustomer question: {question}"}])
        return response.choices[0].message.content


# ------------------------------------------------------------------ agent
AGENT_SYSTEM = """You are the shopping assistant of Novin Kala, an Iranian online store. Answer in Persian.
- Store rules come ONLY from search_docs; products, prices and stock ONLY from search_products or get_product.
- Do every calculation with the calculator tool.
- Give a short final answer with the numbers you used. Never invent prices, rules or numbers."""


class Agent:
    def __init__(self, client, model, tools, system=AGENT_SYSTEM, max_steps=8):
        self.client, self.model, self.tools, self.system, self.max_steps = client, model, tools, system, max_steps
        self.schemas = [tool_schema(t) for t in tools]

    def run(self, task):
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": task}]
        trace, seen = [], set()
        for _ in range(self.max_steps):
            message = self.client.chat.completions.create(model=self.model, messages=messages, tools=self.schemas,
                                                          temperature=0).choices[0].message
            if not message.tool_calls:
                return message.content, trace
            messages.append(message.model_dump(exclude_none=True))
            for call in message.tool_calls:
                key = (call.function.name, call.function.arguments)
                result = (json.dumps({"error": "repeated call; use the earlier result"}) if key in seen
                          else run_tool(call.function.name, call.function.arguments, self.tools))
                seen.add(key)
                trace.append(f"{call.function.name}({call.function.arguments})")
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        return "متأسفانه نتوانستم به پاسخ برسم.", trace


# ------------------------------------------------------------------ order dialogue
class Update(BaseModel):
    intent: Literal["order", "confirm", "deny", "other"]
    product_query: Optional[str]
    choice: Optional[int]
    quantity: Optional[int]
    color: Optional[str]
    city: Optional[str]


@dataclass
class State:
    product_query: Optional[str] = None
    product_id: Optional[str] = None
    candidates: list = field(default_factory=list)
    quantity: Optional[int] = None
    color: Optional[str] = None
    city: Optional[str] = None
    awaiting: Optional[str] = None
    order: Optional[dict] = None


NLU_SYSTEM = """You update the state of an order dialogue for a Persian online store.
Given the current state, the assistant's last question and the user's new message, return what the NEW
message says; leave a field null if the message does not mention it.
- intent: "order" for buying or order details, "confirm" for yes (بله، آره، درسته)، "deny" for no (نه، لغو), else "other".
- product_query: product words as written by the user, or null.
- choice: the number of the option the user picks if the assistant listed numbered options.
- quantity: integer ("دو تا" -> 2). color and city: in Persian."""


def _same_colour(a, b):
    squash = lambda x: (x or "").replace("‌", "").replace(" ", "")
    return squash(a) == squash(b)


class OrderBot:
    def __init__(self, client, model):
        self.client, self.model = client, model
        self.state, self.last = State(), "چه کالایی می‌خواهید سفارش بدهید؟"

    @property
    def active(self):
        return self.state.awaiting is not None

    def turn(self, message):
        s = self.state
        prompt = f"STATE: {json.dumps(asdict(s), ensure_ascii=False)}\nASSISTANT ASKED: {self.last}\nUSER: {message}"
        u = _structured(self.client, self.model, NLU_SYSTEM, prompt, Update)
        if u.product_query and u.product_query != s.product_query:
            s.product_query, s.product_id, s.candidates = u.product_query, None, []
        if u.choice and s.candidates and 1 <= u.choice <= len(s.candidates):
            s.product_id = s.candidates[u.choice - 1]["product_id"]
        for slot in ("quantity", "color", "city"):
            if getattr(u, slot):
                setattr(s, slot, getattr(u, slot))
        self.last = self._act(u)
        return self.last

    def _act(self, u):
        s = self.state
        if s.awaiting == "confirm" and u.intent == "confirm":
            try:
                s.order = place_order(s.product_id, s.quantity, s.city, s.color)
            except ValueError as e:
                s.awaiting = None
                return f"ثبت سفارش ممکن نشد: {e}"
            self.state = State()
            return f"سفارش شما ثبت شد. کد پیگیری: {s.order['order_id']}. مبلغ: {s.order['total']:,} تومان."
        if s.awaiting == "confirm" and u.intent == "deny":
            s.awaiting = "change"
            return "کدام مورد را می‌خواهید تغییر دهید؟"
        if s.product_id is None:
            if not s.product_query:
                s.awaiting = "product"
                return "چه کالایی می‌خواهید سفارش بدهید؟"
            found = [c for c in search_products(s.product_query) if c["stock"] > 0]
            if not found:
                s.awaiting, s.product_query = "product", None
                return "متأسفانه کالای موجودی با این مشخصات پیدا نکردم. لطفاً نام کالا را دقیق‌تر بنویسید."
            if len(found) > 1:
                s.candidates, s.awaiting = found[:3], "choice"
                options = "\n".join(f"{i}. {c['name']} ({c['price_toman']:,} تومان)" for i, c in enumerate(s.candidates, 1))
                return f"کدام کالا را می‌خواهید؟\n{options}"
            s.product_id = found[0]["product_id"]
        product = get_product(s.product_id)
        match = [c for c in product["colors"] if _same_colour(s.color, c)]
        if product["colors"] and not match:
            wrong, s.color, s.awaiting = s.color, None, "color"
            prefix = f"رنگ «{wrong}» برای این کالا موجود نیست. " if wrong else ""
            return f"{prefix}رنگ‌های موجود {product['name']}: {'، '.join(product['colors'])}. کدام را می‌خواهید؟"
        if match:
            s.color = match[0]
        if not s.quantity or s.quantity > product["stock"]:
            note = f"فقط {product['stock']} عدد موجود است. " if s.quantity else ""
            s.quantity, s.awaiting = None, "quantity"
            return f"{note}چند عدد می‌خواهید؟"
        if not s.city:
            s.awaiting = "city"
            return "سفارش به کدام شهر ارسال شود؟"
        s.awaiting = "confirm"
        color = f"، رنگ {s.color}" if s.color else ""
        return (f"سفارش شما: {s.quantity} عدد {product['name']}{color}، ارسال به {s.city}. "
                f"مبلغ کالاها: {product['price_toman'] * s.quantity:,} تومان. تأیید می‌کنید؟")


# ------------------------------------------------------------------ router
class Route(BaseModel):
    route: Literal["policy", "order", "shopping", "chitchat"]


ROUTER_SYSTEM = """Classify the customer's message for a Persian online store assistant.
- policy: a question about store rules: shipping, returns, warranty, payment, instalments, club points,
  account, installation, support hours, gift cards, corporate sales.
- order: the customer wants to buy / place an order now, or is answering questions of an ongoing order
  dialogue (product choice, colour, quantity, city, yes/no confirmation).
- shopping: questions about products, prices, stock, comparisons, order tracking, or calculations that
  combine products, prices and rules.
- chitchat: greetings, thanks and anything unrelated to the store."""


def route(client, model, message, order_in_progress=False):
    context = "An order dialogue is in progress.\n" if order_in_progress else ""
    return _structured(client, model, ROUTER_SYSTEM, f"{context}MESSAGE: {message}", Route).route


# ------------------------------------------------------------------ assistant
class NovinAssistant:
    def __init__(self, client, model):
        self.client, self.model = client, model
        self.rag = DocsRAG(client, model)

        def search_docs(query: str) -> list:
            """Search the Novin Kala policy documents.

            Args:
                query: a short question or keywords in Persian.
            """
            return [{"source": h["id"], "text": h["text"]} for h in self.rag.search(query, 3)]

        self.agent = Agent(client, model, [search_docs] + STORE_TOOLS)
        self.order_bot = OrderBot(client, model)
        self.log = []

    def reply(self, message):
        start = time.time()
        where = route(self.client, self.model, message, self.order_bot.active)
        if where == "order":
            answer = self.order_bot.turn(message)
        elif where == "policy":
            answer = self.rag.answer(message)
        elif where == "shopping":
            answer, _ = self.agent.run(message)
        else:
            answer = "سلام! من دستیار نوین‌کالا هستم. دربارهٔ کالاها، سفارش و قوانین فروشگاه می‌توانم کمکتان کنم."
        self.log.append({"message": message, "route": where, "answer": answer,
                         "seconds": round(time.time() - start, 2)})
        return answer
