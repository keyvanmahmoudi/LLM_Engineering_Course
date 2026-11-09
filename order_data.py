"""Synthetic Persian order messages for the fine-tuning sessions (10 and 11).

Each example is a short, informal customer message and the structured order
the assistant should extract from it:

    {"product_id": "P1001", "quantity": 2, "color": "آبی", "city": "شیراز"}

`quantity` is 1 when the message does not mention it; `color` and `city`
are null when they are not mentioned.
"""
import json
import random

PRODUCTS = {
    "P1001": (["سامسونگ A55", "گلکسی A55", "گوشی A55"], ["مشکی", "آبی", "سفید"]),
    "P1002": (["سامسونگ S24 FE", "گلکسی S24 FE", "اس ۲۴ اف ای"], ["مشکی", "سبز"]),
    "P1003": (["ردمی نوت 14 پرو", "شیائومی نوت 14 پرو", "ردمی نوت ۱۴ پرو"], ["مشکی", "بنفش", "آبی"]),
    "P1004": (["پوکو X7", "شیائومی پوکو X7"], ["مشکی", "زرد"]),
    "P1005": (["آیفون 16", "iPhone 16", "آیفون ۱۶"], ["مشکی", "سفید", "صورتی"]),
    "P1006": (["نوکیا G42", "گوشی نوکیا G42"], ["خاکستری"]),
    "P2001": (["لنوو آیدیاپد", "لپ‌تاپ لنوو IdeaPad", "IdeaPad Slim 3"], ["خاکستری"]),
    "P2002": (["ویووبوک 15", "لپ‌تاپ ایسوس ویووبوک"], ["نقره‌ای"]),
    "P2003": (["مک‌بوک ایر", "MacBook Air", "مک بوک ایر M3"], ["نقره‌ای", "خاکستری"]),
    "P2004": (["ایسوس TUF", "لپ‌تاپ گیمینگ ایسوس TUF"], ["مشکی"]),
    "P3001": (["تلویزیون ال‌جی 55", "تلویزیون ۵۵ اینچ ال جی"], ["مشکی"]),
    "P3002": (["تلویزیون سامسونگ 65", "تی‌وی ۶۵ اینچ سامسونگ"], ["مشکی"]),
    "P3003": (["تلویزیون شیائومی 43", "تی وی شیائومی ۴۳ اینچ"], ["مشکی"]),
    "P4002": (["لباسشویی سامسونگ", "ماشین لباسشویی ۹ کیلویی سامسونگ"], ["سفید", "نقره‌ای"]),
    "P4004": (["کولر گازی گری", "کولر گری ۱۸ هزار"], ["سفید"]),
    "P5001": (["هدفون سونی XM5", "هدفون سونی", "WH-1000XM5"], ["مشکی", "نقره‌ای"]),
    "P5002": (["ردمی بادز 6", "هندزفری ردمی بادز ۶"], ["مشکی", "سفید"]),
    "P5003": (["گلکسی واچ 7", "ساعت سامسونگ واچ ۷"], ["مشکی", "سبز", "نقره‌ای"]),
    "P5004": (["شارژر انکر", "شارژر ۶۵ وات انکر"], ["سفید"]),
    "P5005": (["پاوربانک شیائومی", "پاوربانک ۲۰ هزار شیائومی"], ["مشکی", "آبی"]),
    "P5006": (["براکت تلویزیون", "پایه دیواری تلویزیون"], ["مشکی"]),
}

CITIES = ["تهران", "کرج", "اصفهان", "شیراز", "مشهد", "تبریز", "اهواز", "رشت", "کرمان", "یزد",
          "قم", "ساری", "آمل", "بابل", "همدان", "ارومیه", "زنجان", "قزوین", "اراک", "گرگان"]

QUANTITY_WORDS = {
    1: ["یه", "یک", "یه دونه", "یک عدد", "1 عدد", "۱ عدد"],
    2: ["دو تا", "۲ تا", "2 تا", "دو عدد", "۲ عدد"],
    3: ["سه تا", "۳ تا", "3 تا", "سه عدد"],
    4: ["چهار تا", "۴ تا", "4 تا", "چهار عدد"],
    5: ["پنج تا", "۵ تا", "5 تا", "پنج عدد"],
}

COLOR_PHRASES = ["{c}", "رنگ {c}", "{c} رنگ", "به رنگ {c}"]
CITY_PHRASES = ["برای {city}", "بفرستید {city}", "ارسال به {city}", "آدرسم {city}ه",
                "تحویل در {city}", "به مقصد {city}"]
OPENINGS = ["", "", "سلام، ", "سلام وقت بخیر، ", "درود، ", "ببخشید، "]
CLOSINGS = ["", "", " ممنون", " مرسی", " لطفاً", "، سپاس"]

TEMPLATES = [
    "{q} {p} {color} می‌خوام {city}",
    "{p} {color} دارید؟ {q} لازم دارم {city}",
    "لطفاً {q} {p} {color} {city} بفرستید",
    "می‌خواستم {q} {p} سفارش بدم، {color} {city}",
    "{q} {p} {color} {city}",
    "{city} {q} {p} {color} ثبت کنید",
    "یه سفارش داشتم: {q} {p} {color} {city}",
    "موجوده {p}؟ اگه هست {q} {color} برام بذارید {city}",
]
# Two phrasings that never appear in training: used to test generalisation.
HELD_OUT_TEMPLATES = [
    "structured",
    "واسه {city} {q} {p} {color} می‌خواستم، چطوری ثبت کنم؟",
]


def _render(rng, template):
    pid = rng.choice(list(PRODUCTS))
    names, colors = PRODUCTS[pid]
    product = rng.choice(names)
    quantity = rng.choice([1, 1, 1, 2, 2, 3, 4, 5])
    q_text = rng.choice(QUANTITY_WORDS[quantity]) if (quantity > 1 or rng.random() < 0.6) else ""
    color = rng.choice(colors) if rng.random() < 0.6 else None
    city = rng.choice(CITIES) if rng.random() < 0.7 else None

    if template == "structured":
        parts = [f"کالا: {product}"]
        if q_text:
            parts.append(f"تعداد: {q_text}")
        if color:
            parts.append(f"رنگ: {color}")
        if city:
            parts.append(f"شهر: {city}")
        message = "سفارش جدید - " + " | ".join(parts)
    else:
        color_text = rng.choice(COLOR_PHRASES).format(c=color) if color else ""
        if template.startswith("واسه"):
            city_text = f"واسه {city}" if city else ""
            template = template.replace("واسه {city}", "{city}")
        else:
            city_text = rng.choice(CITY_PHRASES).format(city=city) if city else ""
        message = template.format(q=q_text, p=product, color=color_text, city=city_text)
        message = " ".join(message.split()).replace(" ،", "،").replace("، ،", "،").strip(" ،")
    message = rng.choice(OPENINGS) + message + rng.choice(CLOSINGS)
    order = {"product_id": pid, "quantity": quantity, "color": color, "city": city}
    return {"message": message, "order": order}


def generate(n, seed, templates=TEMPLATES):
    rng = random.Random(seed)
    seen, data = set(), []
    while len(data) < n:
        example = _render(rng, rng.choice(templates))
        if example["message"] not in seen:
            seen.add(example["message"])
            data.append(example)
    return data


# Messages written by hand, in styles that the generator never produces.
REAL_MESSAGES = [
    ("سلام. یه A55 آبی میخواستم واسه آمل. ممنون میشم زودتر بفرستید", {"product_id": "P1001", "quantity": 1, "color": "آبی", "city": "آمل"}),
    ("دوتا هدفون سونی نقره ای برا شرکتمون تو تهران", {"product_id": "P5001", "quantity": 2, "color": "نقره‌ای", "city": "تهران"}),
    ("آیفون ۱۶ صورتی موجود دارید؟ اگه دارید یکی بفرستید مشهد", {"product_id": "P1005", "quantity": 1, "color": "صورتی", "city": "مشهد"}),
    ("برای خونه جدیدمون یه لباسشویی سامسونگ سفید لازم داریم، شیراز هستیم", {"product_id": "P4002", "quantity": 1, "color": "سفید", "city": "شیراز"}),
    ("۳ تا شارژر انکر میخوام", {"product_id": "P5004", "quantity": 3, "color": None, "city": None}),
    ("مک بوک ایر خاکستری - ۱ عدد - اصفهان", {"product_id": "P2003", "quantity": 1, "color": "خاکستری", "city": "اصفهان"}),
    ("پوکو ایکس ۷ زرد دو عدد لطفا", {"product_id": "P1004", "quantity": 2, "color": "زرد", "city": None}),
    ("من از کرج سفارش میدم: پاوربانک شیائومی، چهار تا، مشکی", {"product_id": "P5005", "quantity": 4, "color": "مشکی", "city": "کرج"}),
    ("تلویزیون ۴۳ اینچ شیائومی برای رشت", {"product_id": "P3003", "quantity": 1, "color": None, "city": "رشت"}),
    ("سلام خسته نباشید، ساعت گلکسی واچ ۷ سبز رنگ یکی برام بفرستید یزد", {"product_id": "P5003", "quantity": 1, "color": "سبز", "city": "یزد"}),
    ("لپتاپ گیمینگ TUF ایسوس دارید؟ ۲تا برای کافی‌نت میخوام، قزوین", {"product_id": "P2004", "quantity": 2, "color": None, "city": "قزوین"}),
    ("پنج عدد براکت دیواری تلویزیون برای هتل در گرگان", {"product_id": "P5006", "quantity": 5, "color": None, "city": "گرگان"}),
]


def real_examples():
    return [{"message": m, "order": o} for m, o in REAL_MESSAGES]


if __name__ == "__main__":
    for name, data in [("orders_train.jsonl", generate(1200, seed=1)),
                       ("orders_test.jsonl", generate(150, seed=2) + generate(50, seed=3, templates=HELD_OUT_TEMPLATES)),
                       ("orders_real.jsonl", real_examples())]:
        with open(f"data/{name}", "w", encoding="utf-8") as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(name, len(data))
