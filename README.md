# LLM Engineering Course

<div dir="rtl">

دورهٔ **مهندسی مدل‌های زبانی بزرگ (LLM) و برنامه‌نویسی با کمک هوش مصنوعی** به زبان فارسی، برای کسانی که یادگیری عمیق و Transformer را گذرانده‌اند و می‌خواهند با مدل‌های زبانی بزرگ برنامهٔ واقعی بسازند. این دوره ترم پنجم از یک مسیر شش ترمی است و در ۱۶ جلسهٔ ۹۰ دقیقه‌ای، از کار با LLM از طریق API و اجرای مدل‌های متن‌باز با Ollama تا برنامه‌نویسی با دستیار هوش مصنوعی (Vibe Coding)، پرامپت‌نویسی و ارزیابی، RAG روی اسناد فارسی، فاین‌تیون با LoRA، فراخوانی ابزار، ساخت Agent و سیستم‌های دیالوگ را پوشش می‌دهد. بیشتر جلسه‌ها بخشی از یک پروژهٔ پیوسته را می‌سازند: دستیار هوشمند **نوین‌کالا**، یک فروشگاه اینترنتی فرضی.

برای هر جلسه سه فایل وجود دارد:

- **نوت‌بوک درس** (پوشهٔ `sessions`): توضیحات کامل، مثال‌های قابل اجرا و تمرین‌های پایان جلسه
- **پاسخ تمرین‌ها** (پوشهٔ `solutions`): پاسخ پیشنهادی هر تمرین
- **اسلاید** (پوشهٔ `slides`): ارائهٔ پاورپوینت جلسه برای کلاس

## جلسه‌ها

| جلسه | موضوع | درس | پاسخ‌ها | اسلاید |
|---|---|---|---|---|
| ۱ | شروع کار با مدل‌های زبانی بزرگ: توکن، دما، Ollama و API سازگار با OpenAI | [نوت‌بوک](sessions/Session01_getting_started_with_llms.ipynb) | [پاسخ تمرین‌ها](solutions/Session01_getting_started_with_llms_solutions.ipynb) | [اسلاید](slides/Session01_getting_started_with_llms.pptx) |
| ۲ | وایب کدینگ: برنامه‌نویسی با دستیار هوش مصنوعی | [نوت‌بوک](sessions/Session02_vibe_coding_workflow.ipynb) | [پاسخ تمرین‌ها](solutions/Session02_vibe_coding_workflow_solutions.ipynb) | [اسلاید](slides/Session02_vibe_coding_workflow.pptx) |
| ۳ | بازبینی، دیباگ و تست کد تولیدشده با هوش مصنوعی | [نوت‌بوک](sessions/Session03_reviewing_and_testing_ai_code.ipynb) | [پاسخ تمرین‌ها](solutions/Session03_reviewing_and_testing_ai_code_solutions.ipynb) | [اسلاید](slides/Session03_reviewing_and_testing_ai_code.pptx) |
| ۴ | پرامپت‌نویسی و خروجی ساختاریافته (JSON Schema و Pydantic) | [نوت‌بوک](sessions/Session04_prompt_engineering.ipynb) | [پاسخ تمرین‌ها](solutions/Session04_prompt_engineering_solutions.ipynb) | [اسلاید](slides/Session04_prompt_engineering.pptx) |
| ۵ | ارزیابی برنامه‌های مبتنی بر LLM و LLM-as-a-Judge | [نوت‌بوک](sessions/Session05_evaluating_llm_applications.ipynb) | [پاسخ تمرین‌ها](solutions/Session05_evaluating_llm_applications_solutions.ipynb) | [اسلاید](slides/Session05_evaluating_llm_applications.pptx) |
| ۶ | امبدینگ، جستجوی معنایی، FAISS و Chroma | [نوت‌بوک](sessions/Session06_embeddings_and_vector_search.ipynb) | [پاسخ تمرین‌ها](solutions/Session06_embeddings_and_vector_search_solutions.ipynb) | [اسلاید](slides/Session06_embeddings_and_vector_search.pptx) |
| ۷ | آماده‌سازی اسناد فارسی برای RAG: استخراج، نرمال‌سازی و Chunking | [نوت‌بوک](sessions/Session07_document_processing_for_rag.ipynb) | [پاسخ تمرین‌ها](solutions/Session07_document_processing_for_rag_solutions.ipynb) | [اسلاید](slides/Session07_document_processing_for_rag.pptx) |
| ۸ | ساخت سیستم RAG با ارجاع به منبع و RAG گفتگومحور | [نوت‌بوک](sessions/Session08_building_a_rag_system.ipynb) | [پاسخ تمرین‌ها](solutions/Session08_building_a_rag_system_solutions.ipynb) | [اسلاید](slides/Session08_building_a_rag_system.pptx) |
| ۹ | ارزیابی و بهبود RAG: BM25، جستجوی ترکیبی و Reranker | [نوت‌بوک](sessions/Session09_evaluating_and_improving_rag.ipynb) | [پاسخ تمرین‌ها](solutions/Session09_evaluating_and_improving_rag_solutions.ipynb) | [اسلاید](slides/Session09_evaluating_and_improving_rag.pptx) |
| ۱۰ | فاین‌تیون مدل‌های زبانی با LoRA و QLoRA | [نوت‌بوک](sessions/Session10_fine_tuning_with_lora.ipynb) | [پاسخ تمرین‌ها](solutions/Session10_fine_tuning_with_lora_solutions.ipynb) | [اسلاید](slides/Session10_fine_tuning_with_lora.pptx) |
| ۱۱ | ارزیابی و انتشار مدل فاین‌تیون‌شده: Hugging Face Hub و GGUF | [نوت‌بوک](sessions/Session11_evaluating_and_publishing_models.ipynb) | [پاسخ تمرین‌ها](solutions/Session11_evaluating_and_publishing_models_solutions.ipynb) | [اسلاید](slides/Session11_evaluating_and_publishing_models.pptx) |
| ۱۲ | فراخوانی ابزار (Tool Calling) | به‌زودی | به‌زودی | به‌زودی |
| ۱۳ | ساخت ایجنت از صفر با حلقه ReAct | به‌زودی | به‌زودی | به‌زودی |
| ۱۴ | LangGraph، MCP و ایمنی ایجنت | به‌زودی | به‌زودی | به‌زودی |
| ۱۵ | چت‌بات و سیستم دیالوگ وظیفه‌محور | به‌زودی | به‌زودی | به‌زودی |
| ۱۶ | پروژه پایانی: دستیار هوشمند فروشگاه | به‌زودی | به‌زودی | به‌زودی |

## مسیر یادگیری

| ترم | دوره |
|---|---|
| ۱ | [برنامه‌نویسی پایتون](https://github.com/keyvanmahmoudi/Python_Course) |
| ۲ | [علم داده](https://github.com/keyvanmahmoudi/Data_Science_Course) |
| ۳ | [یادگیری ماشین](https://github.com/keyvanmahmoudi/Machine_Learning_Course) |
| ۴ | [یادگیری عمیق و شبکه‌های عصبی](https://github.com/keyvanmahmoudi/Deep_Learning_Course) |
| ۵ | **مهندسی مدل‌های زبانی بزرگ (LLM) و برنامه‌نویسی با کمک هوش مصنوعی (Vibe Coding)** (همین دوره) |
| ۶ | [ساخت محصول هوش مصنوعی با FastAPI و کسب درآمد](https://github.com/keyvanmahmoudi/AI_Product_Course) |

پیش‌نیاز این دوره، جلسه‌های ۱۳ و ۱۴ دورهٔ یادگیری عمیق (Transformer و Hugging Face) است.

## اجرا

ساده‌ترین راه، باز کردن نوت‌بوک‌ها در Google Colab با دکمهٔ **Open in Colab** بالای هر نوت‌بوک است. نوت‌بوک‌ها از طریق API سازگار با OpenAI با مدل زبانی کار می‌کنند و تابع `setup_llm` در فایل [`llm_utils.py`](llm_utils.py) دو حالت دارد:

- `setup_llm("ollama")`: مدل متن‌باز `qwen2.5:7b` با Ollama روی همان ماشین اجرا می‌شود (رایگان). در Colab از منوی `Runtime → Change runtime type` گزینهٔ **T4 GPU** را انتخاب کنید. اجرای اول برای نصب Ollama و دانلود مدل چند دقیقه طول می‌کشد.
- `setup_llm("api")`: اتصال به هر سرویس سازگار با OpenAI (OpenAI، OpenRouter، درگاه‌های داخلی یا سرور vLLM). نشانی، کلید و نام مدل را با نام‌های `LLM_BASE_URL`، `LLM_API_KEY` و `LLM_MODEL` در بخش **Secrets** کولب ذخیره کنید.

جلسهٔ ۳ به مدل زبانی نیاز ندارد. برای اجرا روی سیستم خودتان، ابتدا [Ollama](https://ollama.com) را نصب کنید:

</div>

```bash
git clone https://github.com/keyvanmahmoudi/LLM_Engineering_Course.git
cd LLM_Engineering_Course
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
ollama pull qwen2.5:7b
jupyter notebook
```

<div dir="rtl">

## داده‌ها

داده‌های دوره در پوشهٔ [`data`](data) قرار دارند و نوت‌بوک‌ها آن‌ها را هنگام اجرا به‌طور خودکار دانلود می‌کنند. همهٔ داده‌ها برای این دوره ساخته شده‌اند و فروشگاه نوین‌کالا، سیاست‌ها، کالاها و نظرات آن فرضی‌اند. توضیح هر فایل در [`data/README.md`](data/README.md) آمده است.

مدل‌های ازپیش‌آموزش‌دیده (Qwen2.5، multilingual-e5، bge-reranker) از Ollama و Hugging Face Hub دانلود می‌شوند.

## اسلایدها

اسلایدها با فونت‌های Vazirmatn و Lalezar ساخته شده‌اند. فایل فونت‌ها در پوشهٔ [`fonts`](fonts) قرار دارد. پیش از باز کردن اسلایدها آن‌ها را نصب کنید.

</div>
