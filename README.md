# LLM Engineering and Generative AI Course

Notebooks and slides for a 16-session course on building applications with large language models. Each session is designed for a 90-minute class and has a Jupyter notebook for the hands-on part and a slide deck (PDF and PowerPoint) for the lecture. The explanations in the notebooks and slides are in Persian, with technical terms in English; all code is in English.

This is the fifth course in a series, after [Python](https://github.com/keyvanmahmoudi/Python_Course), [Data Science](https://github.com/keyvanmahmoudi/Data_Science_Course), [Machine Learning](https://github.com/keyvanmahmoudi/Machine_Learning_Course) and [Deep Learning](https://github.com/keyvanmahmoudi/Deep_Learning_Course). It starts where the deep learning course ends (Transformers and Hugging Face) and covers working with LLMs through an API, AI-assisted programming, prompt engineering and evaluation, retrieval-augmented generation (RAG), fine-tuning with LoRA, tool calling and agents, and dialogue systems. Most sessions build parts of one running project: an assistant for **Novin Kala**, a fictional Iranian online electronics store.

## Contents

| # | Session | Topics | Data | Slides |
|---|---|---|---|---|
| 1 | [Getting started with large language models](01.%20Getting_Started_with_LLMs.ipynb) | Next-token prediction, temperature and sampling, tokenization of Persian text, base vs. instruct models, chat templates, Ollama, the OpenAI-compatible API, streaming, speed and cost | — | [PDF](slides/01.%20Getting_Started_with_LLMs.pdf) · [PPTX](slides/01.%20Getting_Started_with_LLMs.pptx) |
| 2 | [Vibe coding: building software with AI coding assistants](02.%20Vibe_Coding_Workflow.ipynb) | Autocomplete, chat and agentic tools, a professional workflow, specs and test-first prompting, a minimal coding agent loop, project rules files (AGENTS.md, CLAUDE.md), git as a safety net, Continue + Ollama | — | [PDF](slides/02.%20Vibe_Coding_Workflow.pdf) · [PPTX](slides/02.%20Vibe_Coding_Workflow.pptx) |
| 3 | [Reviewing, debugging and testing AI-generated code](03.%20Reviewing_and_Testing_AI_Code.ipynb) | Edge cases in Persian text, parametrised pytest, hallucinated and deprecated APIs, SQL injection, `eval` and secrets in code, tests that lock in bugs | Novin Kala products | [PDF](slides/03.%20Reviewing_and_Testing_AI_Code.pdf) · [PPTX](slides/03.%20Reviewing_and_Testing_AI_Code.pptx) |
| 4 | [Prompt engineering and structured outputs](04.%20Prompt_Engineering.ipynb) | Anatomy of a prompt, zero-shot and few-shot classification, error analysis, chain of thought, JSON mode, JSON Schema with Pydantic, separating instructions from data | Novin Kala reviews | [PDF](slides/04.%20Prompt_Engineering.pdf) · [PPTX](slides/04.%20Prompt_Engineering.pptx) |
| 5 | [Evaluating LLM applications](05.%20Evaluating_LLM_Applications.ipynb) | Test sets, classification metrics, confusion matrices, LLM-as-a-judge with rubrics, judge biases, comparing prompts and models on quality, latency and cost | Novin Kala reviews | [PDF](slides/05.%20Evaluating_LLM_Applications.pdf) · [PPTX](slides/05.%20Evaluating_LLM_Applications.pptx) |
| 6 | [Embeddings and vector search](06.%20Embeddings_and_Vector_Search.ipynb) | Sentence embeddings, cosine similarity, semantic vs. keyword search, multilingual E5, FAISS and Chroma, metadata filters | Novin Kala policies | [PDF](slides/06.%20Embeddings_and_Vector_Search.pdf) · [PPTX](slides/06.%20Embeddings_and_Vector_Search.pptx) |
| 7 | [Preparing Persian documents for RAG](07.%20Document_Processing_for_RAG.ipynb) | Reading PDF and Word files, broken Persian PDF text, normalisation with hazm, fixed-size, recursive and sentence chunking, chunk overlap and metadata | Novin Kala handbook (PDF, DOCX) | [PDF](slides/07.%20Document_Processing_for_RAG.pdf) · [PPTX](slides/07.%20Document_Processing_for_RAG.pptx) |
| 8 | [Building a retrieval-augmented generation system](08.%20Building_a_RAG_System.ipynb) | The RAG pipeline, grounded prompts, citations, refusing when the answer is not in the documents, conversational RAG with query rewriting | Novin Kala policies | [PDF](slides/08.%20Building_a_RAG_System.pdf) · [PPTX](slides/08.%20Building_a_RAG_System.pptx) |
| 9 | [Evaluating and improving RAG](09.%20Evaluating_and_Improving_RAG.ipynb) | Recall@k and MRR, BM25, hybrid search with reciprocal rank fusion, cross-encoder rerankers, faithfulness and answer correctness with an LLM judge | Novin Kala QA set | [PDF](slides/09.%20Evaluating_and_Improving_RAG.pdf) · [PPTX](slides/09.%20Evaluating_and_Improving_RAG.pptx) |
| 10 | Fine-tuning LLMs with LoRA and QLoRA | When to fine-tune, instruction datasets, chat templates, LoRA, 4-bit quantisation, PEFT and TRL `SFTTrainer` | Synthetic Persian orders | *in preparation* |
| 11 | Evaluating and publishing fine-tuned models | Before/after evaluation, merging adapters, model cards, publishing on the Hugging Face Hub, GGUF conversion and serving with Ollama | Synthetic Persian orders | *in preparation* |
| 12 | Tool calling | Tool definitions with JSON Schema, the tool-call loop, parallel calls, argument validation, error handling | Novin Kala products | *in preparation* |
| 13 | Building agents from scratch | The ReAct loop, an agent with search, product and calculator tools, step limits, traces, when not to use an agent | Novin Kala | *in preparation* |
| 14 | LangGraph, MCP and agent safety | State graphs with LangGraph, writing an MCP server and client, prompt injection, least privilege and human confirmation | Novin Kala | *in preparation* |
| 15 | Chatbots and task-oriented dialogue systems | Conversation memory, intent detection, slot filling, dialogue state tracking, clarification questions, an order-taking assistant | Novin Kala | *in preparation* |
| 16 | Final project: an AI shopping assistant | Project guidelines and rubric, a complete assistant that combines RAG, tools and an order dialogue, a Gradio interface | Novin Kala | *in preparation* |

## The running example: Novin Kala

The [`data`](data) folder holds a small, fully synthetic dataset written for this course:

| File | Content | Used in sessions |
|---|---|---|
| [`novinkala_docs/`](data/novinkala_docs) | 12 policy documents (shipping, returns, warranty, payment, ...) in Persian Markdown | 6–9, 13–16 |
| [`novinkala_chunks.jsonl`](data/novinkala_chunks.jsonl) | The documents normalised and split into 42 chunks (output of session 7) | 8, 9, 13–16 |
| [`novinkala_handbook.pdf`](data/novinkala_handbook.pdf), [`.docx`](data/novinkala_handbook.docx) | The same documents as one PDF and one Word file | 7 |
| [`rag_eval.jsonl`](data/rag_eval.jsonl) | 30 questions with reference answers, source documents and the evidence sentence that contains each answer | 5, 7–9 |
| [`products.csv`](data/products.csv) | 24 products with prices, stock and colours | 3, 12–16 |
| [`reviews.csv`](data/reviews.csv) | 48 short customer reviews labelled positive, negative or neutral | 4, 5 |

The store, its policies, phone number and products are invented; prices are illustrative.

## Getting started

Every notebook has an **Open in Colab** badge at the top. The notebooks talk to a language model through the OpenAI-compatible API, using the helper in [`llm_utils.py`](llm_utils.py):

```python
client, MODEL = setup_llm("ollama")   # open model (qwen2.5:7b) served by Ollama on the same machine
client, MODEL = setup_llm("api")      # any OpenAI-compatible service
```

- **Ollama (default, free):** on Colab, switch the runtime to a GPU (`Runtime → Change runtime type → T4 GPU`). The first run installs Ollama and downloads the model, which takes a few minutes.
- **API:** add `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` as Colab secrets (the key icon in the left sidebar) or as environment variables. Any OpenAI-compatible endpoint works: OpenAI, OpenRouter, Iranian gateways or your own vLLM server.

Sessions 1, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16 need a GPU. Sessions 2–5 can run with either back-end on a CPU runtime when an API is used.

To run locally, install [Ollama](https://ollama.com), then:

```bash
git clone https://github.com/keyvanmahmoudi/LLM_Engineering_Course.git
cd LLM_Engineering_Course
pip install openai pydantic pandas matplotlib pytest jupyter
ollama pull qwen2.5:7b
jupyter notebook
```

## Slides

The slides use the free [Vazirmatn](https://github.com/rastikerdar/vazirmatn) font. The PDF files display correctly everywhere; to edit the PowerPoint files, install Vazirmatn first.

## Prerequisites

Python and the material of the [Deep Learning Course](https://github.com/keyvanmahmoudi/Deep_Learning_Course), especially sessions 13 and 14 (the Transformer and pretrained models with Hugging Face). Basic familiarity with git and the command line helps from session 2 onwards.

## Author

**Keyvan Mahmoudi**, PhD student in Natural Language Processing, University of Tehran
[LinkedIn](https://www.linkedin.com/in/keyvan-mahmoudi-7b3280103) · [Hugging Face](https://huggingface.co/KeyvanMahmoudi)

## License

The code in this repository is released under the [MIT License](LICENSE). The slides, notebook text and the Novin Kala dataset may be used for teaching with attribution.
