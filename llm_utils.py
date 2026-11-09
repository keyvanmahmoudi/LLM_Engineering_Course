"""Helpers shared by the notebooks of the LLM Engineering Course.

Every notebook talks to a language model through the OpenAI-compatible
Chat Completions API. Two back-ends are supported:

* ``"ollama"``: an open model served locally by Ollama. On Google Colab the
  helper installs Ollama, starts the server and downloads the model.
* ``"api"``: any OpenAI-compatible endpoint (OpenAI, OpenRouter, an Iranian
  gateway, or a vLLM server). The base URL, key and model name are read from
  the arguments, from Colab secrets or from environment variables
  (``LLM_BASE_URL``, ``LLM_API_KEY``, ``LLM_MODEL``).

Session 1 explains every step that ``setup_llm`` performs.
"""
import os
import shutil
import subprocess
import time
import urllib.request

from openai import OpenAI

OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DATA_URL = "https://raw.githubusercontent.com/keyvanmahmoudi/LLM_Engineering_Course/main/data"


def get_secret(name, default=None):
    """Read a value from Colab secrets, then from the environment."""
    try:
        from google.colab import userdata  # type: ignore
        value = userdata.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.environ.get(name, default)


def ollama_is_running():
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/version", timeout=2):
            return True
    except Exception:
        return False


def start_ollama(context_length=8192, log_path="ollama.log", timeout=60):
    """Install Ollama if needed and start ``ollama serve`` in the background.

    ``context_length`` sets the context window (in tokens) used for every model.
    Ollama's default is small, and longer prompts are silently truncated.
    """
    if ollama_is_running():
        return
    if shutil.which("ollama") is None:
        print("Installing Ollama ...")
        if shutil.which("apt-get") and shutil.which("zstd") is None:
            subprocess.run("apt-get -qq update && apt-get -qq install -y zstd",
                           shell=True, check=False, stdout=subprocess.DEVNULL)
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True,
                       check=True, stdout=subprocess.DEVNULL)
    env = dict(os.environ, OLLAMA_CONTEXT_LENGTH=str(context_length))
    log = open(log_path, "a")
    subprocess.Popen(["ollama", "serve"], stdout=log, stderr=log, env=env)
    start = time.time()
    while not ollama_is_running():
        if time.time() - start > timeout:
            raise RuntimeError(f"Ollama did not start; see {log_path}")
        time.sleep(1)
    print("Ollama server is running.")


def pull_model(model):
    """Download a model into Ollama (does nothing if it is already present)."""
    listed = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
    if model in listed or f"{model}:latest" in listed:
        return
    print(f"Downloading {model} (this happens only once per runtime) ...")
    subprocess.run(["ollama", "pull", model], check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    print("Download finished.")


def setup_llm(backend="ollama", model=None, base_url=None, api_key=None):
    """Return ``(client, model_name)`` for the chosen back-end."""
    if backend == "ollama":
        model = model or DEFAULT_OLLAMA_MODEL
        start_ollama()
        pull_model(model)
        client = OpenAI(base_url=f"{OLLAMA_HOST}/v1", api_key="ollama")
    elif backend == "api":
        base_url = base_url or get_secret("LLM_BASE_URL", "https://api.openai.com/v1")
        api_key = api_key or get_secret("LLM_API_KEY")
        model = model or get_secret("LLM_MODEL")
        if not api_key or not model:
            raise ValueError("Set LLM_API_KEY and LLM_MODEL as Colab secrets or "
                             "environment variables, or pass them as arguments.")
        client = OpenAI(base_url=base_url, api_key=api_key)
    else:
        raise ValueError("backend must be 'ollama' or 'api'")
    print(f"backend={backend}  model={model}")
    return client, model


def chat(client, model, prompt=None, system=None, messages=None, **kwargs):
    """Send one request and return the text of the answer.

    Either ``prompt`` (with an optional ``system`` message) or a full
    ``messages`` list must be given. Extra keyword arguments such as
    ``temperature`` or ``max_tokens`` are passed to the API unchanged.
    """
    if messages is None:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model=model, messages=messages, **kwargs)
    return response.choices[0].message.content


def download_data(*names, target="data"):
    """Download course data files (paths relative to the repository's data folder)."""
    os.makedirs(target, exist_ok=True)
    paths = []
    for name in names:
        path = os.path.join(target, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            urllib.request.urlretrieve(f"{DATA_URL}/{urllib.request.quote(name)}", path)
        paths.append(path)
    return paths[0] if len(paths) == 1 else paths


STORE_DOCS = [
    "01_shipping.md", "02_returns.md", "03_warranty.md", "04_payment.md",
    "05_club.md", "06_orders.md", "07_account.md", "08_installation.md",
    "09_privacy.md", "10_support.md", "11_gift_cards.md", "12_corporate.md",
]


def load_store_docs(target="data"):
    """Download the Novin Kala policy documents and return {file_name: text}."""
    docs = {}
    for name in STORE_DOCS:
        path = download_data(f"novinkala_docs/{name}", target=target)
        with open(path, encoding="utf-8") as f:
            docs[name] = f.read()
    return docs
