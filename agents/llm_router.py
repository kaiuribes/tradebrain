import os
from llama_cpp import Llama

# Map model names to file paths
MODEL_PATHS = {
    "mistral": "models/mistral/mistral.gguf",
    "gpt4all": "models/gpt4all/gpt4all-model.gguf"
}

# Cache for loaded models
LLM_INSTANCES = {}

def call_local_llm(prompt, model_name="mistral", max_tokens=200):
    """
    Call a local LLM (like Mistral or GPT4All) via llama-cpp-python.
    """
    model_path = MODEL_PATHS.get(model_name)
    if not model_path or not os.path.exists(model_path):
        return f"[ERROR] Model file not found: {model_path}"

    # Load model if not already loaded
    if model_name not in LLM_INSTANCES:
        print(f"🔁 Loading model: {model_name}...")
        LLM_INSTANCES[model_name] = Llama(model_path=model_path, n_ctx=1024)

    llm = LLM_INSTANCES[model_name]
    response = llm(prompt, max_tokens=max_tokens, stop=["</s>"])
    return response["choices"][0]["text"].strip()


def call_online_llm(prompt, provider="deepseek"):
    """
    Placeholder for calling a free cloud LLM API like DeepSeek or Gemini.
    """
    return f"[SIMULATED ONLINE {provider.upper()}] Response to: {prompt}"


def route_llm_task(prompt, task_type="general"):
    """
    Route a given prompt to the best LLM based on task_type.
    """
    if task_type == "chart_analysis":
        return call_local_llm(prompt, model_name="mistral")
    elif task_type == "news_reasoning":
        return call_local_llm(prompt, model_name="gpt4all")
    elif task_type == "strategy_decision":
        return call_local_llm(prompt, model_name="mistral")
    elif task_type == "risk_management":
        return call_local_llm(prompt, model_name="gpt4all")
    else:
        return call_local_llm(prompt)
