import os
import requests


class LocalLLM:
    def __init__(
        self,
        host: str | None = None,
        model: str = "llama3.2"
    ):
        self.host = host or os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434"
        )

        self.url = f"{self.host}/api/generate"
        self.model = model

        print(f"LocalLLM initialized")
        print(f"Model: {self.model}")
        print(f"Ollama URL: {self.url}")

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        r = requests.post(self.url, json=payload, timeout=300)
        r.raise_for_status()

        data = r.json()
        return data.get("response", "").strip()
