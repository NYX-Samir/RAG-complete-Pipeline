import requests


class LocalLLM:

    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3.2"):
        self.url = f"{host}/api/generate"
        self.model = model

        print(f"LocalLLM initialized with model: {self.model}")

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        r = requests.post(self.url, json=payload, timeout=120)
        r.raise_for_status()

        data = r.json()
        return data.get("response", "").strip()
