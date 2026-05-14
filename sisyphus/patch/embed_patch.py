from langchain_openai import OpenAIEmbeddings


class OpenAIEmbeddingThrottle(OpenAIEmbeddings):
    """Thin wrapper kept for naming compatibility. No throttling — rate-limiting is handled by the API provider."""
    pass
