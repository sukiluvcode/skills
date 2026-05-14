from langchain_openai import ChatOpenAI


class ChatOpenAIThrottle(ChatOpenAI):
    """Thin wrapper kept for naming compatibility. No throttling — rate-limiting is handled by the API provider."""
    pass
