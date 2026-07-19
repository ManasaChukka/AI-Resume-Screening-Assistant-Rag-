import os
import time
import httpx
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Errors worth retrying: transient network/connection resets, not auth/quota errors.
RETRYABLE_EXCEPTIONS = (
    httpx.ReadError,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.WriteError,
    ConnectionResetError,
    ConnectionError,
)

MAX_RETRIES = 4
BASE_DELAY_SECONDS = 2  # doubles each retry: 2s, 4s, 8s, 16s


def _extract_text(content) -> str:
    """
    Normalize a LangChain message's .content into a plain string.

    Newer Gemini models can return content as a list of structured
    blocks (e.g. [{"type": "text", "text": "...", "extras": {...}}])
    instead of a plain string. This pulls out just the text pieces.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "".join(text_parts)

    return str(content)


def _get_api_key() -> str | None:
    """
    Look up the Google API key from either source:
    1. Streamlit Cloud secrets (used when deployed on share.streamlit.io)
    2. A local .env file (used when running on your own machine)

    Returns:
        The API key string, or None if not found anywhere.
    """
    try:
        import streamlit as st
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        # Not running inside Streamlit, or no secrets.toml configured.
        # That's fine locally — fall through to .env below.
        pass

    return os.getenv("GOOGLE_API_KEY")


def get_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """
    Initialize and return the Gemini chat model.

    Args:
        temperature: Controls randomness of output.
            0.0 = fully deterministic/factual.
            Higher values = more creative/varied output.

    Returns:
        A configured ChatGoogleGenerativeAI instance.

    Raises:
        ValueError: If GOOGLE_API_KEY is not set.
    """
    api_key = _get_api_key()
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Set it in your .env file (local) "
            "or in Streamlit Cloud's Secrets settings (deployed)."
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=api_key,
        temperature=temperature,
    )
    return llm


def ask_llm(prompt: str, temperature: float = 0.0) -> str:
    """
    Send a prompt to Gemini and return its text response.

    Args:
        prompt: The full prompt text to send.
        temperature: Randomness setting for this call.

    Returns:
        The model's text response.
    """
    llm = get_llm(temperature=temperature)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = llm.invoke(prompt)
            return _extract_text(response.content)
        except RETRYABLE_EXCEPTIONS as e:
            last_error = e
            if attempt == MAX_RETRIES:
                break
            delay = BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            print(
                f"[ask_llm] Connection error on attempt {attempt}/{MAX_RETRIES} "
                f"({type(e).__name__}: {e}). Retrying in {delay}s..."
            )
            time.sleep(delay)

    raise ConnectionError(
        "Could not reach Google's Gemini API after multiple retries. "
        "This usually means antivirus/firewall/VPN on this machine is resetting "
        "the HTTPS connection to generativelanguage.googleapis.com, or the "
        "network connection is unstable. Try: (1) disabling antivirus/VPN "
        "temporarily, (2) switching networks (e.g. mobile hotspot) to test, "
        "and (3) verifying GOOGLE_API_KEY in your .env file is correct."
    ) from last_error