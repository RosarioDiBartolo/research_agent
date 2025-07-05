
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from constant import GeminiKey , AIMLKEY, MistralKey

from langchain_mistralai import ChatMistralAI

from langchain.chat_models import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache

from services.llm_log import LoggingLLMWrapper

# Enable caching to file
set_llm_cache(SQLiteCache(database_path=".model_cache.db"))
 
temperature = 0
AiMl = LoggingLLMWrapper("gpt-4o", ChatOpenAI(
    model="gpt-4o",
    temperature=temperature,
    api_key=AIMLKEY,
    base_url="https://api.aimlapi.com/v1",
    verbose=True
))

mistral = LoggingLLMWrapper("devstral-small-2505", ChatMistralAI(
    model="devstral-small-2505",
    api_key=MistralKey,
    temperature=temperature,
    max_retries=2,
))

deeepSeek = LoggingLLMWrapper("deepseek-r1", ChatOllama(
    model="deepseek-r1:1.5b",
    temperature=temperature,
    verbose=False
))

gemma = LoggingLLMWrapper("gemma-2b", ChatOllama(
    model="gemma:2b",
    temperature=temperature,
    verbose=False
))

gemini = LoggingLLMWrapper("gemini-1.5-flash", ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GeminiKey,
    temperature=temperature,
))
