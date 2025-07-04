
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from constant import GeminiKey , AIMLKEY, MistralKey

from langchain_mistralai import ChatMistralAI

AiMl = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        api_key= AIMLKEY,
        base_url="https://api.aimlapi.com/v1",
        verbose=True
    )

print(MistralKey)
mistral = ChatMistralAI(
    model="devstral-small-2505",
    api_key=MistralKey,  # Use the Mistral API key
    temperature=0,
    max_retries=2,
    # other params...
)

deeepSeek =  ChatOllama(
        model= "deepseek-r1:1.5b",
        temperature=0.2,
        verbose=False
    )

gemini = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GeminiKey,  # Use the Gemini API key
            temperature=0.2,
        )