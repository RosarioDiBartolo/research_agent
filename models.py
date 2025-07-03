
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from constant import GeminiKey 




gemini = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GeminiKey,  # Use the Gemini API key
            temperature=0.2,
        )