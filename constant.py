from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()

OpenRouterKey = os.getenv("OPEN_ROUTER_API_KEY") #

GeminiKey = os.getenv("GEMINI_API_KEY") #
AIMLKEY = os.getenv("AIML_API_KEY") #
MistralKey = os.getenv("MISTRAL_API_KEY") #