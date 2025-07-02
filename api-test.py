import argparse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
# Parse command-line arguments
parser = argparse.ArgumentParser(description="Chat with different OpenAI providers.")
parser.add_argument("--provider", type=str, required=True, help="Provider name: e.g., 'openrouter', 'openai'")
args = parser.parse_args()

provider = args.provider.lower()


print(f"Testing provider: {provider}")
# Determine base URL based on provider
match provider:
    case "openrouter":
      from constant import OpenRouterKey

      base_url = "https://openrouter.ai/api/v1"
            
      # Initialize client with selected provider
      client = OpenAI(
          base_url=base_url,
          api_key=OpenRouterKey,  # Use the OpenRouter API key
      )

            # Send a test message
      completion = client.chat.completions.create(
          model="openai/gpt-4o",  # Adjust if model name differs per provider
          messages=[
              {
                  "role": "user",
                  "content": "What is the meaning of life?"
              }
          ]
      )

      print(f"Response from {provider}: {completion.choices[0].message.content}")




    case "gemini":
      from langchain_google_genai import ChatGoogleGenerativeAI
      from constant import GeminiKey

      # Initialize Gemini client
      client = ChatGoogleGenerativeAI(
          model="gemini-2.5-flash",  # Adjust if model name differs per provider
          google_api_key=GeminiKey,  # Use the Gemini API key
      )
      res = client.invoke("Simple test")  # Test connection

      print(f"Response from {provider}: {res}")
    case _:
        raise ValueError(f"Unknown provider: {args.provider}")


