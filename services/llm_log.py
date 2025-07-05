 

import logging


class LoggingLLMWrapper:
    def __init__(self, model_name, llm):
        self.llm = llm
        self.model_name = model_name

    def invoke(self, prompt):
        # You can handle string or HumanMessage prompts
        content = prompt.content if hasattr(prompt, "content") else prompt

        logging.info(f"[PROMPT] [{self.model_name}] Prompt: {content}")

        try:
            response = self.llm.invoke(prompt)
            output = response.content if hasattr(response, "content") else str(response)
            logging.info(f"[RESPONSE] [{self.model_name}] Response: {output}")
            return response
        except Exception as e:
            logging.error(f"[ERROR] [{self.model_name}] Error: {str(e)}")
            raise
    def with_structured_output(self, response_type):
        # Wrap the model with structured output
        wrapped_model = self.llm.with_structured_output(response_type)
        return LoggingLLMWrapper(self.model_name, wrapped_model)