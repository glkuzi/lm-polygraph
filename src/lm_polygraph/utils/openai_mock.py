import openai
import os
import time
import logging
import httpx
import diskcache as dc


log = logging.getLogger()


class OpenAIMock:
    """
    Allows for the implementation of a singleton class to chat with mock model for dataset marking. Always return the input.
    """

    def __init__(
        self,
        openai_model: str = "gpt-4o",
        base_url: str = None,
        cache_path: str = os.path.expanduser("~") + "/.cache",
        timeout: int = 600,
        max_tokens: int = None,
        rewrite_cache: bool = False,
        cache_responses: bool = True,
    ):
        """
        Parameters
        ----------
        openai_model: str
            the model to use in OpenAI to chat.
        """
        api_key = os.environ.get("OPENAI_API_KEY", None)
        if api_key is not None:
            openai.api_key = api_key
        proxy_url = os.environ.get("HTTPS_PROXY", None)
        if proxy_url is not None:
            self.proxy = httpx.Client(proxy=proxy_url)
        else:
            self.proxy = None
        self.openai_model = openai_model

        self.cache_path = os.path.join(cache_path, "openai_chat_cache.diskcache")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        self.base_url = base_url
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.rewrite_cache = rewrite_cache
        self.cache_responses = cache_responses

    def ask(self, message: str) -> str:
        cache_settings = dc.DEFAULT_SETTINGS.copy()
        cache_settings["eviction_policy"] = "none"
        cache_settings["size_limit"] = int(1e12)
        cache_settings["cull_limit"] = 0
        cache_settings["timeout"] = 2400
        if self.cache_responses:
            openai_responses = dc.Cache(self.cache_path, **cache_settings)

            if (self.openai_model, message) in openai_responses and not self.rewrite_cache:
                reply = openai_responses[(self.openai_model, message)]

            else:
                # Ask openai
                if openai.api_key is None:
                    raise Exception(
                        "Cant ask openAI without token. "
                        "Please specify OPENAI_API_KEY in environment parameters."
                    )
                reply = message

                openai_responses[(self.openai_model, message)] = reply
                openai_responses.close()
        else:
            # note - this is a mock class
            reply = message

        if "please provide" in reply.lower():
            return ""
        if "to assist you" in reply.lower():
            return ""
        if "as an ai language model" in reply.lower():
            return ""

        return reply
