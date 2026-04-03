import os
import threading


class ChatClientManager:
    """Manages the DeepSeek chat client (OpenAI-compatible SDK)."""

    def __init__(self, config, cli_model_override=None, logger=None):
        self.config = config
        self.cli_model_override = cli_model_override
        self.logger = logger
        self.client = None
        self.model_init_complete = False
        self._resolved_model_name = None
        self._init_lock = threading.Lock()

    def _log_debug(self, message):
        if self.logger:
            self.logger.debug(message)

    def _log_info(self, message):
        if self.logger:
            self.logger.info(message)

    def _log_warning(self, message):
        if self.logger:
            self.logger.warning(message)

    def _log_error(self, message):
        if self.logger:
            self.logger.error(message)

    def get_model_name(self):
        if self._resolved_model_name:
            return self._resolved_model_name
        return self.cli_model_override or self.config.get("deepseek", {}).get("model", "deepseek-chat")

    def get_client(self):
        if self.model_init_complete and self.client:
            return self.client

        if self.client is None:
            with self._init_lock:
                if self.client is None:
                    self._log_debug("Initializing DeepSeek AI client on demand...")
                    deepseek_key = os.getenv("DEEPSEEK_API_KEY")

                    if deepseek_key:
                        try:
                            from openai import OpenAI

                            self.client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
                            self.model_init_complete = True
                            self._resolved_model_name = self.cli_model_override or self.config.get(
                                "deepseek", {}
                            ).get("model", "deepseek-chat")
                            self._log_info("DeepSeek client (via OpenAI SDK) initialized successfully.")
                        except Exception as exc:
                            self.model_init_complete = False
                            self._log_error(f"DeepSeek initialization failed: {exc}")
                    else:
                        self.model_init_complete = False
                        self._log_error(
                            "DEEPSEEK_API_KEY not found. Set it in your .env file to enable AI chat."
                        )

        return self.client
