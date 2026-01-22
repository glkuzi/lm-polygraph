from lm_polygraph.stat_calculators.extract_claims import ClaimsExtractor
from lm_polygraph.utils.openai_chat import OpenAIChat
from lm_polygraph.utils.openai_mock import OpenAIMock


def load_stat_calculator(config, builder):
    if not hasattr(builder, "chat_model"):
        builder.chat_model = OpenAIChat(
            openai_model=config.openai_model,
            base_url=getattr(config, "base_url", None),
            timeout=getattr(config, "timeout", 600),
            cache_path=config.cache_path,
        )
    if config.get("mock_model", False):
        builder.chat_model = OpenAIMock(
            openai_model=config.openai_model,
            base_url=getattr(config, "base_url", None),
            timeout=getattr(config, "timeout", 600),
            cache_path=config.cache_path,
            cache_responses=False,
        )
        return ClaimsExtractor(
            builder.chat_model,
            language=config.language,
            sent_separators=config.get("sent_separators", ".?!。？！\n"),
            n_threads=getattr(config, "n_threads", 1),
        )

    return ClaimsExtractor(
        builder.chat_model,
        language=config.language,
        n_threads=getattr(config, "n_threads", 1),
    )
