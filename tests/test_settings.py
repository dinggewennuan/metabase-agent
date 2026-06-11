from metabase_agent.config.settings import Settings


def test_openai_base_url_configurable() -> None:
    settings = Settings(OPENAI_BASE_URL="https://example.com/v1")

    assert settings.openai_base_url == "https://example.com/v1"


def test_openai_wire_api_configurable() -> None:
    settings = Settings(OPENAI_WIRE_API="responses")

    assert settings.openai_wire_api == "responses"


def test_openai_model_defaults_to_gpt_5() -> None:
    settings = Settings(OPENAI_MODEL="gpt-5")

    assert settings.openai_model == "gpt-5"


def test_get_settings_is_cached() -> None:
    from metabase_agent.config.settings import get_settings

    get_settings.cache_clear()
    assert get_settings() is get_settings()


def test_metabase_base_url_default_is_neutral() -> None:
    from metabase_agent.config.settings import Settings

    assert Settings(_env_file=None).metabase_base_url == ""


def test_require_token_defaults_false() -> None:
    from metabase_agent.config.settings import Settings

    assert Settings(_env_file=None).agent_require_token is False
