"""The OpenAI Conversation integration."""

from __future__ import annotations

import openai
import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import (
    ConfigEntryNotReady,
    HomeAssistantError,
    ServiceValidationError,
)
from homeassistant.helpers import config_validation as cv, selector
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType

from .const import DOMAIN, LOGGER

SERVICE_GENERATE_IMAGE = "generate_image"
PLATFORMS = (Platform.CONVERSATION,)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

type OpenAIConfigEntry = ConfigEntry[openai.AsyncClient]

# === Begin Heinous Monkey Patching Crimes ===

from homeassistant.helpers import llm as patched_llm

# Blank out the BASE_PROMPT to fix caching issues (https://github.com/home-assistant/core/issues?q=llm%20cache)
# Issues #133687 and #134847, PR #141156
patched_llm.BASE_PROMPT = ('')


# Bastardized copy of _get_exposed_entities from llm.py as of https://github.com/home-assistant/core/blob/2025.3.4/homeassistant/helpers/llm.py
# Fixes issues related to 0-255 brightness representation
# Issues #134848, #134592
def _custom_get_exposed_entities(
    hass: HomeAssistant, assistant: str
) -> dict[str, dict[str, dict[str, Any]]]:
    """Get exposed entities.

    Splits out calendars and scripts.
    """

    from enum import Enum
    from decimal import Decimal
    from homeassistant.helpers import device_registry as dr
    from homeassistant.helpers import area_registry as ar
    from homeassistant.helpers import entity_registry as er
    from homeassistant.components.homeassistant import async_should_expose
    from homeassistant.components.script import DOMAIN as SCRIPT_DOMAIN
    from homeassistant.components.calendar import (
        DOMAIN as CALENDAR_DOMAIN
    )

    area_registry = ar.async_get(hass)
    entity_registry = er.async_get(hass)
    device_registry = dr.async_get(hass)
    interesting_attributes = {
        "temperature",
        "current_temperature",
        "temperature_unit",
        "brightness",
        "humidity",
        "unit_of_measurement",
        "device_class",
        "current_position",
        "percentage",
        "volume_level",
        "media_title",
        "media_artist",
        "media_album_name",
    }

    entities = {}
    data: dict[str, dict[str, Any]] = {
        SCRIPT_DOMAIN: {},
        CALENDAR_DOMAIN: {},
    }

    for state in hass.states.async_all():
        if not async_should_expose(hass, assistant, state.entity_id):
            continue

        description: str | None = None
        entity_entry = entity_registry.async_get(state.entity_id)
        names = [state.name]
        area_names = []

        if entity_entry is not None:
            names.extend(entity_entry.aliases)
            if entity_entry.area_id and (
                area := area_registry.async_get_area(entity_entry.area_id)
            ):
                # Entity is in area
                area_names.append(area.name)
                area_names.extend(area.aliases)
            elif entity_entry.device_id and (
                device := device_registry.async_get(entity_entry.device_id)
            ):
                # Check device area
                if device.area_id and (
                    area := area_registry.async_get_area(device.area_id)
                ):
                    area_names.append(area.name)
                    area_names.extend(area.aliases)

        info: dict[str, Any] = {
            "names": ", ".join(names),
            "domain": state.domain,
            "state": state.state,
        }

        if description:
            info["description"] = description

        if area_names:
            info["areas"] = ", ".join(area_names)

        attributes = {}
        for attr_name, attr_value in state.attributes.items():
            if attr_name in interesting_attributes:
                if attr_name == "brightness" and isinstance(attr_value, (int, float)):
                    # Convert brightness 0–255 to 0–100
                    attributes[attr_name] = round((attr_value / 255.0) * 100)
                elif isinstance(attr_value, (Enum, Decimal, int)):
                    attributes[attr_name] = str(attr_value)
                else:
                    attributes[attr_name] = attr_value

        if attributes:
            info["attributes"] = attributes

        if state.domain in data:
            data[state.domain][state.entity_id] = info
        else:
            entities[state.entity_id] = info

    data["entities"] = entities
    return data

patched_llm._get_exposed_entities = _custom_get_exposed_entities

# === End Monkey Patching Crimes ===

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up OpenAI Conversation."""

    async def render_image(call: ServiceCall) -> ServiceResponse:
        """Render an image with dall-e."""
        entry_id = call.data["config_entry"]
        entry = hass.config_entries.async_get_entry(entry_id)

        if entry is None or entry.domain != DOMAIN:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="invalid_config_entry",
                translation_placeholders={"config_entry": entry_id},
            )

        client: openai.AsyncClient = entry.runtime_data

        try:
            response = await client.images.generate(
                model="dall-e-3",
                prompt=call.data["prompt"],
                size=call.data["size"],
                quality=call.data["quality"],
                style=call.data["style"],
                response_format="url",
                n=1,
            )
        except openai.OpenAIError as err:
            raise HomeAssistantError(f"Error generating image: {err}") from err

        return response.data[0].model_dump(exclude={"b64_json"})

    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_IMAGE,
        render_image,
        schema=vol.Schema(
            {
                vol.Required("config_entry"): selector.ConfigEntrySelector(
                    {
                        "integration": DOMAIN,
                    }
                ),
                vol.Required("prompt"): cv.string,
                vol.Optional("size", default="1024x1024"): vol.In(
                    ("1024x1024", "1024x1792", "1792x1024")
                ),
                vol.Optional("quality", default="standard"): vol.In(("standard", "hd")),
                vol.Optional("style", default="vivid"): vol.In(("vivid", "natural")),
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )
    return True


async def async_setup_entry(hass: HomeAssistant, entry: OpenAIConfigEntry) -> bool:
    """Set up OpenAI Conversation from a config entry."""
    client = openai.AsyncOpenAI(
        api_key=entry.data[CONF_API_KEY],
        http_client=get_async_client(hass),
    )

    # Cache current platform data which gets added to each request (caching done by library)
    _ = await hass.async_add_executor_job(client.platform_headers)

    try:
        await hass.async_add_executor_job(client.with_options(timeout=10.0).models.list)
    except openai.AuthenticationError as err:
        LOGGER.error("Invalid API key: %s", err)
        return False
    except openai.OpenAIError as err:
        raise ConfigEntryNotReady(err) from err

    entry.runtime_data = client

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload OpenAI."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
