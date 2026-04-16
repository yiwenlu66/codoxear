# Pi New-Session Model Source

## Goal

Make the Pi backend section of the `New session` dialog use `~/.pi/agent/models.json` as the source of truth for provider choices and provider-specific model suggestions.

## User Intent

When the user creates a new Pi session, the launch form should behave like Pi itself:

- the `Provider` list should come from `~/.pi/agent/models.json`
- the `Model` suggestions should change when the selected provider changes
- the UI should stop treating Pi models as one flat cross-provider list
- users should still be able to type a model manually when they need a custom or newly added model id

## Current State

Pi launch defaults are assembled in `codoxear/server.py` by `_read_pi_launch_defaults()`.

Current behavior is only partially aligned with Pi config:

- `provider_choices` start from `models.json.providers`, but OAuth-backed providers from `auth.json` are also appended
- `model_choices` are only collected for the configured default provider, not for every provider
- the response payload does not include a provider-to-model mapping
- `web/src/components/new-session/NewSessionDialog.tsx` treats Pi models as a single flat list and cannot update suggestions when the provider changes

This means the Pi launch flow can show providers that do not have model suggestions and can keep suggesting models from the wrong provider after a provider switch.

## Desired Behavior

### Provider source of truth

- Pi `Provider` choices in the new-session dialog come from `~/.pi/agent/models.json.providers`.
- The server should preserve file order so the dialog matches the local Pi configuration.
- `auth.json` may still be relevant to runtime authentication, but it should not add extra provider choices to the launch UI.

### Provider-model linkage

- Each Pi provider should have its own ordered list of model ids derived from that provider's `models[*].id` entries.
- When the selected provider changes, the dialog should immediately swap the `Model` suggestions to that provider's list.
- Model suggestions should never be built from models belonging to a different provider.

### Defaults and fallback rules

- If `settings.json` defines `defaultProvider` and that provider exists in `models.json`, use it as the initial provider.
- If the configured default provider is missing from `models.json`, fall back to the first provider defined in `models.json`.
- If `settings.json` defines `defaultModel` and that model exists under the selected provider, use it as the initial model.
- If the configured default model does not belong to the selected provider, fall back to the first model for that provider, or to an empty value when the provider has no models.

### Manual model entry

- `Model` remains a text input with a datalist, not a hard select.
- Users may still type a custom model id.
- Provider switching updates the datalist suggestions even when the current text value is custom.

### User-edit preservation

- If the user has not manually edited `Model`, changing provider may replace the model value with that provider's default suggestion.
- If the user has manually edited `Model`, changing provider should keep the typed value and only refresh the datalist suggestions.

## Recommended Approach

Extend the Pi launch-defaults payload so the backend returns both:

- the ordered Pi provider list
- a provider-to-model-list mapping for Pi only

This is preferred because:

- `models.json` parsing stays in one place on the server
- the frontend receives the exact structure it needs for linked controls
- session creation validation can stay aligned with the same backend data source
- the change is additive and does not require a separate Pi-specific API endpoint

## API Design

Extend the Pi portion of `new_session_defaults` returned from `GET /api/sessions`.

Current Pi payload is roughly:

```json
{
  "agent_backend": "pi",
  "provider_choice": "macaron",
  "provider_choices": ["macaron", "openai"],
  "model": "gpt-5.4",
  "models": ["gpt-5.4"],
  "reasoning_effort": "high",
  "reasoning_efforts": ["off", "minimal", "low", "medium", "high", "xhigh"]
}
```

Add an additive mapping field:

```json
{
  "agent_backend": "pi",
  "provider_choice": "macaron",
  "provider_choices": ["macaron", "openai"],
  "model": "gpt-5.4",
  "models": ["gpt-5.4", "gpt-5.3-codex", "gpt-5.4-mini"],
  "provider_models": {
    "macaron": ["gpt-5.4", "gpt-5.3-codex", "gpt-5.4-mini"],
    "openai": ["gpt-5.4"]
  },
  "reasoning_effort": "high",
  "reasoning_efforts": ["off", "minimal", "low", "medium", "high", "xhigh"]
}
```

Rules:

- `provider_models` is only added for Pi launch defaults.
- `models` continues to exist for compatibility and should reflect the currently selected initial provider.
- Providers with no valid `models[*].id` entries may still appear in `provider_choices`, but their mapped model list is empty.

## Backend Design

### Parsing `models.json`

In `codoxear/server.py`:

- parse `models.json.providers` once inside `_read_pi_launch_defaults()`
- build an ordered `provider_choices` list from provider keys
- build `provider_models: dict[str, list[str]]` using deduplicated `models[*].id` values per provider
- derive the top-level `models` field from the selected initial provider

### Validation behavior

Keep request validation simple:

- `model_provider` should continue to be validated against the Pi provider choices returned from `models.json`
- `model` should remain a normalized freeform string
- do not reject a typed custom model just because it is absent from the provider's suggested list

This preserves flexibility while still making the UI suggestions correct.

### Compatibility

- Existing Codex launch defaults are unchanged.
- Existing Pi clients that only consume `provider_choices`, `model`, and `models` continue to work.
- The web UI can progressively adopt `provider_models` without a breaking backend transition.

## Frontend Design

### Type updates

In `web/src/lib/types.ts`:

- add `provider_models?: Record<string, string[]>` to `LaunchBackendDefaults`

### Dialog behavior

In `web/src/components/new-session/NewSessionDialog.tsx`:

- for Pi, compute model choices from `provider_models[providerChoice]`
- for other backends, keep using the existing flat `models` list
- when the backend changes, initialize provider and model from the backend defaults as today
- when the Pi provider changes, recompute model suggestions immediately

### Model update rules

Track whether the model field has been manually touched.

- untouched model field: switching Pi provider updates the model value to the first valid suggestion for that provider, or to the configured Pi default when it belongs to that provider
- touched model field: switching Pi provider leaves the current input value alone

This keeps the form helpful without unexpectedly erasing a deliberate custom model.

### UI shape

- keep `Provider` as a select
- keep `Model` as an input with `datalist`
- no additional visible controls are needed

The UX change should feel like a smarter version of the current form rather than a redesign.

## Testing

### Backend tests

Add or update tests around Pi launch defaults to cover:

- provider choices come from `models.json` provider keys
- `auth.json` providers do not appear in Pi launch provider choices
- `provider_models` is built correctly per provider
- top-level `models` reflects the selected initial provider
- invalid configured defaults fall back to valid provider/model values

### Frontend tests

Add or update `web/src/components/new-session/NewSessionDialog.test.tsx` to cover:

- Pi provider options come from backend defaults
- changing Pi provider changes the datalist suggestions
- changing Pi provider updates the model value only when the model field is untouched
- changing Pi provider preserves a manually typed model value

## Out of Scope

- changing how Pi itself resolves providers or authentication
- making model entry fully locked down to a select control
- changing Codex launch defaults or Codex provider behavior
- introducing a dedicated endpoint just for Pi model metadata

## Implementation Notes

- Prefer the smallest additive payload change over a larger refactor of launch defaults.
- Keep the server as the single interpreter of Pi config files.
- Avoid pulling runtime-auth-only providers into UI semantics for provider/model selection.
