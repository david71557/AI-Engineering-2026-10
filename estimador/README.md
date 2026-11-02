# Estimador de proyectos software

Servicio FastAPI que genera estimaciones de proyectos software con un LLM, y un cliente Streamlit con un formulario para usarlo.

- **Contrato tipado**: `EstimationRequest` (`description`, `project_type`, `detail_level`, `output_format`) → `EstimationResponse` (`text`, `prompt_version`).
- **Prompts versionados** con Jinja2 en `app/prompts/estimation/<versión>/` (`system.j2`, `user.j2`, `examples.j2`).
- **Wrapper LiteLLM**: primero OpenAI (`PRIMARY_MODEL`) y, si falla, Claude (`FALLBACK_MODEL`). Caché exact-match en Redis y logs con structlog.

## Requisitos

- [uv](https://docs.astral.sh/uv/)
- Docker y Docker Compose

## Configuración

Crea un `.env` en la raíz del proyecto (no se sube a git). Hace falta al menos una API key; los demás valores son opcionales y estos son sus valores por defecto:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

PRIMARY_MODEL=gpt-4o-mini
FALLBACK_MODEL=claude-haiku-4-5-20251001
LLM_TIMEOUT=30
LLM_RETRIES=2
CACHE_TTL=86400
ESTIMATOR_API_BASE_URL=http://localhost:8000
```

## Levantar el servicio

```bash
uv sync                          # dependencias locales (Streamlit y tests)
docker compose up -d --build     # API en :8000 + Redis en :6379
```

- Docs interactivas: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Logs: `docker compose logs -f estimator`

`app/` está montado como volumen con `--reload`, así que los cambios en código y plantillas se aplican sin reconstruir la imagen. Si cambias el `.env`, recrea el contenedor con `docker compose up -d --force-recreate`.

### Cliente Streamlit

Corre en local, fuera de Docker, y llama a la API en `ESTIMATOR_API_BASE_URL`:

```bash
uv run streamlit run streamlit_app.py
```

### Llamada directa a la API

```bash
curl -X POST http://localhost:8000/api/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{
        "description": "Admin portal for a B2B SaaS: SSO login, customer list, plan upgrades and an audit log.",
        "project_type": "web_saas",
        "detail_level": "medium",
        "output_format": "phases_table"
      }'
```

Valores válidos:

| Campo | Valores |
|---|---|
| `description` | 20–2000 caracteres |
| `project_type` | `mobile_app`, `web_saas`, `internal_tool`, `data_pipeline` |
| `detail_level` | `summary`, `medium`, `detailed` |
| `output_format` | `phases_table`, `line_items`, `narrative` |

Para vaciar la caché y forzar una nueva llamada al LLM: `docker exec estimator-redis redis-cli FLUSHALL`.

## Tests

```bash
uv run pytest
```

Prueban las plantillas de prompt (`tests/prompts/test_estimation_v1.py`): que la descripción va dentro de `<project_description>`, que los bloques condicionales de `output_format` y `detail_level` solo aparecen cuando toca, que se generan bien todas las combinaciones y que el loader falla con variables o versiones inexistentes. No llaman al LLM ni necesitan Docker, y tardan menos de un segundo.

## Nueva versión del prompt

1. Copia `app/prompts/estimation/v1/` a `v2/` y edita las plantillas.
2. Cambia `PROMPT_VERSION` en `app/routers/estimations.py`.
3. Añade sus tests en `tests/prompts/test_estimation_v2.py`.

La versión usada se devuelve en `prompt_version` y aparece en los logs.
