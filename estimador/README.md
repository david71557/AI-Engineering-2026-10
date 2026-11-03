# Estimador de proyectos software

Servicio FastAPI que genera estimaciones de proyectos software con un LLM, y un cliente Streamlit con un formulario para usarlo.

- **Contrato tipado**: `EstimationRequest` (`description`, `project_type`, `detail_level`, `output_format` y, opcional, `reference_projects`) → `EstimationResponse` (`text`, `prompt_version`).
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
| `reference_projects` | Opcional, hasta 5. Cada uno: `name` (1–120 caracteres), `description` (10–1000), `duration_weeks` (> 0), `cost_eur` (≥ 0) y, opcional, `team` (texto libre) |

Con `reference_projects`, el prompt del usuario añade un bloque `<reference_projects>` con un `<reference_project>` por proyecto, y el system prompt pide al LLM que los use para calibrar fases de alcance parecido y que diga en cuál se ha basado. Sin el campo (o con `[]`), los prompts quedan exactamente igual que antes, así que las respuestas ya cacheadas siguen siendo válidas.

```json
"reference_projects": [
  {
    "name": "Physio booking",
    "description": "Booking SaaS for physiotherapy clinics with Stripe deposits.",
    "duration_weeks": 13,
    "cost_eur": 39650,
    "team": "1 designer, 2 developers"
  }
]
```

Para vaciar la caché y forzar una nueva llamada al LLM: `docker exec estimator-redis redis-cli FLUSHALL`.

## Tests

```bash
uv run pytest
```

Prueban las plantillas de prompt (`tests/prompts/test_estimation_v1.py`): que la descripción va dentro de `<project_description>`, que los bloques condicionales de `output_format` y `detail_level` solo aparecen cuando toca, que se generan bien todas las combinaciones y que el loader falla con variables o versiones inexistentes. `tests/prompts/test_estimation_v2.py` comprueba que `v2` solo se diferencia de `v1` en el tono, y `tests/routers/test_estimations.py` prueba el query param `prompt_version` con un LLM falso. No llaman al LLM ni necesitan Docker, y tardan menos de un segundo.

## Versiones del prompt

Cada versión vive en `app/prompts/estimation/<versión>/` y se elige por petición con el query param `prompt_version` (por defecto `v1`):

```bash
curl -X POST "http://localhost:8000/api/v1/estimate?prompt_version=v2" \
  -H "Content-Type: application/json" \
  -d '{"description": "...", "project_type": "web_saas", "detail_level": "medium", "output_format": "phases_table"}'
```

| Versión | Qué cambia |
|---|---|
| `v1` | Consultor senior de estimación, tono técnico. |
| `v2` | Solo cambia el tono: tech lead que asesora a un fundador no técnico. Empieza con una frase de conclusión, usa lenguaje llano y señala el principal factor de coste. Ejemplos, tarifas y reglas son idénticos a `v1`. |

Una versión desconocida devuelve `422` sin llamar al LLM. La versión usada se devuelve en `prompt_version` y aparece en los logs. La caché incluye el system prompt completo en la clave, así que las versiones no comparten respuestas cacheadas.

Cada render emite un evento `prompt_rendered` en los logs, con hashes y longitudes pero nunca el texto del prompt, porque la descripción puede contener datos del cliente:

```
[info] prompt_rendered  prompt_version=v2 template_hash=38c14b36bb1bb00a system_hash=256a9da6e3ac9e7e user_hash=09fcd9ce18ab32a2 system_chars=5389 user_chars=170 reference_projects=0
```

- `template_hash`: hash de todos los `.j2` de la carpeta de la versión. Es igual en todas las peticiones con las mismas plantillas desplegadas; si cambia sin que cambie `prompt_version`, alguien editó la versión sin crear una nueva.
- `system_hash` / `user_hash`: hash del texto renderizado. Para saber qué prompt exacto recibió el LLM, renderiza la misma petición en local y compara con `content_hash()` de `app/prompts/loader.py`.

Para añadir una versión nueva:

1. Copia la carpeta de la última versión a `vN/` y edita las plantillas (incluido el `{% include %}` de `examples.j2`).
2. Añade `VN = "vN"` al enum `PromptVersion` en `app/schemas/estimation.py`.
3. Añade sus tests en `tests/prompts/test_estimation_vN.py`.
