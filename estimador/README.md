# Estimador de proyectos software

Servicio FastAPI que genera estimaciones de proyectos software con un LLM, y un cliente Streamlit para usarlo.

- **Contrato tipado**: `EstimationRequest` (`description`, `project_type`, `detail_level`, `output_format` y, opcional, `reference_projects`) → `EstimationResponse` (`result` estructurado y validado con Pydantic, `prompt_version`, `cached` y metadatos de la llamada).
- **Prompts versionados** con Jinja2 en `app/prompts/<caso de uso>/<versión>/`.
- **Wrapper LiteLLM**: primero OpenAI (`PRIMARY_MODEL`) y, si falla, Claude (`FALLBACK_MODEL`). Logs estructurados con structlog.
- **Guardrails** de entrada (moderación, inyección de prompt, datos personales) y de salida (respuestas de baja confianza).
- **Cachés** exacta y semántica en Redis Stack (solo en el endpoint transaccional).
- **Sesiones conversacionales** (Sesión 5): memoria multi-turno y adjuntos PDF/Word en `/sessions`. Ver [más abajo](#sesión-5--memoria-conversacional-y-adjuntos).

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
REDIS_URL=redis://localhost:6379
CACHE_TTL=86400
SEMANTIC_CACHE_THRESHOLD=0.85
ESTIMATOR_API_BASE_URL=http://localhost:8000

# Sesión 5
MAX_CONVERSATION_TURNS=6        # pares user+assistant que guarda la ventana
MAX_ATTACHMENT_CHARS=60000      # recorte por adjunto extraído
METADATA_EXTRACTOR_MODEL=gpt-4o-mini
```

Deja `REDIS_URL` con `localhost`: dentro de Docker el `docker-compose.yml` ya lo sobrescribe con `redis://redis:6379`, y con `redis` en el `.env` fallan las cachés cuando ejecutas algo fuera de Docker (tests, scripts, uvicorn local).

## Levantar el servicio

```bash
uv sync                          # dependencias locales (Streamlit y tests)
docker compose up -d --build     # API en :8000 + Redis Stack en :6379 (RedisInsight en :8001)
```

- Docs interactivas: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Logs: `docker compose logs -f estimator`

`app/` está montado como volumen con `--reload`, así que los cambios en código y plantillas se aplican sin reconstruir la imagen. **Reconstruye con `--build` cuando cambien las dependencias** del `pyproject.toml` (por ejemplo, tras añadir `pypdf` y `python-docx` en la Sesión 5; sin ellos los adjuntos devuelven 422). Si cambias el `.env`, recrea el contenedor con `docker compose up -d --force-recreate`. Las sesiones viven en memoria: cualquier reinicio de la API las borra.

### Cliente Streamlit

Corre en local, fuera de Docker, y llama a la API en `ESTIMATOR_API_BASE_URL`:

```bash
uv run streamlit run streamlit_app.py
```

Tres pestañas: **New estimation** (formulario transaccional), **Estimations** (historial de estimaciones con coste, duración, confianza, versión de prompt y caché) y **Conversation** (sesiones de la Sesión 5, ver más abajo).

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

Con `reference_projects`, el prompt del usuario añade un bloque `<reference_projects>` y el system prompt pide al LLM que los use para calibrar fases de alcance parecido y que diga en cuál se ha basado.

Para vaciar las cachés y forzar una nueva llamada al LLM: `docker exec estimator-redis redis-cli FLUSHALL`.

## Tests

```bash
uv run pytest
```

No llaman al LLM ni necesitan Docker ni Redis (el LLM, las cachés y la moderación se sustituyen por fakes) y tardan unos segundos. Cubren:

- **Prompts** (`tests/prompts/`): bloques condicionales, todas las combinaciones de formulario, `StrictUndefined`, y que `v2` solo cambia el tono respecto a `v1`.
- **Endpoint transaccional** (`tests/routers/test_estimations.py`): versiones de prompt, guardrails, caché exacta y semántica, errores 400/422/502.
- **Wrapper, esquema, cachés y guardrails** (`tests/test_llm_wrapper_structured.py`, `tests/test_cache_semantic.py`, `tests/test_guardrails_*.py`).
- **Sesiones** (`tests/test_sessions_*.py`, `tests/test_metadata_extractor.py`, `tests/routers/test_sessions.py`).

Los **tests del Paso 7** son de integración con `httpx.AsyncClient` sobre la app ASGI, con todo real salvo el LLM:

```bash
uv run pytest tests/test_sessions_integration.py -v
```

1. Dos peticiones en la misma sesión actualizan `ProjectMetadata`.
2. Un PDF adjunto real cambia la estimación (aparece una fase nueva y sube el coste).
3. Ocho turnos nunca envían al LLM más historial que `MAX_CONVERSATION_TURNS`.

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

Una versión desconocida devuelve `422` sin llamar al LLM. La clave de la caché exacta incluye la versión y el hash de sus plantillas, así que las versiones no comparten respuestas y editar un `.j2` invalida las anteriores.

Cada render emite un evento `prompt_rendered` en los logs, con hashes y longitudes pero nunca el texto del prompt, porque la descripción puede contener datos del cliente:

```
[info] prompt_rendered  prompt_version=v2 template_hash=38c14b36bb1bb00a system_hash=256a9da6e3ac9e7e user_hash=09fcd9ce18ab32a2 system_chars=5389 user_chars=170 reference_projects=0
```

- `template_hash`: hash de todos los `.j2` de la carpeta de la versión. Si cambia sin que cambie `prompt_version`, alguien editó la versión sin crear una nueva.
- `system_hash` / `user_hash`: hash del texto renderizado. Para saber qué prompt exacto recibió el LLM, renderiza la misma petición en local y compara con `content_hash()` de `app/prompts/loader.py`.

Para añadir una versión nueva:

1. Copia la carpeta de la última versión a `vN/` y edita las plantillas (incluido el `{% include %}` de `examples.j2`).
2. Añade `VN = "vN"` al enum `PromptVersion` en `app/schemas/estimation.py`.
3. Añade sus tests en `tests/prompts/test_estimation_vN.py`.

---

## Sesión 5 — Memoria conversacional y adjuntos

Además de `POST /api/v1/estimate` (que no cambia), el servicio soporta **sesiones**: el cliente refina el alcance en varios turnos, adjunta documentación y el servicio recuerda el proyecto entre llamadas.

```
POST /sessions                         → 201 {"session_id": "<uuid v4>"}
GET  /sessions/{session_id}            → 200 {session_id, message_count, max_turns, metadata}
POST /sessions/{session_id}/estimate   → 200 EstimationResponse
     multipart/form-data: transcript (20–80.000 caracteres), project_type, detail_level, output_format, attachments[] (opcional)
```

```bash
curl -X POST http://localhost:8000/sessions                     # {"session_id":"abc-123"}

curl -X POST http://localhost:8000/sessions/abc-123/estimate \
  -F transcript="Queremos estimar un CRM llamado Nimbus en React + Postgres para el equipo de ventas." \
  -F project_type=web_saas -F detail_level=medium -F output_format=phases_table \
  -F attachments=@spec.pdf

curl -X POST http://localhost:8000/sessions/abc-123/estimate \
  -F transcript="Añade un módulo de facturación con Stripe." \
  -F project_type=web_saas -F detail_level=medium -F output_format=phases_table

curl http://localhost:8000/sessions/abc-123                     # ProjectMetadata y tamaño del historial
```

Errores: `404` sesión inexistente (o la API se reinició), `415` adjunto que no es `.pdf`/`.docx`, `422` adjunto ilegible, `400` guardrail de entrada (también sobre el texto de los adjuntos), `502` el LLM no cumple el esquema tras los reintentos.

En Streamlit, la pestaña **Conversation** crea la sesión al cargar la página, envía transcript y adjuntos, muestra el `ProjectMetadata` y el tamaño del historial en el panel lateral, y tiene un botón **New conversation** que crea una sesión nueva.

### Adjuntos: camino B (extracción local)

El texto del PDF/Word se extrae **dentro del servicio** con `pypdf` y `python-docx` (`app/attachments/extractor.py`), se recorta a `MAX_ATTACHMENT_CHARS` y se añade al transcript entre separadores:

```
--- attachment: spec.pdf ---
<texto extraído>
--- end attachment ---
```

Por qué B y no A (Files API de OpenAI/Anthropic):

- **Independiente del proveedor.** El wrapper sigue siendo texto entra / texto sale, así que el fallback OpenAI → Anthropic funciona igual con adjuntos; con la Files API cada proveedor sube y referencia los archivos a su manera.
- **Los guardrails ven el contenido.** Se ejecutan sobre el transcript ya enriquecido: un documento con instrucciones ocultas se bloquea igual que si viniera en el texto.
- **Control del prompt.** Decidimos cuánto texto entra, y una página corrupta o un archivo vacío no rompen la petición.
- **Prepara el módulo 3** (chunking y RAG sobre el texto extraído).

El coste: sin OCR, un PDF escaneado no aporta texto, y se pierde la maquetación (tablas, imágenes).

### Cómo se extrae `project_metadata`: extractor LLM

`ProjectMetadata` (`project_name`, `assumed_team_size`, `mentioned_technologies`, `agreed_scope`) vive fuera del historial y se inyecta cada turno en el system prompt (`app/prompts/conversation/v1/system.j2`) como bloque `<project_metadata>`. En el primer turno el bloque solo indica que aún no hay contexto.

Tras cada estimación validada, `app/sessions/metadata_extractor.py` hace una **segunda llamada** a un modelo barato (`METADATA_EXTRACTOR_MODEL`) con el prompt `app/prompts/metadata_extraction/v1/`: recibe el transcript (con adjuntos), la estimación y la metadata previa, y devuelve un `ProjectMetadata` en JSON validado con Pydantic. Se fusiona con el anterior: los campos simples se sobrescriben solo si vienen informados y las tecnologías se acumulan sin duplicados.

Por qué un extractor LLM y no una heurística con regex: el nombre del proyecto, el tamaño del equipo o el alcance aparecen con paráfrasis muy distintas ("un CRM llamado Nimbus", "1 designer, 2 developers"), `agreed_scope` es un resumen y no un patrón, y una lista de palabras marcaría "Stripe" aunque el usuario diga "sin Stripe". El coste es marginal: 1–2 s y unos 0,0002 $ por turno con `gpt-4o-mini` (estimado). Si la llamada falla, se conserva la metadata previa y la conversación sigue. Contra: una llamada más por turno, y el extractor también puede omitir datos.

### Historial y coherencia entre turnos

- **Ventana deslizante:** `ConversationHistory` guarda como máximo `MAX_CONVERSATION_TURNS` pares user+assistant y descarta los más antiguos de dos en dos. El system prompt no forma parte del historial: se regenera cada turno desde el `ProjectMetadata` actual, así que nunca se pierde. El LLM recibe `[system] + ventana + user actual` (`ConversationHistory.to_messages_list()`).
- **Última estimación como punto de partida:** la sesión guarda la última estimación (`Session.last_estimate`) y el prompt la incluye como `<current_estimate>`, con reglas para re-estimar el proyecto entero, no bajar el total al añadir alcance, quitar solo las fases del módulo eliminado y nombrar el proyecto. Sin esto, gpt-4o-mini estimaba solo lo nuevo y el total bajaba al añadir funcionalidades. Es una mejora del prompt verificada con el LLM real, no una garantía: no hay validador que lo imponga.
- **Memoria en proceso:** `SessionStore` es un `dict` en memoria del worker, sin base de datos ni Redis. Se pierde al reiniciar y no se comparte entre workers; es intencional en esta fase.
- **Sin cachés en el camino conversacional:** cada turno depende del historial y la metadata, así que `cached` es siempre `false`.
