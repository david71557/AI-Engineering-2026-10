import json
from dataclasses import dataclass

from app.schemas.estimation import ExampleFormat


@dataclass
class CanonicalExample:
    """A reference estimation expressed both as structured data and as Markdown.

    The structured fields (`breakdown`, totals, team) are the single source of truth
    used by the JSON and narrative formatters. `estimation_markdown` is precomputed
    so the Markdown formatter stays byte-for-byte identical to the legacy output.
    """

    title: str
    meeting_summary: str
    breakdown: list[tuple[str, int, int]]
    total_hours: int
    total_cost: int
    team: list[str]
    duration_weeks: int
    estimation_markdown: str


CANONICAL_EXAMPLES: list[CanonicalExample] = [
    CanonicalExample(
        title="Inventory Management Web Platform",
        meeting_summary=(
            "The client needs an inventory management web platform for a mid-size warehouse "
            "operation. They want real-time stock tracking, automated reorder alerts when items "
            "fall below a configurable threshold, a role-based access system (admin, warehouse "
            "manager, viewer), CSV/Excel import and export of inventory data, and a dashboard "
            "with key metrics (turnover rate, stock value, low-stock items). They already have "
            "a PostgreSQL database running in AWS that we should integrate with. No mobile app "
            "is needed — the warehouse team will use tablets with the web interface."
        ),
        breakdown=[
            ("Requirements analysis and technical design", 16, 1000),
            ("Database schema design and migrations", 12, 750),
            ("Authentication and role-based access control", 20, 1250),
            ("Product and stock CRUD API", 24, 1500),
            ("Automated reorder alert engine", 16, 1000),
            ("CSV/Excel import and export module", 14, 875),
            ("Dashboard with key metrics and charts", 20, 1250),
            ("Frontend: inventory views and search/filter", 24, 1500),
            ("Frontend: admin panel and user management", 12, 750),
            ("AWS PostgreSQL integration and deployment", 10, 625),
            ("Testing (unit, integration, E2E)", 20, 1250),
            ("Code review, QA, and bug fixing", 12, 750),
        ],
        total_hours=200,
        total_cost=12500,
        team=[
            "1 Senior Backend Developer (lead)",
            "1 Mid-level Full-Stack Developer",
            "1 QA Engineer (part-time, last 3 weeks)",
        ],
        duration_weeks=10,
        estimation_markdown="""\
## Inventory Management Web Platform

### Task Breakdown

| Task | Hours | Cost (EUR) |
|------|------:|------------|
| Requirements analysis and technical design | 16 | 1,000 |
| Database schema design and migrations | 12 | 750 |
| Authentication and role-based access control | 20 | 1,250 |
| Product and stock CRUD API | 24 | 1,500 |
| Automated reorder alert engine | 16 | 1,000 |
| CSV/Excel import and export module | 14 | 875 |
| Dashboard with key metrics and charts | 20 | 1,250 |
| Frontend: inventory views and search/filter | 24 | 1,500 |
| Frontend: admin panel and user management | 12 | 750 |
| AWS PostgreSQL integration and deployment | 10 | 625 |
| Testing (unit, integration, E2E) | 20 | 1,250 |
| Code review, QA, and bug fixing | 12 | 750 |

### Totals

- **Total hours:** 200
- **Total cost:** 12,500 EUR

### Recommended Team

- 1 Senior Backend Developer (lead)
- 1 Mid-level Full-Stack Developer
- 1 QA Engineer (part-time, last 3 weeks)

### Estimated Duration

**10 weeks** with a two-person development team.""",
    ),
    CanonicalExample(
        title="Real Estate Landing Page with CRM Integration",
        meeting_summary=(
            "A real estate agency wants a high-conversion landing page to capture leads for "
            "luxury property listings. The page should feature a hero section with a video "
            "background, a curated property gallery with filtering by price range and location, "
            "a lead capture form, and client testimonials. All leads must be pushed in real-time "
            "to their existing HubSpot CRM via the API. The design must be responsive, "
            "mobile-first, and follow their brand guidelines (they will provide a Figma file). "
            "They also want basic analytics integration with Google Tag Manager and Meta Pixel "
            "for ad campaign tracking."
        ),
        breakdown=[
            ("Requirements review and UX analysis", 8, 500),
            ("UI design adaptation from Figma to code", 16, 800),
            ("Hero section with video background", 8, 500),
            ("Property gallery with filtering", 16, 1000),
            ("Lead capture form with validation", 8, 500),
            ("HubSpot CRM API integration", 14, 875),
            ("Testimonials carousel component", 6, 375),
            ("Google Tag Manager and Meta Pixel setup", 6, 375),
            ("Responsive design and cross-browser testing", 10, 625),
            ("Performance optimization (Core Web Vitals)", 8, 500),
            ("Deployment and DNS configuration", 4, 250),
        ],
        total_hours=104,
        total_cost=6300,
        team=[
            "1 Senior Frontend Developer",
            "1 UI/UX Designer (part-time, first 2 weeks)",
        ],
        duration_weeks=5,
        estimation_markdown="""\
## Real Estate Landing Page with CRM Integration

### Task Breakdown

| Task | Hours | Cost (EUR) |
|------|------:|------------|
| Requirements review and UX analysis | 8 | 500 |
| UI design adaptation from Figma to code | 16 | 800 |
| Hero section with video background | 8 | 500 |
| Property gallery with filtering | 16 | 1,000 |
| Lead capture form with validation | 8 | 500 |
| HubSpot CRM API integration | 14 | 875 |
| Testimonials carousel component | 6 | 375 |
| Google Tag Manager and Meta Pixel setup | 6 | 375 |
| Responsive design and cross-browser testing | 10 | 625 |
| Performance optimization (Core Web Vitals) | 8 | 500 |
| Deployment and DNS configuration | 4 | 250 |

### Totals

- **Total hours:** 104
- **Total cost:** 6,300 EUR

### Recommended Team

- 1 Senior Frontend Developer
- 1 UI/UX Designer (part-time, first 2 weeks)

### Estimated Duration

**5 weeks** with a single frontend developer and part-time design support.""",
    ),
    CanonicalExample(
        title="SaaS Subscription Management Platform (MVP)",
        meeting_summary=(
            "A B2B startup is building a SaaS platform to manage software subscriptions for "
            "SMEs. Core features include: user registration and company onboarding, a dashboard "
            "showing all active subscriptions with renewal dates and monthly spend, the ability "
            "to add/edit/cancel subscriptions, integration with Stripe for payment processing "
            "and invoice generation, a notification system (email alerts 30/7/1 days before "
            "renewal), and an admin panel to manage customer accounts. They want a REST API "
            "with a React frontend. MVP scope — they plan to iterate after launch."
        ),
        breakdown=[
            ("Architecture design and project setup", 12, 750),
            ("User registration and authentication (JWT)", 16, 1000),
            ("Company onboarding flow", 12, 750),
            ("Subscription CRUD API and data model", 20, 1250),
            ("Dashboard: active subscriptions and spend analytics", 24, 1500),
            ("Stripe integration: payments and invoices", 28, 1750),
            ("Email notification system (renewal reminders)", 16, 1000),
            ("Admin panel for customer management", 20, 1250),
            ("React frontend: views, forms, and routing", 40, 2500),
            ("API documentation (OpenAPI/Swagger)", 6, 375),
            ("Testing (unit, integration, Stripe sandbox)", 24, 1500),
            ("Deployment, CI/CD pipeline, and staging env", 16, 1000),
            ("Security audit and hardening", 10, 625),
        ],
        total_hours=244,
        total_cost=15250,
        team=[
            "1 Senior Full-Stack Developer (lead)",
            "1 Mid-level Backend Developer",
            "1 Mid-level Frontend Developer",
            "1 QA Engineer (part-time, last 4 weeks)",
        ],
        duration_weeks=12,
        estimation_markdown="""\
## SaaS Subscription Management Platform (MVP)

### Task Breakdown

| Task | Hours | Cost (EUR) |
|------|------:|------------|
| Architecture design and project setup | 12 | 750 |
| User registration and authentication (JWT) | 16 | 1,000 |
| Company onboarding flow | 12 | 750 |
| Subscription CRUD API and data model | 20 | 1,250 |
| Dashboard: active subscriptions and spend analytics | 24 | 1,500 |
| Stripe integration: payments and invoices | 28 | 1,750 |
| Email notification system (renewal reminders) | 16 | 1,000 |
| Admin panel for customer management | 20 | 1,250 |
| React frontend: views, forms, and routing | 40 | 2,500 |
| API documentation (OpenAPI/Swagger) | 6 | 375 |
| Testing (unit, integration, Stripe sandbox) | 24 | 1,500 |
| Deployment, CI/CD pipeline, and staging env | 16 | 1,000 |
| Security audit and hardening | 10 | 625 |

### Totals

- **Total hours:** 244
- **Total cost:** 15,250 EUR

### Recommended Team

- 1 Senior Full-Stack Developer (lead)
- 1 Mid-level Backend Developer
- 1 Mid-level Frontend Developer
- 1 QA Engineer (part-time, last 4 weeks)

### Estimated Duration

**12 weeks** with a three-person core development team.""",
    ),
    CanonicalExample(
        title="Customer Support AI Chatbot",
        meeting_summary=(
            "A SaaS company wants an AI-powered support chatbot to deflect common questions "
            "from their human support team. The bot must ingest their existing knowledge base "
            "(FAQs, PDFs, and help-center articles), answer in natural language using a RAG "
            "pipeline, keep conversation context across turns, and seamlessly escalate to a "
            "human agent when confidence is low or the user explicitly asks. They want an "
            "embeddable chat widget for their web app and a lightweight admin panel where the "
            "support team can review conversations, mark answers as good or bad, and re-ingest "
            "documents. Analytics on deflection rate, satisfaction, and top intents are required."
        ),
        breakdown=[
            ("Requirements analysis and architecture", 12, 750),
            ("Knowledge base ingestion pipeline (PDF, FAQ, HTML)", 20, 1250),
            ("Embeddings generation and vector store setup", 16, 1000),
            ("RAG retrieval module with reranking", 18, 1125),
            ("LLM integration (OpenAI/Anthropic) with prompts", 16, 1000),
            ("Conversation history and context management", 14, 875),
            ("Embeddable chat widget (React)", 24, 1500),
            ("Human escalation flow", 10, 625),
            ("Analytics dashboard (deflection, CSAT, intents)", 14, 875),
            ("Admin panel for KB management and review", 12, 750),
            ("Evaluation harness and integration tests", 14, 875),
            ("Deployment and observability", 10, 625),
        ],
        total_hours=180,
        total_cost=11250,
        team=[
            "1 Senior Backend Developer (RAG lead)",
            "1 Mid-level Frontend Developer",
            "1 ML Engineer (part-time, first 3 weeks)",
        ],
        duration_weeks=8,
        estimation_markdown="""\
## Customer Support AI Chatbot

### Task Breakdown

| Task | Hours | Cost (EUR) |
|------|------:|------------|
| Requirements analysis and architecture | 12 | 750 |
| Knowledge base ingestion pipeline (PDF, FAQ, HTML) | 20 | 1,250 |
| Embeddings generation and vector store setup | 16 | 1,000 |
| RAG retrieval module with reranking | 18 | 1,125 |
| LLM integration (OpenAI/Anthropic) with prompts | 16 | 1,000 |
| Conversation history and context management | 14 | 875 |
| Embeddable chat widget (React) | 24 | 1,500 |
| Human escalation flow | 10 | 625 |
| Analytics dashboard (deflection, CSAT, intents) | 14 | 875 |
| Admin panel for KB management and review | 12 | 750 |
| Evaluation harness and integration tests | 14 | 875 |
| Deployment and observability | 10 | 625 |

### Totals

- **Total hours:** 180
- **Total cost:** 11,250 EUR

### Recommended Team

- 1 Senior Backend Developer (RAG lead)
- 1 Mid-level Frontend Developer
- 1 ML Engineer (part-time, first 3 weeks)

### Estimated Duration

**8 weeks** with a two-person core team and ML support.""",
    ),
    CanonicalExample(
        title="Multi-tenant SaaS Billing Platform",
        meeting_summary=(
            "A B2B SaaS company needs a billing platform layered on top of Stripe to manage "
            "subscriptions for multiple tenants on a shared infrastructure. Each tenant must be "
            "fully isolated at the data level (row-level security) and have its own admin users, "
            "plans and invoices. The platform handles the full subscription lifecycle (trial, "
            "active, paused, canceled), generates branded PDF invoices, and runs a dunning "
            "workflow on failed payments. They also need a customer-facing portal where end "
            "users can update payment methods and download invoices, and GDPR-compliant data "
            "export and deletion endpoints. All Stripe webhooks must be signed and audited."
        ),
        breakdown=[
            ("Architecture design (multi-tenant strategy)", 16, 1000),
            ("Database schema with tenant_id and RLS", 16, 1000),
            ("Authentication and tenant onboarding", 20, 1250),
            ("Stripe integration: subscriptions and webhooks", 28, 1750),
            ("Subscription lifecycle state machine", 20, 1250),
            ("Invoice generation (PDF + email)", 16, 1000),
            ("Dunning workflow for failed payments", 14, 875),
            ("Admin panel with tenant impersonation", 24, 1500),
            ("Customer-facing billing portal", 24, 1500),
            ("GDPR data export and deletion", 12, 750),
            ("Webhook signing and audit log", 10, 625),
            ("Testing (Stripe test mode, integration, E2E)", 24, 1500),
            ("Deployment, monitoring and runbooks", 16, 1000),
        ],
        total_hours=240,
        total_cost=15000,
        team=[
            "1 Senior Backend Developer (lead)",
            "1 Mid-level Full-Stack Developer",
            "1 DevOps Engineer (part-time)",
            "1 QA Engineer (part-time, last 4 weeks)",
        ],
        duration_weeks=11,
        estimation_markdown="""\
## Multi-tenant SaaS Billing Platform

### Task Breakdown

| Task | Hours | Cost (EUR) |
|------|------:|------------|
| Architecture design (multi-tenant strategy) | 16 | 1,000 |
| Database schema with tenant_id and RLS | 16 | 1,000 |
| Authentication and tenant onboarding | 20 | 1,250 |
| Stripe integration: subscriptions and webhooks | 28 | 1,750 |
| Subscription lifecycle state machine | 20 | 1,250 |
| Invoice generation (PDF + email) | 16 | 1,000 |
| Dunning workflow for failed payments | 14 | 875 |
| Admin panel with tenant impersonation | 24 | 1,500 |
| Customer-facing billing portal | 24 | 1,500 |
| GDPR data export and deletion | 12 | 750 |
| Webhook signing and audit log | 10 | 625 |
| Testing (Stripe test mode, integration, E2E) | 24 | 1,500 |
| Deployment, monitoring and runbooks | 16 | 1,000 |

### Totals

- **Total hours:** 240
- **Total cost:** 15,000 EUR

### Recommended Team

- 1 Senior Backend Developer (lead)
- 1 Mid-level Full-Stack Developer
- 1 DevOps Engineer (part-time)
- 1 QA Engineer (part-time, last 4 weeks)

### Estimated Duration

**11 weeks** with a three-person core team and DevOps support.""",
    ),
]


def select_examples(n: int) -> list[CanonicalExample]:
    """Return the first n canonical examples, capped at the available pool."""
    return CANONICAL_EXAMPLES[: max(0, min(n, len(CANONICAL_EXAMPLES)))]


def format_examples_for_prompt(
    examples: list[CanonicalExample],
    fmt: ExampleFormat = "markdown",
) -> str:
    """Render a list of canonical examples in the requested format for prompt injection."""
    if not examples:
        return ""
    if fmt == "markdown":
        return _format_markdown(examples)
    if fmt == "json":
        return _format_json(examples)
    if fmt == "narrative":
        return _format_narrative(examples)
    raise ValueError(f"Unknown example format: {fmt}")


def _format_markdown(examples: list[CanonicalExample]) -> str:
    parts: list[str] = []
    for i, ex in enumerate(examples, start=1):
        parts.append(
            f"--- EXAMPLE {i} ---\n"
            f"Meeting Summary:\n{ex.meeting_summary}\n\n"
            f"Estimation:\n{ex.estimation_markdown}\n"
        )
    return "\n".join(parts)


def _format_json(examples: list[CanonicalExample]) -> str:
    payload = [
        {
            "meeting_summary": ex.meeting_summary,
            "title": ex.title,
            "breakdown": [
                {"task": task, "hours": hours, "cost_eur": cost}
                for task, hours, cost in ex.breakdown
            ],
            "totals": {"hours": ex.total_hours, "cost_eur": ex.total_cost},
            "team": ex.team,
            "duration_weeks": ex.duration_weeks,
        }
        for ex in examples
    ]
    return "Reference examples (JSON):\n" + json.dumps(payload, indent=2, ensure_ascii=False)


def _format_narrative(examples: list[CanonicalExample]) -> str:
    parts: list[str] = []
    for i, ex in enumerate(examples, start=1):
        items = "; ".join(f"{task} ({hours}h)" for task, hours, _ in ex.breakdown)
        parts.append(
            f"In a previous engagement (#{i}), the client requested: {ex.meeting_summary} "
            f"We proposed '{ex.title}', estimating {ex.total_hours} hours at "
            f"{ex.total_cost:,} EUR over {ex.duration_weeks} weeks with team "
            f"{', '.join(ex.team)}. Major work items included: {items}."
        )
    return "\n\n".join(parts)
