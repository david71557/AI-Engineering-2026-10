ESTIMATION_EXAMPLES = [
    {
        "meeting_summary": (
            "El cliente necesita una plataforma web de gestión de inventario "
            "para administrar productos, controlar existencias, registrar "
            "movimientos y consultar métricas en un dashboard."
        ),
        "estimation": """
## Estimación: Plataforma de Gestión de Inventario

### Desglose de tareas:
1. Diseño UI/UX: 40 horas
2. Backend API (CRUD inventario): 60 horas
3. Autenticación y roles: 20 horas
4. Dashboard con métricas: 30 horas
5. Testing y QA: 25 horas

**Total estimado: 175 horas**
**Equipo recomendado: 2 desarrolladores full-stack + 1 diseñador UX (part-time)**
**Duración estimada: 6-8 semanas**
"""
    },
    {
        "meeting_summary": (
            "El cliente necesita una aplicación web para reservar citas. "
            "Los usuarios podrán consultar la disponibilidad, seleccionar "
            "horarios, cancelar reservas y recibir confirmaciones por correo. "
            "Un administrador gestionará los horarios y las citas."
        ),
        "estimation": """
## Estimación: Sistema de Reservas de Citas

### Desglose de tareas:
1. Análisis de requisitos: 12 horas
2. Diseño UI/UX: 24 horas
3. Modelo de datos y base de datos: 16 horas
4. Backend API de reservas: 40 horas
5. Calendario y disponibilidad: 32 horas
6. Notificaciones por correo: 12 horas
7. Panel de administración: 24 horas
8. Pruebas y QA: 20 horas
9. Despliegue y documentación: 12 horas

**Total estimado: 192 horas**
**Coste orientativo: 9.600 € (tarifa supuesta de 50 €/hora)**
**Equipo recomendado: 1 desarrollador backend, 1 desarrollador frontend y QA parcial**
**Duración estimada: 6-8 semanas**

### Supuestos y riesgos:
- No se incluyen pagos en línea.
- Se utiliza un proveedor de correo existente.
- La sincronización con calendarios externos queda fuera del alcance.
"""
    },
    {
        "meeting_summary": (
            "El cliente solicita una plataforma interna para registrar "
            "incidencias de soporte, asignarlas a técnicos, cambiar sus "
            "estados, añadir comentarios y generar informes mensuales. "
            "Los responsables necesitan consultar métricas por técnico "
            "y por prioridad."
        ),
        "estimation": """
## Estimación: Plataforma de Gestión de Soporte

### Desglose de tareas:
1. Análisis funcional: 12 horas
2. Diseño de interfaz: 20 horas
3. Autenticación y permisos: 16 horas
4. Backend API de incidencias: 36 horas
5. Asignación de técnicos y estados: 24 horas
6. Comentarios e historial de actividad: 16 horas
7. Dashboard e informes: 24 horas
8. Pruebas y QA: 20 horas
9. Despliegue y documentación: 12 horas

**Total estimado: 180 horas**
**Coste orientativo: 9.000 € (tarifa supuesta de 50 €/hora)**
**Equipo recomendado: 1 desarrollador full-stack y QA parcial**
**Duración estimada: 5-7 semanas**

### Supuestos y riesgos:
- La aplicación será de uso interno.
- No se incluye integración con herramientas externas de ticketing.
- Los informes utilizarán los datos registrados en la plataforma.
"""
    },
    {
        "meeting_summary": (
            "El cliente quiere una tienda online para vender productos, "
            "con catálogo, categorías, buscador, carrito, pagos, gestión "
            "de pedidos y un panel administrativo para controlar el stock."
        ),
        "estimation": """
## Estimación: Tienda Online

### Desglose de tareas:
1. Análisis de requisitos: 16 horas
2. Diseño UI/UX: 32 horas
3. Catálogo y buscador de productos: 32 horas
4. Backend API y base de datos: 48 horas
5. Carrito y proceso de compra: 32 horas
6. Integración con pasarela de pago: 24 horas
7. Gestión de pedidos y stock: 24 horas
8. Panel de administración: 24 horas
9. Pruebas de seguridad y QA: 28 horas
10. Despliegue y documentación: 16 horas

**Total estimado: 276 horas**
**Coste orientativo: 13.800 € (tarifa supuesta de 50 €/hora)**
**Equipo recomendado: 2 desarrolladores full-stack y QA parcial**
**Duración estimada: 8-11 semanas**

### Supuestos y riesgos:
- Se utilizará una pasarela de pago de terceros.
- No se incluye la carga masiva inicial de productos.
- Los costes de alojamiento y las comisiones de pago no están incluidos.
"""
    },
]


def format_examples(examples: list[dict]) -> str:
    """Format estimation examples into a string suitable for injection into a system prompt."""
    parts: list[str] = []
    for i, example in enumerate(examples, start=1):
        parts.append(
            f"--- EXAMPLE {i} ---\n"
            f"Meeting Summary:\n{example['meeting_summary']}\n\n"
            f"Estimation:\n{example['estimation']}\n"
        )
    return "\n".join(parts)