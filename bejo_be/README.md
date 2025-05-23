# ISA-95 Based Agentic RAG System

## ISA-95 Knowledge Hierarchy

Sistem ini menggunakan struktur ISA-95 untuk mengorganisir knowledge:

```
Level 4 (Management) ←→ Akses ke Level 1,2,3,4
Level 3 (Planning)   ←→ Akses ke Level 1,2,3
Level 2 (Supervisory)←→ Akses ke Level 1,2
Level 1 (Field)      ←→ Akses ke Level 1 saja
```

### Level Descriptions

| Level | Name                       | Focus Area                                                           | Knowledge Access |
| ----- | -------------------------- | -------------------------------------------------------------------- | ---------------- |
| 1     | **Field & Control System** | Real-time control, sensors, actuators, basic automation              | Level 1 only     |
| 2     | **Supervisory**            | SCADA, HMI, batch control, recipe management                         | Level 1,2        |
| 3     | **Planning**               | Production scheduling, resource allocation, workflow management      | Level 1,2,3      |
| 4     | **Management**             | Business planning, KPIs, enterprise integration, strategic decisions | Level 1,2,3,4    |

## API Usage Examples

### 1. Basic Chat with ISA-95 Context

```python
# Chat untuk pertanyaan Field Level
POST /chat/stream/session123
{
    "message": "How to configure PLC for temperature control?",
    "category": 1,  # Field & Control Level
    "agent_id": "field_engineer"
}

# Response akan menggunakan Level 1 knowledge saja
```

### 2. Cross-Level Analysis

```python
# Pertanyaan yang membutuhkan analisis lintas level
POST /chat/stream/session456
{
    "message": "How does equipment failure at field level impact production planning?",
    "category": 3,  # Planning Level - bisa akses Level 1,2,3
    "agent_id": "production_analyst"
}

# Agent akan menggunakan isa95_level_analysis tool
```

### 3. Knowledge Embedding Berdasarkan Level

```python
# Embed dokumen ke Level 2 (Supervisory)
POST /knowledge/embed
{
    "file_path": "/docs/scada_manual.pdf",
    "category": 2,  # Akan disimpan di bejo-knowledge-level-2
    "chunk_size": 1000
}

# Embed dokumen ke Level 4 (Management)
POST /knowledge/embed
{
    "file_path": "/docs/kpi_dashboard_spec.pdf",
    "category": 4,  # Akan disimpan di bejo-knowledge-level-4
    "chunk_size": 800
}
```

### 4. Integration Points Analysis

```python
POST /chat/stream/session789
{
    "message": "Analyze integration points between SCADA and MES systems",
    "category": 3,  # Planning level untuk akses MES knowledge
    "agent_id": "integration_specialist"
}

# Agent akan menggunakan integration_points_analysis tool
# untuk menganalisis integrasi antara Level 2 (SCADA) dan Level 3 (MES)
```

## Agent Reasoning Examples

### Example 1: Field Level Question

**User Question**: "What are the best practices for PLC programming?"

**Agent Reasoning**:

```
Thought: This is a field-level technical question about PLC programming.
I should search at ISA-95 Level 1 (Field & Control System) since PLCs are basic control devices.

Action: document_retrieval
Action Input: {"query": "PLC programming best practices", "isa_level": 1, "k": 5, "strategy": "basic"}

Observation: Found 3 relevant documents from ISA-95 Field & Control System level...

Final Answer: Based on the field-level documentation, here are the PLC programming best practices:
1. Use structured programming with function blocks
2. Implement proper error handling and diagnostics
3. Follow naming conventions for tags and variables
...
```

### Example 2: Cross-Level Question

**User Question**: "How do production delays at field level affect management KPIs?"

**Agent Reasoning**:

```
Thought: This question involves understanding the impact from field level (Level 1)
up to management level (Level 4). I need cross-level analysis.

Action: isa95_level_analysis
Action Input: {"query": "production delays impact", "start_level": 1, "end_level": 4}

Observation: ISA-95 Cross-Level Analysis shows:
- Level 1: Equipment downtime, sensor failures
- Level 2: Production line monitoring, alarm management
- Level 3: Schedule adjustments, resource reallocation
- Level 4: OEE impact, delivery performance metrics

Final Answer: Production delays cascade through ISA-95 levels as follows:
Field Level (1): Equipment issues cause immediate production stops
Supervisory (2): SCADA systems trigger alarms and notifications
Planning (3): MES systems adjust schedules and resource allocation
Management (4): KPIs show reduced OEE, delayed deliveries, and cost impacts
```

### Example 3: Integration Question

**User Question**: "What data needs to flow from SCADA to ERP system?"

**Agent Reasoning**:

```
Thought: This is about integration between Level 2 (SCADA/Supervisory) and
Level 4 (ERP/Management). I should use integration points analysis.

Action: integration_points_analysis
Action Input: {"domain": "production", "source_level": 2, "target_level": 4}

Observation: Integration analysis shows data flow requirements:
- Production counts and rates
- Equipment status and availability
- Quality parameters
- Energy consumption data

Final Answer: The integration between SCADA (Level 2) and ERP (Level 4) requires:
[Detailed integration specifications based on analysis]
```

## Specialized Tools Usage

### 1. ISA-95 Level Analysis Tool

```python
# Menganalisis dampak lintas level
{
    "tool": "isa95_level_analysis",
    "query": "maintenance scheduling impact",
    "start_level": 1,  # Field level
    "end_level": 4     # Management level
}
```

### 2. Integration Points Tool

```python
# Analisis integrasi antara level
{
    "tool": "integration_points_analysis",
    "domain": "quality",
    "source_level": 1,  # Field sensors
    "target_level": 3   # Quality management system
}
```

### 3. Compliance Check Tool

```python
# Validasi compliance ISA-95
{
    "tool": "isa95_compliance_check",
    "topic": "batch manufacturing system",
    "compliance_standard": "ISA-95",
    "focus_levels": [2, 3]  # Supervisory dan Planning
}
```

## Migration dari Sistem Lama

### Mapping Level Lama ke ISA-95

```python
# Sistem Lama → ISA-95 Mapping
OLD_SYSTEM = {
    1: [1,2,3,4],  # Level 1 akses semua → Management (Level 4)
    2: [2,3,4],    # Level 2 akses 2,3,4 → Planning (Level 3)
    3: [3,4],      # Level 3 akses 3,4 → Supervisory (Level 2)
    4: [4]         # Level 4 akses 4 → Field (Level 1)
}

ISA95_SYSTEM = {
    1: [1],        # Field → Level 1 saja
    2: [1,2],      # Supervisory → Level 1,2
    3: [1,2,3],    # Planning → Level 1,2,3
    4: [1,2,3,4]   # Management → Semua level
}
```

### Update API Calls

```python
# Lama
POST /ask/thread123
{
    "input": "How to configure PLC?",
    "category": 4  # Level 4 = paling spesifik
}

# Baru (ISA-95)
POST /chat/stream/thread123
{
    "message": "How to configure PLC?",
    "category": 1,  # Level 1 = Field & Control
    "agent_id": "field_engineer"
}
```

## Best Practices

### 1. Choosing the Right ISA-95 Level

| Question Type                  | Recommended Level | Reasoning                       |
| ------------------------------ | ----------------- | ------------------------------- |
| "How to program PLC?"          | Level 1           | Field devices and basic control |
| "SCADA alarm configuration?"   | Level 2           | Supervisory control systems     |
| "Production scheduling rules?" | Level 3           | Planning and optimization       |
| "KPI dashboard requirements?"  | Level 4           | Management and business metrics |

### 2. Strategy Selection

- **basic**: Single collection, fastest response
- **hierarchical**: Weighted multi-level search, best for cross-level understanding
- **comprehensive**: All available levels, most thorough but slower

### 3. Agent Specialization

```python
# Specialized agents untuk domain tertentu
agents = {
    "field_engineer": {"focus_levels": [1], "expertise": "control_systems"},
    "process_operator": {"focus_levels": [1,2], "expertise": "operations"},
    "production_planner": {"focus_levels": [2,3], "expertise": "scheduling"},
    "plant_manager": {"focus_levels": [3,4], "expertise": "management"}
}
```

## Monitoring & Debugging

### Trace Agent Reasoning

```python
# Enable detailed logging untuk debugging
POST /chat/stream/session123?debug=true
{
    "message": "Analyze equipment efficiency",
    "category": 3
}

# Response includes intermediate steps:
# 1. Tool selection reasoning
# 2. ISA-95 level justification
# 3. Search results from each level
# 4. Cross-level impact analysis
```

### Performance Metrics

```python
GET /health/detailed

# Response includes:
{
    "isa95_levels": {
        "level_1": {"collections": 1, "documents": 1500},
        "level_2": {"collections": 1, "documents": 800},
        "level_3": {"collections": 1, "documents": 600},
        "level_4": {"collections": 1, "documents": 300}
    },
    "agent_performance": {
        "avg_reasoning_steps": 3.2,
        "tool_usage_distribution": {
            "document_retrieval": 45,
            "isa95_level_analysis": 25,
            "integration_points_analysis": 20,
            "compliance_check": 10
        }
    }
}
```
