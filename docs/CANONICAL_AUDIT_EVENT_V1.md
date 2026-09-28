# Canonical DCL Audit Event v1.0 — инвентаризация контракта

**Статус: PROPOSAL / DRAFT.** Это черновик-предложение по инвентаризации уже существующих в репозитории (и в установленном пакете `dcl-core`) описаний. Файл **не** является замороженной спецификацией, **не** вводит новую JSON Schema, **не** меняет production-код, тесты, серверы, базы данных и платёжную логику. Сам по себе этот документ **не фиксирует** схему.

**Контекст даты:** 2026-09-28.

**Правило чтения:** если ни один источник не фиксирует пункт, стоит **UNSPECIFIED**. Пробелы не заполняются правдоподобными инженерными догадками. Цитаты — то, что источник действительно говорит.

---

## 1. Статус

| | |
| --- | --- |
| Тип документа | PROPOSAL / DRAFT |
| Дата | 2026-09-28 |
| Меняет production? | Нет |
| Замораживает новую схему? | Нет |
| Отдельная frozen JSON Schema Canonical DCL Audit Event v1.0 в репозитории | **NOT FOUND IN REPO** |

В репозитории есть локальный Python-builder `create_audit_event` и архитектурный чеклист v0.1. Они **называют** артефакт «Canonical DCL Audit Event v1.0», но ни один найденный источник не помечает себя как единственный authoritative JSON Schema / RFC / замороженный контракт этого события.

---

## 2. Инвентаризация источников

Для каждого источника: путь, что он фактически определяет, и **каким артефактом** он себя считает.

### 2.1 Источники, которые претендуют на Canonical DCL Audit Event v1.0

| Путь | Символы | Что фактически определяет | Заявляемый артефакт |
| --- | --- | --- | --- |
| `agent_control/canonical_audit.py` | `CANONICAL_EVENT_TYPE`, `CANONICAL_SCHEMA_VERSION`, `create_audit_event` | Python-builder `dict`. Константы `event_type = "dcl.audit.evaluated"`, `schema_version = "1.0"`. Проверки `trace_id`, `verdict ∈ {COMMIT, NO_COMMIT}`, непустые `policy_id` / `policy_version`, отличие `event_id` от переданных `payment_id` / `tx_hash` / `receipt_id`. Опциональные ключи добавляются только если аргумент не `None`. | Модуль называет себя *«Canonical DCL Audit Event v1.0 builder»* и *«local builder for the existing Canonical DCL Audit Event v1.0 contract»*. Одновременно пишет, что инспекция репозитория **не нашла** in-tree `create_audit_event` (историческая заметка внутри самого модуля) и что модуль *«does not invent a second competing schema»*. |
| `docs/agent-control-architecture-v0.1.md` §9 | чеклист полей | Повторяет чеклист builder: `event_type`, `schema_version`, required `trace_id`, distinct `event_id`, `policy_id` + `policy_version`, contextual `agent_id`, optional `payment` / `proof` / `integration` / `metadata`. Говорит, когда событие **не** эмитируется. | Называет артефакт **Canonical DCL Audit Event v1.0**. Явно отделяет его от `ChainState`. Архитектура v0.1 заявляет, что **не меняет** Canonical DCL Audit Event v1.0. |
| `agent_control/orchestrator.py` | `AgentControlOrchestrator.handle`, `audit_event_builder` | Единственный in-repo **вызов** builder после DCL-вердикта. Заполняет конкретный набор kwargs; `policy_version` может стать `"unknown"`; всегда передаёт `metadata={...}`. | Не определяет контракт. Пишет `audit_event` в `ControlFlowResult` «only after a DCL verdict». |
| `agent_control/__init__.py` | реэкспорт | Реэкспортирует `CANONICAL_EVENT_TYPE`, `CANONICAL_SCHEMA_VERSION`, `create_audit_event`. | Пакет заявляет, что **не заменяет** Canonical DCL Audit Event v1.0. |
| `test_agent_control.py` | `test_dcl_no_commit_does_not_execute_but_emits_canonical_event`, `test_create_audit_event_rejects_colliding_event_id`, `test_trace_id_propagates_through_control_flow_and_canonical_event` | Assert'ы на поля, которые возвращает builder / orchestrator. | Тесты поведения v0.1, **не** JSON Schema. |

### 2.2 Источники, которые явно говорят «это НЕ Canonical DCL Audit Event»

| Путь | Символы | Что фактически определяет | Артефакт |
| --- | --- | --- | --- |
| `agent_control/policy.py` | `LocalBlockRecord` | Локальная запись блока: `record_type = "local.policy.blocked"`, `trace_id`, `agent_id`, `action_type`, `reason`, `rule`, `policy_verdict`. | **Не** Canonical DCL Audit Event v1.0. Docstring: *MUST NOT be passed through `create_audit_event`*. |
| `docs/agent-control-architecture-v0.1.md` §9, Inspection notes | таблица поверхностей | `dcl_core.ChainState.append` — «Tamper-evident row; **not** Canonical Audit Event v1.0». | ChainState ≠ canonical audit event. |
| `agent_control/canonical_audit.py` (модульный docstring) | — | Production evaluation эмитит *другой* артефакт: строка ChainState. | ChainState ≠ canonical audit event. |
| `agent_control/dcl.py` | `EvaluatePolicyDCLGuard`, `_serialize_for_evaluate_policy` | Адаптер `evaluate_policy`; `json.dumps(..., sort_keys=True)` сериализует **вход** в движок, не audit event. *Does not append to ChainState*. | DCL evaluation adapter, не audit event. |

### 2.3 Production: таблица `chain` (не `dcl_audit_events`)

| Путь | Что фактически определяет | Артефакт |
| --- | --- | --- |
| **NOT FOUND IN REPO:** идентификатор `dcl_audit_events` | Поиск по всему дереву (`*.py`, `*.sql`, `*.md`, JSON, YAML) — **ноль совпадений**. | Таблица / модуль `dcl_audit_events` **не найдена**. Колонки **не выдуманы**. |
| Установленный пакет `dcl-core` (`dcl_core.chain.ChainState`), зависимость `requirements.txt`: `dcl-core @ git+https://github.com/Fronesis-Labs/dcl-core.git@v0.1.1`. Исходник **не** лежит в git-дереве `dcl-webhook`. Наблюдаемый `__version__` пакета: `"0.1.0"`. | `CREATE TABLE chain (...)`: `idx`, `tx_hash`, `prev_hash`, `verdict`, `input_hash`, `policy_hash`, `agent_id`, `reason`, `confidence`, `task_type`, `timestamp`, `drift_context`. Hash: `sha256` от `"\|".join(...)` в `_content_for_hash`. | **ChainState row**. Архитектура и builder называют его другим артефактом. |
| `webhook_server.py` | `_chain.append(...)`; `GET /audit/{tx_hash}` и `/deep` читают `get_by_tx`; `GET /chain/export`. HTTP `EvaluateResponse` — транспортный ответ, не canonical event. | Production wrap вокруг ChainState + HTTP. |
| `mcp_server.py` | То же `ChainState.append` / `get_by_tx`; модели `AuditResult` / `AuditDeepResult`. | MCP wrap вокруг ChainState. |
| `bazaar_server.py` | `ChainState.append`; JSON Schema `_EVALUATE_OUTPUT_SCHEMA` для HTTP EvaluateResponse. | Bazaar wrap вокруг ChainState + HTTP schema **ответа evaluate**, не canonical event. |
| `sentinel_audit.py` | `chain.append(...)` с `verdict` `PASS`/`FAIL` (не `COMMIT`/`NO_COMMIT`). | Sentinel → ChainState. |
| `payment_log.sql`, `payment_logger.py` | Sibling-таблица `chain_payments` (`tx_hash`, `route`, `amount_usdc`, `network`, `logged_at`). Комментарий: **не** трогать таблицу `chain`. | Платёжный лог по `tx_hash`, не audit event v1.0. |
| `README.md` | Пример JSON ответа evaluate; `GET /audit/{tx_hash}` описан как post-action audit **по `tx_hash`**. | Документация Trust Oracle / ChainState / HTTP, не canonical event builder. |

### 2.4 Прочие поверхности (сравнение, не canonical event)

| Путь | Что определяет | Артефакт |
| --- | --- | --- |
| `telemetry.py` | `DecisionEvent` (в т.ч. `session_id` как SHA-256 UUID сессии, `policy_id` как SHA-256 path+content). | Телеметрия AI Decision Behavior Graph. **Не** Canonical DCL Audit Event. Поля `session_id_fingerprint` нет. |
| `sentinel_db.py` | Таблицы `skills`, `sentinel_events` (`event_type`, `scan_type`, `verdict`, …). | Sentinel persistence. `event_type` здесь — не `dcl.audit.evaluated`. |
| `agent_control/context.py` | `ControlContext`: required `trace_id`, `agent_id`; `policy_id` default `"default"`; `policy_version` optional. | Контрольный контекст предложения, не audit event. |
| `agent_control/dcl.py` | `DCLEvaluation`: `verdict`, `reason`, `confidence`, `policy_id`, `policy_version`, `tx_hash`, `raw`. | Результат оценки DCL, вход к builder, не само событие. |
| `.well-known/agent.json`, `server.json` | MCP/agent tool metadata; `dcl_audit_decode` — decode **chain** по `tx_hash`. | Реестр инструментов, не schema события. |
| `bazaar_server.py` `GET /` | Ссылается на `static/index.html`. Каталог `static/` в дереве репозитория **не найден**. | Landing page, не Transparency Board. |

### 2.5 Искали и **не нашли** в репозитории

| Запрос | Результат |
| --- | --- |
| `dcl_audit_events` | **NOT FOUND IN REPO** |
| `session_id_fingerprint` | **NOT FOUND IN REPO** |
| `Transparency Board` / `transparency_board` / `TransparencyBoard` / `transparency` (как продукт/UI) | **NOT FOUND IN REPO** (слово `board` встречается только в BIP39-списке `dcl_crypto.py`) |
| Frozen JSON Schema Canonical DCL Audit Event v1.0 | **NOT FOUND IN REPO** |
| TypeScript-файлы / in-repo `@fronesis-labs/dcl-sdk` | **NOT FOUND IN REPO** (README ссылается на внешний пакет) |
| `create_audit_event` вне `agent_control/` | Только `agent_control/canonical_audit.py` (плюс импорт/тесты/архитектура) |

---

## 3. Сравнительная матрица поверхностей

Один ли это артефакт? **Нет** — источники, которые вообще высказываются, разделяют их.

| | Production `dcl_audit_events` | ChainState (`dcl_core.chain`, таблица `chain`) | `create_audit_event` (`agent_control/canonical_audit.py`) | Transparency Board |
| --- | --- | --- | --- | --- |
| Существует в репо? | **NOT FOUND IN REPO** | Да, как зависимость + вызовы. DDL в установленном `dcl_core/chain.py`, не в git-дереве webhook. | Да | **NOT FOUND IN REPO** |
| Имя типа события | — | Нет поля `event_type`. Строка цепочки. | `event_type = "dcl.audit.evaluated"` | — |
| Версия схемы | — | Нет `schema_version` у строки. Пакет `__version__` — версия библиотеки, не события. | `schema_version = "1.0"` | — |
| Поля, которые **этот** источник реально несёт / отображает | неизвестны (таблицы нет) | `index`, `tx_hash`, `prev_hash`, `verdict`, `input_hash`, `policy_hash`, `agent_id`, `reason`, `confidence`, `task_type`, `timestamp` (float), `drift_context` | Всегда: `event_type`, `schema_version`, `event_id`, `trace_id`, `agent_id`, `policy_id`, `policy_version`, `verdict`, `timestamp` (ISO-строка). Условно: `reason`, `confidence`, `payment_id`, `tx_hash`, `receipt_id`, `payment`, `proof`, `integration`, `metadata` | неизвестны (реализации нет) |
| Тот же артефакт, что Canonical DCL Audit Event v1.0? | Нечего сравнивать | Источники явно говорят **нет** | Builder **заявляет**, что строит именно его | Нечего сравнивать |

Кратко по связанным HTTP/MCP decode-поверхностям (это **чтение ChainState**, не canonical event):

- `webhook_server.audit_decode` / `mcp_server.dcl_audit_decode` отдают из `get_by_tx`: `tx_hash`, `agent_id`, `verdict`, `reason`, `confidence`, `task_type`, `timestamp`, `chain_index` (= `entry["index"]`), `prev_hash`, плюс вычисленные `chain_integrity`, `seal_text`, `verify_url`.
- Deep-вариант добавляет `tampered_at_index`, `tamper_reason`, `drift_context`.
- Ни один decode-путь не читает и не пишет `event_id`, `trace_id`, `event_type`, `schema_version`, `payment`, `proof`, `integration`, `metadata` canonical event.

---

## 4. Поля: карточки

Для каждого имени: статус контракта честный. «Builder» = поведение `create_audit_event`. «Архитектура» = `docs/agent-control-architecture-v0.1.md`. Колонка «каноническая сериализация» относится к **Canonical DCL Audit Event**, если не сказано иное. Для события она везде **UNSPECIFIED** (см. §5): в репозитории нет определённого byte layout события.

### 4.1 `event_id`

| Колонка | Значение |
| --- | --- |
| required / optional | Builder: параметр `event_id: str \| None = None`; если `None`, подставляется `str(uuid.uuid4())`. В выходном `dict` ключ **всегда есть**. Архитектура не говорит «required», говорит «distinct from `payment_id` / `tx_hash` / `receipt_id`». Контракт required/optional сверх этого: **UNSPECIFIED**. |
| type | Builder-аннотация: `str \| None` на входе; в `dict` всегда `str`. Контрактный тип: **UNSPECIFIED** (нет JSON Schema). |
| semantic meaning | Builder/архитектура: должен отличаться от `payment_id` / `tx_hash` / `receipt_id`. Иных семантики («UUID v4», «стабильный id», «идемпотентный ключ») источники **не фиксируют**. |
| producer | `create_audit_event` (caller или `uuid.uuid4`). Orchestrator **не** передаёт `event_id`. ChainState **не** хранит `event_id`. |
| authoritative? | **UNSPECIFIED**. Никто не назван authoritative источником именно `event_id`. |
| participates in canonical serialization? | Canonical event: **UNSPECIFIED** (сериализация события не определена). ChainState: поле отсутствует. |
| participates in event identity/hash? | Distinctness-проверка — не hash. Hash canonical event: **UNSPECIFIED**. ChainState `tx_hash` не включает `event_id`. |

Цитата (builder): `event_id distinct from payment_id / tx_hash / receipt_id`.

### 4.2 `trace_id`

| Колонка | Значение |
| --- | --- |
| required / optional | Builder: required (`if not trace_id: raise ValueError`). Архитектура: *«`trace_id` required (same id as the proposal)»*. `ControlContext`: required. |
| type | Python `str`. Контрактный JSON-тип: **UNSPECIFIED**. |
| semantic meaning | Архитектура/context: тот же id, что у proposal; должен propagate end-to-end. Формат (UUID vs opaque string): **UNSPECIFIED**. |
| producer | `Agent.propose(..., trace_id=...)` → `ControlContext.trace_id` → orchestrator передаёт в builder. Production servers **не** пишут `trace_id` в ChainState. |
| authoritative? | **UNSPECIFIED** для контракта события. Архитектура требует тот же id, что у proposal. |
| participates in canonical serialization? | Canonical event: **UNSPECIFIED**. ChainState: поле отсутствует. |
| participates in event identity/hash? | Canonical event hash: **UNSPECIFIED**. |

### 4.3 `session_id_fingerprint`

| Колонка | Значение |
| --- | --- |
| required / optional | **UNSPECIFIED**. Имя **не встречается** ни в одном файле репозитория. |
| type | **UNSPECIFIED** |
| semantic meaning | **UNSPECIFIED**. Рядом есть **другой** артефакт: `telemetry.DecisionEvent.session_id` (*«SHA-256 UUID сессии»*) — это не это поле и не canonical event. |
| producer | **UNSPECIFIED** |
| authoritative? | **UNSPECIFIED** |
| participates in canonical serialization? | **UNSPECIFIED** |
| participates in event identity/hash? | **UNSPECIFIED** |

### 4.4 `agent_id`

| Колонка | Значение |
| --- | --- |
| required / optional | Builder: параметр без default (вызвать без него нельзя), **пустая строка не проверяется**. `ControlContext`: required, пустая запрещена. Архитектурный чеклист §9 **не** помечает `agent_id` как required, говорит что identity contextual. Production evaluate: `Optional[str] = "unknown"`. Контракт события: **конфликт источников, см. §6**. |
| type | Python `str`. JSON Schema события: **UNSPECIFIED**. |
| semantic meaning | Архитектура: *«contextual identity, not implicit verification (ERC-8004 / identity-guard remain separate)»*. Builder: *«`agent_id` is contextual and is NOT implicitly verified»*. |
| producer | Orchestrator: `context.agent_id`. ChainState.append: `agent_id` с HTTP/MCP request (часто default `"unknown"`). |
| authoritative? | Архитектура: ERC-8004 / identity-guard — отдельные authoritative компоненты для identity. Для поля canonical event: **UNSPECIFIED**. |
| participates in canonical serialization? | Canonical event: **UNSPECIFIED**. ChainState: **да**, входит в `_content_for_hash` (это hash **ChainState**, не audit-event). |
| participates in event identity/hash? | Canonical event: **UNSPECIFIED**. ChainState: да, в `tx_hash`. |

### 4.5 `policy_id`

| Колонка | Значение |
| --- | --- |
| required / optional | Builder: required, непустой. Архитектура: *«`policy_id` + `policy_version` identify the policy actually applied»* — не использует слова required/optional. ChainState **не имеет** колонки `policy_id`; есть `policy_hash`. Telemetry `policy_id` — SHA-256 path+content, другой артефакт. |
| type | Python `str`. Контракт: **UNSPECIFIED**. |
| semantic meaning | Архитектура/builder: идентифицирует применённую политику (вместе с `policy_version`). Формат (имя builtin vs hash): **UNSPECIFIED**. Orchestrator может подставить `dcl_result.policy_id or context.policy_id`. |
| producer | Orchestrator / caller builder. `EvaluatePolicyDCLGuard` ставит `policy_id=context.policy_id`. Production ChainState пишет `policy_hash = sha256hex(policy_yaml)[:16]`, не `policy_id`. |
| authoritative? | **UNSPECIFIED** для canonical event. |
| participates in canonical serialization? | Canonical event: **UNSPECIFIED**. ChainState: поля нет. |
| participates in event identity/hash? | Canonical event: **UNSPECIFIED**. |

### 4.6 `policy_version`

| Колонка | Значение |
| --- | --- |
| required / optional | Builder: required, непустой. Архитектура: вместе с `policy_id` идентифицирует политику. `ControlContext.policy_version`: `str \| None = None`. Orchestrator: `dcl_result.policy_version or context.policy_version or "unknown"`. `evaluate_policy` возвращает YAML `version` или `"unknown"`. |
| type | Python `str`. HTTP EvaluateResponse: `str`. Контракт события: **UNSPECIFIED**. |
| semantic meaning | Архитектура: версия **фактически применённой** политики. Fallback `"unknown"` у orchestrator / parse error у `evaluate_policy` этому утверждению может противоречить (см. §6). ChainState **не** хранит `policy_version`; в `drift_context` production иногда кладёт `"policy_version_hash": policy_hash` — это **не** то же поле. |
| producer | DCL adapter / `evaluate_policy` / orchestrator fallback / caller builder. |
| authoritative? | **UNSPECIFIED**. |
| participates in canonical serialization? | Canonical event: **UNSPECIFIED**. ChainState: поля нет. |
| participates in event identity/hash? | Canonical event: **UNSPECIFIED**. |

### 4.7 `verdict`

| Колонка | Значение |
| --- | --- |
| required / optional | Builder: обязателен и **должен** быть `"COMMIT"` или `"NO_COMMIT"`. Архитектура: событие только после DCL-вердикта, для обоих значений. ChainState: `verdict TEXT NOT NULL` без enum в DDL. Sentinel пишет `"PASS"`/`"FAIL"` в ту же колонку ChainState. |
| type | Builder: `str` с закрытым множеством `{COMMIT, NO_COMMIT}` на этом builder. Контракт за пределами builder: **UNSPECIFIED** (ChainState принимает другие строки). |
| semantic meaning | Архитектура: enforcement decision DCL. HTTP bazaar: *«COMMIT: the checked action may proceed. NO_COMMIT: do not execute»*. Builder: *Call for both COMMIT and NO_COMMIT. Do not call when DCL was not evaluated.* |
| producer | DCL (`evaluate_policy` / `DCLGuard`) → orchestrator → builder. Production: `evaluate_policy` или детекторы → `ChainState.append`. |
| authoritative? | Архитектура: *«DCL is … authoritative for COMMIT / NO_COMMIT once invoked»* — про решение DCL, не про schema события. Для поля события: **UNSPECIFIED**. |
| participates in canonical serialization? | Canonical event: **UNSPECIFIED**. ChainState: **да**, в `_content_for_hash`. |
| participates in event identity/hash? | Canonical event: **UNSPECIFIED**. ChainState: да. |

### 4.8 `timestamp`

| Колонка | Значение |
| --- | --- |
| required / optional | Builder: параметр optional (`str \| None = None`); в `dict` **всегда** есть (`datetime.now(timezone.utc).isoformat()` если не передан). Архитектурный чеклист §9 **не упоминает** `timestamp`. Контракт required: **UNSPECIFIED**. |
| type | Builder: ISO-строка. ChainState / HTTP EvaluateResponse: `float` Unix time. Два разных runtime-типа; контракт события: **UNSPECIFIED**. |
| semantic meaning | Источники не фиксируют, **какой** момент часовой (старт evaluate vs append vs «sealed»). ChainState docstring: timestamp внутри hash — то же значение, что в колонке. HTTP `EvaluateResponse.timestamp = time.time()` вызывается **после** `append` (второй `time.time()`). |
| producer | Builder default `datetime.now`; ChainState `time.time()` внутри `append`; HTTP отдельно `time.time()`. Orchestrator **не** передаёт `timestamp`. |
| authoritative? | **UNSPECIFIED**. |
| participates in canonical serialization? | Canonical event: **UNSPECIFIED**. ChainState: **да** (`f"{timestamp:.6f}"`). |
| participates in event identity/hash? | Canonical event: **UNSPECIFIED**. ChainState: да, в `tx_hash`. |

### 4.9 `payment`

| Колонка | Значение |
| --- | --- |
| required / optional | Builder и архитектура: **optional** (*«remain optional»*). В `dict` попадает только если аргумент не `None`. Orchestrator **не** передаёт `payment`. |
| type | Builder: `Mapping[str, Any] \| None`, копируется в `dict`. Внутренняя схема: **UNSPECIFIED**. |
| semantic meaning | **UNSPECIFIED** (кроме того, что объект optional). Не путать с `chain_payments` и с x402 `request.state.tx_hash`. |
| producer | Только caller `create_audit_event`. Orchestrator не producer. |
| authoritative? | **UNSPECIFIED**. |
| participates in canonical serialization? | **UNSPECIFIED**. |
| participates in event identity/hash? | **UNSPECIFIED**. Не участвует в ChainState hash (поля нет). |

### 4.10 `proof`

| Колонка | Значение |
| --- | --- |
| required / optional | Builder и архитектура: **optional**. Orchestrator не передаёт. |
| type | `Mapping[str, Any] \| None` → `dict`. Внутренняя схема: **UNSPECIFIED**. |
| semantic meaning | **UNSPECIFIED**. Слово «proof» в README/MCP относится к ChainState/`tx_hash` (*«Don't trust the agent. Trust the proof.»*), не к этому объекту. |
| producer | Только caller builder. |
| authoritative? | **UNSPECIFIED**. |
| participates in canonical serialization? | **UNSPECIFIED**. |
| participates in event identity/hash? | **UNSPECIFIED**. |

### 4.11 `integration`

| Колонка | Значение |
| --- | --- |
| required / optional | Builder и архитектура: **optional**. Orchestrator не передаёт. |
| type | `Mapping[str, Any] \| None` → `dict`. Внутренняя схема: **UNSPECIFIED**. |
| semantic meaning | **UNSPECIFIED**. |
| producer | Только caller builder. |
| authoritative? | **UNSPECIFIED**. |
| participates in canonical serialization? | **UNSPECIFIED**. |
| participates in event identity/hash? | **UNSPECIFIED**. |

### 4.12 `metadata`

| Колонка | Значение |
| --- | --- |
| required / optional | Builder и архитектура: **optional**. Orchestrator **всегда** передаёт dict (`action_type`, `behavior_signal`), поэтому на пути orchestrator ключ фактически всегда присутствует. Это поведение оркестратора, не отдельное утверждение контракта. |
| type | `Mapping[str, Any] \| None` → `dict`. Схема содержимого: **UNSPECIFIED**. Orchestrator кладёт `action_type` и `behavior_signal` (dict или `None`). |
| semantic meaning | Контракт: **UNSPECIFIED**. Orchestrator использует как носитель `action_type` и advisory signal. |
| producer | Orchestrator (на своём пути) или caller builder. |
| authoritative? | **UNSPECIFIED**. |
| participates in canonical serialization? | **UNSPECIFIED**. |
| participates in event identity/hash? | **UNSPECIFIED**. |

### 4.13 Прочие поля, которые реальные источники действительно эмитят

Ниже поля **не** из обязательного списка prompt, но они есть у реальных артефактов. Для Canonical DCL Audit Event контракт по ним указан отдельно.

#### `event_type`

| Колонка | Значение |
| --- | --- |
| required / optional | Builder всегда пишет `"dcl.audit.evaluated"`. Архитектура фиксирует то же значение. Есть ли контракт «обязательное поле JSON»: **UNSPECIFIED** сверх этих двух источников. |
| type | Python `str`. Константа `CANONICAL_EVENT_TYPE`. |
| semantic meaning | Имя типа canonical evaluation event (как сказано источниками). |
| producer | `create_audit_event`. |
| authoritative? | **UNSPECIFIED**. |
| canonical serialization / event hash | **UNSPECIFIED** / **UNSPECIFIED**. |
| Другие артефакты | `LocalBlockRecord.record_type = "local.policy.blocked"`. `sentinel_events.event_type` — другие строки (`rescan_paid` и т.д.). ChainState поля `event_type` не имеет. |

#### `schema_version`

| Колонка | Значение |
| --- | --- |
| required / optional | Builder всегда `"1.0"`. Архитектура то же. Иное: **UNSPECIFIED**. |
| type | Python `str`. Константа `CANONICAL_SCHEMA_VERSION = "1.0"`. Смысл semver vs литерал: **UNSPECIFIED**. |
| semantic meaning | Источники называют это версией схемы Canonical DCL Audit Event v1.0. Frozen schema file отсутствует. |
| producer | `create_audit_event`. |
| authoritative? | **UNSPECIFIED**. |
| canonical serialization / event hash | **UNSPECIFIED** / **UNSPECIFIED**. |
| ChainState | Поля нет. |

#### `reason`

| Колонка | Значение |
| --- | --- |
| required / optional | Builder: optional (`None` → ключ отсутствует). Orchestrator передаёт `dcl_result.reason` (у `DCLEvaluation` тип `str`, default `""`), поэтому ключ на этом пути обычно есть, даже если пустая строка. Архитектура §6: provenance включает reason. Чеклист §9 reason **не** перечисляет. ChainState: `reason TEXT NOT NULL`. |
| type | Python `str` у builder/ChainState. |
| semantic meaning | Объяснение вердикта (формулировки HTTP/MCP). Для canonical event сверх этого: **UNSPECIFIED**. |
| producer | DCL / `evaluate_policy` / детекторы / orchestrator. |
| authoritative? | **UNSPECIFIED**. |
| ChainState hash | **Да** (поле ChainState). Canonical event hash: **UNSPECIFIED**. |

#### `confidence`

| Колонка | Значение |
| --- | --- |
| required / optional | Builder: optional. ChainState: `REAL NOT NULL`. HTTP EvaluateResponse: обязательно в bazaar required-списке. |
| type | Builder: `float \| None`. Диапазон 0..1 упоминается у HTTP/MCP, не у canonical builder. |
| semantic meaning | Bazaar: *«Phrase-policy score from 0 to 1. Not a probability that the action is safe.»* Архитектура: provenance. |
| producer | `evaluate_policy` / DCL / orchestrator. |
| ChainState hash | Да. Canonical event hash: **UNSPECIFIED**. |

#### `payment_id`

| Колонка | Значение |
| --- | --- |
| required / optional | Builder: optional. Архитектура упоминает только как то, от чего `event_id` должен отличаться. Orchestrator не передаёт. |
| type | `str \| None`. |
| semantic meaning | **UNSPECIFIED** (кроме distinctness от `event_id`). |
| producer | Caller builder, если передал. |
| hash | **UNSPECIFIED** для события. В ChainState нет. |

#### `tx_hash`

| Колонка | Значение |
| --- | --- |
| required / optional | Builder: optional. Архитектура: optional chain `tx_hash` from Trust Oracle record; `event_id` distinct from it. ChainState: `tx_hash TEXT UNIQUE NOT NULL` — identity **строки цепи**. |
| type | `str`. Формат ChainState: `"0x" + sha256hex(content)` (64 hex). Builder не проверяет формат. |
| semantic meaning | `payment_logger.py`: *chain.py's `tx_hash` = sha256 of the audit CONTENT (protocol-level, no chain involved)* — это **не** on-chain payment hash. Bazaar: *«Hash of this audit record. Not the payment transaction.»* |
| producer | `ChainState.append` вычисляет; DCLEvaluation/orchestrator может прокинуть в canonical event. |
| authoritative? | Архитектура: ChainState rows *«stay authoritative for the tamper-evident chain»*. Это про цепь, не про canonical event. |
| ChainState hash | `tx_hash` **есть результат** hash, не вход `_content_for_hash`. Canonical event hash: **UNSPECIFIED**. |

#### `receipt_id`

| Колонка | Значение |
| --- | --- |
| required / optional | Builder: optional. Упоминается только в правиле distinctness `event_id`. Orchestrator не передаёт. В ChainState / HTTP evaluate **нет**. |
| type | `str \| None`. |
| semantic meaning | **UNSPECIFIED**. |
| producer | Caller builder. |
| hash | **UNSPECIFIED**. |

#### Поля только ChainState / decode (не ключи `create_audit_event`)

| Поле | Где | required (этого артефакта) | type (как в источнике) | смысл (как сказано) | producer | canonical event serialization / hash | ChainState serialization / `tx_hash` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `idx` / `index` | DDL `idx`; dict key `"index"` | NOT NULL PK | `INTEGER` / Python int | Позиция в цепи (`chain_index` в HTTP) | `ChainState.append` | поле отсутствует у canonical event → **UNSPECIFIED** | **да** (`str(idx)`) |
| `prev_hash` | ChainState, decode | NOT NULL | `TEXT` / str | Hash предыдущей записи; genesis `"0"*64` | `ChainState.append` | **UNSPECIFIED** | **да** |
| `input_hash` | ChainState, HTTP evaluate | NOT NULL | str; production часто `"0x" + sha256hex(response)[:16]` | Hash входа; raw content не хранится (MCP Field) | servers перед `append` | **UNSPECIFIED** | **да** |
| `policy_hash` | ChainState | NOT NULL | str; production `sha256hex(policy_yaml)[:16]` | Hash политики YAML (усечённый в вызовах servers) | servers | **UNSPECIFIED** | **да** |
| `task_type` | ChainState, decode | NOT NULL | str | Тег задачи | servers / sentinel | **UNSPECIFIED** | **да** |
| `drift_context` | ChainState, deep audit | NOT NULL (JSON text, default `{}`) | dict после `json.loads` | Deep: *«Extended forensic metadata captured at evaluation time»* | `append(..., drift_context=)` | **UNSPECIFIED** | **да** (`json.dumps(..., sort_keys=True)`) |

#### `LocalBlockRecord` (контраст, не DCL audit event)

| Поле | Значение в `policy.py` |
| --- | --- |
| `record_type` | default `"local.policy.blocked"` |
| `trace_id`, `agent_id`, `action_type`, `reason`, `rule`, `policy_verdict` | см. `as_dict()` |
| Это Canonical DCL Audit Event? | **Нет** (явный docstring). |
| Hash / canonical serialization события | неприменимо; не определены |

#### HTTP EvaluateResponse (транспорт, не canonical event)

Поля примера README / `EvaluateResponse`: `verdict`, `confidence`, `reason`, `tx_hash`, `chain_index`, `input_hash`, `policy_version`, `timestamp` (float), `pipeline_id` (webhook), `drift_mode`, `drift_score`, `seal_text`, `verify_url`. Bazaar `_EVALUATE_OUTPUT_SCHEMA` required: `verdict`, `confidence`, `reason`, `tx_hash`, `chain_index`. Это schema **HTTP-ответа evaluate**, не Canonical DCL Audit Event v1.0.

#### `telemetry.DecisionEvent` (не canonical event)

Среди прочего: `install_id`, `session_id`, `pipeline_id`, `policy_id` (hash), `timestamp` (float), `verdict`, `confidence`, `deterministic_trace_hash`, … Producer: `telemetry.Collector.record_decision` из webhook. Hash участия в ChainState: нет. Canonical event: не этот dataclass.

#### `chain_payments` (`payment_log.sql`)

`tx_hash`, `route`, `amount_usdc`, `network`, `logged_at`. Не canonical event. Join key — ChainState `tx_hash`.

---

## 5. Фиксированные разделы

### 5.1 `event_id`

Builder всегда кладёт строку: переданную или `uuid.uuid4()`. Правило: не совпадать с **переданными** `payment_id` / `tx_hash` / `receipt_id` (проверка только если те значения не `None` и равны `event_id`). ChainState, `dcl_audit_events`, Transparency Board это поле не определяют. Стабильность, алгоритм, namespace UUID, участие в hash события: **UNSPECIFIED**.

### 5.2 `trace_id`

Требуется builder, архитектурой и `ControlContext`. Должен совпадать с id proposal (архитектура). Production ChainState `trace_id` не хранит. Формат: **UNSPECIFIED**.

### 5.3 `session_id_fingerprint`

**UNSPECIFIED.** **NOT FOUND IN REPO.** Не путать с `telemetry.DecisionEvent.session_id`.

### 5.4 `agent_id`

Есть у builder, context, ChainState, HTTP. Семантика: contextual, без implicit verification (builder + архитектура). Required-статус расходится (см. §6). ChainState включает `agent_id` в свой content-hash. Canonical event hash: **UNSPECIFIED**.

### 5.5 `policy_id`

Есть у builder / context / DCLEvaluation / telemetry (другой смысл). Нет у ChainState (там `policy_hash`). Архитектура: вместе с `policy_version` идентифицирует применённую политику. Orchestrator может взять id из context, если DCL не вернул. **UNSPECIFIED**, является ли это «actually applied» в смысле архитектуры.

### 5.6 `policy_version`

Builder требует непустую строку. Orchestrator может записать `"unknown"`. `evaluate_policy` тоже может вернуть `"unknown"`. ChainState колонку не хранит. HTTP EvaluateResponse поле имеет. Контракт «обязательное поле события» vs fallback: конфликт, не разрешён (§6).

### 5.7 `verdict`

Для canonical builder — только `COMMIT` | `NO_COMMIT`, обязательно. Событие не строится при LOCAL_BLOCK и DCL_UNAVAILABLE. ChainState хранит `verdict` без этого enum (Sentinel: `PASS`/`FAIL`). DCL authoritative для решения COMMIT/NO_COMMIT **после вызова** (архитектура) — это не заявление authoritative schema события.

### 5.8 `timestamp`

У canonical builder — ISO-строка UTC `datetime.now` (если не передали). У ChainState и HTTP — `float` Unix. Чеклист архитектуры §9 поле не перечисляет. Какой clock / какой instant: **UNSPECIFIED**.

### 5.9 `payment`

Optional mapping у builder и в чеклисте. Orchestrator не заполняет. Внутренняя структура, связь с x402 / `chain_payments`: **UNSPECIFIED**.

### 5.10 `proof`

Optional mapping у builder и в чеклисте. Orchestrator не заполняет. Семантика: **UNSPECIFIED**. Не отождествлять с ChainState `tx_hash`, пока источник явно не приравняет (никто не приравнивает).

### 5.11 `integration`

Optional mapping у builder и в чеклисте. Orchestrator не заполняет. Семантика: **UNSPECIFIED**.

### 5.12 `metadata`

Optional mapping у builder и в чеклисте. Orchestrator всегда передаёт свой dict. Схема ключей контрактом не зафиксирована.

### 5.13 Canonical serialization (Canonical DCL Audit Event)

**Не определена.**

Что **есть** в репозитории:

- `create_audit_event` возвращает обычный Python `dict`. Порядок ключей — порядок вставки в литерал + условные `if value is not None`. Это **не** спецификация канонической сериализации (нет byte layout, нет обязательного key order для wire format, нет правила excluded fields, нет encoding).
- `json.dumps(..., sort_keys=True)` в `agent_control/dcl.py` относится к **входу** `evaluate_policy`, не к audit event.
- Frozen JSON Schema события: **NOT FOUND**.

Что **есть**, но у **другого** артефакта (ChainState):

- `ChainState._content_for_hash`: каноническая, order-fixed сериализация через `"|"`.join фиксированного списка полей; `drift_context` как `json.dumps(..., sort_keys=True)`; `confidence` и `timestamp` как `:.6f`.
- Это сериализация **строки цепи**, не Canonical DCL Audit Event, пока источник явно не приравняет их. Архитектура и builder говорят, что это разные артефакты.

### 5.14 Event hash (Canonical DCL Audit Event)

**Не определён.**

- `create_audit_event` **не** считает hash события. Нет алгоритма, нет списка полей, входящих в identity hash.
- Правило «`event_id` distinct from …» — не hash.
- ChainState `tx_hash = "0x" + sha256hex(content)` — hash **ChainState**. `payment_logger.py` и `dcl_core.chain` описывают его как sha256 **содержимого записи цепи**. Архитектура: production chain rows authoritative **для tamper-evident chain**, не утверждает равенство с hash canonical event.
- README: *«`tx_hash` is recomputed from the record's own fields»* — про chain record / dcl-core / dcl-sdk, не про `create_audit_event`.
- Telemetry `deterministic_trace_hash` — другой артефакт.

---

## 6. Конфликты между источниками

Ни один источник не помечен как authoritative **для схемы Canonical DCL Audit Event v1.0**. Authority этой схемы: **UNSPECIFIED**. Ниже — разногласия; победитель **не** выбирается.

1. **Есть ли frozen контракт?** Builder говорит, что реализует *existing* v1.0 contract и не изобретает вторую схему. В репозитории нет отдельной JSON Schema / RFC. Архитектура v0.1 говорит, что **не меняет** Canonical DCL Audit Event v1.0, и одновременно описывает builder в `agent_control/`. Источник «существующего» контракта вне этих файлов **не найден**.

2. **`policy_version` required vs `"unknown"`.** Builder: `if not policy_version: raise ValueError`. Orchestrator: `dcl_result.policy_version or context.policy_version or "unknown"`. Архитектура: поля идентифицируют *policy actually applied*. Строка `"unknown"` проходит builder, но не доказывает applied version. `evaluate_policy` тоже возвращает `"unknown"` при parse error / отсутствии YAML version.

3. **`verdict` closed set vs ChainState.** Builder отвергает всё кроме COMMIT/NO_COMMIT. Таблица `chain` не ограничивает enum. Sentinel пишет `PASS`/`FAIL` в ту же колонку.

4. **`timestamp` тип и часы.** ISO `str` (builder) vs Unix `float` (ChainState, HTTP). HTTP timestamp evaluate ≠ обязательно timestamp строки цепи (два `time.time()`). Архитектура §9 `timestamp` не включает.

5. **`agent_id` required.** Context: обязателен и непустой. Builder: аргумент обязателен, пустота не валидируется. Чеклист §9 не говорит required. Production evaluate: optional, default `"unknown"`.

6. **`metadata` optional vs always-on orchestrator.** Чеклист/builder: optional. Orchestrator всегда передаёт объект.

7. **`reason` / `confidence` optional vs NOT NULL на цепи.** Builder опускает ключ при `None`. ChainState требует оба поля. Orchestrator почти всегда передаёт `reason` как `str`.

8. **Идентификатор политики.** Canonical event: `policy_id` + `policy_version`. ChainState: `policy_hash` (и иногда `policy_version_hash` внутри `drift_context`). Telemetry: `policy_id` = SHA-256 path+content. Это не одно поле и не сказано, что они эквивалентны.

9. **Identity записи.** Canonical event: `event_id` (UUID default). ChainState: `tx_hash` (content hash). Архитектура требует, чтобы они не совпадали, если оба присутствуют; не говорит, что одно является другим.

10. **Где «audit» в production.** README/MCP `GET /audit/{tx_hash}` и `dcl_audit_decode` читают **ChainState**. Canonical event туда не пишется (`create_audit_event` нет в server modules — это говорит сама архитектура).

11. **`dcl_core` версия.** `requirements.txt` pin `@v0.1.1`; установленный пакет `__version__ == "0.1.0"`. К схеме события это не приравнивается; фиксируется как наблюдение о зависимости ChainState.

---

## 7. Что этот черновик **не** решает

Этот файл **не** решает и **не** выбирает:

- какая поверхность authoritative для Canonical DCL Audit Event v1.0;
- JSON Schema, required-набор, типы JSON, форматы (`date-time` vs unix, UUID);
- каноническую сериализацию и hash **события**;
- равенство или маппинг между событиями и ChainState / HTTP EvaluateResponse / telemetry / Sentinel / `chain_payments`;
- вводить таблицу `dcl_audit_events` или Transparency Board;
- смысл и схему `payment`, `proof`, `integration`, `metadata`, `session_id_fingerprint`, `receipt_id`, `payment_id`;
- победителя конфликтов §6;
- менять Python, тесты, production, git.

Любой пункт, не зафиксированный цитатой источника, остаётся **UNSPECIFIED**.
