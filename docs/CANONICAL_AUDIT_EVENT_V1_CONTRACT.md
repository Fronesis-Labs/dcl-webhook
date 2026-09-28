# Canonical DCL Audit Event v1.0 — Semantic Contract

## 1. Status banner

| | |
| --- | --- |
| Status | **FROZEN v1.0** |
| Frozen? | **Yes.** Canonical DCL Audit Event v1.0 is frozen at the semantic and canonicalization level defined in this document. Items explicitly marked **OPEN** remain intentionally unspecified and must not be inferred by implementations. |
| JSON Schema? | **No.** This is not a JSON Schema and does not introduce one. |
| Production impact | **None.** This document does not modify production, Python code, tests, databases, ChainState, `telemetry.DecisionEvent`, `agent_control/canonical_audit.py`, or `docs/CANONICAL_AUDIT_EVENT_V1.md`. |
| Evidence inventory | `docs/CANONICAL_AUDIT_EVENT_V1.md` (inventory only; not an authoritative schema; **untouched**) |
| Date | 2026-09-28 |
| Frozen | 2026-09-28 |

Canonical DCL Audit Event v1.0 is frozen at the semantic and canonicalization level defined in this document. Items explicitly marked OPEN remain intentionally unspecified and must not be inferred by implementations.

Labels **FROZEN** below are approved v1.0 contract rules. Labels **OPEN** remain intentionally unspecified and were not promoted.

Implementations may be cited as evidence. They are not schema authorities.

---

## 2. How to read labels

Every decision in this document carries exactly one of these labels (words used as written):

| Label | Meaning |
| --- | --- |
| **FROZEN** | An approved v1.0 contract rule. Code that contradicts a **FROZEN** rule is **IMPLEMENTATION-SPECIFIC**, not a silent winner. |
| **OPEN** | Intentionally unspecified. This document does not settle it. Do not invent a rule. Implementations must not infer a value. |
| **IMPLEMENTATION-SPECIFIC** | Behavior that exists in code, ChainState, telemetry, the orchestrator, storage, or rendering, but is **not** part of this contract. |
| **UNKNOWN** | Repository evidence is insufficient. No meaning is invented. |

Field names in this document are identifiers (`event_id`, `policy_id`, …). They are not prose.

---

## 3. Canonical fields (minimal set)

This section lists the **canonical** identity, applied-policy, agent, session, decision, and time fields of this contract. Optional extension objects (`payment`, `proof`, `integration`, `metadata`) are **not** canonical fields; they are optional extensions (§4, §12).

`payment_id`, `tx_hash`, and `receipt_id` are **not** canonical fields. They appear only as distinctness constraints for `event_id` (§14). Whether they should eventually become explicit event fields remains **OPEN**.

| Identifier | Role | Required when a canonical event is emitted? | Status |
| --- | --- | --- | --- |
| `event_type` | Identity literal | Yes | **FROZEN** = `dcl.audit.evaluated` |
| `schema_version` | Identity literal | Yes | **FROZEN** = `1.0` |
| `event_id` | Identity of one emitted audit event | Yes | **FROZEN** semantics; concrete format **OPEN** |
| `trace_id` | Identity / correlation with the evaluation/execution trace | Yes | **FROZEN** required; exact format **OPEN** |
| `agent_id` | Contextual agent identifier | No — **OPTIONAL** | **FROZEN** optional/contextual; presence does **not** imply verified identity |
| `session_id_fingerprint` | Contextual fingerprint of a session | No — **OPTIONAL** | **FROZEN** optional; algorithm/format **OPEN** |
| `policy_id` | Policy actually evaluated by DCL | Yes | **FROZEN** applied-policy semantics |
| `policy_version` | Version of the policy actually evaluated by DCL | Yes | **FROZEN** applied-policy semantics |
| `verdict` | Canonical decision | Yes | **FROZEN** vocabulary = `COMMIT` \| `NO_COMMIT` |
| `timestamp` | Event time | Yes | **FROZEN** UTC ISO-8601 representation; some details remain **OPEN** (§11) |

Fields **not** in this table are not canonical v1.0 fields in this contract.

---

## 4. Optional extensions

These four names are **OPTIONAL EXTENSIONS**, not canonical fields. Optional membership is **FROZEN**. Inner schemas are **UNKNOWN** (no authoritative evidence) and therefore remain **OPEN**.

| Identifier | Optional? | Inner schema |
| --- | --- | --- |
| `payment` | **FROZEN** optional extension | **UNKNOWN** / **OPEN** |
| `proof` | **FROZEN** optional extension | **UNKNOWN** / **OPEN** |
| `integration` | **FROZEN** optional extension | **UNKNOWN** / **OPEN** |
| `metadata` | **FROZEN** optional extension | **UNKNOWN** / **OPEN** |

**Omission (FROZEN serialization requirement of this contract):** if an optional extension has no value, the canonical representation **MUST OMIT** the field. Emitting `null` is not a valid canonical representation of absence.

The orchestrator always supplying a `metadata` dict is **IMPLEMENTATION-SPECIFIC**. Always-on orchestrator metadata does **not** make `metadata` a required canonical field.

Full discussion: §12.

---

## 5. Identity

### 5.1 `event_type`

| | |
| --- | --- |
| Contract value | `dcl.audit.evaluated` (literal) |
| Status | **FROZEN** |

Other `event_type` / `record_type` strings (`local.policy.blocked`, Sentinel `sentinel_events.event_type`, absence of `event_type` on ChainState) are **IMPLEMENTATION-SPECIFIC** to those artifacts.

### 5.2 `schema_version`

| | |
| --- | --- |
| Contract value | `1.0` (literal) |
| Status | **FROZEN** |

Whether `1.0` is semver, a display label, or a wire token beyond the literal is **OPEN**. ChainState has no `schema_version`; that absence is **IMPLEMENTATION-SPECIFIC** to ChainState.

### 5.3 `event_id`

| | |
| --- | --- |
| Status | **FROZEN** semantics (this subsection). Concrete format: **OPEN**. |
| Role | Required identity field. Identifies **one emitted audit event**. |
| Distinctness | **MUST** be distinct from `payment_id`, `tx_hash`, and `receipt_id` (§14). |
| Collision resistance | `event_id` is collision-resistant. |
| Replay | `event_id` is **NOT** replay-idempotent: a new evaluation or retry **may** receive a new `event_id`. |
| Format | **OPEN** — do not invent UUID, namespace, or encoding. |

`event_id` is **not** claimed to be globally unique.

Global uniqueness requires an event store or equivalent persistence constraint and is not established by the event builder alone.

**IMPLEMENTATION-SPECIFIC (not contract format):** `create_audit_event` always emits `event_id`; if the caller omits it, the builder defaults to `str(uuid.uuid4())`. Orchestrator does not pass `event_id`. ChainState does not store `event_id`. Builder `uuid4` is not the frozen format.

Exact future replay / idempotency mechanism (beyond: not replay-idempotent; a retry may get a new `event_id`) remains **OPEN**.

### 5.4 `trace_id`

| | |
| --- | --- |
| Status | **FROZEN** required identity / correlation field. Exact format: **OPEN**. |
| Role | Connects the event to the evaluation / execution trace. |
| Required | Yes, when a canonical event is emitted. |
| Format | **OPEN** — do not invent UUID, namespace, or encoding. |

The control-plane claim “same id as the proposal” is architecture of agent-control, not a frozen wire-format rule for this event. Status of that extra semantic as a v1.0 event rule: **OPEN**.

Production ChainState does not store `trace_id` (**IMPLEMENTATION-SPECIFIC** to ChainState). Builder/context non-empty-string checks are **IMPLEMENTATION-SPECIFIC**.

---

## 6. Applied policy

### 6.1 Names and requirement

`policy_id` and `policy_version` are **REQUIRED** when a canonical event is emitted.

| Identifier | Contract rule | Status |
| --- | --- | --- |
| `policy_id` | **MUST** identify the policy **actually evaluated by DCL**. **MUST NOT** be synthesized from the requested policy when DCL evaluated another policy. The literal fallback value `"unknown"` **MUST NOT** be used. | **FROZEN** |
| `policy_version` | **MUST** identify the version of the policy **actually evaluated by DCL**. The literal fallback value `"unknown"` **MUST NOT** be used. | **FROZEN** |

Identifier **format** (builtin name, path, hash, YAML `version` field, other): **OPEN**. No format is invented.

### 6.2 Contradiction (recorded history; contract rule is not weakened)

The **FROZEN** rule above is the contract. The following code paths contradict it and remain **IMPLEMENTATION-SPECIFIC**. They are not silent winners and are not edited by this document.

| Source | What it does |
| --- | --- |
| Architecture / builder docstring | `policy_id` + `policy_version` identify the policy actually applied |
| `AgentControlOrchestrator.handle` | `policy_id=dcl_result.policy_id or context.policy_id` — fallback to requested/context policy |
| `AgentControlOrchestrator.handle` | `policy_version=dcl_result.policy_version or context.policy_version or "unknown"` |
| `audit_logic.evaluate_policy` | On YAML parse error, returns `policy_version="unknown"`; if YAML has no `version`, uses `"unknown"` |

Builder requires non-empty `policy_id` and `policy_version` strings; `"unknown"` is non-empty and therefore passes the builder. That is **IMPLEMENTATION-SPECIFIC**, not approval of `"unknown"` as an applied-policy identifier.

ChainState has `policy_hash`, not `policy_id` / `policy_version`. Telemetry `DecisionEvent.policy_id` is a SHA-256 of path+content. Those are **IMPLEMENTATION-SPECIFIC** to those artifacts and are not this event’s fields.

---

## 7. Agent context

### 7.1 `agent_id`

| | |
| --- | --- |
| Required / optional | **FROZEN** **OPTIONAL** contextual field |
| Presence implies verified identity? | **No.** Status: **FROZEN** |

Presence of `agent_id` **MUST NOT** imply verified identity.

**IMPLEMENTATION-SPECIFIC (not this contract’s requiredness):** `ControlContext` requires a non-empty `agent_id`; `create_audit_event` requires the argument but allows `""`; architecture checklist treats identity as contextual; production HTTP evaluate defaults `agent_id="unknown"`. Those surfaces do not change the **FROZEN** rule that `agent_id` is optional and unverified.

Production default `"unknown"` for ChainState/HTTP `agent_id` is **IMPLEMENTATION-SPECIFIC**, not a contract value.

---

## 8. Session context

### 8.1 `session_id_fingerprint`

| | |
| --- | --- |
| Required / optional | **FROZEN** **OPTIONAL** contextual field |
| Meaning | A fingerprint of a session. Status of that role: **FROZEN**. |
| Algorithm / format | **OPEN** and **IMPLEMENTATION-SPECIFIC**. Also **UNKNOWN** in this repository (no producer defines it). Do not invent an algorithm. |

Do **not** copy `telemetry.DecisionEvent.session_id` SHA-256 semantics into this field.

### 8.2 Lookalike (not this field)

`telemetry.DecisionEvent.session_id` is a different field (SHA-256 of a session UUID in telemetry). That field is **IMPLEMENTATION-SPECIFIC** to telemetry. It is **not** `session_id_fingerprint`.

The identifier `session_id_fingerprint` does not appear in repository producers. Absence of evidence is **UNKNOWN**; it does not invent an algorithm.

---

## 9. Decision

### 9.1 `verdict`

| | |
| --- | --- |
| Required | **FROZEN** **REQUIRED** |
| v1.0 vocabulary | Exactly `COMMIT` and `NO_COMMIT`. Status: **FROZEN** |

`PASS` and `FAIL` are **not** part of the canonical v1.0 verdict vocabulary. Do **not** inherit `PASS`/`FAIL` from ChainState. `PASS`/`FAIL` is an implementation/history artifact of ChainState and is **IMPLEMENTATION-SPECIFIC**.

### 9.2 Evidence and contradiction (recorded, not resolved in code)

Named canonical event sources (architecture and `create_audit_event`) use only `COMMIT` and `NO_COMMIT`. Builder rejects any other string. No evidence equates ChainState/Sentinel `PASS`/`FAIL` with `COMMIT`/`NO_COMMIT`.

| Source | Verdict values |
| --- | --- |
| Architecture + canonical builder | `COMMIT`, `NO_COMMIT` |
| ChainState column `verdict` | `TEXT NOT NULL`, no enum in DDL |
| `sentinel_audit.py` → `ChainState.append` | `PASS` / `FAIL` |

`PASS` / `FAIL` remain **IMPLEMENTATION-SPECIFIC** to ChainState / Sentinel. They are not inherited into this event.

---

## 10. Time

### 10.1 `timestamp`

| | |
| --- | --- |
| Required | **FROZEN** **REQUIRED** |
| Representation | **FROZEN**: UTC ISO-8601 |

timestamp MUST be a UTC ISO-8601 string (offset Z or equivalent UTC designator) before it participates in canonical serialization; a Unix numeric timestamp is not a valid v1.0 representation.

Implementations **MUST NOT** silently use Unix timestamps as the v1.0 representation.

### 10.2 Details that remain OPEN

Do not invent a profile beyond UTC ISO-8601.

| Detail | Status |
| --- | --- |
| Whether fractional seconds are required | **OPEN** |
| The exact instant the timestamp denotes (start of evaluate vs emit vs other) | **OPEN** |

### 10.3 Why not inherited

| Source | Representation | Label |
| --- | --- | --- |
| `create_audit_event` | Always emits; default `datetime.now(timezone.utc).isoformat()` | **IMPLEMENTATION-SPECIFIC** |
| ChainState / HTTP `EvaluateResponse` | Unix `float` (`time.time()`); ChainState hash uses `f"{timestamp:.6f}"` | **IMPLEMENTATION-SPECIFIC** |

This contract does **not** inherit Unix float from ChainState/HTTP. ChainState Unix time **MUST NOT** silently become Canonical DCL Audit Event v1.0 timestamp semantics.

---

## 11. Canonicalization

Canonical serialization is **REQUIRED** before any implementation may be called frozen against this contract. Deterministic canonical serialization requirements below are **FROZEN**.

### 11.1 FROZEN serialization requirements

Canonical serialization **MUST**:

- use UTF-8 encoding
- use deterministic field ordering
- use lexicographic ordering of object keys
- contain no insignificant whitespace
- omit optional fields when absent (not emit `null`) — same omission rule as §4
- require `timestamp` already normalized to the UTC ISO-8601 representation in §10 **before** it participates in canonical serialization
- produce exactly one deterministic byte representation for the same event

### 11.2 Event hash

Do **not** invent a hash algorithm.

| Topic | Status |
| --- | --- |
| Hash algorithm | **OPEN** — not approved; remains intentionally unspecified |
| Bytes a future hash MUST cover | **FROZEN** constraint on any future hash (this subsection) |

ChainState’s existing sha256 is **NOT** the canonical audit-event hash.

A future event hash **MUST** be computed over the canonical serialized event, not over the ChainState pipe-delimited string.

The constraint “future hash is over canonical bytes, not the pipe string” is **FROZEN** for any future hash. The hash **algorithm** itself stays **OPEN**.

Builder does not compute an event hash. That absence is **IMPLEMENTATION-SPECIFIC** to the current sketch, not a frozen algorithm.

`json.dumps(..., sort_keys=True)` in `agent_control/dcl.py` serializes **input** to `evaluate_policy`, not this event. **IMPLEMENTATION-SPECIFIC** to the DCL adapter.

A frozen JSON Schema for this event: **not found** in the repository (inventory). This file does not create one.

Python `dict` insertion order in the builder is **IMPLEMENTATION-SPECIFIC**, not a substitute for the **FROZEN** lexicographic canonical-byte rule above.

---

## 12. Optional extensions (detail)

| Identifier | Optional status | Inner schema / keys | Semantic meaning beyond “optional object” |
| --- | --- | --- | --- |
| `payment` | **FROZEN** optional extension | **UNKNOWN** / **OPEN** | **UNKNOWN** |
| `proof` | **FROZEN** optional extension | **UNKNOWN** / **OPEN** | **UNKNOWN**. Do not equate with ChainState `tx_hash` or marketing “proof” copy. |
| `integration` | **FROZEN** optional extension | **UNKNOWN** / **OPEN** | **UNKNOWN** |
| `metadata` | **FROZEN** optional extension | **UNKNOWN** / **OPEN** | **UNKNOWN** |

Orchestrator always passes `metadata={"action_type": ..., "behavior_signal": ...}`. **IMPLEMENTATION-SPECIFIC.** Does not make `metadata` required. Does not freeze those inner keys.

Builder copies a mapping only if the argument is not `None`. **IMPLEMENTATION-SPECIFIC** relative to Python objects. The **FROZEN** canonical rule is omit-when-absent, not `null`.

Do not invent keys inside these objects.

---

## 13. Identity and replay (summary)

Semantics in §5 for `event_id` and `trace_id` are **FROZEN**:

- `event_id` is a required identity field for one emitted audit event; collision-resistant; distinct from `payment_id`, `tx_hash`, and `receipt_id`; not replay-idempotent (a retry may receive a new `event_id`).
- `trace_id` is a required identity/correlation field connecting the event to the evaluation/execution trace.

This contract does **not** claim global uniqueness of `event_id`.

Global uniqueness requires an event store or equivalent persistence constraint and is not established by the event builder alone.

| Topic | Status |
| --- | --- |
| `event_id` concrete format | **OPEN** |
| `trace_id` exact format | **OPEN** |
| Global event-store uniqueness mechanism | **OPEN** |
| Exact future replay / idempotency mechanism | **OPEN** (beyond the **FROZEN** non-idempotent `event_id` rule) |

---

## 14. Relationships / distinctness

These relationship rules are **FROZEN**.

1. `event_id` is distinct from `payment_id`.
2. `event_id` is distinct from `tx_hash`.
3. `event_id` is distinct from `receipt_id`.
4. ChainState sha256 (`tx_hash` / `_content_for_hash` pipe-delimited string) is **not** the canonical audit-event hash.
5. ChainState is **not** the schema authority for this event.

`payment_id`, `tx_hash`, and `receipt_id` are distinctness constraints. They are **not** required canonical fields (§3).

**Should they eventually become explicit event fields?** **OPEN**. Do not add them as required canonical fields.

| Source | Treatment |
| --- | --- |
| This contract’s canonical set | Does not include them as canonical fields |
| `create_audit_event` | Optional kwargs; keys added only if not `None` (**IMPLEMENTATION-SPECIFIC**) |
| Architecture | Mentions them as things `event_id` must be distinct from |
| Orchestrator | May pass `tx_hash=dcl_result.tx_hash`; does not pass `payment_id` or `receipt_id` (**IMPLEMENTATION-SPECIFIC**) |

---

## 15. Authority

The Canonical DCL Audit Event v1.0 contract is independent of:

- ChainState
- `telemetry.DecisionEvent`
- current database schemas
- `agent_control/canonical_audit.py`

ChainState is **NOT** the schema authority.

ChainState’s `PASS`/`FAIL` vocabulary, Unix timestamp, `policy_hash`, and pipe-delimited sha256 **MUST NOT** silently become Canonical DCL Audit Event v1.0 semantics.

`agent_control/canonical_audit.py` is a **reference implementation sketch only** until it conforms to this contract.

This document is the **FROZEN v1.0** semantic and canonicalization contract for decisions labeled **FROZEN**. Items labeled **OPEN** remain intentionally unspecified. The evidence inventory `docs/CANONICAL_AUDIT_EVENT_V1.md` is **not** an authoritative schema. No JSON Schema / RFC for Canonical DCL Audit Event v1.0 was found in-tree.

Python object representation, database storage, ChainState adapters, and production persistence are **IMPLEMENTATION-SPECIFIC**.

---

## 16. Contradictions recorded (not resolved in code)

This document records contradictions as history. The **CONTRACT** rule is labeled **FROZEN**. The **code** behavior is **IMPLEMENTATION-SPECIFIC**. This document does not edit code to fix the contradiction.

1. **Applied policy vs `"unknown"` / requested fallback.** **FROZEN** rule: identify the policy actually evaluated; do not synthesize from the requested policy; do not use literal `"unknown"`. Orchestrator and `evaluate_policy` can emit `"unknown"` or fall back to `context.policy_id`. That code is **IMPLEMENTATION-SPECIFIC**.

2. **`timestamp` representations.** **FROZEN** rule: UTC ISO-8601 string before canonical serialization; Unix numeric timestamp is not valid v1.0. ChainState/HTTP Unix float is **IMPLEMENTATION-SPECIFIC**.

3. **`agent_id` requiredness.** **FROZEN** rule: optional contextual; presence does not imply verified identity. `ControlContext` required non-empty; builder allows `""`; production evaluate defaults `"unknown"`. Those surfaces are **IMPLEMENTATION-SPECIFIC**.

4. **`verdict` closed set vs ChainState.** **FROZEN** vocabulary: `COMMIT` \| `NO_COMMIT`. ChainState/Sentinel `PASS`/`FAIL` is **IMPLEMENTATION-SPECIFIC**.

5. **`metadata` optional vs always-on orchestrator.** **FROZEN** rule: `metadata` is an optional extension; omit when absent. Orchestrator always passes a dict. Orchestrator always-on metadata is **IMPLEMENTATION-SPECIFIC**.

6. **`event_id` format.** Concrete format **OPEN**. Builder `uuid4` default is **IMPLEMENTATION-SPECIFIC**, not the frozen format.

7. **Hash.** ChainState pipe-delimited sha256 is **IMPLEMENTATION-SPECIFIC** and is **not** the canonical audit-event hash. Future hash-over-canonical-bytes is a **FROZEN** constraint; algorithm remains **OPEN**.

Additional inventory conflicts (reason/confidence NOT NULL on ChainState vs optional on builder; policy_id vs policy_hash vs telemetry hash; event_id vs ChainState `tx_hash` as record identity) are **not** inherited. They corroborate that ChainState is a different artifact (**FROZEN** relationship).

---

## 17. Decision matrix

### FROZEN (approved v1.0 decisions)

- `event_type`
- `schema_version`
- `event_id` semantics (required identity of one emitted event; collision-resistant; not replay-idempotent; not claimed globally unique by the builder)
- `trace_id` required
- `agent_id` optional / contextual (presence does not imply verified identity)
- `session_id_fingerprint` optional (fingerprint of a session)
- applied policy semantics (actually evaluated by DCL; no requested-policy synthesis; no literal `"unknown"`)
- `policy_id`
- `policy_version`
- verdict vocabulary (`COMMIT` \| `NO_COMMIT` only)
- timestamp requirement (REQUIRED; UTC ISO-8601 as specified in §10)
- optional extension membership (`payment`, `proof`, `integration`, `metadata`)
- deterministic canonical serialization requirements (§11.1), including omit-when-absent (not `null`)
- `event_id` distinctness rules (distinct from `payment_id`, `tx_hash`, `receipt_id`)
- future event hash, if any, MUST be over canonical serialized bytes, not the ChainState pipe-delimited string (algorithm still **OPEN**)
- ChainState is not schema authority; ChainState sha256 is not the canonical audit-event hash

### OPEN

- `trace_id` format
- `event_id` concrete format
- `session_id_fingerprint` algorithm
- inner schemas of `payment` / `proof` / `integration` / `metadata`
- event hash algorithm
- global event-store uniqueness mechanism
- exact future replay / idempotency mechanism
- whether `payment_id` / `tx_hash` / `receipt_id` should eventually become explicit event fields rather than only distinctness constraints
- whether fractional seconds are required on `timestamp`
- which instant `timestamp` denotes
- whether `schema_version` `1.0` is semver vs a wire token beyond the literal
- whether “same id as the proposal” is a v1.0 `trace_id` event rule

### IMPLEMENTATION-SPECIFIC

- Python object representation
- database storage
- ChainState adapter
- production persistence
- Transparency Board rendering (as a category; no Transparency Board exists in this repository)
- orchestrator policy fallback / `"unknown"`
- ChainState `PASS`/`FAIL`, Unix time, `policy_hash`, and pipe-delimited sha256
- builder `uuid4` default for `event_id`
- orchestrator always-on `metadata`
- builder/context non-empty string checks; HTTP/`ControlContext` `agent_id` defaults and requiredness
- `telemetry.DecisionEvent` fields including hashed `session_id` and hashed `policy_id`
- `LocalBlockRecord` (`local.policy.blocked`); Sentinel persistence / `sentinel_events`

### UNKNOWN

- inner schemas of `payment` / `proof` / `integration` / `metadata` (insufficient evidence; do not invent keys)
- `session_id_fingerprint` algorithm (no producer in repo; do not invent; do not copy telemetry SHA-256)
- Transparency Board: **NOT FOUND** in this repository. Rendering remains **IMPLEMENTATION-SPECIFIC** as a category; absence does not invent a board or a rendering contract
- any other meaning for which repository evidence is insufficient

---

## 18. Freeze Gate

A human has approved the Freeze Gate for the **FROZEN** set in this document.

Canonical DCL Audit Event v1.0 is frozen at the semantic and canonicalization level defined in this document. Items explicitly marked OPEN remain intentionally unspecified and must not be inferred by implementations.

Checked boxes record human approval of decisions that were already labeled **FROZEN**. Unchecked boxes cover **OPEN** decisions; they remain **OPEN**, were not approved as specified, and were not promoted to **FROZEN**.

### Human approval checklist

- [x] Core event identity — approved as **FROZEN** (`event_type`, `schema_version`, `event_id` semantics, `trace_id` required). Concrete `event_id` / `trace_id` formats remain **OPEN**.
- [x] Required vs optional fields — approved as **FROZEN**
- [x] Applied policy semantics — approved as **FROZEN**
- [x] Verdict vocabulary — approved as **FROZEN**
- [x] Timestamp representation — approved as **FROZEN** (UTC ISO-8601). Fractional seconds and which instant remain **OPEN**.
- [x] Canonical serialization rules — approved as **FROZEN**
- [x] Optional-field omission semantics — approved as **FROZEN**
- [x] `session_id_fingerprint` semantics — approved as **FROZEN** (optional fingerprint of a session). Algorithm/format remains **OPEN** and was not promoted.
- [ ] event hash decision — remains **OPEN** (algorithm not specified; not promoted)
- [ ] event-store uniqueness/replay semantics — remains **OPEN** (not promoted)
- [ ] extension schemas — remains **OPEN** (inner schemas **UNKNOWN** / **OPEN**; not promoted)
