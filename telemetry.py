"""
DCL Telemetry v2 — AI Decision Behavior Graph

Архитектурный принцип: content-agnostic, structure-only.
Храним структуру решения — никогда не храним содержимое.

Что собирается:
  - Decision metadata     (модель, версия, pipeline, timestamp)
  - Decision context      (task_type, confidence, latency, retry/fallback)
  - Evidence layer        (RAG sources, verification steps, deterministic trace)
  - Outcome               (verdict, human override, error type)
  - Behavioral patterns   (sequences, drift fingerprint, policy effectiveness)

Что НЕ собирается — никогда:
  - Prompts, completions, responses (любой контент)
  - Raw inputs любого вида
  - API keys, tokens, credentials
  - IP адреса, имена пользователей, email, PII
  - Пути к файлам, названия организаций

Privacy compliance:
  GDPR Art.25 (privacy by design) — все ID необратимо хэшируются
  EU AI Act Art.13 — структура решения логируется без контента

Управление: DCL_TELEMETRY=false в env OR collector.disable()
"""

import hashlib
import json
import os
import platform
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional


# ════════════════════════════════════════════════════════════════════════════════
# Privacy helpers
# ════════════════════════════════════════════════════════════════════════════════

def _hash(value: str) -> str:
    """Односторонний SHA-256[:16]. Восстановление невозможно."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _hash_version(version: str) -> str:
    """Хэш версионной строки. Клиентские схемы версионирования не раскрываются."""
    return hashlib.sha256(version.encode("utf-8")).hexdigest()[:12]


def _install_id() -> str:
    """
    Стабильный анонимный ID установки (~/.dcl_install_id).
    Не привязан к пользователю. Пользователь может удалить файл для сброса.
    """
    id_file = os.path.join(os.path.expanduser("~"), ".dcl_install_id")
    if os.path.exists(id_file):
        try:
            val = open(id_file).read().strip()
            if val:
                return val
        except OSError:
            pass
    new_id = uuid.uuid4().hex
    try:
        open(id_file, "w").write(new_id)
    except OSError:
        pass
    return new_id


# ════════════════════════════════════════════════════════════════════════════════
# DecisionEvent — атомарная единица AI Decision Behavior Graph
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class DecisionEvent:
    """
    Структурная запись одного AI decision.
    Контент не присутствует — только метрики и паттерны.

    Группы полей намеренно соответствуют патентной формуле DCL v2/v3:
    каждая группа — отдельный зависимый пункт формулы.
    """

    # ── [1] Identity ──────────────────────────────────────────────────────────
    install_id: str          # SHA-256 ID установки
    session_id: str          # SHA-256 UUID сессии
    pipeline_id: str         # SHA-256 ID pipeline/конфигурации

    # ── [2] Decision metadata ─────────────────────────────────────────────────
    model_provider: str      # "ollama"|"openai"|"anthropic"|"custom" — публичная категория
    model_name: str          # "llama3.2:1b"|"gpt-4o"|"claude-3-5-sonnet" — публичная
    model_version_hash: str  # SHA-256 строки версии модели/endpoint
    prompt_version_hash: str # SHA-256 версии prompt template (содержимое не собирается)
    policy_id: str           # SHA-256 path+content policy файла
    timestamp: float         # Unix timestamp
    dcl_version: str         # Версия DCL Evaluator

    # ── [3] Decision context ──────────────────────────────────────────────────
    task_type: str           # "classification"|"reasoning"|"summarization"|
                             # "generation"|"extraction"|"qa"|"unknown"
    confidence: float        # 0.0 – 1.0
    latency_ms: int          # время LLM inference + evaluation (мс)
    retry_count: int         # retry до валидного ответа (>0 = нестабильность)
    fallback_triggered: bool # сработал ли fallback на другую модель

    # ── [4] Evidence ──────────────────────────────────────────────────────────
    rag_source_count: int         # кол-во RAG источников (0 = RAG не используется)
    rag_source_diversity: float   # entropy разнообразия источников (0.0–1.0)
    verification_steps: int       # кол-во шагов верификации DCL engine
    deterministic_trace_hash: str # SHA-256 трейса выполнения — без деталей
    chain_length_at_decision: int # длина commitment chain на момент решения

    # ── [5] Outcome ───────────────────────────────────────────────────────────
    verdict: str                  # "COMMIT" | "NO_COMMIT"
    error_type: Optional[str]     # "forbidden_pattern"|"low_confidence"|"drift"|
                                  # "policy_violation"|"timeout"|"parse_error"|None
    human_override: bool          # пользователь изменил NO_COMMIT → COMMIT
                                  # КРИТИЧНО: сигнал false positive политики

    # ── [6] Drift metrics ─────────────────────────────────────────────────────
    drift_score: Optional[float]  # Z-test score (None если baseline не установлен)
    drift_mode: Optional[str]     # "NORMAL"|"WARNING"|"ESCALATION"|"BLOCK"
    is_drift_onset: bool          # момент перехода NORMAL → WARNING/ESCALATION
    pre_drift_window: list        # confidence scores за N решений ДО onset
                                  # Заполняется только если is_drift_onset=True
                                  # "Fingerprint деградации" — уникальный актив

    # ── [7] Behavioral patterns ───────────────────────────────────────────────
    decision_sequence: list       # последние N вердиктов: ["COMMIT","COMMIT","NO_COMMIT"]
    confidence_sequence: list     # последние N confidence: [0.91, 0.87, 0.43]
    sequence_window: int          # размер окна
    policy_trigger_rate: float    # NO_COMMIT rate для этой policy в сессии
    session_eval_index: int       # порядковый номер в сессии
    session_eval_pct: float       # относительная позиция 0.0–1.0 (post-hoc)

    # ── Platform ──────────────────────────────────────────────────────────────
    platform_os: str = field(default_factory=lambda: platform.system())
    # "Windows"|"Darwin"|"Linux"


# ════════════════════════════════════════════════════════════════════════════════
# SessionSummary — агрегат сессии
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class SessionSummary:
    """
    Сессионный агрегат. Отправляется при закрытии приложения.
    Позволяет анализировать session-level паттерны отдельно от event-level.
    """
    # Identity
    install_id: str
    session_id: str
    dcl_version: str
    platform_os: str
    timestamp: float

    # Volume
    session_duration_s: int
    total_evaluations: int
    commit_count: int
    no_commit_count: int
    human_override_count: int      # кол-во human overrides = proxy quality политики

    # Confidence distribution
    avg_confidence: float
    min_confidence: float
    confidence_std: float          # стандартное отклонение — мера нестабильности модели

    # Drift
    drift_events: int
    drift_onset_count: int         # сколько раз начинался drift в сессии
    max_drift_mode: str            # наихудший режим: NORMAL|WARNING|ESCALATION|BLOCK

    # Reliability signals
    total_retries: int
    fallback_count: int

    # RAG
    avg_rag_sources: float
    rag_used: bool

    # Diversity
    policy_count: int              # разных политик в сессии
    models_used: list              # список model_name
    task_types_used: list          # список task_type

    # Policy health
    avg_policy_trigger_rate: float # средняя агрессивность политик
    has_pre_drift_data: bool       # есть ли drift fingerprint — ценность для ML


# ════════════════════════════════════════════════════════════════════════════════
# TelemetryCollector
# ════════════════════════════════════════════════════════════════════════════════

class TelemetryCollector:
    """
    Content-agnostic сборщик AI Decision Behavior Graph.

    Гарантии:
    - Контент никогда не попадает в collector — только метрики
    - Async батчевая отправка — UI не блокируется
    - Offline-first: буферизация при недоступности сети
    - Полный opt-out одной командой или переменной окружения
    """

    ENDPOINT = "https://telemetry.fronesislabs.io/v1/events"
    BATCH_SIZE = 50
    FLUSH_INTERVAL_S = 300

    def __init__(self, enabled: bool = True):
        self._enabled = (
            enabled
            and os.environ.get("DCL_TELEMETRY", "true").lower() != "false"
        )
        self._install_id = _install_id()
        self._session_id = uuid.uuid4().hex
        self._session_start = time.time()
        self._queue: queue.Queue = queue.Queue(maxsize=1000)
        self._buffer_file = os.path.join(
            os.path.expanduser("~"), ".dcl_telemetry_buffer.jsonl"
        )
        self._evaluations: list[DecisionEvent] = []

        # Behavioral state
        self._decision_window: list[str] = []
        self._confidence_window: list[float] = []
        self._window_size = 10
        self._pre_drift_buffer: list[float] = []
        self._prev_drift_mode: Optional[str] = None
        self._policy_stats: dict = {}
        self._last_no_commit_index: Optional[int] = None

        if self._enabled:
            self._start_background_flush()
            self._flush_buffer()

    # ── Public API ────────────────────────────────────────────────────────────

    def is_enabled(self) -> bool:
        return self._enabled

    def enable(self):
        self._enabled = True
        self._start_background_flush()

    def disable(self):
        """Полное отключение + удаление локального буфера."""
        self._enabled = False
        try:
            if os.path.exists(self._buffer_file):
                os.remove(self._buffer_file)
        except OSError:
            pass

    def record_decision(
        self,
        # Outcome
        verdict: str,
        confidence: float,
        latency_ms: int,
        error_type: Optional[str] = None,

        # Model
        model_provider: str = "unknown",
        model_name: str = "unknown",
        model_version: str = "",
        prompt_version: str = "",

        # Pipeline
        policy_path: Optional[str] = None,
        pipeline_id: str = "",
        task_type: str = "unknown",

        # Drift
        drift_score: Optional[float] = None,
        drift_mode: Optional[str] = None,

        # Reliability
        retry_count: int = 0,
        fallback_triggered: bool = False,

        # Evidence
        rag_source_count: int = 0,
        rag_source_diversity: float = 0.0,
        verification_steps: int = 1,
        deterministic_trace: str = "",
        chain_length: int = 0,
    ) -> None:
        """
        Записать AI decision. Вызывается из evaluation engine.

        Контент (prompt, response) не принимается намеренно —
        collector принимает только метрики и категории.
        """
        if not self._enabled:
            return

        policy_id = _hash(policy_path or "default")
        eval_index = len(self._evaluations)

        # Temporal sequences
        decision_seq = list(self._decision_window)
        confidence_seq = list(self._confidence_window)
        self._decision_window.append(verdict)
        self._confidence_window.append(confidence)
        if len(self._decision_window) > self._window_size:
            self._decision_window.pop(0)
            self._confidence_window.pop(0)

        # Pre-drift fingerprint
        pre_drift: list[float] = []
        is_drift_onset = False
        if drift_mode in (None, "NORMAL"):
            self._pre_drift_buffer.append(confidence)
            if len(self._pre_drift_buffer) > self._window_size:
                self._pre_drift_buffer.pop(0)
        elif drift_mode in ("WARNING", "ESCALATION", "BLOCK"):
            if self._prev_drift_mode in (None, "NORMAL"):
                is_drift_onset = True
                pre_drift = list(self._pre_drift_buffer)
                self._pre_drift_buffer = []
        self._prev_drift_mode = drift_mode

        # Policy effectiveness
        if policy_id not in self._policy_stats:
            self._policy_stats[policy_id] = {"total": 0, "no_commit": 0}
        self._policy_stats[policy_id]["total"] += 1
        if verdict == "NO_COMMIT":
            self._policy_stats[policy_id]["no_commit"] += 1
            self._last_no_commit_index = eval_index
        stats = self._policy_stats[policy_id]
        trigger_rate = stats["no_commit"] / stats["total"] if stats["total"] else 0.0

        event = DecisionEvent(
            install_id=self._install_id,
            session_id=_hash(self._session_id),
            pipeline_id=_hash(pipeline_id or "default"),
            model_provider=model_provider,
            model_name=model_name,
            model_version_hash=_hash_version(model_version or model_name),
            prompt_version_hash=_hash_version(prompt_version or "v0"),
            policy_id=policy_id,
            timestamp=time.time(),
            dcl_version="1.1.0",
            task_type=task_type,
            confidence=round(confidence, 3),
            latency_ms=latency_ms,
            retry_count=retry_count,
            fallback_triggered=fallback_triggered,
            rag_source_count=rag_source_count,
            rag_source_diversity=round(rag_source_diversity, 3),
            verification_steps=verification_steps,
            deterministic_trace_hash=_hash(deterministic_trace or verdict),
            chain_length_at_decision=chain_length,
            verdict=verdict,
            error_type=error_type,
            human_override=False,
            drift_score=round(drift_score, 4) if drift_score is not None else None,
            drift_mode=drift_mode,
            is_drift_onset=is_drift_onset,
            pre_drift_window=[round(c, 3) for c in pre_drift],
            decision_sequence=decision_seq,
            confidence_sequence=[round(c, 3) for c in confidence_seq],
            sequence_window=self._window_size,
            policy_trigger_rate=round(trigger_rate, 3),
            session_eval_index=eval_index,
            session_eval_pct=0.0,
        )

        self._evaluations.append(event)
        try:
            self._queue.put_nowait(asdict(event))
        except queue.Full:
            pass

    def record_override(self) -> None:
        """
        Вызвать когда пользователь overrides NO_COMMIT вердикт.
        Проставляет human_override=True на последнее NO_COMMIT событие.
        Этот сигнал — основа для auto-tuning policy engine (DCL v2).
        """
        if not self._enabled or self._last_no_commit_index is None:
            return
        idx = self._last_no_commit_index
        if idx < len(self._evaluations):
            self._evaluations[idx].human_override = True
        self._last_no_commit_index = None

    def flush_session(self) -> Optional[SessionSummary]:
        """
        Финализировать сессию при закрытии приложения.
        Рассчитывает session_eval_pct post-hoc, собирает SessionSummary.
        """
        if not self._evaluations:
            return None

        total = len(self._evaluations)
        for ev in self._evaluations:
            ev.session_eval_pct = round(ev.session_eval_index / max(total - 1, 1), 3)

        confidences = [e.confidence for e in self._evaluations]
        avg_conf = sum(confidences) / total
        conf_std = round((sum((c - avg_conf) ** 2 for c in confidences) / total) ** 0.5, 3)

        rank = {"NORMAL": 0, "WARNING": 1, "ESCALATION": 2, "BLOCK": 3}
        max_drift = max(
            (e.drift_mode for e in self._evaluations if e.drift_mode),
            key=lambda m: rank.get(m, 0),
            default="NORMAL",
        )

        summary = SessionSummary(
            install_id=self._install_id,
            session_id=_hash(self._session_id),
            dcl_version="1.1.0",
            platform_os=platform.system(),
            timestamp=time.time(),
            session_duration_s=int(time.time() - self._session_start),
            total_evaluations=total,
            commit_count=sum(1 for e in self._evaluations if e.verdict == "COMMIT"),
            no_commit_count=sum(1 for e in self._evaluations if e.verdict == "NO_COMMIT"),
            human_override_count=sum(1 for e in self._evaluations if e.human_override),
            avg_confidence=round(avg_conf, 3),
            min_confidence=round(min(confidences), 3),
            confidence_std=conf_std,
            drift_events=sum(1 for e in self._evaluations if e.drift_mode and e.drift_mode != "NORMAL"),
            drift_onset_count=sum(1 for e in self._evaluations if e.is_drift_onset),
            max_drift_mode=max_drift,
            total_retries=sum(e.retry_count for e in self._evaluations),
            fallback_count=sum(1 for e in self._evaluations if e.fallback_triggered),
            avg_rag_sources=round(sum(e.rag_source_count for e in self._evaluations) / total, 2),
            rag_used=any(e.rag_source_count > 0 for e in self._evaluations),
            policy_count=len({e.policy_id for e in self._evaluations}),
            models_used=list({e.model_name for e in self._evaluations}),
            task_types_used=list({e.task_type for e in self._evaluations}),
            avg_policy_trigger_rate=round(
                sum(e.policy_trigger_rate for e in self._evaluations) / total, 3
            ),
            has_pre_drift_data=any(e.pre_drift_window for e in self._evaluations),
        )

        if self._enabled:
            try:
                self._queue.put_nowait({"_type": "session_summary", **asdict(summary)})
            except queue.Full:
                pass

        return summary

    # ── Background flush ──────────────────────────────────────────────────────

    def _start_background_flush(self):
        threading.Thread(target=self._flush_loop, daemon=True).start()

    def _flush_loop(self):
        while self._enabled:
            time.sleep(self.FLUSH_INTERVAL_S)
            self._send_batch()

    def _send_batch(self):
        batch = []
        try:
            while len(batch) < self.BATCH_SIZE:
                batch.append(self._queue.get_nowait())
        except queue.Empty:
            pass
        if not batch:
            return
        try:
            import urllib.request
            data = json.dumps({"events": batch}).encode()
            req = urllib.request.Request(
                self.ENDPOINT, data=data,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status != 200:
                    raise ValueError(f"HTTP {r.status}")
        except Exception:
            self._write_to_buffer(batch)

    def _write_to_buffer(self, events: list):
        try:
            with open(self._buffer_file, "a") as f:
                for ev in events:
                    f.write(json.dumps(ev) + "\n")
        except OSError:
            pass

    def _flush_buffer(self):
        """При старте — отправить события из прошлых offline сессий."""
        if not os.path.exists(self._buffer_file):
            return
        try:
            lines = open(self._buffer_file).readlines()
            events = [json.loads(l) for l in lines if l.strip()]
            if events:
                import urllib.request
                data = json.dumps({"events": events}).encode()
                req = urllib.request.Request(
                    self.ENDPOINT, data=data,
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                urllib.request.urlopen(req, timeout=5)
            os.remove(self._buffer_file)
        except Exception:
            pass


# ════════════════════════════════════════════════════════════════════════════════
# Singleton
# ════════════════════════════════════════════════════════════════════════════════

_collector: Optional[TelemetryCollector] = None


def get_collector(enabled: bool = True) -> TelemetryCollector:
    global _collector
    if _collector is None:
        _collector = TelemetryCollector(enabled=enabled)
    return _collector
