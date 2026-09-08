# DCL Update Sentinel — техспек v3 (финальный для MVP, Cursor)

Контекст: новый платный продукт в линейке DCL. Реализуется как **новый роут в существующем `dcl-webhook`**, использующий функции из `dcl-core` (audit engine). Отдельный репозиторий не создаётся.

Позиционирование: 9 существующих бесплатных скиллов — демонстрация возможностей DCL. **Sentinel — managed security service**: "DCL continuously monitors the software your agent depends on and detects malicious or risky updates before they enter production."

```
                    GitHub update
                         │
                         ▼
                  Update Sentinel
                         │
                         ▼
                    DCL Auditor
                         │
                 ┌───────┴────────┐
                 ▼                ▼
              verdict           score
                 │                │
                 └───────┬────────┘
                         ▼
                  Regression?
                    /       \
                  NO         YES
                  │           │
                  ▼           ▼
             last_known    BLOCKED
                good          │
                              ▼
                       /sentinel/status
```

---

## 1. Модель монетизации (финал)

### Track A: Subscription (Human / Company)
- **$49 / 30 дней / skill**, x402 USDC на Base.
- Покрывает: registration, baseline-аудит, continuous monitoring, **все автоматические webhook-триггерные rescan'ы** (сколько бы patch-релизов автор ни выпустил за 30 дней), алерты, history retention.
- **Важно**: автоматический rescan после GitHub-релиза НИКОГДА не тарифицируется отдельно — иначе клиент получает неожиданный микроплатёж на каждый patch. Это входит в подписку как entitlement.

### Track B: Pay-per-call (Agent-to-agent / manual actions only)
Разовая оплата, **без credits, без баланса, без аккаунтов** — каждый вызов сам по себе x402-транзакция:
- `POST /sentinel/scan` со `scan_type=update_rescan` — `$0.50` (ручной/по требованию, вне цикла подписки)
- `scan_type=deep_scan` — `$2`
- `scan_type=forensic_audit` — `$10`

Это применяется **только** к ручным/emergency вызовам. Автоматические webhook-rescan'ы — не сюда, см. Track A.

### Auto-block — MVP-реализация
- **Не вмешиваемся в чужой x402 payment flow.** Никакой интеграции с Circuit Breaker сейчас — явно отложено на будущее.
- При регрессии Sentinel просто переводит `skills.status = 'blocked'`.
- Потребитель сам решает, что делать, вызвав `GET /sentinel/status/{skill}`:
```json
{
  "status": "blocked",
  "reason": "security_regression",
  "version": "2.4.1",
  "baseline_version": "2.4.0"
}
```

---

## 2. Регрессия — точное условие

```
regression = (current.verdict == "FAIL") OR (current.score < policy.score_threshold)
```

Пример: baseline verdict=PASS score=0.94; новая версия verdict=PASS score=0.71; threshold=0.80 → **REGRESSION**, несмотря на формальный PASS. Порог `score_threshold` задаётся в `policy` при регистрации (дефолт — зафиксировать разумное значение, например 0.80, если владелец не указал своё).

---

## 3. Версионирование — три раздельных состояния (не перезаписывать!)

```
immutable_baseline  -- первая доверенная версия, устанавливается один раз при регистрации, НИКОГДА не переписывается
last_known_good      -- последняя версия, успешно прошедшая проверку (обновляется при каждом non-regression скане)
current               -- последняя просканированная версия (обновляется при каждом скане, включая regression)
```

Критично: если новая версия — регрессия, `current` обновляется (чтобы видеть, что происходит), но `last_known_good` **не трогается** и `immutable_baseline` **не трогается никогда**. Это защищает от сценария, где скомпрометированная версия случайно становится новой точкой доверия.

---

## 4. Схема БД (SQLite, та же база что у dcl-webhook)

```sql
CREATE TABLE skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_full_name TEXT NOT NULL UNIQUE,
    owner_ref TEXT NOT NULL,

    -- три раздельных версионных состояния
    immutable_baseline_version TEXT,
    immutable_baseline_verdict TEXT,
    immutable_baseline_score REAL,
    immutable_baseline_audited_at TIMESTAMP,

    last_known_good_version TEXT,
    last_known_good_verdict TEXT,
    last_known_good_score REAL,
    last_known_good_audited_at TIMESTAMP,

    current_version TEXT,
    current_verdict TEXT,
    current_score REAL,
    current_audited_at TIMESTAMP,

    status TEXT DEFAULT 'unregistered',   -- unregistered / active / blocked / paused / expired
    block_reason TEXT,                     -- 'security_regression' и т.п., NULL если не blocked

    policy_json TEXT,                      -- {"score_threshold": 0.80, ...}
    plan_type TEXT,                        -- 'subscription' | NULL (pay-per-call не требует записи плана)
    plan_expires_at TIMESTAMP,

    webhook_secret TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE sentinel_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_id INTEGER REFERENCES skills(id),
    event_type TEXT,       -- registration / renewal / rescan_auto / rescan_paid / regression_blocked / unblocked
    scan_type TEXT,          -- 'update_rescan' | 'deep_scan' | 'forensic_audit' | NULL
    version TEXT,
    verdict TEXT,
    score REAL,
    amount_paid REAL,
    payer_ref TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 5. Эндпоинты

### `POST /sentinel/register` (x402 — Track A)
Вход: `repo_full_name`, `owner_ref`, опц. `policy` (score_threshold).
1. x402: 402 → $49 USDC на Base → verified.
2. Baseline-скан через dcl-core.
3. Записывает `immutable_baseline_*` = `last_known_good_*` = `current_*` = результат baseline-скана.
4. `status='active'`, `plan_type='subscription'`, `plan_expires_at=now()+30d`.
5. Возвращает `webhook_secret` для настройки GitHub webhook.

### `POST /sentinel/renew` (x402 — Track A)
Продление на 30 дней. Версионные состояния не трогает.

### `POST /sentinel/webhook/<webhook_secret>` (бесплатно в рамках активной подписки)
1. Verify HMAC-подпись.
2. Проверить `status=='active'` и `plan_expires_at > now()` — иначе no-op + уведомление о необходимости renew.
3. Rate-limit.
4. Скачать tarball релиза → audit через dcl-core → получить verdict+score.
5. Обновить `current_*`.
6. Проверить регрессию (раздел 2):
   - Если НЕТ регрессии → обновить `last_known_good_*` = `current_*`; событие `rescan_auto`.
   - Если ЕСТЬ регрессия → `status='blocked'`, `block_reason='security_regression'`; `last_known_good_*` НЕ трогать; событие `regression_blocked`; отправить алерт owner'у.

### `POST /sentinel/scan` (x402 — Track B, разовая оплата)
Вход: `repo_full_name`, `scan_type` (`update_rescan`/`deep_scan`/`forensic_audit`), `payer_ref`.
1. x402: 402 по цене scan_type → оплата → verified.
2. Синхронный скан через dcl-core, результат в ответе на тот же запрос (чистый machine-to-machine).
3. Не требует активной подписки — доступен даже для незарегистрированного skill.
4. Обновляет `current_*` (если skill уже зарегистрирован); не меняет `status` skill'а автоматически по regression-логике Track A (это ручная проверка, не встроенный монитор) — но результат возвращается сразу вызывающему, который сам решает, что делать.
5. Событие `rescan_paid`.

### `GET /sentinel/status/<repo_full_name>`
Отвечает на 5 вопросов:
```json
{
  "registered": true,
  "monitoring_active": true,
  "status": "blocked",
  "reason": "security_regression",
  "immutable_baseline_version": "2.4.0",
  "last_known_good_version": "2.4.0",
  "current_version": "2.4.1",
  "last_scanned_at": "...",
  "plan_type": "subscription",
  "plan_expires_at": "..."
}
```

---

## 6. Переиспользование существующего кода

- Audit-функция: та же, что в `/audit/` (dcl_core.py, v0.1.x pinned).
- x402 payment flow: паттерн с 18 уже работающих paid tools в mcp_server.py, wallet `0xb790ed3796194E5511C44411CF045F67E069cdC0` на Base.
- Rate limiting: sliding-window limiter — на `/sentinel/webhook/` и `/sentinel/scan`.
- PM2: роуты в существующем процессе `dcl-webhook`, `pm2 restart dcl-webhook --update-env`.

---

## 7. Явно отложено (не делать сегодня)

- Интеграция Sentinel → Circuit Breaker → x402 (блокировка реального payment flow агента).
- Credits/пакеты для pay-per-call.
- Мультиканальные алерты (сегодня — один канал, например webhook-callback на owner_ref).
- `/sentinel/renew` можно сделать после базового потока, если время поджимает.

---

## 8. План на сегодня (MVP)

1. Миграция БД: `skills`, `sentinel_events` (со всеми тремя версионными состояниями).
2. `/sentinel/register` — x402 $49/30d, baseline-скан, инициализация baseline=last_known_good=current.
3. `/sentinel/webhook/` — verify → rescan → regression-check (verdict OR threshold) → update last_known_good ИЛИ blocked.
4. `/sentinel/scan` — x402 pay-per-call, синхронный скан, 3 цены/scan_type.
5. `/sentinel/status/<repo>` — полный ответ по разделу 5.
6. Деплой, `pm2 restart dcl-webhook --update-env`.
7. E2E тест: register → эмулировать non-regression webhook release (проверить last_known_good обновился) → эмулировать regression release (проверить status=blocked, last_known_good НЕ изменился, immutable_baseline НЕ изменился) → отдельно вызвать `/sentinel/scan` напрямую с оплатой.
