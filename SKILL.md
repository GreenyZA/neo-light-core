# Token Optimiser Skill

**Name:** token-optimiser  
**Description:** Budget-aware router T1 local → T2 free cloud → T3 paid. Plan first, confidence gate, learn after. Always loaded into Monica — intercepts agent calls and routes through the 3-tier budget system.

**Invoke:** Automatic on every non-trivial agent call. Explicit: `/token-optimiser` or "route this with token optimiser".

---

## When Invoked

Receive task. Always produce short Socratic plan + Token Budget Plan before multi-step work.

---

## Tiers

| Tier | Backend | Cost | Best for | Weak at |
|------|---------|------|----------|---------|
| **T1** | Local Ollama (qwen2.5:7b, qwen2.5-coder:7b, qwen2.5:3b) | $0 electricity | Private drafts, simple code, lists, transforms | Hard reasoning, huge context |
| **T2** | Free cloud (OmniRoute / OpenRouter / NVIDIA free / NOUS free tier) | $0 free tiers | Stronger free codegen, multi-model fallback | Quotas, flaky providers |
| **T3** | Grok / Solar / high-tier | Paid plan | Security, architecture, full context, final judgment | Burning tokens on grunt work |

---

## Confidence Gates

| Confidence | Action |
|------------|--------|
| ≥ 80% | T1 or free T2 — cheap path |
| 60–79% | Hybrid or ask human — cheap first, light polish |
| < 60% | T3 with stripped prompt + "cheap already attempted" note |

---

## Token Budget Plan (always before multi-step work)

```
- Real goal:
- Good enough:
- Cheapest viable tier + model:
- Confidence %:
- Risk if cheap path fails:
```

---

## How It Hooks Into Monica

The token-optimiser is loaded as an **always-on pre-routing layer**. When Monica receives a request:

1. Token-optimiser intercepts the task description
2. Runs confidence assessment (keyword matching + scope analysis)
3. Produces a Token Budget Plan
4. Routes to the selected tier (T1/T2/T3)
5. The appropriate Worker(s) execute on that tier's backend
6. After task completion, append Learning Log entry

**Toggle:**
- Enable: `token-optimiser.enable()`
- Disable: `token-optimiser.disable()` — instantly reverts to default routing (T2 as baseline)

---

## Dependent Skills

- **board-of-advisors** — Token-optimiser occupies the "Budget" seat. Board launches sub-agents, votes on routing.
- **stop-slop** — Run on marketing copy and user-facing docs during T2/T3 phases.
- **caveman** — Ultra-short mode for tight iteration and cheap routing summaries.

---

## Learning Log

After each routed task, append:

```
## YYYY-MM-DDTHH:MM:SS | task description
- Route: T1/T2/T3
- Confidence: X%
- Outcome: worked / partial / failed
- Next change: ...
```

---

## Safety

- Never route security/auth/credentials below T3 confidence cap (55%)
- Never commit API keys — keys only in local untracked paths
- Feature branches only — never push main via agent
- Backup before edits: `file.YYYYMMDDHHMMSS.bkp`
