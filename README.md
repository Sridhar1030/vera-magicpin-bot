# Vera AI Bot — magicpin AI Challenge

## Approach

**Architecture**: Single FastAPI server with LLM-powered message composition and a deterministic multi-turn state machine.

Every message is composed by feeding the **full 4-context stack** (category, merchant, trigger, customer) into a carefully engineered system prompt. The prompt encodes the 5 scoring dimensions directly — specificity, category fit, merchant fit, trigger relevance, engagement compulsion — so the LLM optimizes for what the judge actually measures.

**Key design decisions**:

1. **Context-grounded, never fabricated** — The composer prompt injects only data that was pushed via `/v1/context`. No hardcoded facts, no hallucinated numbers. If the judge injects fresh digest items or updated performance data mid-test, the bot uses the latest version automatically (atomic version replacement on push).

2. **Deterministic multi-turn state machine** — Auto-reply detection, intent transitions, and hostile exits are handled via regex pattern matching (zero-latency, no LLM needed). This guarantees consistent behavior: auto-reply escalates in 3 steps (acknowledge → wait 24h → end), hostile always exits gracefully, and intent commitment always switches to action mode.

3. **Category voice routing** — The system prompt includes voice rules per category (peer-clinical for dentists, warm-practical for salons, fellow-operator for restaurants, coach-to-member for gyms, trustworthy-precise for pharmacies). The LLM matches tone without separate prompt templates per vertical.

4. **Parallel trigger processing** — `/v1/tick` evaluates all available triggers concurrently via `asyncio.gather`, sorted by urgency. Suppression keys prevent duplicate sends. This keeps total tick latency under 3 seconds even with 5+ triggers.

## Model Choice

**Groq (Llama 3.3 70B Versatile)** — chosen for three reasons:

- **Speed**: ~1-2s per composition. The judge's tick/reply latency budget is 10s; Groq consistently lands under 3s even with parallel calls.
- **Quality**: 70B parameter model produces specific, voice-matched, context-grounded messages competitive with frontier models for structured composition tasks.
- **Reliability**: Groq's inference infrastructure has high uptime and consistent latency — critical for a bot that must stay live for 3 days of evaluation.

## Tradeoffs

| Decision | Upside | Downside |
|---|---|---|
| Single system prompt (not per-trigger-type) | Simpler, fewer failure modes, handles unseen trigger kinds | Slightly less specialized than dedicated prompt templates |
| Regex-based auto-reply/intent/hostile detection | Instant, deterministic, no LLM cost | May miss creative phrasings (mitigated by broad pattern set) |
| In-memory context store | Zero-latency reads, no external dependency | State lost on restart (acceptable for 3-day eval window) |
| Hindi-English code-mix encouraged in prompt | Matches real Indian merchant communication patterns | Quality depends on LLM's Hindi capability |
| Fallback composition when LLM is unavailable | Bot never times out, always returns valid JSON | Fallback messages score lower (~5/dimension vs ~8/dimension) |

## Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/healthz` | GET | Liveness + context count |
| `/v1/metadata` | GET | Bot identity and approach |
| `/v1/context` | POST | Receive and version-store contexts |
| `/v1/tick` | POST | Compose proactive messages from active triggers |
| `/v1/reply` | POST | Handle merchant replies with state-aware routing |

## Run Locally

```bash
pip install -r requirements.txt
export LLM_API_KEY=your-groq-key
python3 bot.py
```
