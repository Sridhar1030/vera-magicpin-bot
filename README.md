# Vera AI Bot — magicpin AI Challenge

**Team**: master pushers | **Member**: Sridhar Pillai | **Contact**: sridharpillai75@gmail.com

## Approach

**Architecture**: Single FastAPI server with LLM-powered message composition and a deterministic multi-turn state machine.

Every message is composed by feeding the **full 4-context stack** (category, merchant, trigger, customer) into a carefully engineered system prompt. The prompt encodes the 5 scoring dimensions directly — specificity, category fit, merchant fit, trigger relevance, engagement compulsion — so the LLM optimizes for what the judge actually measures.

### Key Design Decisions

1. **Context-grounded, never fabricated** — The composer prompt injects only data pushed via `/v1/context`. No hardcoded facts, no hallucinated numbers. If the judge injects fresh digest items or updated performance data mid-test, the bot uses the latest version automatically (atomic version replacement on push).

2. **Derived metrics computed before LLM call** — Before calling the LLM, the bot pre-computes comparative insights: CTR gap vs peer average, estimated affected customer counts for supply alerts, seasonal reframing for expected dips. This gives the LLM concrete numbers to cite rather than asking it to do math.

3. **Trigger-kind dispatch with shape templates** — Each trigger kind (`research_digest`, `supply_alert`, `perf_dip`, `active_planning_intent`, `recall_due`, etc.) has a prescribed message shape in the system prompt. Research digests lead with source+finding+page. Supply alerts lead with batch numbers and compute affected customers. Planning intents deliver draft artifacts. This ensures structural consistency across runs.

4. **Deterministic multi-turn state machine** — Auto-reply detection, intent transitions, and hostile exits are handled via regex pattern matching (zero-latency, no LLM needed). Auto-reply escalates in 3 steps (acknowledge → wait 24h → end), hostile always exits gracefully, and intent commitment switches to action mode with a concrete deliverable.

5. **Category voice routing** — The system prompt includes voice rules per category:
   - **Dentists**: peer-clinical register, journal citations, "Dr. {name}"
   - **Salons**: warm-practical, fellow-aesthetician, emoji-light
   - **Restaurants**: fellow-operator, covers/footfall/AOV language
   - **Gyms**: coach-to-operator, ad spend/retention/conversion
   - **Pharmacies**: trustworthy-precise, molecule names, batch numbers

6. **Grounded fallbacks** — When the LLM is unavailable or rate-limited, the bot uses trigger-specific deterministic fallbacks that extract real data from context (merchant name, performance numbers, offer titles, peer stats) rather than generic templates.

## Model Choice

**Gemini 2.5 Flash** — primary model, chosen for:

- **Quality**: Strong structured-output adherence, good Hindi-English code-mix, follows complex system prompts faithfully
- **Speed**: Sub-3s composition latency, well within the judge's 30-second tick budget
- **Cost**: Free tier with generous token limits

**Groq (Llama 3.3 70B)** — fallback provider, used when Gemini quota is exhausted. ~1-2s latency, strong quality for composition tasks.

### Rate Limit Resilience

- Retry with exponential backoff (up to 3 retries on 429)
- Concurrency semaphore limits parallel LLM calls
- Trigger-specific grounded fallbacks ensure valid output even when LLM is fully unavailable
- Suppression keys prevent duplicate sends across ticks

## Tradeoffs

| Decision | Upside | Downside |
|---|---|---|
| Single system prompt (not per-trigger templates) | Handles unseen trigger kinds, fewer failure modes | Slightly less specialized per-trigger |
| Regex-based auto-reply/intent/hostile detection | Instant, deterministic, zero LLM cost | May miss creative phrasings (mitigated by broad pattern set) |
| In-memory context store | Zero-latency reads, no external dependency | State lost on restart (acceptable for eval window) |
| Hindi-English code-mix in prompt | Matches real Indian merchant communication | Quality depends on LLM's Hindi capability |
| Pre-computed derived metrics | LLM cites real numbers, higher specificity scores | Adds ~10ms computation per trigger |
| Grounded fallbacks over empty responses | Bot never times out, always returns valid JSON | Fallback messages score lower than LLM-composed ones |

## Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/healthz` | GET | Liveness probe + loaded context counts |
| `/v1/metadata` | GET | Team identity, model, approach |
| `/v1/context` | POST | Receive and version-store category/merchant/customer/trigger contexts |
| `/v1/tick` | POST | Compose proactive messages from available triggers (≤20 actions) |
| `/v1/reply` | POST | Handle merchant/customer replies with state-aware routing |

## Run Locally

```bash
pip install -r requirements.txt
export LLM_PROVIDER=gemini
export LLM_API_KEY=your-gemini-key
python3 bot.py
```

## Deployment

Deployed on Render at `https://vera-bot-k7al.onrender.com`

```bash
# Health check
curl https://vera-bot-k7al.onrender.com/v1/healthz

# Metadata
curl https://vera-bot-k7al.onrender.com/v1/metadata
```
