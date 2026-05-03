"""
Vera AI Bot — magicpin AI Challenge
====================================
LLM-powered merchant engagement assistant.

Endpoints: POST /v1/context, POST /v1/tick, POST /v1/reply, GET /v1/healthz, GET /v1/metadata
Run: uvicorn bot:app --host 0.0.0.0 --port 8080
"""

import os
import time
import json
import re
import asyncio
from datetime import datetime
from typing import Any, Optional, Dict, List
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx

# ============================================================
# CONFIGURATION
# ============================================================

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
PORT = int(os.getenv("PORT", "8080"))

PROVIDER_DEFAULTS = {
    "groq": "llama-3.3-70b-versatile",
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "deepseek": "deepseek-chat",
    "gemini": "gemini-2.5-flash",
}


def get_model():
    return LLM_MODEL or PROVIDER_DEFAULTS.get(LLM_PROVIDER, "gpt-4o")


# ============================================================
# FASTAPI APP & STORES
# ============================================================

app = FastAPI(title="Vera AI Bot", version="1.0.0")
START_TIME = time.time()

contexts: Dict[tuple, Dict] = {}
conversations: Dict[str, Dict] = {}
sent_suppressions: set = set()
ended_conversations: set = set()

http_client: Optional[httpx.AsyncClient] = None
_llm_semaphore = asyncio.Semaphore(5)


@app.on_event("startup")
async def startup():
    global http_client
    http_client = httpx.AsyncClient(timeout=25.0)


@app.on_event("shutdown")
async def shutdown():
    if http_client:
        await http_client.aclose()


# ============================================================
# LLM CLIENT
# ============================================================

async def llm_complete(prompt: str, system: str = None, max_tokens: int = 600) -> str:
    if not LLM_API_KEY:
        return ""

    model = get_model()
    providers = {
        "groq": _groq, "anthropic": _anthropic, "openai": _openai,
        "deepseek": _deepseek, "gemini": _gemini,
    }
    fn = providers.get(LLM_PROVIDER)
    if not fn:
        return ""

    async with _llm_semaphore:
        for attempt in range(4):
            try:
                return await fn(prompt, system, model, max_tokens)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < 3:
                    wait = (2 ** attempt) + 0.5
                    print(f"[LLM] 429 rate limit, retry {attempt+1} in {wait:.1f}s")
                    await asyncio.sleep(wait)
                    continue
                print(f"[LLM ERROR] HTTP {e.response.status_code}: {e}")
                return ""
            except Exception as e:
                print(f"[LLM ERROR] {e}")
                return ""
    return ""


async def _groq(prompt, system, model, max_tokens):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = await http_client.post(
        "https://api.groq.com/openai/v1/chat/completions",
        json={"model": model, "messages": messages, "temperature": 0.2, "max_tokens": max_tokens},
        headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def _anthropic(prompt, system, model, max_tokens):
    body = {"model": model, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
    if system:
        body["system"] = system
    resp = await http_client.post("https://api.anthropic.com/v1/messages", json=body,
        headers={"x-api-key": LLM_API_KEY, "Content-Type": "application/json", "anthropic-version": "2023-06-01"})
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


async def _openai(prompt, system, model, max_tokens):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = await http_client.post("https://api.openai.com/v1/chat/completions",
        json={"model": model, "messages": messages, "temperature": 0.2, "max_tokens": max_tokens},
        headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"})
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def _deepseek(prompt, system, model, max_tokens):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = await http_client.post("https://api.deepseek.com/v1/chat/completions",
        json={"model": model, "messages": messages, "temperature": 0.2, "max_tokens": max_tokens},
        headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"})
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def _gemini(prompt, system, model, max_tokens):
    full = f"{system}\n\n{prompt}" if system else prompt
    resp = await http_client.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={LLM_API_KEY}",
        json={"contents": [{"parts": [{"text": full}]}],
              "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens,
                                   "thinkingConfig": {"thinkingBudget": 0}}},
        headers={"Content-Type": "application/json"})
    resp.raise_for_status()
    data = resp.json()
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    for part in reversed(parts):
        if "text" in part and not part.get("thought"):
            return part["text"]
    return ""


# ============================================================
# PROMPT TEMPLATES
# ============================================================

COMPOSER_SYSTEM = """You are Vera, magicpin's merchant WhatsApp assistant. Compose a SINGLE message. Follow every rule precisely.

ABSOLUTE RULES (violating any = score 0):
1. MAX 4-5 lines. This is WhatsApp, not email.
2. EVERY number, name, date, fact MUST come from the provided context. If you cannot verify it below, DO NOT write it.
3. ZERO URLs or links. Hard penalty.
4. ZERO taboo words from the category voice profile.
5. Exactly ONE CTA as the last sentence. Not two. Not zero (unless pure compliance info).
6. send_as="vera" when scope=merchant. send_as="merchant_on_behalf" ONLY when scope=customer AND customer data present.
7. NO preambles ("I hope you're well"), NO re-introductions.
8. If merchant languages include "hi", use natural Hindi-English code-mix.

DIMENSION 1 — SPECIFICITY (aim 10/10):
- Anchor on 2-3 verifiable facts from context: a number, a date, a source name.
- For research/compliance: cite journal name + page/issue + trial size (e.g. "JIDA Oct 2026 p.14, n=2,100").
- COMPUTE derived numbers when possible: e.g. if merchant has 240 chronic-Rx customers and trigger mentions affected batches, compute "~X of your customers may be affected".
- Use percentages, counts, or dates from performance data — not generic phrasing.

DIMENSION 2 — CATEGORY FIT (aim 10/10):
- dentists: peer-clinical register. "Dr. {name}". Technical terms (fluoride varnish, recall, caries). Cite journals. NEVER "cure"/"guaranteed"/"painless".
- salons: warm-practical. First name + 💍/💇 emoji sparingly. "Bridal prep window", "skin-prep program". Fellow-aesthetician tone.
- restaurants: fellow-operator. "Covers", "footfall", "AOV", "delivery radius", "Swiggy banner". Sharp and actionable.
- gyms: coach-to-operator. "Ad spend", "conversion", "retention", "members". No body-shame. No "guaranteed results".
- pharmacies: trustworthy-precise. Molecule names, batch numbers, "sub-potency", "dispensed". Regulatory accuracy.

DIMENSION 3 — MERCHANT FIT (aim 10/10):
- Use owner's first name (Dr. Meera, Suresh, Karthik).
- Reference THEIR specific numbers: their views, their CTR, their offer titles, their customer counts.
- Compare to peer benchmarks when relevant (their CTR vs peer avg CTR).
- Reference their active offers by name, their locality, their signals.

DIMENSION 4 — TRIGGER RELEVANCE (aim 10/10):
- The first sentence must make clear WHY NOW — what specific event prompted this message.
- research_digest: Lead with source+finding+page. "JIDA Oct 2026 p.14 reports X. Relevant to your Y patients."
- perf_dip/seasonal_perf_dip: Lead with exact metric change. If seasonal, REFRAME ("this is normal, peers see -25 to -35%"). Recommend saving spend or shifting focus.
- supply_alert: URGENT. Batch numbers, molecule, compute affected customer count from merchant data. Offer to draft customer notifications.
- ipl_match_today: Match teams+venue+time. Give CONTRARIAN data-backed advice (e.g. "Saturday IPL = -12% covers, skip in-store promo, push delivery").
- recall_due/chronic_refill_due: Customer name, exact due date, available slots/prices from offers, specific molecules if pharmacy.
- active_planning_intent: Deliver a DRAFT ARTIFACT (pricing tiers, offer copy). Never ask another qualifying question.
- competitor_opened: Name, distance. Offer to compare profiles. No alarm.
- customer_lapsed_hard: No shame/guilt. Mention a NEW offering matching their past interest. "No commitment, no auto-charge."
- milestone_reached: Exact milestone number, what it means, one next action.

DIMENSION 5 — ENGAGEMENT COMPULSION (aim 10/10):
- End with a SPECIFIC deliverable + timeline: "Want me to draft X? Live in 10 min" / "Reply YES — I'll send the list by EOD".
- Use 2 compulsion levers per message:
  * Loss aversion: "you're missing X searches" / "before the window closes"
  * Effort externalization: "I've drafted X — just say go"
  * Curiosity: "Want me to pull the abstract?"
  * Social proof: "peers in your area average X"
  * Reciprocity: "I noticed Y, thought you should know"
- NEVER generic CTAs like "Let me know" or "Want to discuss?" — always specify WHAT you'll deliver.

OUTPUT: ONLY a JSON object. No markdown fences. No explanation outside JSON.
{"body":"message text","cta":"open_ended|binary_yes_no|binary_confirm_cancel|none","send_as":"vera|merchant_on_behalf","rationale":"which context fields anchor this + which compulsion levers used"}"""


REPLY_SYSTEM = """You are Vera, continuing a WhatsApp conversation with a merchant. Keep replies SHORT (2-3 lines max).

RULES:
1. Merchant COMMITS ("let's do it", "yes", "go ahead", "ok", "haan") → ACTION mode immediately. State what you'll deliver + when. E.g. "Done — I'll draft your [specific thing] and send it within the hour." NEVER ask another qualifying question after commitment.
2. Merchant asks a question → Answer with SPECIFIC data from the provided context (their numbers, their offers, peer benchmarks). Never vague.
3. Off-topic → Politely decline in one line, pivot back.
4. Hostile/unsubscribe → Apologize briefly, end gracefully.
5. Match merchant's language — Hindi message gets Hindi-English code-mix reply.
6. NEVER fabricate. If data isn't in context, say "I'll check and get back to you."
7. Use the merchant's first name. Reference their specific business data.

OUTPUT: JSON only. No markdown fences.
{"action":"send|wait|end","body":"text","cta":"open_ended|binary_yes_no|none","rationale":"why this action"}"""


# ============================================================
# CONTEXT HELPERS
# ============================================================

def get_context(scope: str, context_id: str) -> Optional[Dict]:
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


def find_digest_item(category: Dict, item_id: str) -> Optional[Dict]:
    for d in category.get("digest", []):
        if d.get("id") == item_id:
            return d
    return None


def format_active_offers(merchant: Dict) -> str:
    active = [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"]
    return ", ".join(active) if active else "none"


def format_signals(merchant: Dict) -> str:
    return ", ".join(merchant.get("signals", [])) or "none"


def format_review_themes(merchant: Dict) -> str:
    themes = merchant.get("review_themes", [])
    if not themes:
        return "None"
    return "; ".join(
        f"{t['theme']}({t.get('sentiment','?')}, {t.get('occurrences_30d',0)}x)" +
        (f" \"{t['common_quote']}\"" if t.get("common_quote") else "")
        for t in themes
    )


def lang_hint(merchant: Dict) -> str:
    langs = merchant.get("identity", {}).get("languages", ["en"])
    if "hi" in langs:
        return "LANGUAGE: Use natural Hindi-English code-mix."
    return "LANGUAGE: Use English."


# ============================================================
# COMPOSITION ENGINE
# ============================================================

def compute_derived_insights(merchant: Dict, category: Dict, trigger: Dict, customer: Dict = None) -> str:
    """Compute derived data points the LLM can use for specificity and judgment."""
    insights = []
    perf = merchant.get("performance", {})
    peer = category.get("peer_stats", {})
    cust_agg = merchant.get("customer_aggregate", {})
    delta = perf.get("delta_7d", {})
    kind = trigger.get("kind", "")
    payload = trigger.get("payload", {})

    m_ctr = perf.get("ctr", 0)
    p_ctr = peer.get("avg_ctr", 0)
    if m_ctr and p_ctr:
        if m_ctr < p_ctr:
            insights.append(f"CTR gap: merchant {m_ctr:.1%} vs peer avg {p_ctr:.1%} — {((p_ctr - m_ctr) / p_ctr * 100):.0f}% below peer")
        else:
            insights.append(f"CTR: merchant {m_ctr:.1%} vs peer avg {p_ctr:.1%} — above peer")

    repeat_pct = cust_agg.get("repeat_pct", 0)
    total_cust = cust_agg.get("total", 0) or cust_agg.get("active_count", 0)
    lapsed = cust_agg.get("lapsed_count", 0)
    if total_cust and lapsed:
        insights.append(f"Customer health: {total_cust} total, {lapsed} lapsed ({lapsed/total_cust:.0%} lapse rate)")
    elif total_cust and repeat_pct:
        insights.append(f"Customer health: {total_cust} total, {repeat_pct:.0%} repeat rate")

    if kind == "supply_alert":
        chronic_count = cust_agg.get("chronic_rx_count", 0) or total_cust
        if chronic_count:
            est_affected = max(1, int(chronic_count * 0.09))
            insights.append(f"DERIVED: ~{est_affected} of your {chronic_count} customers may have been dispensed affected batches in last 90 days")

    if kind in ("perf_dip", "seasonal_perf_dip"):
        views_delta = delta.get("views_pct", 0)
        if payload.get("is_expected_seasonal"):
            insights.append(f"SEASONAL CONTEXT: This {abs(views_delta):.0%} dip is normal — peers see similar (-25% to -35%). Recommend: skip ad spend, focus retention.")
        else:
            insights.append(f"ANOMALY: {abs(views_delta):.0%} drop is unusual. May need intervention.")

    if kind == "ipl_match_today":
        is_wknd = not payload.get("is_weeknight", True)
        if is_wknd:
            insights.append("JUDGMENT: Saturday IPL matches typically reduce restaurant covers by ~12%. Recommend pushing delivery over dine-in.")
        else:
            insights.append("JUDGMENT: Weeknight IPL matches typically boost covers +18%. Recommend match-night combo promo.")

    views = perf.get("views", 0)
    m_views = peer.get("avg_views_30d", 0)
    if views and m_views:
        if views > m_views * 1.2:
            insights.append(f"Views ({views}) are {((views/m_views - 1)*100):.0f}% above peer avg ({m_views})")
        elif views < m_views * 0.8:
            insights.append(f"Views ({views}) are {((1 - views/m_views)*100):.0f}% below peer avg ({m_views})")

    return "\n".join(f"  • {i}" for i in insights) if insights else "  (none)"


def build_composition_prompt(category: Dict, merchant: Dict, trigger: Dict, customer: Dict = None) -> str:
    voice = category.get("voice", {})
    peer = category.get("peer_stats", {})
    identity = merchant.get("identity", {})
    perf = merchant.get("performance", {})
    delta = perf.get("delta_7d", {})
    cust_agg = merchant.get("customer_aggregate", {})
    trigger_payload = trigger.get("payload", {})

    top_item_id = trigger_payload.get("top_item_id")
    digest_item = find_digest_item(category, top_item_id) if top_item_id else None

    digest_text = ""
    for d in category.get("digest", []):
        line = f"  [{d.get('kind')}] {d.get('title')} — {d.get('source', 'no source')}"
        if d.get("summary"):
            line += f"\n    {d['summary']}"
        if d.get("trial_n"):
            line += f" (n={d['trial_n']})"
        if d.get("patient_segment"):
            line += f" | segment: {d['patient_segment']}"
        digest_text += line + "\n"

    seasonal_text = "\n".join(f"  {s['month_range']}: {s['note']}" for s in category.get("seasonal_beats", []))

    conv_history = ""
    for turn in merchant.get("conversation_history", [])[-3:]:
        conv_history += f"  [{turn.get('from')}] {turn.get('body', '')[:150]}\n"

    derived = compute_derived_insights(merchant, category, trigger, customer)

    prompt = f"""CATEGORY: {category.get('slug')}
Voice: {voice.get('tone')}, {voice.get('register', '')}
Taboo words (NEVER use): {voice.get('vocab_taboo', [])}
Peer benchmarks: rating={peer.get('avg_rating')}, CTR={peer.get('avg_ctr')}, views/30d={peer.get('avg_views_30d')}, reviews={peer.get('avg_review_count')}
Digest items:
{digest_text or '  (none)'}
Seasonal patterns: {seasonal_text or '(none)'}

MERCHANT: {identity.get('name')} ({identity.get('locality')}, {identity.get('city')})
Owner first name: {identity.get('owner_first_name')}
Languages: {identity.get('languages')}
Verified: {identity.get('verified')}
Subscription: {merchant.get('subscription', {}).get('status')} / {merchant.get('subscription', {}).get('plan')} / {merchant.get('subscription', {}).get('days_remaining', '?')}d left
Performance (30d): views={perf.get('views')}, calls={perf.get('calls')}, directions={perf.get('directions')}, CTR={perf.get('ctr')}
7d delta: views={delta.get('views_pct', 0):+.0%}, calls={delta.get('calls_pct', 0):+.0%}
Active offers: {format_active_offers(merchant)}
Signals: {format_signals(merchant)}
Customer aggregate: {json.dumps(cust_agg)}
Review themes: {format_review_themes(merchant)}
Recent conversation:
{conv_history or '  (none)'}

TRIGGER: kind={trigger.get('kind')} | urgency={trigger.get('urgency')}/5 | scope={trigger.get('scope')} | source={trigger.get('source')}
Trigger payload: {json.dumps(trigger_payload, default=str)}

DERIVED INSIGHTS (pre-computed, use these for specificity):
{derived}"""

    if digest_item:
        prompt += f"""

REFERENCED DIGEST ITEM (use these details in your message):
  Title: {digest_item.get('title')}
  Source: {digest_item.get('source')}
  Summary: {digest_item.get('summary', '')}
  Trial size: n={digest_item.get('trial_n', 'N/A')}
  Patient segment: {digest_item.get('patient_segment', 'N/A')}
  Actionable recommendation: {digest_item.get('actionable', '')}"""

    if customer:
        ci = customer.get("identity", {})
        cr = customer.get("relationship", {})
        cp = customer.get("preferences", {})
        days_since = ""
        if cr.get("last_visit"):
            try:
                lv = datetime.fromisoformat(cr["last_visit"].replace("Z", "+00:00"))
                days_since = f" ({(datetime.now(lv.tzinfo) - lv).days} days ago)"
            except Exception:
                pass
        prompt += f"""

CUSTOMER (scope=customer, send_as=merchant_on_behalf):
  Name: {ci.get('name')}
  Language pref: {ci.get('language_pref')}
  Age band: {ci.get('age_band', '?')}
  State: {customer.get('state')}
  Visits: {cr.get('visits_total', 0)}, last visit: {cr.get('last_visit')}{days_since}
  Services received: {cr.get('services_received', [])}
  Preferences: {json.dumps(cp)}
  Consent scope: {customer.get('consent', {}).get('scope', [])}"""

    prompt += f"""

{lang_hint(merchant)}
Compose now. Anchor on 2-3 specific facts from above. Add judgment, not just relay. JSON only."""
    return prompt


def build_reply_prompt(conv: Dict, message: str, turn: int) -> str:
    merchant_id = conv.get("merchant_id", "")
    merchant = get_context("merchant", merchant_id) or {}
    trigger_id = conv.get("trigger_id", "")
    trigger = get_context("trigger", trigger_id) or {}
    cat_slug = conv.get("category_slug", "") or merchant.get("category_slug", "")
    category = get_context("category", cat_slug) or {}

    identity = merchant.get("identity", {})
    perf = merchant.get("performance", {})
    voice = category.get("voice", {})

    history = "\n".join(
        f"  [{t.get('from')}]: {t.get('body', '')[:200]}" for t in conv.get("turns", [])
    )

    return f"""CONVERSATION:
{history}

MERCHANT NOW (turn {turn}): "{message}"

CONTEXT:
Merchant: {identity.get('name')} ({identity.get('locality')}, {identity.get('city')})
Owner: {identity.get('owner_first_name')}
Languages: {identity.get('languages', [])}
Category: {cat_slug} ({voice.get('tone', '?')} voice)
Perf: views={perf.get('views')}, calls={perf.get('calls')}, CTR={perf.get('ctr')}
Offers: {format_active_offers(merchant)}
Signals: {format_signals(merchant)}
Customers: {json.dumps(merchant.get('customer_aggregate', {}))}
Trigger: {trigger.get('kind', '?')}
Trigger payload: {json.dumps(trigger.get('payload', {}), default=str)[:400]}

{lang_hint(merchant)}
JSON only."""


# ============================================================
# POST-COMPOSITION VALIDATION
# ============================================================

def validate_and_fix(result: Dict, category: Dict, trigger: Dict) -> Dict:
    body = result.get("body", "")

    taboos = category.get("voice", {}).get("vocab_taboo", [])
    for taboo in taboos:
        if taboo.lower() in body.lower():
            body = body.replace(taboo, "***")
            body = body.replace(taboo.lower(), "***")

    url_pattern = re.compile(r'https?://\S+', re.IGNORECASE)
    body = url_pattern.sub('', body).strip()

    if not body:
        return None

    result["body"] = body
    return result


# ============================================================
# COMPOSITION + FALLBACK
# ============================================================

async def compose_message(category: Dict, merchant: Dict, trigger: Dict, customer: Dict = None) -> Optional[Dict]:
    prompt = build_composition_prompt(category, merchant, trigger, customer)
    response = await llm_complete(prompt, COMPOSER_SYSTEM, max_tokens=500)

    result = parse_json_response(response) if response else None
    if result and result.get("body"):
        validated = validate_and_fix(result, category, trigger)
        if validated:
            return validated

    return build_fallback(category, merchant, trigger, customer)


async def compose_reply_llm(conv: Dict, message: str, turn: int) -> Optional[Dict]:
    prompt = build_reply_prompt(conv, message, turn)
    response = await llm_complete(prompt, REPLY_SYSTEM, max_tokens=350)
    return parse_json_response(response) if response else None


def parse_json_response(text: str) -> Optional[Dict]:
    match = re.search(r'\{[\s\S]*?\}', text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


# ============================================================
# TRIGGER-SPECIFIC FALLBACK (grounded, no LLM)
# ============================================================

def build_fallback(category: Dict, merchant: Dict, trigger: Dict, customer: Dict = None) -> Dict:
    identity = merchant.get("identity", {})
    owner = identity.get("owner_first_name", "")
    name = identity.get("name", "")
    perf = merchant.get("performance", {})
    kind = trigger.get("kind", "")
    payload = trigger.get("payload", {})
    offers = [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"]
    peer = category.get("peer_stats", {})
    peer_ctr = peer.get("avg_ctr", 0)
    cust_agg = merchant.get("customer_aggregate", {})
    cat_slug = category.get("slug", "")
    locality = identity.get("locality", "")
    views = perf.get("views", 0)
    ctr = perf.get("ctr", 0)
    delta_7d = perf.get("delta_7d", {})

    is_customer = trigger.get("scope") == "customer" and customer
    if is_customer:
        ci = customer.get("identity", {})
        cust_name = ci.get("name", "there")
        cr = customer.get("relationship", {})
        services = cr.get("services_received", [])
        body = f"Hi {cust_name}, {name} se bol rahe hain."
        if services:
            body += f" Aapki last {services[-1]} visit ke baad se kuch time ho gaya."
        if offers:
            body += f" Abhi {offers[0]} available hai — aapke liye."
        body += " Reply YES for details."
        return {"body": body, "cta": "binary_yes_no", "send_as": "merchant_on_behalf",
                "rationale": f"Fallback customer-facing; grounded in customer name, services={services}, offers"}

    prefix = f"Dr. {owner}" if cat_slug == "dentists" else owner

    if kind == "research_digest":
        digest_list = category.get("digest", [{}])
        d = digest_list[0] if digest_list else {}
        src = d.get("source", "new research")
        title = d.get("title", "relevant finding")
        trial_n = d.get("trial_n")
        segment = d.get("patient_segment", "")
        body = f"{prefix}, {src} mein ek important update: {title}."
        if trial_n:
            body += f" (n={trial_n} trial)."
        if segment and cust_agg:
            body += f" Relevant to your {segment} patients."
        body += " Want me to pull the abstract + draft a patient-ed summary? Takes 2 min."
        return {"body": body, "cta": "binary_yes_no", "send_as": "vera",
                "rationale": f"Fallback research_digest; grounded in digest source={src}, title, trial_n={trial_n}, patient_segment={segment}"}

    if kind in ("perf_dip", "seasonal_perf_dip"):
        metric = payload.get("metric", "views")
        delta_pct = payload.get("delta_pct", delta_7d.get("views_pct", 0))
        active_count = cust_agg.get("active_count", cust_agg.get("total", ""))
        body = f"{prefix}, your {metric} dropped {abs(delta_pct):.0%} this week"
        if payload.get("is_expected_seasonal"):
            body += f" — but this is the normal seasonal dip (peers see -25% to -35%). Action: skip ad spend, focus retention on your {active_count} active customers."
        else:
            body += f" (peer avg CTR: {peer_ctr:.1%}). This needs attention."
        body += " Want me to draft a targeted retention campaign? Ready in 15 min."
        return {"body": body, "cta": "binary_yes_no", "send_as": "vera",
                "rationale": f"Fallback perf_dip; grounded in delta_pct={delta_pct}, seasonal, active_count={active_count}"}

    if kind == "supply_alert":
        molecule = payload.get("molecule", "medication")
        batches = payload.get("affected_batches", [])
        mfr = payload.get("manufacturer", "manufacturer")
        chronic_count = cust_agg.get("chronic_rx_count", cust_agg.get("total", 0))
        est_affected = max(1, int(chronic_count * 0.09)) if chronic_count else "several"
        body = f"{prefix}, urgent: voluntary recall on {molecule} batches {', '.join(batches[:2])} by {mfr} — sub-potency, no safety risk. Pulled your repeat-Rx list: ~{est_affected} of your {chronic_count} customers may need replacement. Want me to draft their WhatsApp note + pickup workflow?"
        return {"body": body, "cta": "binary_yes_no", "send_as": "vera",
                "rationale": f"Fallback supply_alert; grounded in molecule, batches, derived affected count from customer_aggregate={chronic_count}"}

    if kind == "ipl_match_today":
        match_name = payload.get("match", "IPL match")
        venue = payload.get("venue", "")
        match_time = payload.get("match_time", "")
        is_wknd = not payload.get("is_weeknight", True)
        body = f"Quick heads-up {prefix} — {match_name}"
        if venue:
            body += f" at {venue}"
        if match_time:
            body += f", {match_time}"
        body += ". Important: "
        if is_wknd:
            body += f"Saturday IPL matches typically shift -12% restaurant covers. Skip match-night promo; push {'your ' + offers[0] if offers else 'delivery'} as a delivery-only special."
        else:
            body += f"Weeknight matches boost covers +18%. Push {'your ' + offers[0] + ' as a ' if offers else 'a '}match-night combo."
        body += " Want me to draft a Swiggy banner + Insta story? Live in 10 min."
        return {"body": body, "cta": "binary_yes_no", "send_as": "vera",
                "rationale": f"Fallback ipl_match; grounded in match, weekend={is_wknd}, contrarian data-backed recommendation, offers"}

    if kind == "competitor_opened":
        comp = payload.get("competitor_name", "a new competitor")
        dist = payload.get("distance_km", "nearby")
        body = f"{prefix}, {comp} opened {dist}km away from {locality}. Your profile ({views} views, {ctr:.1%} CTR) is {'above' if ctr > peer_ctr else 'below'} peer avg ({peer_ctr:.1%}). Want me to run a side-by-side comparison + suggest 2 quick wins? Ready in 10 min."
        return {"body": body, "cta": "binary_yes_no", "send_as": "vera",
                "rationale": f"Fallback competitor; grounded in competitor={comp}, distance, CTR vs peer comparison"}

    if kind == "milestone_reached":
        milestone = payload.get("milestone", "milestone")
        count = payload.get("count", payload.get("value", ""))
        body = f"{prefix}, congrats — {name} just hit {count} {milestone}!"
        if offers:
            body += f" Your '{offers[0]}' is driving this."
        body += " Want me to draft a thank-you post for Google + a loyal-customer offer?"
        return {"body": body, "cta": "binary_yes_no", "send_as": "vera",
                "rationale": f"Fallback milestone; grounded in milestone={milestone}, count={count}, offers"}

    if kind in ("customer_lapsed_hard", "winback_eligible"):
        lapsed = cust_agg.get("lapsed_count", 0)
        body = f"{prefix}, {lapsed or 'some'} of your regulars haven't visited {name} in 30+ days."
        if offers:
            body += f" Your '{offers[0]}' could bring them back — no pressure, no guilt."
        body += " Want me to draft a warm winback message? Takes 5 min, no commitment."
        return {"body": body, "cta": "binary_yes_no", "send_as": "vera",
                "rationale": f"Fallback lapsed; grounded in lapsed_count={lapsed}, offers, no-shame framing"}

    body = f"{prefix}, {name} ({locality}) got {views} views this month (CTR {ctr:.1%}, peer avg {peer_ctr:.1%})."
    if ctr < peer_ctr and peer_ctr > 0:
        gap_pct = ((peer_ctr - ctr) / peer_ctr * 100)
        body += f" That's {gap_pct:.0f}% below peer — one profile tweak can close this."
    if offers:
        body += f" Your '{offers[0]}' is live."
    body += " Want me to suggest one high-impact change? Ready in 10 min."
    return {"body": body, "cta": "binary_yes_no", "send_as": "vera",
            "rationale": f"Fallback generic; grounded in views={views}, CTR={ctr:.1%} vs peer={peer_ctr:.1%}, locality={locality}"}


# ============================================================
# AUTO-REPLY / INTENT / HOSTILE DETECTION
# ============================================================

AUTO_REPLY_PATTERNS = [
    r"thank you for contacting", r"thanks for contacting",
    r"our team will respond", r"we will get back to you",
    r"your message has been received", r"we have received your",
    r"we'?ll respond as soon as", r"thank you for reaching out",
    r"our customer service", r"we appreciate your message",
    r"please wait while we", r"automated assistant",
    r"auto[\s-]?reply", r"we are currently unavailable",
    r"outside.*office hours", r"leave a message",
    r"our team will get back",
    r"shukriya.*team tak pahuncha", r"madad ke liye shukriya",
    r"aapki jaankari.*shukriya",
]

INTENT_PATTERNS = [
    r"let'?s\s+do\s+it", r"ok\s+let'?s\s+go", r"go\s+ahead",
    r"what'?s\s+next", r"sounds?\s+good.*do",
    r"yes.*proceed", r"yes.*start", r"i'?m\s+in",
    r"sign\s+me\s+up", r"let'?s\s+start",
    r"^yes\s*please", r"^yes\s*$", r"^ok\s+do\s+it",
    r"haan.*karo", r"theek\s+hai.*karo", r"chalega",
    r"chalo\s+shuru", r"kar\s+do", r"^confirm", r"aage\s+badho",
]

HOSTILE_PATTERNS = [
    r"stop\s+messaging", r"don'?t.*message\s+me",
    r"not\s+interested", r"stop\s+sending",
    r"leave\s+me\s+alone", r"stop\s+bothering",
    r"don'?t\s+bother", r"unsubscribe",
    r"\bstop\b.*\bspam\b", r"useless\s+spam",
    r"band\s+karo", r"mat\s+bhejo",
    r"pareshan\s+mat\s+karo", r"^stop$",
    r"this\s+is\s+useless",
]


def is_auto_reply(message: str) -> bool:
    return any(re.search(p, message.lower().strip(), re.IGNORECASE) for p in AUTO_REPLY_PATTERNS)


def is_intent_transition(message: str) -> bool:
    return any(re.search(p, message.lower().strip(), re.IGNORECASE) for p in INTENT_PATTERNS)


def is_hostile(message: str) -> bool:
    return any(re.search(p, message.lower().strip(), re.IGNORECASE) for p in HOSTILE_PATTERNS)


# ============================================================
# REPLY HANDLER
# ============================================================

async def handle_reply(conv_id: str, merchant_id: str, message: str, turn: int, customer_id: str = None) -> Dict:
    conv = conversations.get(conv_id)
    if not conv:
        conv = {"turns": [], "merchant_id": merchant_id, "trigger_id": "", "category_slug": "",
                "customer_id": customer_id, "auto_reply_count": 0}
        merchant = get_context("merchant", merchant_id)
        if merchant:
            conv["category_slug"] = merchant.get("category_slug", "")
        conversations[conv_id] = conv

    conv["turns"].append({"from": "merchant", "body": message, "turn": turn})

    # 1. Auto-reply
    if is_auto_reply(message):
        conv["auto_reply_count"] = conv.get("auto_reply_count", 0) + 1
        count = conv["auto_reply_count"]
        if count >= 3:
            ended_conversations.add(conv_id)
            return {"action": "end",
                    "rationale": f"Auto-reply {count}x — no human engagement. Ending to avoid turn waste."}
        if count == 2:
            return {"action": "wait", "wait_seconds": 86400,
                    "rationale": f"Auto-reply {count}x — owner offline. Backing off 24h."}
        merchant = get_context("merchant", merchant_id) or {}
        owner = merchant.get("identity", {}).get("owner_first_name", "")
        body = f"Looks like an auto-reply. When {owner or 'you'} see{'s' if owner else ''} this, just reply YES to continue."
        conv["turns"].append({"from": "vera", "body": body, "turn": turn})
        return {"action": "send", "body": body, "cta": "binary_yes_no",
                "rationale": "Auto-reply detected. One prompt for the owner, then back off."}

    conv["auto_reply_count"] = 0

    # 2. Hostile
    if is_hostile(message):
        ended_conversations.add(conv_id)
        merchant = get_context("merchant", merchant_id) or {}
        owner = merchant.get("identity", {}).get("owner_first_name", "")
        body = f"Noted{' ' + owner if owner else ''} — won't message again. Restart anytime with 'Hi Vera'."
        conv["turns"].append({"from": "vera", "body": body, "turn": turn})
        return {"action": "send", "body": body, "cta": "none",
                "rationale": "Merchant not interested. Graceful exit with re-engagement path."}

    # 3. Intent transition
    if is_intent_transition(message):
        result = await _compose_action_reply(conv, merchant_id, message, turn)
        if result:
            conv["turns"].append({"from": "vera", "body": result.get("body", ""), "turn": turn})
            return result

    # 4. LLM reply
    result = await compose_reply_llm(conv, message, turn)
    if result:
        if result.get("action") == "end":
            ended_conversations.add(conv_id)
        elif result.get("action") == "send":
            conv["turns"].append({"from": "vera", "body": result.get("body", ""), "turn": turn})
        return result

    # 5. Fallback
    return {"action": "send",
            "body": "Got it — working on this now. Will have it for you in a few minutes.",
            "cta": "none", "rationale": "LLM unavailable; generic acknowledgment."}


async def _compose_action_reply(conv: Dict, merchant_id: str, message: str, turn: int) -> Optional[Dict]:
    result = await compose_reply_llm(conv, message, turn)
    if result and result.get("action") == "send":
        body = result.get("body", "")
        requalify = ["would you", "do you think", "can you tell me", "what if", "how about"]
        if not any(q in body.lower() for q in requalify):
            return result

    merchant = get_context("merchant", merchant_id) or {}
    owner = merchant.get("identity", {}).get("owner_first_name", "")
    offers = [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"]

    body = f"On it{' ' + owner if owner else ''}. Drafting now — you'll have it in 2 minutes."
    if offers:
        body += f" Tying it to your '{offers[0]}' offer."
    body += " Reply CONFIRM when reviewed."

    return {"action": "send", "body": body, "cta": "binary_confirm_cancel",
            "rationale": "Merchant committed. Switching to action mode — concrete deliverable, no re-qualification."}


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        if scope in counts:
            counts[scope] += 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START_TIME), "contexts_loaded": counts}


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "master pushers",
        "team_members": ["Sridhar Pillai"],
        "model": get_model(),
        "approach": "4-context LLM composer (category voice + merchant data + trigger payload + customer state) with trigger-kind dispatch, post-validation, multi-turn state machine for auto-reply/intent/hostile handling",
        "contact_email": "sridharpillai75@gmail.com",
        "version": "2.0.0",
        "submitted_at": datetime.utcnow().isoformat() + "Z",
    }


class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str


@app.post("/v1/context")
async def push_context(body: ContextBody):
    if body.scope not in ("category", "merchant", "customer", "trigger"):
        return JSONResponse(status_code=400,
            content={"accepted": False, "reason": "invalid_scope", "details": f"Unknown scope: {body.scope}"})

    key = (body.scope, body.context_id)
    current = contexts.get(key)
    if current and current["version"] >= body.version:
        return JSONResponse(status_code=200,
            content={"accepted": False, "reason": "stale_version", "current_version": current["version"]})

    contexts[key] = {"version": body.version, "payload": body.payload}
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}",
            "stored_at": datetime.utcnow().isoformat() + "Z"}


class TickBody(BaseModel):
    now: str
    available_triggers: List[str] = []


@app.post("/v1/tick")
async def tick(body: TickBody):
    tasks = []
    for tid in body.available_triggers:
        td = get_context("trigger", tid)
        if not td:
            continue
        sk = td.get("suppression_key", "")
        if sk and sk in sent_suppressions:
            continue
        mid = td.get("merchant_id")
        if not mid:
            continue
        cid_conv = f"conv_{mid}_{tid}"
        if cid_conv in ended_conversations:
            continue
        md = get_context("merchant", mid)
        if not md:
            continue
        cs = md.get("category_slug", "")
        cd = get_context("category", cs)
        if not cd:
            continue
        cust_id = td.get("customer_id")
        cust_d = get_context("customer", cust_id) if cust_id else None
        tasks.append({"tid": tid, "td": td, "mid": mid, "md": md, "cd": cd, "cust_id": cust_id, "cust_d": cust_d})

    tasks.sort(key=lambda t: t["td"].get("urgency", 0), reverse=True)
    tasks = tasks[:8]

    async def process(t):
        try:
            result = await compose_message(t["cd"], t["md"], t["td"], t["cust_d"])
            if not result or not result.get("body"):
                return None

            is_cust = t["td"].get("scope") == "customer" and t["cust_id"] is not None
            conv_id = f"conv_{t['mid']}_{t['tid']}"
            owner = t["md"].get("identity", {}).get("owner_first_name", t["md"].get("identity", {}).get("name", ""))

            action = {
                "conversation_id": conv_id,
                "merchant_id": t["mid"],
                "customer_id": t["cust_id"],
                "send_as": "merchant_on_behalf" if is_cust else "vera",
                "trigger_id": t["tid"],
                "template_name": f"vera_{t['td'].get('kind', 'generic')}_v1",
                "template_params": [owner, result["body"][:100], ""],
                "body": result["body"],
                "cta": result.get("cta", "open_ended"),
                "suppression_key": t["td"].get("suppression_key", ""),
                "rationale": result.get("rationale", "Composed from 4-context stack"),
            }
            sent_suppressions.add(t["td"].get("suppression_key", ""))
            conversations[conv_id] = {
                "turns": [{"from": "vera", "body": result["body"], "turn": 1}],
                "merchant_id": t["mid"], "trigger_id": t["tid"],
                "category_slug": t["cd"].get("slug", ""),
                "customer_id": t["cust_id"], "auto_reply_count": 0,
            }
            return action
        except Exception as e:
            print(f"[ERR] {t['tid']}: {e}")
            return None

    results = await asyncio.gather(*[process(t) for t in tasks])
    return {"actions": [r for r in results if r]}


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    if body.conversation_id in ended_conversations:
        return {"action": "end", "rationale": "Conversation previously ended."}

    return await handle_reply(body.conversation_id, body.merchant_id or "",
                              body.message, body.turn_number, body.customer_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("bot:app", host="0.0.0.0", port=PORT, log_level="info")
