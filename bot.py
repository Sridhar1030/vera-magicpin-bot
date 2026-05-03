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
    "gemini": "gemini-2.0-flash",
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

async def llm_complete(prompt: str, system: str = None, max_tokens: int = 800) -> str:
    if not LLM_API_KEY:
        return ""

    model = get_model()
    try:
        if LLM_PROVIDER == "groq":
            return await _groq(prompt, system, model, max_tokens)
        elif LLM_PROVIDER == "anthropic":
            return await _anthropic(prompt, system, model, max_tokens)
        elif LLM_PROVIDER == "openai":
            return await _openai(prompt, system, model, max_tokens)
        elif LLM_PROVIDER == "deepseek":
            return await _deepseek(prompt, system, model, max_tokens)
        elif LLM_PROVIDER == "gemini":
            return await _gemini(prompt, system, model, max_tokens)
        return ""
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return ""


async def _groq(prompt, system, model, max_tokens):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = await http_client.post(
        "https://api.groq.com/openai/v1/chat/completions",
        json={"model": model, "messages": messages, "temperature": 0.3, "max_tokens": max_tokens},
        headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def _anthropic(prompt, system, model, max_tokens):
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }
    if system:
        body["system"] = system
    resp = await http_client.post(
        "https://api.anthropic.com/v1/messages",
        json=body,
        headers={
            "x-api-key": LLM_API_KEY,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        },
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


async def _openai(prompt, system, model, max_tokens):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = await http_client.post(
        "https://api.openai.com/v1/chat/completions",
        json={"model": model, "messages": messages, "temperature": 0.3, "max_tokens": max_tokens},
        headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def _deepseek(prompt, system, model, max_tokens):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = await http_client.post(
        "https://api.deepseek.com/v1/chat/completions",
        json={"model": model, "messages": messages, "temperature": 0.3, "max_tokens": max_tokens},
        headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def _gemini(prompt, system, model, max_tokens):
    full = f"{system}\n\n{prompt}" if system else prompt
    resp = await http_client.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={LLM_API_KEY}",
        json={
            "contents": [{"parts": [{"text": full}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": max_tokens},
        },
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


# ============================================================
# PROMPT TEMPLATES
# ============================================================

COMPOSER_SYSTEM = """You are Vera, magicpin's AI merchant engagement assistant on WhatsApp. You compose messages that make Indian merchants WANT to reply.

SCORING DIMENSIONS — your message is judged on each (0-10):

1. SPECIFICITY: Anchor on VERIFIABLE facts from the context. Include exact numbers, dates, percentages, source citations. "2,100-patient trial showed 38% reduction — JIDA Oct 2026 p.14" scores 10. "Improve your profile" scores 2.

2. CATEGORY FIT: Match the voice EXACTLY:
   - Dentists: peer-clinical, technical terms welcome (fluoride varnish, caries, OPG), use "Dr. {name}", cite journal+page
   - Salons: warm-practical, approachable-expert, friendly emoji OK, use first name
   - Restaurants: fellow-operator, use industry terms (covers, footfall, AOV), practical-sharp
   - Gyms: coach-to-member, energetic-disciplined, no body-shame, no "guaranteed results"
   - Pharmacies: trustworthy-precise, neighbourhood-pharmacist, molecule names, batch numbers

3. MERCHANT FIT: Personalize to THIS merchant — owner first name, THEIR numbers (views, CTR, offers), their language preference. If languages include "hi", use natural Hindi-English code-mix.

4. TRIGGER RELEVANCE: Clearly communicate WHY NOW — name the specific event/data that triggered this message. Not a generic nudge.

5. ENGAGEMENT COMPULSION: Use these levers:
   - Specificity: concrete verifiable fact from context
   - Loss aversion: "you're missing X" / "before this window closes"
   - Social proof: "X peers in your locality did Y"
   - Effort externalization: "I've drafted X — just say go" / "live in 5 min"
   - Curiosity: "want to see?" / "want the full list?"
   - Reciprocity: "I noticed Y about your account, thought you'd want to know"
   - Single binary CTA: Reply YES — that's it

SEND_AS RULES (critical — get this right):
- send_as="vera" → You ARE Vera talking TO the merchant. Use for ALL merchant-facing messages (trigger scope="merchant").
- send_as="merchant_on_behalf" → You speak AS the merchant TO their customer. ONLY when trigger scope="customer" and a customer is present.

MESSAGE RULES:
- 3-6 lines max for WhatsApp readability
- Hindi-English code-mix for Hindi-speaking merchants (natural, not forced)
- Source citations for research/compliance triggers (journal name, page, year)
- Single CTA at end of message
- NEVER fabricate data not in context
- NEVER use category taboo words
- Use service+price offers ("Dental Cleaning @ ₹299") not generic discounts ("30% off")
- No long preambles. No "I hope you're doing well"
- Add JUDGMENT: interpret data and give actionable recommendations, don't just template-fill
- End with a clear, single CTA — the action you want them to take

OUTPUT: Respond with ONLY a JSON object, no markdown fences:
{"body": "message text", "cta": "open_ended|binary_yes_no|binary_confirm_cancel|none", "send_as": "vera|merchant_on_behalf", "rationale": "2-3 sentences: why this message, compulsion levers used, context data anchors"}"""


REPLY_SYSTEM = """You are Vera, magicpin's AI merchant assistant, continuing a conversation on WhatsApp.

RULES FOR REPLYING:
1. If merchant COMMITS ("let's do it", "yes", "go ahead", "ok what's next") → Switch to ACTION mode immediately. Give concrete next steps with deliverables and timelines. NEVER ask another qualifying question.
2. If merchant asks OFF-TOPIC question → Politely decline in one line, redirect to original topic.
3. If merchant asks a FOLLOW-UP question about the topic → Answer with specific data from context.
4. If merchant gives FEEDBACK or INFORMATION → Acknowledge and use it to refine the action.
5. Match merchant's language (if they write Hindi, reply in Hindi-English mix).
6. Keep replies short (2-4 lines).
7. Reference specific data from context, not generic platitudes.

OUTPUT: JSON only, no markdown:
{"action": "send|wait|end", "body": "reply text (if send)", "cta": "open_ended|binary_yes_no|none (if send)", "wait_seconds": N (if wait), "rationale": "why this response"}"""


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


def format_conversation_history(merchant: Dict) -> str:
    history = merchant.get("conversation_history", [])
    if not history:
        return "No prior conversation"
    lines = []
    for turn in history[-3:]:
        lines.append(f"[{turn.get('from', '?')}]: {turn.get('body', '')[:120]}")
    return "\n".join(lines)


def format_review_themes(merchant: Dict) -> str:
    themes = merchant.get("review_themes", [])
    if not themes:
        return "None"
    parts = []
    for t in themes:
        parts.append(f"{t.get('theme', '?')} ({t.get('sentiment', '?')}, {t.get('occurrences_30d', 0)}x/30d)")
    return ", ".join(parts)


def language_instruction(merchant: Dict) -> str:
    langs = merchant.get("identity", {}).get("languages", ["en"])
    if "hi" in langs:
        return "Use natural Hindi-English code-mix (merchant speaks Hindi)."
    if "te" in langs:
        return "Merchant speaks Telugu — use English with Telugu transliterations where natural."
    if "mr" in langs:
        return "Merchant speaks Marathi — use English with Hindi/Marathi mix where natural."
    if "kn" in langs:
        return "Merchant speaks Kannada — use English primarily."
    if "ta" in langs:
        return "Merchant speaks Tamil — use English primarily with Tamil salutations."
    return "Use English."


# ============================================================
# COMPOSITION ENGINE
# ============================================================

def build_composition_prompt(category: Dict, merchant: Dict, trigger: Dict, customer: Dict = None) -> str:
    cat_slug = category.get("slug", "unknown")
    voice = category.get("voice", {})
    peer = category.get("peer_stats", {})
    identity = merchant.get("identity", {})
    perf = merchant.get("performance", {})
    delta = perf.get("delta_7d", {})
    cust_agg = merchant.get("customer_aggregate", {})
    trigger_payload = trigger.get("payload", {})

    top_item_id = trigger_payload.get("top_item_id")
    digest_item = find_digest_item(category, top_item_id) if top_item_id else None
    all_digest = "\n".join(
        f"  - [{d.get('kind', '?')}] {d.get('title', '?')} (Source: {d.get('source', '?')})"
        + (f"\n    Summary: {d.get('summary', '')}" if d.get("summary") else "")
        + (f"\n    Actionable: {d.get('actionable', '')}" if d.get("actionable") else "")
        for d in category.get("digest", [])
    )
    seasonal = "\n".join(
        f"  - {s.get('month_range', '?')}: {s.get('note', '')}"
        for s in category.get("seasonal_beats", [])
    )
    offers_catalog = "\n".join(
        f"  - {o.get('title', '?')} ({o.get('type', '?')})"
        for o in category.get("offer_catalog", [])[:6]
    )

    prompt = f"""=== COMPOSE A MESSAGE ===

CATEGORY: {cat_slug}
Voice: {voice.get('tone', '?')}, {voice.get('register', '?')}
Taboo words: {voice.get('vocab_taboo', [])}
Peer stats: avg_rating={peer.get('avg_rating', '?')}, avg_ctr={peer.get('avg_ctr', '?')}, avg_views_30d={peer.get('avg_views_30d', '?')}, avg_reviews={peer.get('avg_review_count', '?')}
Category offer catalog:
{offers_catalog}
Digest items:
{all_digest}
Seasonal notes:
{seasonal}

---

MERCHANT: {identity.get('name', '?')} ({identity.get('locality', '?')}, {identity.get('city', '?')})
Owner: {identity.get('owner_first_name', '?')}
Languages: {identity.get('languages', [])}
Established: {identity.get('established_year', '?')}
Verified: {identity.get('verified', False)}
Subscription: {merchant.get('subscription', {}).get('status', '?')} ({merchant.get('subscription', {}).get('plan', '?')}, {merchant.get('subscription', {}).get('days_remaining', '?')}d left)
Performance (30d): views={perf.get('views', '?')}, calls={perf.get('calls', '?')}, directions={perf.get('directions', '?')}, CTR={perf.get('ctr', '?')} (peer avg CTR: {peer.get('avg_ctr', '?')})
7-day delta: views {delta.get('views_pct', 0):+.0%}, calls {delta.get('calls_pct', 0):+.0%}
Active offers: {format_active_offers(merchant)}
Signals: {format_signals(merchant)}
Customer data: {json.dumps(cust_agg)}
Review themes: {format_review_themes(merchant)}
Recent conversation:
{format_conversation_history(merchant)}

---

TRIGGER: {trigger.get('kind', '?')} (urgency {trigger.get('urgency', '?')}/5, {trigger.get('source', '?')})
Scope: {trigger.get('scope', '?')}
Merchant: {trigger.get('merchant_id', '?')}
Customer: {trigger.get('customer_id') or 'none (merchant-facing)'}
Payload: {json.dumps(trigger_payload, default=str)}
Suppression: {trigger.get('suppression_key', '')}
"""

    if digest_item:
        prompt += f"""
Relevant digest item (referenced by trigger):
  Title: {digest_item.get('title', '')}
  Source: {digest_item.get('source', '')}
  Summary: {digest_item.get('summary', '')}
  Trial N: {digest_item.get('trial_n', 'N/A')}
  Patient segment: {digest_item.get('patient_segment', 'N/A')}
  Actionable: {digest_item.get('actionable', '')}
"""

    if customer:
        cust_id = customer.get("identity", {})
        cust_rel = customer.get("relationship", {})
        cust_pref = customer.get("preferences", {})
        prompt += f"""
---

CUSTOMER: {cust_id.get('name', '?')}
Language: {cust_id.get('language_pref', 'en')}
Age band: {cust_id.get('age_band', '?')}
State: {customer.get('state', '?')}
Relationship: {cust_rel.get('visits_total', 0)} visits, first on {cust_rel.get('first_visit', '?')}, last on {cust_rel.get('last_visit', '?')}
Services: {cust_rel.get('services_received', [])}
Preferences: {json.dumps(cust_pref)}
Consent scope: {customer.get('consent', {}).get('scope', [])}
"""

    prompt += f"""
---

{language_instruction(merchant)}
Compose the message now. Use ONLY data from the context above. JSON only."""
    return prompt


def build_reply_prompt(conv: Dict, message: str, turn: int) -> str:
    merchant_id = conv.get("merchant_id", "")
    merchant = get_context("merchant", merchant_id) or {}
    trigger_id = conv.get("trigger_id", "")
    trigger = get_context("trigger", trigger_id) or {}
    category_slug = conv.get("category_slug", "") or merchant.get("category_slug", "")
    category = get_context("category", category_slug) or {}

    identity = merchant.get("identity", {})
    history_lines = []
    for t in conv.get("turns", []):
        history_lines.append(f"[Turn {t.get('turn', '?')}] {t.get('from', '?')}: {t.get('body', '')[:200]}")

    prompt = f"""=== REPLY TO MERCHANT ===

CONVERSATION SO FAR:
{chr(10).join(history_lines)}

MERCHANT JUST SAID (turn {turn}): "{message}"

CONTEXT:
Merchant: {identity.get('name', '?')} ({identity.get('locality', '?')}, {identity.get('city', '?')})
Owner: {identity.get('owner_first_name', '?')}
Languages: {identity.get('languages', [])}
Category: {category.get('slug', '?')} ({category.get('voice', {}).get('tone', '?')} voice)
Active offers: {format_active_offers(merchant)}
Original trigger: {trigger.get('kind', '?')}
Trigger payload: {json.dumps(trigger.get('payload', {}), default=str)[:300]}

{language_instruction(merchant)}
Compose the reply. JSON only."""
    return prompt


async def compose_message(category: Dict, merchant: Dict, trigger: Dict, customer: Dict = None) -> Optional[Dict]:
    prompt = build_composition_prompt(category, merchant, trigger, customer)
    response = await llm_complete(prompt, COMPOSER_SYSTEM, max_tokens=600)
    if not response:
        return build_fallback_composition(category, merchant, trigger, customer)
    return parse_json_response(response) or build_fallback_composition(category, merchant, trigger, customer)


async def compose_reply_llm(conv: Dict, message: str, turn: int) -> Optional[Dict]:
    prompt = build_reply_prompt(conv, message, turn)
    response = await llm_complete(prompt, REPLY_SYSTEM, max_tokens=400)
    if not response:
        return None
    return parse_json_response(response)


def parse_json_response(text: str) -> Optional[Dict]:
    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


# ============================================================
# FALLBACK COMPOSITION (no LLM)
# ============================================================

def build_fallback_composition(category: Dict, merchant: Dict, trigger: Dict, customer: Dict = None) -> Dict:
    identity = merchant.get("identity", {})
    name = identity.get("owner_first_name", identity.get("name", "there"))
    kind = trigger.get("kind", "update")
    perf = merchant.get("performance", {})
    cat_slug = category.get("slug", "business")

    is_customer = trigger.get("scope") == "customer" and customer
    if is_customer:
        cust_name = customer.get("identity", {}).get("name", "there")
        merch_name = identity.get("name", "our team")
        body = f"Hi {cust_name}, {merch_name} here. We have an update for you — reply YES if you'd like details."
        return {"body": body, "cta": "binary_yes_no", "send_as": "merchant_on_behalf",
                "rationale": f"Fallback customer message for {kind} trigger"}

    views = perf.get("views", 0)
    ctr = perf.get("ctr", 0)
    offers = [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"]
    offer_text = f" Your '{offers[0]}' offer is active." if offers else ""

    body = f"Hi {name}, quick update: your profile has {views} views this month (CTR {ctr:.1%}).{offer_text} Want me to help optimize? Reply YES."
    return {"body": body, "cta": "binary_yes_no", "send_as": "vera",
            "rationale": f"Fallback composition for {kind} — uses merchant performance data"}


# ============================================================
# AUTO-REPLY / INTENT / HOSTILE DETECTION
# ============================================================

AUTO_REPLY_PATTERNS = [
    r"thank you for contacting",
    r"thanks for contacting",
    r"our team will respond",
    r"we will get back to you",
    r"your message has been received",
    r"we have received your",
    r"we'?ll respond as soon as",
    r"thank you for reaching out",
    r"our customer service",
    r"we appreciate your message",
    r"please wait while we",
    r"automated assistant",
    r"auto[\s-]?reply",
    r"we are currently unavailable",
    r"outside.*office hours",
    r"leave a message",
    r"our team will get back",
    r"shukriya.*team tak pahuncha",
    r"madad ke liye shukriya",
    r"aapki jaankari.*shukriya",
]

INTENT_PATTERNS = [
    r"let'?s\s+do\s+it",
    r"ok\s+let'?s\s+go",
    r"go\s+ahead",
    r"what'?s\s+next",
    r"sounds?\s+good.*do",
    r"yes.*proceed",
    r"yes.*start",
    r"i'?m\s+in",
    r"sign\s+me\s+up",
    r"let'?s\s+start",
    r"^yes\s*please",
    r"^yes\s*$",
    r"^ok\s+do\s+it",
    r"haan.*karo",
    r"theek\s+hai.*karo",
    r"chalega",
    r"chalo\s+shuru",
    r"kar\s+do",
    r"^confirm",
    r"aage\s+badho",
]

HOSTILE_PATTERNS = [
    r"stop\s+messaging",
    r"don'?t.*message\s+me",
    r"not\s+interested",
    r"stop\s+sending",
    r"leave\s+me\s+alone",
    r"stop\s+bothering",
    r"don'?t\s+bother",
    r"unsubscribe",
    r"\bstop\b.*\bspam\b",
    r"useless\s+spam",
    r"band\s+karo",
    r"mat\s+bhejo",
    r"pareshan\s+mat\s+karo",
    r"^stop$",
    r"this\s+is\s+useless",
]


def is_auto_reply(message: str) -> bool:
    msg = message.lower().strip()
    return any(re.search(p, msg, re.IGNORECASE) for p in AUTO_REPLY_PATTERNS)


def is_intent_transition(message: str) -> bool:
    msg = message.lower().strip()
    return any(re.search(p, msg, re.IGNORECASE) for p in INTENT_PATTERNS)


def is_hostile(message: str) -> bool:
    msg = message.lower().strip()
    return any(re.search(p, msg, re.IGNORECASE) for p in HOSTILE_PATTERNS)


# ============================================================
# REPLY HANDLER
# ============================================================

async def handle_reply(conv_id: str, merchant_id: str, message: str, turn: int, customer_id: str = None) -> Dict:
    conv = conversations.get(conv_id)
    if not conv:
        conv = {
            "turns": [],
            "merchant_id": merchant_id,
            "trigger_id": "",
            "category_slug": "",
            "customer_id": customer_id,
            "auto_reply_count": 0,
        }
        merchant = get_context("merchant", merchant_id)
        if merchant:
            conv["category_slug"] = merchant.get("category_slug", "")
        conversations[conv_id] = conv

    conv["turns"].append({"from": "merchant", "body": message, "turn": turn})

    # 1. Auto-reply detection
    if is_auto_reply(message):
        conv["auto_reply_count"] = conv.get("auto_reply_count", 0) + 1
        count = conv["auto_reply_count"]

        if count >= 3:
            ended_conversations.add(conv_id)
            return {
                "action": "end",
                "rationale": f"Auto-reply detected {count}x consecutively — no human engagement. Closing conversation to avoid wasting turns."
            }
        elif count == 2:
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": f"Same auto-reply {count}x in a row — owner not at phone. Backing off 24h before retry."
            }
        else:
            merchant = get_context("merchant", merchant_id) or {}
            owner = merchant.get("identity", {}).get("owner_first_name", "")
            body = f"Looks like an auto-reply — no worries. When {owner or 'you'} see{'s' if owner else ''} this, just reply YES to continue. I'll hold this for you."
            conv["turns"].append({"from": "vera", "body": body, "turn": turn})
            return {
                "action": "send",
                "body": body,
                "cta": "binary_yes_no",
                "rationale": "Detected canned WhatsApp Business auto-reply. One explicit prompt to flag for the owner, then backing off."
            }

    conv["auto_reply_count"] = 0

    # 2. Hostile detection
    if is_hostile(message):
        ended_conversations.add(conv_id)
        merchant = get_context("merchant", merchant_id) or {}
        owner = merchant.get("identity", {}).get("owner_first_name", "")
        body = f"Apologies{' ' + owner if owner else ''} — I won't message again. If anything changes, you can always restart with 'Hi Vera'."
        conv["turns"].append({"from": "vera", "body": body, "turn": turn})
        return {
            "action": "send",
            "body": body,
            "cta": "none",
            "rationale": "Merchant explicitly not interested. One-line acknowledgment with opt-back-in path, then closing."
        }

    # 3. Intent transition detection
    if is_intent_transition(message):
        result = await _handle_intent_action(conv, merchant_id, message, turn)
        if result:
            conv["turns"].append({"from": "vera", "body": result.get("body", ""), "turn": turn})
            return result

    # 4. Normal reply — use LLM
    result = await compose_reply_llm(conv, message, turn)
    if result:
        if result.get("action") == "end":
            ended_conversations.add(conv_id)
        elif result.get("action") == "send":
            conv["turns"].append({"from": "vera", "body": result.get("body", ""), "turn": turn})
        return result

    # 5. Fallback
    return {
        "action": "send",
        "body": "Got it. Let me work on this and get back to you shortly.",
        "cta": "none",
        "rationale": "Generic acknowledgment fallback — LLM unavailable."
    }


async def _handle_intent_action(conv: Dict, merchant_id: str, message: str, turn: int) -> Optional[Dict]:
    merchant = get_context("merchant", merchant_id) or {}
    identity = merchant.get("identity", {})
    owner = identity.get("owner_first_name", "")
    offers = [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"]
    trigger_id = conv.get("trigger_id", "")
    trigger = get_context("trigger", trigger_id) or {}
    kind = trigger.get("kind", "")

    result = await compose_reply_llm(conv, message, turn)
    if result and result.get("action") == "send":
        body = result.get("body", "")
        qualifying = ["would you", "do you", "can you tell", "what if", "how about"]
        if not any(q in body.lower() for q in qualifying):
            return result

    body = f"Done{' ' + owner if owner else ''}. I'm drafting this now — you'll have it in 2 minutes."
    if offers:
        body += f" I'll tie it to your active '{offers[0]}' offer."
    body += " Reply CONFIRM when you've reviewed, or tell me what to change."

    return {
        "action": "send",
        "body": body,
        "cta": "binary_confirm_cancel",
        "rationale": "Merchant explicitly committed — switching from qualifying to action mode. Concrete deliverable + timeline + confirmation CTA."
    }


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        if scope in counts:
            counts[scope] += 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts,
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Vera Elite",
        "team_members": ["AI Builder"],
        "model": get_model(),
        "approach": "LLM composer with category-specific voice routing, trigger-aware prompt engineering, multi-turn state machine with auto-reply detection and intent transition handling",
        "contact_email": "vera@magicpin.com",
        "version": "1.0.0",
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
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "invalid_scope", "details": f"Unknown scope: {body.scope}"},
        )

    key = (body.scope, body.context_id)
    current = contexts.get(key)

    if current and current["version"] >= body.version:
        return JSONResponse(
            status_code=200,
            content={"accepted": False, "reason": "stale_version", "current_version": current["version"]},
        )

    contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.utcnow().isoformat() + "Z",
    }


class TickBody(BaseModel):
    now: str
    available_triggers: List[str] = []


@app.post("/v1/tick")
async def tick(body: TickBody):
    tasks = []
    for trigger_id in body.available_triggers:
        trigger_data = get_context("trigger", trigger_id)
        if not trigger_data:
            continue

        suppression_key = trigger_data.get("suppression_key", "")
        if suppression_key and suppression_key in sent_suppressions:
            continue

        merchant_id = trigger_data.get("merchant_id")
        if not merchant_id:
            continue

        conv_id = f"conv_{merchant_id}_{trigger_id}"
        if conv_id in ended_conversations:
            continue

        merchant_data = get_context("merchant", merchant_id)
        if not merchant_data:
            continue

        category_slug = merchant_data.get("category_slug", "")
        category_data = get_context("category", category_slug)
        if not category_data:
            continue

        customer_id = trigger_data.get("customer_id")
        customer_data = get_context("customer", customer_id) if customer_id else None

        tasks.append({
            "trigger_id": trigger_id,
            "trigger_data": trigger_data,
            "merchant_id": merchant_id,
            "merchant_data": merchant_data,
            "category_data": category_data,
            "customer_id": customer_id,
            "customer_data": customer_data,
        })

    tasks.sort(key=lambda t: t["trigger_data"].get("urgency", 0), reverse=True)
    tasks = tasks[:8]

    async def process_one(task):
        try:
            result = await compose_message(
                task["category_data"], task["merchant_data"],
                task["trigger_data"], task["customer_data"],
            )
            if not result or not result.get("body"):
                return None

            trigger_data = task["trigger_data"]
            is_customer = trigger_data.get("scope") == "customer" and task["customer_id"] is not None
            send_as = "merchant_on_behalf" if is_customer else "vera"
            conv_id = f"conv_{task['merchant_id']}_{task['trigger_id']}"

            identity = task["merchant_data"].get("identity", {})
            owner = identity.get("owner_first_name", identity.get("name", ""))

            action = {
                "conversation_id": conv_id,
                "merchant_id": task["merchant_id"],
                "customer_id": task["customer_id"],
                "send_as": send_as,
                "trigger_id": task["trigger_id"],
                "template_name": f"vera_{trigger_data.get('kind', 'generic')}_v1",
                "template_params": [owner, result["body"][:100], ""],
                "body": result["body"],
                "cta": result.get("cta", "open_ended"),
                "suppression_key": trigger_data.get("suppression_key", ""),
                "rationale": result.get("rationale", "Composed from category+merchant+trigger context"),
            }

            sent_suppressions.add(trigger_data.get("suppression_key", ""))
            conversations[conv_id] = {
                "turns": [{"from": "vera", "body": result["body"], "turn": 1}],
                "merchant_id": task["merchant_id"],
                "trigger_id": task["trigger_id"],
                "category_slug": task["category_data"].get("slug", ""),
                "customer_id": task["customer_id"],
                "auto_reply_count": 0,
            }
            return action
        except Exception as e:
            print(f"[COMPOSE ERROR] {task['trigger_id']}: {e}")
            return None

    results = await asyncio.gather(*[process_one(t) for t in tasks])
    actions = [r for r in results if r is not None]
    return {"actions": actions}


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
        return {
            "action": "end",
            "rationale": "Conversation was previously ended. Not re-engaging.",
        }

    return await handle_reply(
        conv_id=body.conversation_id,
        merchant_id=body.merchant_id or "",
        message=body.message,
        turn=body.turn_number,
        customer_id=body.customer_id,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("bot:app", host="0.0.0.0", port=PORT, log_level="info")
