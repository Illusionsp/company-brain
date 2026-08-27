# retrieval/rag_pipeline.py
# Generation layer.
# Key features:
#   - "I don't know" detection — never hallucinates confidently
#   - Multi-turn conversation memory per session
#   - Source citations in every answer
#   - 4 AI providers: Groq (free), Gemini (free), Anthropic, OpenAI

import logging
from typing import List
import httpx
from config import settings
from retrieval.vector_store import SearchResult

logger   = logging.getLogger(__name__)
_sessions: dict[str, list] = {}

SYSTEM_PROMPT = """You are Company Brain — an internal AI assistant.
You answer questions ONLY from the provided document excerpts.

Strict rules:
1. Use ONLY information from the provided context. Never use outside knowledge.
2. If the answer is NOT clearly in the context, respond EXACTLY with:
   "I don't have that information in the provided documents. You may want to check with [relevant team/person] or upload more documentation."
3. Never guess, assume, or make up information. Accuracy matters more than completeness.
4. Keep answers concise — 2-4 sentences for most questions.
5. Always end your answer with: "Source: [document name(s) used]"
6. For off-topic questions: "I can only answer questions about the uploaded documents."

Remember: saying "I don't know" when you don't know is better than a confident wrong answer."""


async def generate_answer(
    question:   str,
    chunks:     List[SearchResult],
    session_id: str = "default",
) -> str:
    if not chunks:
        return (
            "I couldn't find relevant information in the documents to answer your question. "
            "Try uploading more documentation or rephrasing your question."
        )

    context = "\n\n---\n\n".join([
        f"[Document: {c.doc_name}]\n{c.content}" for c in chunks
    ])

    history  = _sessions.get(session_id, [])
    messages = [{"role": m["role"], "content": m["content"]} for m in history[-12:]]
    messages.append({
        "role": "user",
        "content": f"Context from documents:\n\n{context}\n\n---\n\nQuestion: {question}",
    })

    provider = settings.ai_provider()
    if provider == "none":
        return (
            "⚠️ No AI key configured.\n\n"
            "Add one of these to your .env:\n"
            "• GROQ_API_KEY (free) — console.groq.com\n"
            "• GEMINI_API_KEY (free) — aistudio.google.com"
        )

    try:
        if provider == "groq":       ans = await _groq(messages)
        elif provider == "gemini":   ans = await _gemini(question, context)
        elif provider == "anthropic": ans = await _anthropic(messages)
        else:                         ans = await _openai(messages)
    except httpx.HTTPStatusError as e:
        logger.error(f"AI {provider} error {e.response.status_code}")
        return f"AI error ({e.response.status_code}). Check your API key."
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return "Something went wrong. Please try again."

    _sessions.setdefault(session_id, [])
    _sessions[session_id] += [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": ans},
    ]
    return ans


async def _groq(messages):
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post("https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
            json={"model": "llama-3.3-70b-versatile", "max_tokens": settings.MAX_TOKENS,
                  "temperature": settings.TEMPERATURE,
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


async def _gemini(question, context):
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={settings.GEMINI_API_KEY}",
            json={"contents": [{"parts": [{"text": prompt}]}],
                  "generationConfig": {"temperature": settings.TEMPERATURE, "maxOutputTokens": settings.MAX_TOKENS}})
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]


async def _anthropic(messages):
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key": settings.ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01"},
            json={"model": "claude-sonnet-4-6", "max_tokens": settings.MAX_TOKENS,
                  "system": SYSTEM_PROMPT, "messages": messages})
        r.raise_for_status()
        return r.json()["content"][0]["text"]


async def _openai(messages):
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post("https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
            json={"model": "gpt-4o-mini", "max_tokens": settings.MAX_TOKENS,
                  "temperature": settings.TEMPERATURE,
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def clear_session(session_id: str):
    _sessions.pop(session_id, None)


def session_turns(session_id: str) -> int:
    return len(_sessions.get(session_id, [])) // 2
