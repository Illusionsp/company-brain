# bot/telegram_bot.py
# Telegram bot — team members ask questions from their phones.
#
# Setup:
#   1. Message @BotFather → /newbot → copy token
#   2. Set TELEGRAM_BOT_TOKEN in .env
#   3. Run: python bot/telegram_bot.py
#
# Commands:
#   /start   — welcome + instructions
#   /docs    — list indexed documents
#   /clear   — reset conversation history
#   /help    — show commands
#   (text)   — ask any question

import asyncio, logging, httpx, os
from dotenv import load_dotenv
load_dotenv()

logger    = logging.getLogger(__name__)
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
API_BASE  = os.getenv("API_BASE_URL", "http://localhost:8000")


async def send(chat_id: int, text: str):
    if not BOT_TOKEN:
        return
    async with httpx.AsyncClient(timeout=10) as c:
        await c.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
        )


async def handle(msg: dict):
    chat_id    = msg["chat"]["id"]
    text       = msg.get("text", "").strip()
    session_id = f"tg_{chat_id}"

    if not text:
        return

    if text.startswith("/start"):
        await send(chat_id, (
            "🧠 *Company Brain*\n\n"
            "I answer questions from your company's uploaded documents.\n\n"
            "Just ask me anything — I'll search the knowledge base and answer with source citations.\n\n"
            "*Commands:*\n"
            "/docs — list available documents\n"
            "/clear — start a new conversation\n"
            "/help — show this message"
        ))
        return

    if text.startswith("/docs"):
        async with httpx.AsyncClient(timeout=10) as c:
            r    = await c.get(f"{API_BASE}/documents")
            data = r.json()
        docs = data.get("documents", [])
        if docs:
            lines = "\n".join([f"• {d['doc_name']} ({d['chunks']} chunks)" for d in docs])
            await send(chat_id, f"📚 *Knowledge Base ({len(docs)} documents):*\n\n{lines}")
        else:
            await send(chat_id, "📭 No documents yet. Upload via the web dashboard.")
        return

    if text.startswith("/clear"):
        async with httpx.AsyncClient(timeout=10) as c:
            await c.delete(f"{API_BASE}/sessions/{session_id}")
        await send(chat_id, "✅ Conversation cleared. Ask me anything!")
        return

    if text.startswith("/help"):
        await send(chat_id, (
            "📖 *How to use Company Brain:*\n\n"
            "• Type any question — I'll search and answer from the documents\n"
            "• I remember context — ask follow-up questions naturally\n"
            "• If I don't know, I'll say so — I never make things up\n\n"
            "/docs — see what documents I know about\n"
            "/clear — reset the conversation"
        ))
        return

    # Regular question → call RAG API
    await send(chat_id, "🔍 _Searching documents..._")

    try:
        async with httpx.AsyncClient(timeout=40) as c:
            r = await c.post(
                f"{API_BASE}/chat",
                json={"question": text, "session_id": session_id},
            )

        if r.status_code == 200:
            data    = r.json()
            answer  = data["answer"]
            sources = data.get("sources", [])
            src_txt = f"\n\n📄 *Source:* {', '.join(sources)}" if sources else ""
            await send(chat_id, f"{answer}{src_txt}")

        elif r.status_code == 400:
            await send(chat_id, f"❌ {r.json().get('detail', 'Error')}")
        else:
            await send(chat_id, "❌ Something went wrong. Please try again.")

    except httpx.TimeoutException:
        await send(chat_id, "⏱ Taking longer than expected. Please try again in 30 seconds.")
    except Exception as e:
        logger.error(f"Telegram error: {e}")
        await send(chat_id, "❌ Connection error. Please try again.")


async def run():
    if not BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set — Telegram bot disabled")
        return

    logger.info("🤖 Telegram bot starting...")
    offset = 0

    async with httpx.AsyncClient(timeout=35) as c:
        while True:
            try:
                r       = await c.get(
                    f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates",
                    params={"offset": offset, "timeout": 30},
                )
                updates = r.json().get("result", [])
                for upd in updates:
                    offset = upd["update_id"] + 1
                    if "message" in upd:
                        await handle(upd["message"])

            except httpx.TimeoutException:
                pass
            except httpx.NetworkError as e:
                logger.warning(f"Network: {e}")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Bot error: {e}")
                await asyncio.sleep(2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    asyncio.run(run())
