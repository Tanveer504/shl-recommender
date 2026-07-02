import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from models import ChatRequest, ChatResponse, Recommendation
from agent import get_agent_response
from catalog import load_catalog
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="SHL Assessment Recommender")
try:
    CATALOG = load_catalog()
    logger.info(f"Loaded catalog: {len(CATALOG)} assessments")
except Exception as e:
    logger.error(f"Failed to load catalog: {e}")
    CATALOG = []
@app.get("/health")
def health():
    return {"status": "ok", "catalog_size": len(CATALOG)}
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not CATALOG:
        raise HTTPException(status_code=500, detail="Catalog failed to load on startup")
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")
    for msg in request.messages:
        if msg.role not in ("user", "assistant"):
            raise HTTPException(status_code=400, detail=f"Invalid role: {msg.role}")
    # Hard guard: evaluator caps at 8 turns total
    total_turns = sum(1 for m in request.messages if m.role == "user")
    if total_turns > 8:
        return ChatResponse(
            reply="We have covered a lot of ground. Here is your final shortlist based on everything discussed.",
            recommendations=[],
            end_of_conversation=True
        )
    try:
        result = get_agent_response(request.messages)
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        return ChatResponse(
            reply="I encountered an error. Please try again.",
            recommendations=[],
            end_of_conversation=False
        )
    recommendations = [
        Recommendation(
            name=item["name"],
            url=item["url"],
            test_type=item["test_type"]
        )
        for item in result["selected"]
    ]
    action = result["action"]
    # "refine" is a mid-conversation update, not a final answer — only "recommend" ends it
    end_of_conversation = (
        action == "recommend" and len(recommendations) > 0
    )
    return ChatResponse(
        reply=result["reply"],
        recommendations=recommendations,
        end_of_conversation=end_of_conversation
    )
