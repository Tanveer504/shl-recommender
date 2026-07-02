import logging
from fastapi import FastAPI, HTTPException
from models import ChatRequest, ChatResponse, Recommendation
from agent import get_agent_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SHL Assessment Recommender")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    for msg in request.messages:
        if msg.role not in ("user", "assistant"):
            raise HTTPException(status_code=400, detail=f"Invalid role: {msg.role}")

    total_turns = sum(1 for m in request.messages if m.role == "user")
    if total_turns > 8:
        return ChatResponse(
            reply="We have covered a lot of ground. Here is your final shortlist.",
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
    end_of_conversation = (
        action in ("recommend", "refine") and len(recommendations) > 0
    )

    return ChatResponse(
        reply=result["reply"],
        recommendations=recommendations,
        end_of_conversation=end_of_conversation
    )
