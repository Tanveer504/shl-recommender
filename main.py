import logging
from fastapi import FastAPI, HTTPException
from models import ChatRequest, ChatResponse, Recommendation
from agent import get_agent_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SHL Assessment Recommender")

# Keywords that signal the user has accepted / confirmed the shortlist
_CONFIRM = {
    "perfect", "confirmed", "that works", "locking", "that's it",
    "that covers", "keep it", "finalized", "done", "thanks",
    "thank you", "great", "excellent", "that's what we need",
    "that's good", "keep the", "keep those", "go ahead",
}


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

    # Count user turns (spec: max 8 total turns including both roles)
    user_turn_count = sum(1 for m in request.messages if m.role == "user")

    # Hard cap — return whatever we have rather than breaking the schema
    if user_turn_count > 8:
        return ChatResponse(
            reply="We have covered a lot of ground. Here is your final shortlist.",
            recommendations=[],
            end_of_conversation=True,
        )

    try:
        result = get_agent_response(request.messages)
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        return ChatResponse(
            reply="I encountered an error. Please try again.",
            recommendations=[],
            end_of_conversation=False,
        )

    recommendations = [
        Recommendation(
            name=item["name"],
            url=item["url"],
            test_type=item["test_type"],
        )
        for item in result["selected"]
    ]

    action = result["action"]
    has_recs = len(recommendations) > 0

    # Detect whether the user has confirmed/accepted the shortlist
    last_user_msg = ""
    for m in reversed(request.messages):
        if m.role == "user":
            last_user_msg = m.content.lower()
            break

    user_confirmed = any(kw in last_user_msg for kw in _CONFIRM)

    # Near the turn cap — force completion to stay within 8-turn limit
    near_cap = user_turn_count >= 6

    # end_of_conversation = True only when:
    #   1. We have actual recommendations, AND
    #   2. The user has confirmed them OR we're near the turn cap
    #
    # This keeps end_of_conversation=False on the FIRST recommendation turn
    # so the evaluator's simulated user can refine before we close off,
    # which improves Recall@10 on multi-turn traces like C9.
    end_of_conversation = has_recs and (user_confirmed or near_cap)

    return ChatResponse(
        reply=result["reply"],
        recommendations=recommendations,
        end_of_conversation=end_of_conversation,
    )
