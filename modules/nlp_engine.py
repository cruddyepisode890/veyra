"""
VEYRA AI — NLP Intent Recognition & Response Generator
Rule-based intent matching with keyword / regex patterns.

Supported intents:
    greeting, farewell, time, date, weather, calculate,
    name, capabilities, joke, compliment, status, music,
    reminder, search
"""

import re
import math
import random
from datetime import datetime


# ── Intent definitions ────────────────────────────────────────
INTENTS: dict[str, dict] = {
    "greeting": {
        "patterns": [
            r"\b(hello|hi|hey|greetings|good\s*morning|good\s*afternoon|good\s*evening|howdy|what's\s*up|sup)\b"
        ],
        "responses": [
            "Hello! How can I help you today?",
            "Hey there! What can I do for you?",
            "Hi! I'm Veyra, your personal assistant. What do you need?",
            "Greetings! I'm ready to assist you.",
            "Hey! Voice verified — what would you like?",
        ],
    },
    "farewell": {
        "patterns": [
            r"\b(bye|goodbye|see\s*you|later|quit|exit|stop|good\s*night|take\s*care)\b"
        ],
        "responses": [
            "Goodbye! Have a great day!",
            "See you later! Stay safe.",
            "Bye! Feel free to talk to me anytime.",
            "Take care! I'll be here when you need me.",
        ],
    },
    "time": {
        "patterns": [
            r"\b(what\s*time|current\s*time|tell\s*me\s*the\s*time|what'?s?\s*the\s*time|time\s*now|time\s*is\s*it)\b"
        ],
        "responses": None,  # dynamic
    },
    "date": {
        "patterns": [
            r"\b(what\s*date|today'?s?\s*date|current\s*date|what\s*day|what\s*is\s*today|today\s*is)\b"
        ],
        "responses": None,
    },
    "weather": {
        "patterns": [
            r"\b(weather|temperature|forecast|rain|sunny|cloudy|hot\s*outside|cold\s*outside|humidity)\b"
        ],
        "responses": [
            "I don't have real-time weather data right now, but I recommend checking a weather app for the latest forecast!",
            "For live weather updates, try asking a weather service. I'm focused on keeping your assistant secure!",
            "I can't check the weather yet, but that's a great feature idea for a future update!",
        ],
    },
    "calculate": {
        "patterns": [
            r"\b(calculate|compute|what\s*is|how\s*much\s*is|solve|math|plus|minus|times|divided|multiply|add|subtract)\b"
        ],
        "responses": None,
    },
    "name": {
        "patterns": [
            r"\b(what\s*is\s*your\s*name|who\s*are\s*you|your\s*name|introduce\s*yourself|what\s*are\s*you|tell\s*me\s*about\s*yourself)\b"
        ],
        "responses": [
            "I'm Veyra AI — your personalized voice assistant that only responds to your voice!",
            "My name is Veyra. I'm an AI assistant secured by speaker authentication.",
            "I'm Veyra AI. I verify your identity by voice before processing any commands.",
        ],
    },
    "capabilities": {
        "patterns": [
            r"\b(what\s*can\s*you\s*do|your\s*capabilities|help\s*me|what\s*do\s*you\s*do|your\s*features|abilities|commands)\b"
        ],
        "responses": [
            "I can tell you the time and date, do basic calculations, tell jokes, and more — but only if I recognize your voice!",
            "I'm a voice-authenticated assistant. I can help with time, dates, calculations, and general conversation. Only your voice unlocks my abilities!",
            "I support greetings, time/date queries, calculations, jokes, and status checks — all secured by your unique voice profile.",
        ],
    },
    "joke": {
        "patterns": [
            r"\b(joke|funny|make\s*me\s*laugh|humor|tell\s*me\s*something\s*funny|amuse\s*me)\b"
        ],
        "responses": [
            "Why don't scientists trust atoms? Because they make up everything!",
            "What do you call a fake noodle? An impasta!",
            "Why did the scarecrow win an award? Because he was outstanding in his field!",
            "What do you call a bear with no teeth? A gummy bear!",
            "Why don't programmers like nature? It has too many bugs!",
            "What's a computer's favorite snack? Microchips!",
            "I told my AI a joke about UDP… but I'm not sure it got it.",
            "Why was the math book sad? It had too many problems!",
            "Parallel lines have so much in common. It's a shame they'll never meet.",
        ],
    },
    "compliment": {
        "patterns": [
            r"\b(you'?re?\s*(great|awesome|amazing|cool|smart|brilliant)|good\s*job|well\s*done|nice|thank\s*you|thanks)\b"
        ],
        "responses": [
            "Thank you! I'm here to help you anytime.",
            "That's kind of you! I'm always working to be better.",
            "You're welcome! Let me know if you need anything else.",
            "Glad I could help! Your voice is the key to my world 🔐",
        ],
    },
    "status": {
        "patterns": [
            r"\b(how\s*are\s*you|how\s*do\s*you\s*feel|are\s*you\s*okay|your\s*status|system\s*status|you\s*doing)\b"
        ],
        "responses": [
            "I'm running perfectly! All systems operational.",
            "I'm doing great, thanks for asking! Ready to assist.",
            "All good here! Voice authentication is active and I'm listening.",
            "Systems nominal — speaker encoder, Whisper, and NLP all green.",
        ],
    },
    "music": {
        "patterns": [
            r"\b(play\s*music|sing|song|music|tune)\b"
        ],
        "responses": [
            "I can't play music yet, but I appreciate the request! Try a music streaming app.",
            "Music playback isn't in my repertoire yet — maybe in a future update!",
        ],
    },
    "search": {
        "patterns": [
            r"\b(search|google|look\s*up|find\s*information|wiki)\b"
        ],
        "responses": [
            "I don't have internet search capabilities, but I can help with what I know!",
            "Web search isn't available right now, but I can answer basic questions.",
        ],
    },
}


# ── Public API ────────────────────────────────────────────────
def recognize_intent(text: str) -> dict:
    """
    Determine the user's intent from transcribed text.

    Returns:
        dict with ``intent``, ``confidence`` (0–1), and ``response``.
    """
    text_lower = text.lower().strip()

    if not text_lower:
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "response": "I didn't catch that. Could you please repeat?",
        }

    best_intent: str | None = None
    best_score: float = 0.0

    for intent_name, intent_data in INTENTS.items():
        for pattern in intent_data["patterns"]:
            matches = re.findall(pattern, text_lower)
            if matches:
                # Score = matched-token ratio (simple but effective)
                word_count = max(len(text_lower.split()), 1)
                score = len(matches) / word_count
                if score > best_score:
                    best_score = score
                    best_intent = intent_name

    if best_intent is None:
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "response": (
                f'I heard: "{text}". '
                "I'm not sure how to help with that, but I'm learning!"
            ),
        }

    response = _generate_response(best_intent, text_lower)
    confidence = round(min(best_score + 0.5, 1.0), 2)

    return {
        "intent": best_intent,
        "confidence": confidence,
        "response": response,
    }


# ── Response generators ──────────────────────────────────────
def _generate_response(intent: str, text: str) -> str:
    if intent == "time":
        now = datetime.now()
        return f"The current time is {now.strftime('%I:%M %p')}."

    if intent == "date":
        now = datetime.now()
        return f"Today is {now.strftime('%A, %B %d, %Y')}."

    if intent == "calculate":
        return _handle_calculation(text)

    responses = INTENTS[intent].get("responses")
    if responses:
        return random.choice(responses)
    return "I'm not sure how to respond to that."


def _handle_calculation(text: str) -> str:
    """Parse and safely evaluate a simple arithmetic expression."""
    # Strip surrounding natural language
    cleaned = text
    for phrase in (
        "what is", "calculate", "compute", "how much is",
        "solve", "what's", "tell me",
    ):
        cleaned = cleaned.replace(phrase, "")

    # Map spoken operators to symbols
    replacements = {
        "plus": "+", "minus": "-", "times": "*",
        "multiplied by": "*", "multiply": "*",
        "divided by": "/", "divide by": "/", "divide": "/",
        "add": "+", "subtract": "-", "mod": "%",
        "power": "**", "to the power of": "**",
    }
    for word, sym in replacements.items():
        cleaned = cleaned.replace(word, sym)

    # Keep only digits, operators, dots, parens, spaces
    math_expr = re.sub(r"[^0-9\.\+\-\*\/\%\(\)\s]", "", cleaned).strip()

    if not math_expr:
        return "I couldn't identify a math expression. Try saying something like 'calculate 5 plus 3'."

    try:
        # Safe eval — no builtins, only math constants
        result = eval(math_expr, {"__builtins__": {}}, {"pi": math.pi, "e": math.e})
        if isinstance(result, float) and result == int(result):
            result = int(result)
        return f"The answer is {result}."
    except Exception:
        return "I couldn't compute that. Could you rephrase the math problem?"
