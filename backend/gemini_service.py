from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List


class GeminiInterviewAssistant:
    def __init__(self, api_key: str = "", model: str = "gemini-1.5-flash") -> None:
        self.api_key = (api_key or "").strip()
        self.model = model.strip() or "gemini-1.5-flash"

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def configure_api_key(self, api_key: str) -> bool:
        self.api_key = (api_key or "").strip()
        return self.configured

    def generate_question_answer(self, prior_questions: List[str]) -> Dict[str, Any]:
        if not self.configured:
            return {
                "question": "Tell me about yourself.",
                "ideal_answer": "Introduce your background, key strengths, and one relevant achievement.",
                "key_points": ["background", "strengths", "achievement"],
                "source": "fallback",
            }

        prompt = (
            "You are an interview coach. Generate ONE interview question and its ideal answer. "
            "Avoid repeating prior questions. Return strict JSON only with keys: "
            "question (string), ideal_answer (string), key_points (array of strings, max 6).\n"
            f"Prior questions: {prior_questions!r}"
        )
        data = self._call_json(prompt)
        question = str(data.get("question", "")).strip()
        ideal_answer = str(data.get("ideal_answer", "")).strip()
        key_points_raw = data.get("key_points", [])
        key_points = [str(x).strip() for x in key_points_raw if str(x).strip()]

        if not question or not ideal_answer:
            return {
                "question": "Why are you interested in this role?",
                "ideal_answer": "Show role fit, company motivation, and how your skills will create impact.",
                "key_points": ["role fit", "company motivation", "impact"],
                "source": "fallback",
            }

        return {
            "question": question,
            "ideal_answer": ideal_answer,
            "key_points": key_points[:6],
            "source": "gemini",
        }

    def evaluate_answer(self, question: str, ideal_answer: str, user_answer: str) -> Dict[str, Any]:
        if not user_answer.strip():
            return {
                "score": 0.0,
                "strengths": [],
                "gaps": ["No answer detected."],
                "feedback": "Please provide a complete answer with concrete details.",
                "source": "rule",
            }

        if not self.configured:
            return self._heuristic_eval(ideal_answer=ideal_answer, user_answer=user_answer)

        prompt = (
            "You are an interview evaluator. Compare the candidate answer with the ideal answer and return strict JSON only. "
            "Required keys: score (0-100 number), strengths (array of strings), gaps (array of strings), feedback (string).\n"
            f"Question: {question}\n"
            f"Ideal answer: {ideal_answer}\n"
            f"Candidate answer: {user_answer}"
        )
        data = self._call_json(prompt)

        try:
            score = float(data.get("score", 0.0))
        except Exception:
            score = 0.0
        score = max(0.0, min(100.0, score))

        strengths = [str(x).strip() for x in data.get("strengths", []) if str(x).strip()]
        gaps = [str(x).strip() for x in data.get("gaps", []) if str(x).strip()]
        feedback = str(data.get("feedback", "")).strip()
        if not feedback:
            feedback = "Good structure. Add more concrete examples and measurable outcomes."

        return {
            "score": score,
            "strengths": strengths[:5],
            "gaps": gaps[:5],
            "feedback": feedback,
            "source": "gemini",
        }

    def _heuristic_eval(self, ideal_answer: str, user_answer: str) -> Dict[str, Any]:
        ideal_tokens = set(self._tokenize(ideal_answer))
        user_tokens = set(self._tokenize(user_answer))
        if not ideal_tokens:
            return {
                "score": 60.0,
                "strengths": ["Answered the question."],
                "gaps": ["No reference answer was available for exact comparison."],
                "feedback": "Add concrete examples and results to improve quality.",
                "source": "rule",
            }

        overlap = len(ideal_tokens.intersection(user_tokens))
        score = round((overlap / max(1, len(ideal_tokens))) * 100.0, 1)
        score = max(0.0, min(100.0, score))

        if score >= 75:
            strengths = ["Covered most expected key points."]
            gaps = ["Can improve by adding measurable outcomes."]
        elif score >= 45:
            strengths = ["Partially covered expected points."]
            gaps = ["Missed some important points from the ideal answer."]
        else:
            strengths = ["Provided an attempt to answer."]
            gaps = ["Answer was too far from expected key points."]

        return {
            "score": score,
            "strengths": strengths,
            "gaps": gaps,
            "feedback": "Use a clearer structure: context, action, and measurable result.",
            "source": "rule",
        }

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]{3,}", (text or "").lower())

    def _call_json(self, prompt: str) -> Dict[str, Any]:
        if not self.configured:
            return {}

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{urllib.parse.quote(self.model)}:generateContent?key={urllib.parse.quote(self.api_key)}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(exc)
            raise RuntimeError(f"Gemini HTTP error: {exc.code} {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"Gemini request failed: {exc}") from exc

        try:
            parsed = json.loads(body)
            text = (
                parsed.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            if not text:
                return {}
            return json.loads(text)
        except Exception as exc:
            raise RuntimeError(f"Gemini response parse failed: {exc}") from exc
