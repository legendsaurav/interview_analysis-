from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Dict, List


class InterviewState(str, enum.Enum):
    IDLE = "IDLE"
    ASKING_QUESTION = "ASKING_QUESTION"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    RESPONDING = "RESPONDING"


@dataclass
class InterviewStateMachine:
    expected_answer_seconds: float = 20.0
    max_answer_seconds: float = 60.0
    silence_threshold: float = 2.5
    post_speech_pause_seconds: float = 7.0

    state: InterviewState = InterviewState.IDLE
    active_question: str = "Tell me about yourself."
    question_bank: List[str] = field(
        default_factory=lambda: [
            "Tell me about yourself.",
            "Why are you interested in this role?",
            "Describe a challenging project you worked on.",
            "What is one technical decision you are proud of?",
            "How do you handle feedback and improve your work?",
        ]
    )
    question_index: int = 0
    listening_started_at: float = 0.0
    speech_pause_started_at: float = 0.0
    waiting_confirmation: bool = False
    session_transcript: List[str] = field(default_factory=list)

    def ask_question(self, question: str | None = None) -> Dict[str, object]:
        if question:
            self.active_question = question
        elif self.question_bank:
            self.active_question = self.question_bank[self.question_index]
        self.state = InterviewState.ASKING_QUESTION
        self.listening_started_at = time.time()
        self.speech_pause_started_at = 0.0
        self.waiting_confirmation = False

        # Immediately move to LISTENING after prompting.
        self.state = InterviewState.LISTENING
        return {
            "state": self.state.value,
            "question": self.active_question,
            "prompt": self.active_question,
        }

    def update_from_audio(self, event: Dict[str, object]) -> Dict[str, object]:
        now = time.time()
        feedback: List[str] = []
        actions: List[str] = []

        speaking_state = str(event.get("speaking_state", "paused"))
        speaking_duration = float(event.get("speaking_duration", 0.0))
        transcript = str(event.get("transcript", "")).strip()

        if transcript:
            self.session_transcript.append(transcript)

        if self.state == InterviewState.IDLE:
            return {
                "state": self.state.value,
                "feedback": feedback,
                "actions": actions,
            }

        if self.state == InterviewState.LISTENING:
            elapsed = max(0.0, now - self.listening_started_at)

            if speaking_state == "speaking":
                self.speech_pause_started_at = 0.0

            if speaking_state == "finished" and self.speech_pause_started_at == 0.0:
                # Start a grace timer after speech ends; if user resumes talking, timer is cleared.
                self.speech_pause_started_at = now

            if self.speech_pause_started_at > 0.0 and (now - self.speech_pause_started_at) >= self.post_speech_pause_seconds:
                self.state = InterviewState.RESPONDING
                self.speech_pause_started_at = 0.0
                feedback.append("No further speech detected. Moving to the next question.")
                actions.append("next_question")

            if elapsed > self.max_answer_seconds or speaking_duration > self.max_answer_seconds:
                self.state = InterviewState.RESPONDING
                self.speech_pause_started_at = 0.0
                feedback.append("Let's move to the next question.")
                actions.append("next_question")

        return {
            "state": self.state.value,
            "feedback": feedback,
            "actions": actions,
        }

    def generate_response(self, transcript: str, speaking_duration: float, vision: Dict[str, object]) -> str:
        words = len(transcript.split()) if transcript else 0
        eye = float(vision.get("eye_contact_score", 0.5))
        move = float(vision.get("head_movement_score", 0.5))

        suggestions: List[str] = []
        if words < 15 or speaking_duration < 8:
            suggestions.append("Can you elaborate more?")
        if words > 120 or speaking_duration > self.expected_answer_seconds * 1.5:
            suggestions.append("Let's keep it concise.")
        if eye < 0.45:
            suggestions.append("Maintain better eye contact.")
        if move < 0.35:
            suggestions.append("Try to keep your posture steadier.")

        if not suggestions:
            suggestions.append("Good response. Let's continue.")

        self.state = InterviewState.RESPONDING
        return " ".join(suggestions)

    def reset_for_next_question(self, question: str | None = None) -> Dict[str, object]:
        self.waiting_confirmation = False
        self.listening_started_at = time.time()
        self.speech_pause_started_at = 0.0
        self.state = InterviewState.LISTENING
        if question:
            self.active_question = question
        else:
            self._advance_default_question()
        return {
            "state": self.state.value,
            "question": self.active_question,
        }

    def _advance_default_question(self) -> None:
        if not self.question_bank:
            return
        self.question_index = (self.question_index + 1) % len(self.question_bank)
        self.active_question = self.question_bank[self.question_index]
