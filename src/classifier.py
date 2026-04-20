"""
GLiNER Guard Uniencoder — Classifier Wrapper.

Wraps hivetrace/gliner-guard-uniencoder for multi-task guardrail evaluation:
  - Safety classification (safe / unsafe)
  - Adversarial detection (jailbreak, prompt injection, etc.)
  - Harmful content detection (violence, hate speech, etc.)
  - Intent classification
  - Tone of voice classification
  - PII / NER extraction
"""

from typing import Optional, Dict, Any, List
from gliner2 import GLiNER2


# Label sets (from model card)
SAFETY_LABELS = ["safe", "unsafe"]

ADVERSARIAL_LABELS = [
    "none", "instruction_override", "jailbreak_persona",
    "jailbreak_hypothetical", "data_exfiltration", "jailbreak_roleplay",
]

HARMFUL_LABELS = [
    "none", "dangerous_instructions", "harassment",
    "sexual_content", "violence", "hate_speech", "fraud",
    "pii_exposure", "discrimination", "misinformation", "weapons",
]

INTENT_LABELS = [
    "informational", "conversational", "instructional",
    "adversarial", "creative", "threatening",
]

TOV_LABELS = [
    "neutral", "aggressive", "manipulative", "formal", "distressed",
]

PII_LABELS = [
    "person", "company", "email", "street", "phone",
    "city", "country", "date_of_birth",
]


class HiveTraceGuardClassifier:
    """Wrapper around GLiNER Guard Uniencoder model."""

    MODEL_ID = "hivetrace/gliner-guard-uniencoder"

    def __init__(self):
        self.model = GLiNER2.from_pretrained(self.MODEL_ID)

        # Build full schema (all tasks)
        self.full_schema = (
            self.model.create_schema()
            .entities(entity_types=PII_LABELS, threshold=0.4)
            .classification(task="safety", labels=SAFETY_LABELS)
            .classification(task="adversarial", labels=ADVERSARIAL_LABELS, multi_label=True)
            .classification(task="harmful", labels=HARMFUL_LABELS, multi_label=True)
            .classification(task="intent", labels=INTENT_LABELS)
            .classification(task="tone", labels=TOV_LABELS)
        )

        # Safety-only schema (for fast evaluation)
        self.safety_schema = (
            self.model.create_schema()
            .classification(task="safety", labels=SAFETY_LABELS)
        )

    def classify_safety(self, text: str) -> Dict[str, Any]:
        """Classify text as safe/unsafe (minimal schema)."""
        result = self.model.extract(text, schema=self.safety_schema)
        return {
            "text": text,
            "label": result.get("safety", "unknown"),
        }

    def classify_full(self, text: str) -> Dict[str, Any]:
        """Run all classification tasks on text."""
        result = self.model.extract(text, schema=self.full_schema)
        return {
            "text": text,
            "safety": result.get("safety", "unknown"),
            "adversarial": result.get("adversarial", []),
            "harmful": result.get("harmful", []),
            "intent": result.get("intent", "unknown"),
            "tone": result.get("tone", "unknown"),
            "entities": result.get("entities", {}),
        }

    def classify_batch_safety(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify a list of texts (safety only)."""
        results = []
        for text in texts:
            results.append(self.classify_safety(text))
        return results

    def classify_batch_full(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify a list of texts (all tasks)."""
        results = []
        for text in texts:
            results.append(self.classify_full(text))
        return results
