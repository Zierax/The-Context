# ============================================================================
# entity_extractor.py — Spectral Memory Manifold Co-Processor
# ENTITY EXTRACTOR: Pluggable ABC + deterministic heuristic extractor
# No LLM calls. No spaCy. Pure regex-based SVO extraction.
# ============================================================================

import re
from abc import ABC, abstractmethod
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Verb and noun word lists for deterministic POS simulation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Common English relation verbs (is-a, has-a, part-of, domain-specific)
RELATION_VERBS: set[str] = {
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "having",
    "contains", "contain", "contained", "containing",
    "includes", "include", "included", "including",
    "refers", "refer", "referred", "referring",
    "calls", "call", "called", "calling",
    "depends", "depend", "depended", "depending",
    "requires", "require", "required", "requiring",
    "uses", "use", "used", "using",
    "defines", "define", "defined", "defining",
    "implements", "implement", "implemented", "implementing",
    "extends", "extend", "extended", "extending",
    "produces", "produce", "produced", "producing",
    "creates", "create", "created", "creating",
    "transforms", "transform", "transformed", "transforming",
    "computes", "compute", "computed", "computing",
    "links", "link", "linked", "linking",
    "maps", "map", "mapped", "mapping",
    "connects", "connect", "connected", "connecting",
    "triggers", "trigger", "triggered", "triggering",
    "follows", "follow", "followed", "following",
    "precedes", "precede", "preceded", "preceding",
    "generates", "generate", "generated", "generating",
    "stores", "store", "stored", "storing",
    "retrieves", "retrieve", "retrieved", "retrieving",
    "processes", "process", "processed", "processing",
    "sends", "send", "sent", "sending",
    "receives", "receive", "received", "receiving",
}

# Prepositions that typically end relation phrases
RELATION_PREPOSITIONS: set[str] = {
    "to", "from", "into", "by", "with", "for", "as", "of", "in", "on", "at",
}

# Capitalised words are likely proper nouns / named entities
CAPITALISED_PATTERN = re.compile(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*")


class EntityExtractor(ABC):
    """Abstract base class for entity extractors.

    All implementations must provide a deterministic `extract` method
    that returns (Subject, Predicate, Object) triples from text.
    """

    @abstractmethod
    def extract(self, text: str) -> list[tuple[str, str, str]]:
        """Extract (Subject, Predicate, Object) triples from text.

        Args:
            text: Raw text string to extract triples from.

        Returns:
            List of (subject, predicate, object) triples as strings.
            Returns empty list if no triples can be extracted.
        """
        raise NotImplementedError("Subclasses must implement extract()")


class HeuristicExtractor(EntityExtractor):
    """Deterministic heuristic extractor using regex patterns and word lists.

    Extracts Subject-Verb-Object triples by:
    1. Splitting text into sentences.
    2. Finding verb phrases using a predefined relation verb list.
    3. Extracting the noun phrases before and after the verb.
    4. Normalising to lowercase and stripping punctuation.

    No external dependencies. Fully deterministic.
    """

    # Sentence splitting pattern
    SENTENCE_PATTERN = re.compile(
        r"[^.!?]+[.!?]",
        re.UNICODE,
    )

    # Word tokenizer
    WORD_PATTERN = re.compile(r"\b\w+(?:'\w+)?\b", re.UNICODE)

    # Verb phrase pattern: captures [SUBJECT] [VERB] [OBJECT]
    # where subject and object are sequences of words
    # Kept word limit at 8 for precision; expanded verb list
    TRIPLE_PATTERN = re.compile(
        r"(\b(?:\w+\s+){0,8}?\w+)\s+"
        r"(is|are|was|were|has|have|had|contains?|includes?|refers?|"
        r"calls?|depends?|requires?|uses?|defines?|implements?|extends?|"
        r"produces?|creates?|transforms?|computes?|links?|maps?|connects?|"
        r"triggers?|generates?|stores?|retrieves?|processes?)\s+"
        r"(\b(?:\w+\s+){0,8}?\w+)",
        re.IGNORECASE,
    )

    def __init__(self, min_word_length: int = 2) -> None:
        """Initialise the heuristic extractor.

        Args:
            min_word_length: Minimum word length to consider as a valid
                subject/object (filters out short noise). Default 2.
        """
        self.min_word_length = min_word_length

    def _normalise(self, phrase: str) -> str:
        """Normalise a phrase to lowercase with stripped punctuation.

        Args:
            phrase: Raw phrase string.

        Returns:
            Normalised, stripped, lowercase string.
        """
        return phrase.strip().lower().strip(".,;:!?\"'()[]{}")

    def _is_valid_concept(self, phrase: str) -> bool:
        """Check if a phrase is a valid concept for the knowledge graph.

        A valid concept has at least one alphabetic word longer than
        min_word_length and isn't entirely stop words.

        Args:
            phrase: Normalised phrase string.

        Returns:
            True if the phrase is a valid concept.
        """
        if not phrase or len(phrase) < self.min_word_length:
            return False
        words = phrase.split()
        meaningful = [
            w
            for w in words
            if len(w) >= self.min_word_length and w.isalpha()
        ]
        return len(meaningful) >= 1

    def extract(self, text: str) -> list[tuple[str, str, str]]:
        """Extract (Subject, Predicate, Object) triples from text.

        Uses pattern matching and word-list-based POS simulation for full
        determinism. Returns normalised triples only.

        Args:
            text: Raw text string to analyse.

        Returns:
            List of (subject, predicate, object) triples.
            Empty list if no valid triples found.

        Raises:
            ValueError: If text exceeds 1,000,000 characters.
        """
        if len(text) > 1_000_000:
            raise ValueError(
                f"Text too long ({len(text)} chars). Max 1,000,000."
            )

        if not text or not text.strip():
            return []

        triples: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()

        # Method 1: Split into sentences and extract SVO from each
        sentences = self.SENTENCE_PATTERN.findall(text)
        for sentence in sentences:
            # Skip sentences that are too long (likely not real sentences)
            if len(sentence.split()) > 30:
                continue
            
            # Try to find "X verb Y" patterns — split on verb phrase
            for verb in ["is", "are", "was", "were", "has", "have", "had",
                         "governs", "combines", "enables", "serves", "consists",
                         "provides", "uses", "relies", "operates", "stores"]:
                parts = sentence.split(f" {verb} ", 1)
                if len(parts) == 2:
                    subject = self._normalise(parts[0])
                    obj = self._normalise(parts[1])
                    
                    # Clean up subject (remove leading articles)
                    subject_words = subject.split()
                    if subject_words and subject_words[0] in {"the", "a", "an", "this", "that", "it", "they", "we"}:
                        subject = " ".join(subject_words[1:])
                    
                    # Clean up object (remove trailing clauses, limit to 5 words)
                    obj_words = obj.split()
                    if len(obj_words) > 5:
                        obj = " ".join(obj_words[:5])
                    
                    # Remove trailing punctuation
                    obj = obj.rstrip(".,;:!?")
                    
                    if not subject or not obj:
                        continue
                    if not self._is_valid_concept(subject) or not self._is_valid_concept(obj):
                        continue
                    if subject in {"the", "a", "an", "this", "that", "it", "they", "we"}:
                        continue
                    if obj in {"the", "a", "an", "this", "that", "it", "they", "we"}:
                        continue

                    triple = (subject, verb, obj)
                    if triple not in seen:
                        seen.add(triple)
                        triples.append(triple)

        # Method 2: Handle "X of Y" patterns (e.g., "definition of X")
        of_pattern = re.compile(
            r"\b(\w+(?:\s+\w+){0,2})\s+of\s+(\w+(?:\s+\w+){0,2})\b",
            re.IGNORECASE,
        )
        for match in of_pattern.finditer(text):
            subject = self._normalise(match.group(1))
            obj = self._normalise(match.group(2))
            if not self._is_valid_concept(subject) or not self._is_valid_concept(obj):
                continue
            triple = (obj, "has", subject)  # obj has subject (inverse of "of")
            if triple not in seen:
                seen.add(triple)
                triples.append(triple)

        logger.debug(
            "HeuristicExtractor.extract",
            text_length=len(text),
            n_triples=len(triples),
        )
        return triples
