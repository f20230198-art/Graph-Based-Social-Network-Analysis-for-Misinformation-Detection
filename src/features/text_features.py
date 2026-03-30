"""
Text feature extraction: RoBERTa embeddings, sentiment, emotion, linguistic style.
Total output: 797-dimensional feature vector per article.
"""

import torch
import numpy as np
import re
from typing import List, Dict


def extract_linguistic_features(text: str) -> np.ndarray:
    """
    Extract 12 linguistic style features from text.

    Returns:
        np.ndarray of shape (12,)
        [cap_ratio, punct_density, excl_count, ques_count, avg_word_len,
         type_token_ratio, readability, ner_density, url_count,
         hashtag_count, mention_count, emoji_count]
    """
    if not text or not isinstance(text, str):
        return np.zeros(12)

    words = text.split()
    num_words = max(len(words), 1)
    num_chars = max(len(text), 1)

    # Capitalization ratio
    upper_chars = sum(1 for c in text if c.isupper())
    cap_ratio = upper_chars / num_chars

    # Punctuation density
    punct_count = sum(1 for c in text if c in ".,;:!?-\"'()[]{}...")
    punct_density = punct_count / num_words

    # Exclamation and question counts
    excl_count = text.count("!")
    ques_count = text.count("?")

    # Average word length
    avg_word_len = np.mean([len(w) for w in words]) if words else 0

    # Type-token ratio (vocabulary diversity)
    unique_words = set(w.lower() for w in words)
    type_token_ratio = len(unique_words) / num_words

    # Readability (simplified Flesch-Kincaid approximation)
    syllable_count = sum(count_syllables(w) for w in words)
    sentence_count = max(len(re.split(r"[.!?]+", text)), 1)
    readability = (
        0.39 * (num_words / sentence_count)
        + 11.8 * (syllable_count / num_words)
        - 15.59
    )

    # Named entity density (simplified: count capitalized words)
    ner_density = sum(1 for w in words if w[0].isupper() and len(w) > 1) / num_words

    # URL, hashtag, mention counts
    url_count = text.count("[URL]")
    hashtag_count = text.count("#")
    mention_count = text.count("[USER]") + text.count("@")

    # Emoji count (simplified)
    emoji_count = len(re.findall(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]", text))

    return np.array([
        cap_ratio, punct_density, excl_count, ques_count,
        avg_word_len, type_token_ratio, readability, ner_density,
        url_count, hashtag_count, mention_count, emoji_count
    ], dtype=np.float32)


def count_syllables(word: str) -> int:
    """Approximate syllable count for a word."""
    word = word.lower().strip(".,!?;:'\"")
    if not word:
        return 1
    count = 0
    vowels = "aeiouy"
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def batch_extract_linguistic(texts: List[str]) -> np.ndarray:
    """Extract linguistic features for a batch of texts."""
    return np.array([extract_linguistic_features(t) for t in texts])


class TextFeatureExtractor:
    """
    Extracts all text features (797-d total):
    - RoBERTa embeddings: 768-d
    - Sentiment: 3-d
    - Emotion: 6-d
    - Linguistic: 12-d
    - Claim-specific: 8-d
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.roberta_tokenizer = None
        self.roberta_model = None
        self.sentiment_pipeline = None
        self.emotion_pipeline = None

    def load_models(self):
        """Load all pretrained models. Call this before extraction."""
        from transformers import AutoTokenizer, AutoModel, pipeline

        print("Loading RoBERTa...")
        self.roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.roberta_model = AutoModel.from_pretrained("roberta-base").to(self.device)
        self.roberta_model.eval()

        print("Loading sentiment model...")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=0 if self.device == "cuda" else -1,
            top_k=None,
        )

        print("Loading emotion model...")
        self.emotion_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=0 if self.device == "cuda" else -1,
            top_k=None,
        )

        print("All text models loaded.")

    def extract_roberta_embedding(self, text: str) -> np.ndarray:
        """Extract 768-d RoBERTa [CLS] embedding."""
        inputs = self.roberta_tokenizer(
            text, return_tensors="pt", max_length=256,
            truncation=True, padding="max_length"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.roberta_model(**inputs)

        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        return cls_embedding

    def extract_sentiment(self, text: str) -> np.ndarray:
        """Extract 3-d sentiment probabilities [negative, neutral, positive]."""
        result = self.sentiment_pipeline(text[:512])[0]
        label_map = {"negative": 0, "neutral": 1, "positive": 2}
        probs = np.zeros(3)
        for item in result:
            label = item["label"].lower()
            if label in label_map:
                probs[label_map[label]] = item["score"]
        return probs.astype(np.float32)

    def extract_emotion(self, text: str) -> np.ndarray:
        """Extract 6-d emotion probabilities."""
        result = self.emotion_pipeline(text[:512])[0]
        labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
        probs = np.zeros(6)
        for item in result:
            label = item["label"].lower()
            if label in labels:
                probs[labels.index(label)] = item["score"]
        return probs.astype(np.float32)

    def extract_all(self, text: str) -> np.ndarray:
        """Extract complete 797-d feature vector."""
        roberta = self.extract_roberta_embedding(text)     # 768
        sentiment = self.extract_sentiment(text)            # 3
        emotion = self.extract_emotion(text)                # 6
        linguistic = extract_linguistic_features(text)      # 12
        claim = np.zeros(8, dtype=np.float32)               # 8 (placeholder)

        return np.concatenate([roberta, sentiment, emotion, linguistic, claim])
