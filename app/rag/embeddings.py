"""
Vertex AI Embeddings for generating vector representations of code.

Uses Google's textembedding-gecko model for embeddings, keeping everything on GCP.
"""

from typing import List
from vertexai.language_models import TextEmbeddingModel
import logging
import time

from app.metrics import track_embedding_call

logger = logging.getLogger(__name__)


class VertexAIEmbeddings:
    """Wrapper for Vertex AI text embedding API."""

    MODEL = "text-embedding-005"  # Latest embedding model
    MAX_TOKENS_PER_TEXT = 2048  # Max tokens per individual text
    MAX_TOKENS_PER_BATCH = 12000  # Stay well under 20K API limit (token estimate is approximate)
    MAX_ITEMS_PER_BATCH = 250  # Max inputs per API call

    def __init__(self):
        """Initialize the embeddings client."""
        self._model = TextEmbeddingModel.from_pretrained(self.MODEL)
        logger.info(f"Vertex AI Embeddings initialized with model: {self.MODEL}")

    def count_tokens(self, text: str) -> int:
        """Approximate token count (4 chars per token)."""
        return len(text) // 4

    def truncate_to_tokens(self, text: str, max_tokens: int = None) -> str:
        """
        Truncate text to fit within token limit.

        Args:
            text: Input text to truncate
            max_tokens: Maximum tokens (defaults to model max)

        Returns:
            Truncated text
        """
        if max_tokens is None:
            max_tokens = self.MAX_TOKENS_PER_TEXT

        # Approximate: 4 chars per token
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text

        return text[:max_chars]

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        # Truncate if necessary
        text = self.truncate_to_tokens(text)

        embeddings = self._model.get_embeddings([text])
        return embeddings[0].values

    def _create_token_aware_batches(self, texts: List[str]) -> List[List[str]]:
        """
        Split texts into batches that respect both item count and token limits.

        Args:
            texts: List of texts to batch

        Returns:
            List of batches, each batch is a list of texts
        """
        batches = []
        current_batch = []
        current_tokens = 0

        for text in texts:
            text_tokens = self.count_tokens(text)

            # Check if adding this text would exceed limits
            would_exceed_tokens = (current_tokens + text_tokens) > self.MAX_TOKENS_PER_BATCH
            would_exceed_items = len(current_batch) >= self.MAX_ITEMS_PER_BATCH

            if current_batch and (would_exceed_tokens or would_exceed_items):
                # Start a new batch
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(text)
            current_tokens += text_tokens

        # Don't forget the last batch
        if current_batch:
            batches.append(current_batch)

        return batches

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.

        Batches by token count to stay under API limits.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Truncate all texts first
        truncated_texts = [self.truncate_to_tokens(t) for t in texts]

        # Create token-aware batches
        batches = self._create_token_aware_batches(truncated_texts)
        logger.info(f"Processing {len(truncated_texts)} texts in {len(batches)} batches")

        # Process each batch
        all_embeddings = []
        for batch_idx, batch in enumerate(batches):
            batch_tokens = sum(self.count_tokens(t) for t in batch)
            batch_start = time.time()
            try:
                embeddings = self._model.get_embeddings(batch)
                all_embeddings.extend([e.values for e in embeddings])

                # Track successful embedding call
                track_embedding_call(
                    batch_size=len(batch),
                    duration=time.time() - batch_start,
                    success=True
                )
                logger.info(f"Batch {batch_idx + 1}/{len(batches)}: {len(batch)} texts, ~{batch_tokens} tokens")
            except Exception as e:
                logger.error(f"Error embedding batch {batch_idx}: {e}")
                # Return empty embeddings for failed batch
                all_embeddings.extend([[0.0] * 768 for _ in batch])

                # Track failed embedding call
                track_embedding_call(
                    batch_size=len(batch),
                    duration=time.time() - batch_start,
                    success=False
                )

        return all_embeddings
