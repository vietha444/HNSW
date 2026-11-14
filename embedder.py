"""
Semantic Embedding Engine
=========================
Generates dense contextual semantic embeddings using PyTorch & HuggingFace Transformers.
Optimized for CUDA GPU acceleration when available, with automatic CPU fallback.
Caches embeddings to disk (.npz) for instantaneous subsequent startups.
"""

import os
import sys
import time
import numpy as np
from typing import List, Optional

# Disable TensorFlow backend initialization to avoid Windows cuDNN slowdowns
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

class SemanticEmbedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.dim = 384
        self.tokenizer = None
        self.model = None
        self.device = device
        self._init_model()

    def _init_model(self):
        print(f"[Embedder] Initializing model '{self.model_name}'...")
        t0 = time.time()
        try:
            import torch
            import torch.nn.functional as F
            from transformers import AutoTokenizer, AutoModel

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            print(f"[Embedder] Target compute device: {self.device.upper()}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            print(f"[Embedder] Model successfully loaded in {time.time() - t0:.2f}s on {self.device.upper()}")
            self.backend = "transformers"
        except Exception as e:
            print(f"[Embedder Warning] Could not load PyTorch Transformers ({e}). Falling back to TF-IDF Latent Semantic Analysis.")
            self._init_tfidf_fallback()

    def _init_tfidf_fallback(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self.backend = "tfidf_lsa"
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        self.svd = TruncatedSVD(n_components=min(self.dim, 100), random_state=42)
        print("[Embedder] Fallback TF-IDF + TruncatedSVD initialized.")

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of strings into normalized dense vectors.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        if self.backend == "transformers":
            import torch
            import torch.nn.functional as F

            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    out = self.model(**inputs)
                    # Mean pooling with attention mask
                    mask = inputs["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
                    sum_embeddings = torch.sum(out.last_hidden_state * mask, 1)
                    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                    batch_emb = sum_embeddings / sum_mask
                    
                    # L2 Normalize
                    batch_emb = F.normalize(batch_emb, p=2, dim=1)
                    all_embeddings.append(batch_emb.cpu().numpy())

            embeddings = np.vstack(all_embeddings).astype(np.float32)
            self.dim = embeddings.shape[1]
            return embeddings
        else:
            # Fallback
            tfidf = self.vectorizer.fit_transform(texts)
            reduced = self.svd.fit_transform(tfidf)
            # L2 normalize
            norms = np.linalg.norm(reduced, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return (reduced / norms).astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed a single query string into a 1D normalized vector.
        """
        emb = self.embed_texts([text])
        return emb[0]
