"""
Camada de Cache Distribuído e Otimização de Latência — Passo 32.

Implementa:
1. Cache de respostas completas (hash da query + tenant_id -> payload da resposta gerada).
2. Cache de embeddings de consulta (texto da query -> vetor denso), evitando inferência repetida.
3. Conexão resiliente com Redis com fallback automático em memória (LRU/TTL) se o Redis estiver offline.
4. Métricas de observabilidade: contador de hits, misses e cache hit rate.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from src.config import settings

# ---------------------------------------------------------------------------
# Fallback em Memória (LRU com TTL)
# ---------------------------------------------------------------------------

class InMemoryCache:
    """Cache em memória para fallback resiliente quando Redis estiver offline."""

    def __init__(self, max_size: int = 2000):
        self._store: Dict[str, Tuple[str, float]] = {}  # key -> (json_value, expire_at)
        self._max_size = max_size

    def get(self, key: str) -> Optional[str]:
        if key not in self._store:
            return None
        val, expire_at = self._store[key]
        if time.time() > expire_at:
            del self._store[key]
            return None
        return val

    def set(self, key: str, value: str, ttl_seconds: int = 3600) -> None:
        if len(self._store) >= self._max_size:
            # Evita estouro de memória: remove as 10% chaves mais antigas/expiradas
            now = time.time()
            expired = [k for k, (_, exp) in self._store.items() if now > exp]
            for k in expired:
                del self._store[k]
            if len(self._store) >= self._max_size:
                # Remove os primeiros itens inseridos
                for k in list(self._store.keys())[: int(self._max_size * 0.1)]:
                    del self._store[k]

        self._store[key] = (value, time.time() + ttl_seconds)

    def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# Gerenciador de Conexão Redis / Fallback
# ---------------------------------------------------------------------------

class CacheManager:
    """Gerencia operações de cache com Redis ou fallback em memória."""

    def __init__(self):
        self.redis_client = None
        self.in_memory = InMemoryCache()
        self.hits: int = 0
        self.misses: int = 0
        self._init_redis()

    def _init_redis(self):
        """Tenta inicializar cliente Redis se configurado."""
        try:
            import redis

            client = redis.Redis.from_url(
                settings.redis_url,
                socket_timeout=1.0,
                socket_connect_timeout=1.0,
                decode_responses=True,
            )
            # Teste de ping rápido
            client.ping()
            self.redis_client = client
        except Exception:
            self.redis_client = None

    @property
    def backend_name(self) -> str:
        return "redis" if self.redis_client is not None else "in_memory"

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 4) if total > 0 else 0.0

    def reset_metrics(self):
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "backend": self.backend_name,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "enabled": settings.enable_cache,
        }

    def clear(self):
        self.in_memory.clear()
        self.reset_metrics()
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception:
                pass

    def _hash_key(self, prefix: str, key_data: str) -> str:
        """Gera chave consistente usando hash SHA-256."""
        h = hashlib.sha256(key_data.strip().lower().encode("utf-8")).hexdigest()[:32]
        return f"{prefix}:{h}"

    # --- Cache de Respostas de Geração ---

    def get_response(self, query: str, tenant_id: str = "") -> Optional[Dict[str, Any]]:
        if not settings.enable_cache:
            return None

        cache_key = self._hash_key("rag:resp", f"{tenant_id}:{query}")
        raw = None

        if self.redis_client:
            try:
                raw = self.redis_client.get(cache_key)
            except Exception:
                raw = None

        if raw is None:
            raw = self.in_memory.get(cache_key)

        if raw is not None:
            self.hits += 1
            try:
                return json.loads(raw)
            except Exception:
                return None

        self.misses += 1
        return None

    def set_response(
        self,
        query: str,
        tenant_id: str,
        response_payload: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        if not settings.enable_cache:
            return

        ttl = ttl_seconds or settings.cache_response_ttl
        cache_key = self._hash_key("rag:resp", f"{tenant_id}:{query}")
        serialized = json.dumps(response_payload, ensure_ascii=False)

        if self.redis_client:
            try:
                self.redis_client.set(cache_key, serialized, ex=ttl)
            except Exception:
                pass

        self.in_memory.set(cache_key, serialized, ttl_seconds=ttl)

    # --- Cache de Embeddings de Query ---

    def get_embedding(self, query: str, model_name: str) -> Optional[List[float]]:
        if not settings.enable_cache:
            return None

        cache_key = self._hash_key("rag:emb", f"{model_name}:{query}")
        raw = None

        if self.redis_client:
            try:
                raw = self.redis_client.get(cache_key)
            except Exception:
                raw = None

        if raw is None:
            raw = self.in_memory.get(cache_key)

        if raw is not None:
            self.hits += 1
            try:
                return json.loads(raw)
            except Exception:
                return None

        self.misses += 1
        return None

    def set_embedding(
        self,
        query: str,
        model_name: str,
        vector: List[float],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        if not settings.enable_cache:
            return

        ttl = ttl_seconds or settings.cache_embedding_ttl
        cache_key = self._hash_key("rag:emb", f"{model_name}:{query}")
        serialized = json.dumps(vector)

        if self.redis_client:
            try:
                self.redis_client.set(cache_key, serialized, ex=ttl)
            except Exception:
                pass

        self.in_memory.set(cache_key, serialized, ttl_seconds=ttl)


# Instância singleton global
cache = CacheManager()
