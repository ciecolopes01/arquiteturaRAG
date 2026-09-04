# ADR 0003: Pool de Conexões Thread-Safe e Mutexes em Singletons

## Status
Aceito

## Data
2026-09-04

## Contexto
O módulo `bm25_index.py` utilizava uma única conexão global `_conn` com PostgreSQL (`autocommit=False`).
Sob múltiplos requests concorrentes (ex: servidores web ASGI, endpoints FastAPI com workers assíncronos ou threads de streaming), requisições concorrentes compartilhavam a mesma transação no banco de dados. Qualquer chamada a `conn.rollback()` (usada, por exemplo, no fallback de sintaxe tsquery) abortava as operações de outras threads ativas.

Além disso, os singletons de módulo com lazy loading (`_model`, `_client`, `_reranker`, `_GRAPH_INDEX`) não possuíam mecanismos de sincronização, permitindo inicializações concorrentes duplicadas que consumiam RAM e VRAM em excesso.

## Decisão
1. Adotar `psycopg2.pool.ThreadedConnectionPool(minconn=1, maxconn=10)` em `src/retrieval/bm25_index.py`.
2. Encapsular a aquisição e devolução de conexões em um context manager `@contextmanager def get_db_connection()`, com garantia de devolução ao pool em blocos `finally:`.
3. Proteger todos os carregadores de singletons com mutexes de módulo (`threading.Lock()`).

## Consequências
- **Positivas:** Operações simultâneas de busca e indexação isolam transações completamente sem interferência mútua.
- **Positivas:** Zero vazamento de conexões abertas no PostgreSQL.
- **Positivas:** Carregamento thread-safe garantido para modelos de ML pesados em memória.
