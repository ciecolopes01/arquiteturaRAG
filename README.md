# RAG Platform — Passo 0: Setup do ambiente

## O que este passo entrega

- `docker-compose.yml` com Qdrant, PostgreSQL e MinIO
- `.env.example` com as variáveis necessárias
- Estrutura de pastas do projeto (`src/`, `tests/`, `data/seed_corpus/`)
- `src/config.py` — carregamento e validação de configuração
- `tests/test_infra.py` — script que confirma as 3 conexões

## Como rodar

1. Copiar o arquivo de ambiente e ajustar se necessário (os defaults já funcionam para dev local):
   ```bash
   cp .env.example .env
   ```

2. Subir os containers:
   ```bash
   docker-compose up -d
   ```

3. Confirmar que os 3 estão rodando:
   ```bash
   docker ps
   ```
   Deve aparecer `rag_qdrant`, `rag_postgres`, `rag_minio`.

4. Criar um ambiente virtual e instalar as dependências:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # no Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

5. Rodar o teste de infraestrutura:
   ```bash
   python tests/test_infra.py
   ```

## Critério de "pronto"

O script deve imprimir `OK` para os 3 serviços e terminar com:
```
Passo 0 concluído — infraestrutura pronta.
```

Se algum falhar, a mensagem de erro indica o que checar (geralmente: container não subiu, ou porta já em uso por outro processo na sua máquina — nesse caso ajuste a porta no `docker-compose.yml` e no `.env`).

## Acessos manuais (para inspeção visual, opcional)

- Qdrant: http://localhost:6333/dashboard
- MinIO Console: http://localhost:9001 (usuário/senha: `minioadmin` / `minioadmin_dev`)
- Postgres: `psql postgresql://rag:rag_dev_password@localhost:5432/rag_platform`

## Próximo passo

Passo 1 — Canonical Document Model (não depende de Docker, só do repositório criado aqui).
