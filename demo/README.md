# 🚀 Demo Local Interativa — RAG Multimodal Enterprise

Aplicação web completa e interativa para demonstração em tempo real das capacidades de engenharia do projeto **Arquitetura RAG**: ingestão multimodal real, chunking com estratégias avançadas (*Parent-Child*, *UUIDv5* e *Semântico*), governança com *Fail-Closed ACL*, e busca híbrida tri-canal com Fusão RRF ponderada.

---

## 📸 O que esta Demo faz:

1. **Upload & Parse de Arquivos Reais**:
   - Faça upload de **PDFs** (com detecção de colunas e extração de tabelas), **Planilhas XLSX**, **Documentos DOCX**, **Imagens** (com OCR), ou **Textos/Markdown**.
   - O backend executa o **parser canônico real** do projeto (`src/parsing/`).

2. **Chunking Engine em Tempo Real**:
   - Escolha a estratégia:
     - **Parent-Child (Hierárquico)**: Gera chunks filhos de 350 caracteres vinculados a macro-chunks pais de 1500 caracteres para preservação integral de contexto.
     - **Recursivo Determinístico (UUIDv5)**: Divisão com hashing idempotente para reindexação segura.
     - **Semântico**: Quebras baseadas em parágrafos e continuidade temática.
   - Veja a tabela de chunks gerados, seus identificadores e a relação Pai/Filho.

3. **Busca Híbrida Real (Tri-Canal RRF)**:
   - Digite perguntas para pesquisar nos documentos que acabou de subir.
   - A busca executa **3 canais em paralelo**:
     - Canal Denso (Similaridade semântica)
     - Canal Léxico (BM25 com TF-IDF e matching exato de termos)
     - Canal em Grafo (Entidades e triplas)
   - Exibe a tabela de **Fusão RRF (Reciprocal Rank Fusion)**: $\text{Score} = \sum \frac{w_i}{k + \text{rank}_i}$.

4. **Teste de Governança e Segurança (ACL Fail-Closed)**:
   - Alterne o seletor de `Tenant ID` para testar o isolamento multi-tenant: se o usuário não possuir as credenciais autorizadas, **zero dados são retornados** e a consulta é rejeitada em nível de índice!

5. **Geração de Resposta Contextualizada**:
   - Se sua chave `ANTHROPIC_API_KEY` estiver no `.env`, a resposta é gerada pelo **Claude 3.5 Sonnet** usando o contexto expandido do chunk pai.
   - Caso contrário, a demo gera uma síntese local precisa baseada nas evidências extraídas do RRF.

---

## 🛠️ Como Executar em Localhost

### Opção 1: Inicialização em 1 Clique (Windows)
Basta dar **dois cliques** no arquivo:
```bash
demo/run_demo.bat
```
O script ativará o ambiente virtual `.venv`, iniciará o servidor FastAPI na porta `8000` e abrirá o seu navegador automaticamente em `http://localhost:8000`!

---

### Opção 2: Inicialização via Terminal (PowerShell / Bash)

1. **Ative seu ambiente virtual:**
   ```bash
   # Windows (PowerShell):
   .venv\Scripts\Activate.ps1

   # Linux / macOS:
   source .venv/bin/activate
   ```

2. **Inicie o servidor da demo:**
   ```bash
   python -m uvicorn demo.server:app --host 127.0.0.1 --port 8000 --reload
   ```

3. **Abra no seu navegador:**
   👉 [http://localhost:8000](http://localhost:8000)

---

## 📁 Estrutura da Demo

```
demo/
├── README.md              # Este guia completo de instruções
├── run_demo.bat           # Script 1-clique para Windows
├── run_demo.sh            # Script 1-clique para Linux/macOS
├── server.py              # Servidor backend FastAPI (endpoints /api/upload e /api/search)
├── static/
│   └── index.html         # Frontend moderno com Tailwind CSS, drag & drop e dashboard RRF
└── uploads/               # Diretório temporário para arquivos enviados pelo usuário
```

---

## 🧪 Roteiro de Teste Recomendado para Apresentações:

1. **Teste com PDF ou Planilha:**
   - Arraste um PDF financeiro ou planilha `.xlsx` na área de upload.
   - Deixe a estratégia como **Parent-Child**.
   - Clique em **"Processar Arquivo & Gerar Chunks"**.
   - Observe os logs do parser, a contagem de elementos e a lista de chunks filhos gerados com a referência ao ID do pai.

2. **Teste de Busca Híbrida:**
   - Na seção de busca, faça uma pergunta específica sobre um dado ou valor presente no arquivo.
   - Observe o ranking RRF unificando os scores do canal denso e do BM25.
   - Veja a resposta sintetizada pelo assistente citando o documento de origem.

3. **Teste do Bloqueio de Segurança (ACL):**
   - Mude o seletor `Testar Segurança ACL` para `tenant_desconhecido`.
   - Clique em **Buscar**.
   - Veja a recusa imediata **403 Fail-Closed** sem chamada ao LLM nem vazamento de dados!
