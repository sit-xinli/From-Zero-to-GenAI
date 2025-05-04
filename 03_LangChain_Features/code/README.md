# LangChain Feature Examples

This directory contains Python scripts demonstrating various features of the LangChain framework, aimed at helping teams understand base concepts.

## Files

- `01_langchain_basics.py`: Covers fundamental concepts:

  - Initializing LLMs (`ChatOpenAI`)
  - Basic LLM calls
  - Prompt Templates (`PromptTemplate`)
  - Basic Chains (`LLMChain`)
  - Sequential Chains (`SequentialChain`)
  - Memory (`ConversationBufferMemory`)
  - Output Parsers (`CommaSeparatedListOutputParser`, `PydanticOutputParser`)

- `02_langchain_retrieval_agents.py`: Covers retrieval and agent concepts:

  - Document Loading (in-memory) & Text Splitting (`RecursiveCharacterTextSplitter`)
  - Embeddings (`OpenAIEmbeddings`)
  - Vector Stores (`FAISS`) & Similarity Search
  - Agents (`initialize_agent`, `ZERO_SHOT_REACT_DESCRIPTION`)
  - Tools (`DuckDuckGoSearchResults`, Custom Python Functions)
  - Streaming LLM output

- `03_langchain_advanced.py`: Covers more advanced concepts:

  - Explicit Retrievers (`vector_store.as_retriever()`)
  - RetrievalQA Chain
  - Summarization Chain (`load_summarize_chain`)
  - Router Chains (`MultiPromptChain`, `LLMRouterChain`)
  - Advanced Memory (`ConversationSummaryMemory`, `VectorStoreRetrieverMemory`)
  - Callbacks (`StdOutCallbackHandler`) & LangSmith Introduction
  - Evaluation (`QAEvalChain`)

- `requirements.txt`: Lists the required Python packages.
- `.env` (You need to create this file): To store your API keys securely.

## Setup

1.  **Clone the repository or download the code.**
2.  **Create a Python virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    - Add `python-dotenv` to `requirements.txt`:
      ```
      langchain
      langchain-openai
      faiss-cpu
      pydantic
      python-dotenv
      ```
4.  **Set up API Key:**

    - Create a file named `.env` in this `code` directory.
    - Add your OpenAI API key to the `.env` file:
      ```
      OPENAI_API_KEY='your_openai_api_key_here'
      ```
    - Replace `'your_openai_api_key_here'` with your actual key.

5.  **(Optional) Set up LangSmith for Tracing:**
    - If you want to use LangSmith for debugging and monitoring (demonstrated conceptually in `03_langchain_advanced.py`):
      - Sign up at [https://smith.langchain.com/](https://smith.langchain.com/).
      - Create an API Key from the LangSmith settings page.
      - Set the following environment variables (you can add them to your `.env` file or export them in your terminal):
        ```
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_API_KEY='your_langsmith_api_key_here'
        LANGCHAIN_PROJECT='Your-Project-Name' # Optional, but recommended
        ```

## Running the Examples

Execute the Python scripts from your terminal:

```bash
python3 01_langchain_basics.py
```

```bash
python3 02_langchain_retrieval_agents.py
```

```bash
python3 03_langchain_advanced.py
```

Observe the output in the console to understand how each LangChain component works.
