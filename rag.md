Good intro
https://www.youtube.com/watch?v=T-D1OfcDW1M

### Why RAG?
LLM Challenges: *No source* (Needs evidence), and they are *Out of date*.

In RAG, your data is loaded and prepared for queries or "indexed". User queries act on the index, which filters your data down to the most relevant context. This context and your query then go to the LLM along with a prompt, and the LLM provides a response.

![[basic_rag.png]]

## Stages within RAG

There are five key stages within RAG, which in turn will be a part of any larger application you build. These are:

- **Loading**: this refers to getting your data from where it lives -- whether it's text files, PDFs, another website, a database, or an API -- into your pipeline. [LlamaHub](https://llamahub.ai/) provides hundreds of connectors to choose from.
	- [**Nodes and Documents**](https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/): A `Document` is a container around any data source - for instance, a PDF, an API output, or retrieve data from a database. A `Node` is the atomic unit of data in LlamaIndex and represents a "chunk" of a source `Document`. Nodes have metadata that relate them to the document they are in and to other nodes.
	- [**Connectors**](https://docs.llamaindex.ai/en/stable/module_guides/loading/connector/): A data connector (often called a `Reader`) ingests data from different data sources and data formats into `Documents` and `Nodes`.
    
- **Indexing**: this means creating a data structure that allows for querying the data. For LLMs this nearly always means creating `vector embeddings`, numerical representations of the meaning of your data, as well as numerous other metadata strategies to make it easy to accurately find contextually relevant data.
	- [**Indexes**](https://docs.llamaindex.ai/en/stable/module_guides/indexing/): Once you've ingested your data, LlamaIndex will help you index the data into a structure that's easy to retrieve. This usually involves generating `vector embeddings` which are stored in a specialized database called a `vector store`. Indexes can also store a variety of metadata about your data.
	- [**Embeddings**](https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings/) LLMs generate numerical representations of data called `embeddings`. When filtering your data for relevance, LlamaIndex will convert queries into embeddings, and your vector store will find data that is numerically similar to the embedding of your query.
    
- **Storing**: once your data is indexed you will almost always want to store your index, as well as other metadata, to avoid having to re-index it.
    
- **Querying**: for any given indexing strategy there are many ways you can utilize LLMs and LlamaIndex data structures to query, including sub-queries, multi-step queries and hybrid strategies.
	- [**Retrievers**](https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/): A retriever defines how to efficiently retrieve relevant context from an index when given a query. Your retrieval strategy is key to the relevancy of the data retrieved and the efficiency with which it's done.
	- [**Routers**](https://docs.llamaindex.ai/en/stable/module_guides/querying/router/): A router determines which retriever will be used to retrieve relevant context from the knowledge base. More specifically, the `RouterRetriever` class, is responsible for selecting one or multiple candidate retrievers to execute a query. They use a selector to choose the best option based on each candidate's metadata and the query.
	- [**Node Postprocessors**](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/): A node postprocessor takes in a set of retrieved nodes and applies transformations, filtering, or re-ranking logic to them.
	- [**Response Synthesizers**](https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/): A response synthesizer generates a response from an LLM, using a user query and a given set of retrieved text chunks.
    
- **Evaluation**: a critical step in any pipeline is checking how effective it is relative to other strategies, or when you make changes. Evaluation provides objective measures of how accurate, faithful and fast your responses to queries are.

## When doing a RAG
What do we need to check when creating a RAG?
https://youtu.be/Kp_AGKtql_U?si=ZL1iPq8luOdnt8kR&t=1395
#### 1.  Vector DB Creation
- **Embedding Selection:** What embedding models to use?
	- https://platform.openai.com/docs/guides/embeddings/embedding-models
	- https://huggingface.co/BAAI/bge-small-en-v1.5/tree/main?ref=hackernoon.com
- **Data Selection:** How do I select the data?
- Distance Metric
- Index type

#### 2. Retrieval
- How many chunks? (top k)
- Retrieval Method
	- Naive RAG - top 3 / top k
	- Triad of metrics: https://capture.dropbox.com/FXiMdM06bQH7uzpL
	- Auto-merging retrieval
	- Sentence-window retrieval
- Dynamic Retrieval
	- Rerankers
- Chunk Size?
- Context Filters
- 

#### 3. Completion Stage
- Local model
- Model size
- Temperature
- Logit bias
- Function calling?

## Libraries
- LlamaIndex
- LangChain

### LlamaIndex vs LangChain

- Opt for LlamaIndex if you are building an application with a keen focus on search and retrieval efficiency and simplicity, where high throughput and processing of large datasets are essential.
- Choose LangChain if you aim to construct more complex, flexible LLM applications that might include custom query processing pipelines, multimodal integration, and a need for highly adaptable performance tuning.

- https://www.useready.com/blog/rag-wars-llama-index-vs-langchain-showdown/
- https://docs.pinecone.io/integrations/llamaindex

#### How to improve RAG(Retrieval Augmented Generation) performance
https://medium.com/@sthanikamsanthosh1994/how-to-improve-rag-retrieval-augmented-generation-performance-2a42303117f8

## Examples
https://hackernoon.com/heres-why-extraction-matters-the-most
https://hackernoon.com/lite/heres-why-extraction-matters-the-most

For our experiments, we ran using 3 different extractors ([pypdf](https://pypi.org/project/pypdf/?ref=hackernoon.com), [pymupdf](https://pypi.org/project/pymupdf/?ref=hackernoon.com), and [unstructured](https://pypi.org/project/unstructured/?ref=hackernoon.com)), 5 different chunking and indexing strategies ([recursive chunking](https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter?ref=hackernoon.com), [parent-document chunking](https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever?ref=hackernoon.com), [semantic chunking](https://python.langchain.com/docs/modules/data_connection/document_transformers/semantic-chunker?ref=hackernoon.com), and [questions-answered multi-vector embedding](https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector?ref=hackernoon.com#hypothetical-queries) with both recursive and semantic chunking), 2 different embeddings (OpenAI [ada-002](https://platform.openai.com/docs/guides/embeddings/embedding-models?ref=hackernoon.com) and [bge-en-small](https://huggingface.co/BAAI/bge-small-en-v1.5/tree/main?ref=hackernoon.com)) and 2 different search strategies (similarity and maximum-marginal relevance). We ran the experiment and evaluation on the cross-product of all of these factors.

## Evaluation

As more and more developers and businesses adopt RAG for building GenAI applications, evaluating their effectiveness is becoming increasingly important.

- https://medium.com/@zilliz_learn/optimizing-rag-applications-a-guide-to-methodologies-metrics-and-evaluation-tools-for-enhanced-a9ae3d9c7149
- https://arxiv.org/pdf/2307.03109.pdf
- https://news.ycombinator.com/item?id=37579771
- 
### RAG evaluation tools
- [Ragas](https://docs.ragas.io/en/latest/concepts/testset_generation.html)
- LlamaIndex
- [TruEra’s TruLens](https://venturebeat.com/ai/truera-launches-free-tool-for-testing-llm-apps-for-hallucinations/)
	- Example: https://github.com/truera/trulens/blob/main/trulens_eval/examples/expositional/vector-dbs/milvus/milvus_evals_build_better_rags.ipynb
- Phoenix
- DeepEval
- LangSmith
	- There are a variety of frameworks for evaluating RAG and LLM usage, such as [TruLens](https://www.trulens.org/?ref=hackernoon.com) and [RAGAS](https://docs.ragas.io/en/latest/index.html?ref=hackernoon.com) – at the time of writing, LangSmith supports evaluation but doesn’t consider the context, making it less suited for evaluating RAG.
- Open AI Evals
- PromptFlow
- [W&B Prompts](https://venturebeat.com/ai/weights-biases-new-llmops-capabilities-ai-development-model-monitoring/)
- [Arize’s Pheonix](https://venturebeat.com/ai/arize-launches-phoenix-an-open-source-library-to-monitor-llm-hallucinations/)
- [MLFlow](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)

### TruLens for Evaluation
https://www.trulens.org/
- Score and Answer Relevance

Evaluate and Iterate
- Start with LlamaIndex Basic RAG
- Evaluate with TruLens RAG Triad
	- Failure modes related to context size
- Iterate with LlamaIndex Sentence Window
- Re-evaluate with TruLens RAG Triad
	- Do we see improvements in context relevance?
	- What about other metrics?
- Experiment with different window sizes
	- What window size results in the best eval metrics?
	- Too Small: Not relevant context
	- Too Big: Irrelevant context can cause not so great scores in groundedness and answer context


#### Feedback function
Provides a score after reviewing an LLM app's input, outputs and intermediate response. 

**Structure of Feedback Functions**
- https://capture.dropbox.com/8IL5h9mdGa60cN3b
- https://capture.dropbox.com/FmkKSIKBXuIcB8tT

**Feedback Functions can be implemented in different ways**
![[NGjDB_eQ.png]]
- Traditional NLP Evals
	- BLEU scores. Weakness: Quite syntactic. Overlap of words.
- MLM Evals
- LLM Evals
- Human Evals: Similar to Ground Truth Evals. May not be as much as an expert. Degree of confidence is lower. 
- Ground Truth Evals: Expensive to collect.  
	- Human Expert would give a score.


### Some of the metrics
- **Groundedness**
	- Context -> Response
	- Is the response supported by the context?
	- Checks the factual support for each statement within a response, addressing the issue of LLMs potentially generating embellished or factually incorrect statements. 
- **Context Relevance**
	- Query -> Context
	- How good is the retrieval?
	- Is the retrieved context relevant to the query?
	- Involves verifying the pertinence of retrieved content to the input query, which is crucial to preventing irrelevant context from leading to inaccurate answers; this is assessed using a Language Model to generate a context relevance score
	- https://capture.dropbox.com/gLAnAjMGdfSAScPg
- **Answer Relevance**
	- Query -> Response
	- Is the final response useful?
	- Is the response relevant to the query?
	- Ensures the response not only addresses the user's question but does so in a directly applicable manner.

More metrics:
- Honest:
	- Answer Relevance
	- Embedding distance
	- BLEU, ROUGE
	- Summarization quality
	- Context Relevance
	- Groundedness
	- Custom Evaluations
- Harmless
	- PII Detection
	- Toxicity
	- Stereotyping
	- Jailbreaks
	- Custom Evaluations
- Helpful
	- Sentiment
	- Language mismatch
	- Conceseness
	- Coherence
	- Custom Evaluations






### Other Links

- Intro to agents: https://towardsdatascience.com/intro-to-llm-agents-with-langchain-when-rag-is-not-enough-7d8c08145834
- https://blog.langchain.dev/evaluating-rag-pipelines-with-ragas-langsmith/
- https://blog.stackademic.com/how-to-evaluate-your-chatbot-a21bc4191bcb
- RAG ANALYSIS https://colab.research.google.com/drive/1ycGgTPvDRLuEOvK6kOfsxg_nKkki6ncG?ref=hackernoon.com#scrollTo=VH-7g3CdqSOI
- https://lablab.ai/t/trulens-tutorial-langchain-chatbot
- https://community.sap.com/t5/technology-blogs-by-members/generative-ai-with-sap-rag-evaluations/ba-p/13572136

![[Captura-de-pantalla-2024-01-04-a-las-18.28.42.png]]









