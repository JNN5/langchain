from opensearch_embeddings import OpenSearchEmpeddings
from opensearch_client import MyOpenSearchVectorSearch, Indicies
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.llms.bedrock import Bedrock


embeddings = OpenSearchEmpeddings()  # Bedrock is using more dimensions

vector_store = MyOpenSearchVectorSearch.init(
    index=Indicies.CA_EMBEDDINGS_INDEX,
    embeddings=embeddings,
)

retriever = vector_store.as_retriever(
    search_kwargs={
        "space_type": "l2",
        "vector_field": "othertext_embedding",
        "text_field": "othertext",
        "metadata_field": "*",
        "k": 100,
    }
)


def format_docs(docs):
    formatted_docs = "\n\n".join([d.page_content for d in docs])[:8191]
    # print(f"Formatted docs:\n{formatted_docs}")
    return formatted_docs


# RAG prompt
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


def get_llm():
    model_kwargs = {  # AI21
        "maxTokens": 1024,
        "temperature": 0,
        "topP": 0.5,
        "stopSequences": ["Human:"],
        "countPenalty": {"scale": 0},
        "presencePenalty": {"scale": 0},
        "frequencyPenalty": {"scale": 0},
    }

    llm = Bedrock(
        model_id="ai21.j2-ultra-v1",  # set the foundation model
        model_kwargs=model_kwargs,
    )  # configure the properties for Claude

    return llm


# RAG
model = get_llm()
chain = (
    RunnableParallel(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
    )
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)


# ---
# Tests
# ---


def get_docs(query):
    return vector_store.similarity_search(
        query,
        search_type="script_scoring",
        space_type="l2",
        vector_field="othertext_embedding",
        text_field="othertext",
        metadata_field="*",
    )


def semantic_query(query_text):
    query = {
        "size": 5,
        "query": {
            "neural": {
                "othertext_embedding": {
                    "query_text": query_text,
                    "k": 5,
                }
            }
        },
    }
    return vector_store.client.search(
        body=query, index=Indicies.CA_EMBEDDINGS_INDEX.value
    )


def invoke(text):
    return chain.invoke({"question": text})
