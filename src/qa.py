from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from .config import settings

PROMPT_TEMPLATE = """Use the following context to answer the question.

If the answer is not in the context, say you don't know.

Quote key figures precisely.

Keep answers concise.

Do not invent content.

Context:
{context}

Question:
{question}

Answer:
"""

def build_llm():
    return ChatGroq(
        groq_api_key=settings.GROQ_API_KEY,
        model_name=settings.GROQ_MODEL,
        temperature=0,
        timeout=60,
    )

def build_qa_chain(retriever):
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    llm = build_llm()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa
