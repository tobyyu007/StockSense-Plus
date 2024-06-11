# https://github.com/vndee/local-rag-example

from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from sentence_transformers import SentenceTransformer


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="llama3")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100, add_start_index=True,)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] The following is a conversation with an AI Large Language Model. 
            The AI has been trained to answer questions, provide recommendations, and help with decision making. 
            The AI follows user requests. The AI thinks outside the box. [/INST] </s> 
            [INST] Question: {question} 
            Answer: [/INST]
            """
        )
        self.chain = ({"question": RunnablePassthrough()}
                    | self.prompt
                    | self.model
                    | StrOutputParser())


    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        # vector_store = Chroma.from_documents(documents=chunks, embedding=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'))
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        self.prompt = PromptTemplate.from_template(
            """
            Use the following context as your learned knowledge, inside <context></context> XML tags.
            <context>
                {context}
            </context>

            When answer to user:
            - If you don't know, just say that you don't know.
            - If you don't know when you are not sure, ask for clarification.
            Avoid mentioning that you obtained the information from the context.
            And answer according to the language of the user's question.

            Given the context information, answer the query.
            Query: {question}
            """
        )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())
        

    def ask(self, query: str):
        if not self.chain:
            return self.model.invoke(query)

        print(query, self.retriever)
        return self.chain.invoke(query)


    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
