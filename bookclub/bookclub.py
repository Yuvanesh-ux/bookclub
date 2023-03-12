from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import PagedPDFSplitter
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import numpy as np

class Bookclub:
    def __init__(self, open_api_key):
        self.open_api_key = open_api_key
        self.llm = OpenAI(temperature=0, openai_api_key=open_api_key)
    def qa_bot(self, doc_path, embedding_path = ""):
        """
        """

        def book_embeddor(doc_path, embedding_path):
            loader = PagedPDFSplitter(doc_path)
            documents = loader.load_and_split()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(documents)

            print("Embedding!")
            embeddings = OpenAIEmbeddings(openai_api_key=self.open_api_key)

            docsearch = Chroma.from_documents(texts, embeddings, collection_name="lotf-text")

            return docsearch
        
        lotf_agent = VectorDBQA.from_chain_type(llm=self.llm, chain_type="stuff", vectorstore=book_embeddor(doc_path=doc_path, embedding_path=embedding_path))

        tools = [
            Tool(
                name = "Lord of The Flies QA System",
                func=lotf_agent.run,
                description="useful for when you need to answer questions about Lord of the Flies. Input should be a fully formed question."
            ),
        ]

        print("Creating agent")
        agent = initialize_agent(tools, self.llm, agent="zero-shot-react-description", verbose=True)

        return agent

if __name__ == '__main__':
    load_dotenv()

    bookclub = Bookclub(os.getenv('OPENAI_API_KEY'))
    lotf_agent = bookclub.qa_bot('data/lotf_text.pdf')

    print(lotf_agent.run("What happened during the latter half of chapter 3"))

