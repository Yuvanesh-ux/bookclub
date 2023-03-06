from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import PagedPDFSplitter
from langchain.document_loaders import WebBaseLoader
from langchain.agents import initialize_agent, Tool
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
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
            # try:
            embeddings = np.load(embedding_path, allow_pickle=True)
            print("Found embeddings on disk")

            # except BaseException:
            #     print("Embeddings not found! embedding using OpenAI")
            #     embeddings = OpenAIEmbeddings(openai_api_key=self.open_api_key)
            #     np.save(f'{doc_path[:-4]}_embeddings.npy', np.array(embeddings))

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
    bookclub = Bookclub("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")
    lotf_agent = bookclub.qa_bot('./lotf_text.pdf', './lotf_text_embeddings.npy')

    print(lotf_agent.run("Who is jack"))

