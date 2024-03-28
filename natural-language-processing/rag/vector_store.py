import os
import pymongo
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings


class VectorStore:
    def __init__(self):
        uri = os.environ["MONGO_CLUSTER_URI"]
        self.client = pymongo.MongoClient(uri)
        self.db_name = "langchain_db"
        self.col_name = "profile"
        self.search_index = "all-MiniLM-L6-v2"
        self.embeddings = SentenceTransformerEmbeddings(model_name=self.search_index)
        self.vector_search = MongoDBAtlasVectorSearch(
            collection=self.client[self.db_name][self.col_name], 
            embedding=self.embeddings,
            index_name=self.search_index
        )

    def add_document(self, doc: str):
        self.vector_search.add_documents([doc])

