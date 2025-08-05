# retriever2 usees FAISS instead of Chroma
#
#  
from typing import Iterable, List, Optional
from functools import wraps, lru_cache
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from bidi.algorithm import get_display
import arabic_reshaper
import time
from langchain_community.vectorstores import FAISS
from functools import wraps
import logging


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        logging.getLogger(__name__).debug(
            "%s executed in %.4f seconds", func.__name__, duration
        )
        return result
    return wrapper

@lru_cache(maxsize=1024)
def process_farsi_text_cached(text: str) -> str:
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)


class Retriever:
    @timer
    def __init__(self, 
                 new_vdb=True, 
                 embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                                                       model_kwargs={"local_files_only": True}
)
                ) -> None:
        self.new_vdb = new_vdb
        self.vdb = None
        self.retrieved_docs = None
        self.is_persian = True
        self.embedding_model=embedding_model

    @timer
    def create_vdb(self,
        raw_texts: Iterable[str],
        chunk_size: int,
        chunk_overlap: int,
        vdb_dir: str = "faiss_vdb",) -> None:
        if not raw_texts:
            raise ValueError("raw_texts must be a non-empty iterable of strings")
        # If not creating a new VDB and one already exists, skip rebuilding
        if not self.new_vdb:
            try:
                self.vdb = FAISS.load_local(
                    folder_path=vdb_dir,
                    embeddings=self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                logging.getLogger(__name__).debug("Reusing existing vector DB at %s", vdb_dir)
                return
            except Exception:  # fallback to rebuilding if load fails
                logging.getLogger(__name__).warning(
                    "Failed to load existing VDB; rebuilding because new_vdb=False but load failed."
                )

        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs: List[Document] = []
        for text in raw_texts:
            docs.extend(Document(page_content=chunk) for chunk in splitter.split_text(text))

        # Build (and auto-persist) vector DB
        self.vdb = FAISS.from_documents(
            documents=docs,
            embedding=self.embedding_model
            )
        self.vdb.save_local(vdb_dir)
        logging.getLogger(__name__).debug("Built new vector DB at %s", vdb_dir)


    @timer
    def load_vdb(self, query: str, vdb_dir: str = "faiss_vdb", k: int = 3) -> List[Document]:
        if not query:
            raise ValueError("query must be non-empty")
        self.vdb = FAISS.load_local(
                    folder_path=vdb_dir,
                    embeddings=self.embedding_model,
                    allow_dangerous_deserialization=True
                                    )       
        # self.retrieved_docs = self.vdb.similarity_search_with_relevance_scores(query, k=k)
        # for doc in self.retrieved_docs:
            # print(doc)
        self.retrieved_docs = self.vdb.similarity_search(query, k=k)

        return self.retrieved_docs

    @timer
    def display(self, retrieved_docs: Iterable[Document]) -> None:
        for doc in retrieved_docs:
            content = (
                process_farsi_text_cached(doc.page_content) if self.is_persian else doc.page_content
            )
            print(content, '\n', '*'*150)