import time
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    get_response_synthesizer,
)
from llama_index.core.embeddings import resolve_embed_model
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index_demo import db


class LLaMaGo:
    # Default database name
    data_path = "data"
    data_name = "paul_graham_essay"
    db_name = "vector_db"
    embed_model = "local:BAAI/bge-small-en-v1.5"
    model_timeout = 600.0
    model_embed_dim = 384  # llama2 embedding dimension

    def __init__(self):
        Settings.embed_model = resolve_embed_model(self.embed_model)
        Settings.llm = Ollama(model="llama2", request_timeout=self.model_timeout)

    def __call__(self, question=""):
        if self.query_engine == None:
            print(f"Query engine is not initialized.")
            return
        streaming_response = self.query_engine.query(question)
        #streaming_response.print_response_stream()
        return streaming_response

    def create_index(self):
        self.documents = SimpleDirectoryReader(self.data_path).load_data()
        self.index = VectorStoreIndex.from_documents(self.documents)
        self.query_engine = self.index.as_query_engine(streaming=True)

    def create_index_with_db(self):
        vector_store = self._get_vector_store_from_pg(should_clear_table=True)
        if vector_store == None:
            print(f"No vector_store craeted.")
            return
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # Create index from storage_context
        self.documents = SimpleDirectoryReader(self.data_path).load_data()
        self.index = VectorStoreIndex.from_documents(
            self.documents, storage_context=storage_context, show_progress=True
        )
        self.query_engine = self.index.as_query_engine(streaming=True)

    def use_db_as_index(self):
        vector_store = self._get_vector_store_from_pg()
        self.index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        self.query_engine = self.index.as_query_engine(streaming=True)

    def _get_vector_store_from_pg(self, should_clear_table=False):
        # Prepare db connection for saving embedding info.
        pgdb = db.PgDatabase(self.db_name)
        pgdb.check_vector_db()
        # Clear exist data on current table.
        if should_clear_table == True:
            pgdb.clear_table(f"data_{self.data_name}")
        # Get url and do some stuff.
        db_url = pgdb.get_url_params()
        vector_store = PGVectorStore.from_params(
            database=self.db_name,
            host=db_url.host,
            password=db_url.password,
            port=db_url.port,
            user=db_url.username,
            table_name=self.data_name,
            embed_dim=self.model_embed_dim,
        )
        return vector_store
