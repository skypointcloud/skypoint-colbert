from cassandra.cluster import Session
from constants import (
    QUERY_COLBERT_ANN_CQL,
    QUERY_COLBERT_PARTS_CQL,
    QUERY_PARTS_BY_PK,
)


class AstraDB:
    """
    Singleton class to manage the connection to the AstraDB instance for Colbert embeddings
    It uses previously setup connection to the AstraDB instance
    Prepared statements are created for the queries
    
    session: Session - The session object for the connection
    keyspace: str - The keyspace in which the tables are present
    text_table: str - The table in which the text data is stored
    embedding_table: str - The table in which the embeddings are stored
    timeout: int - The timeout for the queries
    """
    _instances: dict = {}

    def __new__(
        cls,
        session: Session,
        keyspace: str,
        text_table: str,
        embedding_table: str,
        timeout: int = 60,
        **kwargs: dict,
    ) -> "AstraDB":
        if keyspace not in cls._instances:
            instance = object.__new__(cls)
            instance.__init__(session, keyspace, text_table, embedding_table, **kwargs)
            cls._instances[keyspace] = instance
        return cls._instances[keyspace]

    def __init__(
        self,
        session: Session,
        keyspace: str,
        text_table: str,
        embedding_table: str,
        timeout: int = 60,
        **kwargs: dict,
    ):
        self.keyspace = keyspace
        self.text_table = text_table
        self.embedding_table = embedding_table
        self.session = session
        self.session.default_timeout = timeout

        self.query_colbert_ann_stmt = self.session.prepare(
            QUERY_COLBERT_ANN_CQL.format(keyspace=keyspace, table=embedding_table)
        )
        self.query_colbert_parts_stmt = self.session.prepare(
            QUERY_COLBERT_PARTS_CQL.format(keyspace=keyspace, table=embedding_table)
        )
        self.query_part_by_pk_stmt = self.session.prepare(
            QUERY_PARTS_BY_PK.format(keyspace=keyspace, table=text_table)
        )

