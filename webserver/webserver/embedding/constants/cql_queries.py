QUERY_COLBERT_ANN_CQL = """
        SELECT title, part
        FROM {keyspace}.{table}
        ORDER BY bert_embedding ANN OF ?
        LIMIT ?
        """

QUERY_COLBERT_PARTS_CQL = """
        SELECT title, part, bert_embedding
        FROM {keyspace}.{table}
        WHERE title = ? AND part = ?
        """

QUERY_PARTS_BY_PK = """
        SELECT body
        FROM {keyspace}.{table}
        WHERE title = ? AND part = ?
        """
