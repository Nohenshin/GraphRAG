from pyvis.network import Network

class GraphService:
    def build_graph(self, chunks, triplets, max_nodes=30) -> str:
        net = Network(height="350px", width="100%", bgcolor="#ffffff", font_color="black")
        node_ids = set()
        chunk_nodes = []

        # Thêm các chunk
        if chunks:
            for idx, chunk in enumerate(chunks[:10]):
                if isinstance(chunk, tuple) and len(chunk) >= 2:
                    cid = chunk[0]
                    text = chunk[1][:50] + "..." if len(chunk[1]) > 50 else chunk[1]
                elif isinstance(chunk, dict):
                    cid = chunk.get("id", f"chunk_{idx}")
                    text = chunk.get("text", "")[:50] + "..."
                else:
                    cid = f"chunk_{idx}"
                    text = str(chunk)[:50]
                node_id = f"C_{cid}"
                net.add_node(node_id, label=text, color="#4b8bff", size=15, title=cid)
                node_ids.add(node_id)
                chunk_nodes.append(node_id)

        # Thêm triplets
        for subj, rel, obj, chunk_id, _ in triplets[:10]:
            subj_id = f"T_{subj}"
            obj_id = f"T_{obj}"
            if subj_id not in node_ids:
                net.add_node(subj_id, label=subj, color="#4bff4b", size=20)
                node_ids.add(subj_id)
            if obj_id not in node_ids:
                net.add_node(obj_id, label=obj, color="#4bff4b", size=20)
                node_ids.add(obj_id)
            net.add_edge(subj_id, obj_id, label=rel, title=rel)
            for cnode in chunk_nodes:
                if cnode.startswith(f"C_{chunk_id}"):
                    net.add_edge(cnode, subj_id, title="mentions")
                    net.add_edge(cnode, obj_id, title="mentions")
                    break

        return net.generate_html()
    
    def get_chunk_graph(self, chunk_id: str) -> str:
        """
        Xây dựng đồ thị cho một chunk cụ thể:
        - Node trung tâm là Chunk (hiển thị text)
        - Các node Term (HAS_TERM)
        - Các node Entity (MENTIONS_ENTITY)
        - Các relationship giữa Entity (RELATES_TO)
        """
        from graphrag.connectors.neo4j_connection import get_connection
        neo4j = get_connection()
        
        # 1. Lấy chunk text
        chunk_query = "MATCH (c:Chunk {id: $chunk_id}) RETURN c.text AS text, c.id AS id"
        chunk_result = neo4j.run_query(chunk_query, {"chunk_id": chunk_id})
        if not chunk_result:
            return "<p>Chunk not found</p>"
        chunk_text = chunk_result[0]["text"][:100] + "..." if len(chunk_result[0]["text"]) > 100 else chunk_result[0]["text"]
        
        # 2. Lấy các Term liên kết với chunk
        term_query = """
        MATCH (c:Chunk {id: $chunk_id})-[:HAS_TERM]->(t:Term)
        RETURN t.text AS term, t.type AS type
        """
        terms = neo4j.run_query(term_query, {"chunk_id": chunk_id})
        
        # 3. Lấy các Entity liên kết với chunk
        entity_query = """
        MATCH (c:Chunk {id: $chunk_id})-[:MENTIONS_ENTITY]->(e:Entity)
        RETURN e.name AS entity
        """
        entities = neo4j.run_query(entity_query, {"chunk_id": chunk_id})
        entity_names = [row["entity"] for row in entities]
        
        # 4. Lấy các relationship giữa các entity đó
        relations = []
        if entity_names:
            relation_query = """
            MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity)
            WHERE e1.name IN $entities AND e2.name IN $entities
            RETURN e1.name AS source, r.name AS relation, e2.name AS target
            """
            relations = neo4j.run_query(relation_query, {"entities": entity_names})
        
        # 5. Xây dựng pyvis network
        net = Network(height="400px", width="100%", bgcolor="#ffffff", font_color="black")
        
        # Thêm node Chunk
        net.add_node("chunk", label=f"Chunk: {chunk_text}", color="#ff6b6b", size=30, shape="box")
        
        # Thêm các node Term (màu xanh dương)
        for term_row in terms:
            term = term_row["term"]
            net.add_node(term, label=term, color="#4b8bff", size=20)
            net.add_edge("chunk", term, label="HAS_TERM", title="HAS_TERM")
        
        # Thêm các node Entity (màu xanh lá)
        for entity in entity_names:
            net.add_node(entity, label=entity, color="#4bff4b", size=25)
            net.add_edge("chunk", entity, label="MENTIONS", title="MENTIONS")
        
        # Thêm các relationship giữa các entity
        for rel in relations:
            source = rel["source"]
            target = rel["target"]
            relation = rel["relation"]
            net.add_edge(source, target, label=relation, title=relation)
        
        return net.generate_html()