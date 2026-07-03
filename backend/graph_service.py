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