"""
HNSW Semantic Document Search Engine - Web Application
======================================================
Flask REST API and Web Interface for real-time semantic document search,
HNSW multi-layer routing inspection, and algorithm benchmarking.
"""

import os
import sys
import time
import json
import numpy as np

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from hnsw import HNSWIndex
from embedder import SemanticEmbedder
from corpus import DocumentStore
from pdf_processor import extract_chunks_from_pdf
from sklearn.decomposition import PCA
CACHE_FILE = os.path.join(BASE_DIR, "embeddings_cache.npz")
CORPUS_FILE = os.path.join(BASE_DIR, "corpus_data.json")

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=os.path.join(BASE_DIR, "static"))
CORS(app)

# Global State
store = None
embedder = None
hnsw_index = None
cached_embeddings = {} # doc_id -> vector
pca_model = None
doc_coordinates = {} # doc_id -> {"x": float, "z": float}
pca_scale_x = 35.0
pca_scale_z = 35.0


def update_3d_coordinates():
    global pca_model, doc_coordinates, pca_scale_x, pca_scale_z
    if not cached_embeddings:
        return
    doc_ids = list(cached_embeddings.keys())
    vectors = np.array([cached_embeddings[i] for i in doc_ids], dtype=np.float32)
    
    n_samples = len(doc_ids)
    if n_samples < 2:
        doc_coordinates = {doc_ids[0]: {"x": 0.0, "z": 0.0}}
        return

    n_comp = min(2, n_samples)
    pca_model = PCA(n_components=n_comp, random_state=42)
    coords_2d = pca_model.fit_transform(vectors)
    
    if n_comp == 1:
        coords_2d = np.hstack([coords_2d, np.zeros((n_samples, 1))])

    std_x = float(coords_2d[:, 0].std()) + 1e-9
    std_z = float(coords_2d[:, 1].std()) + 1e-9
    mean_x = float(coords_2d[:, 0].mean())
    mean_z = float(coords_2d[:, 1].mean())

    pca_scale_x = 42.0 / std_x
    pca_scale_z = 42.0 / std_z

    doc_coordinates = {}
    for idx, did in enumerate(doc_ids):
        nx = float((coords_2d[idx, 0] - mean_x) * pca_scale_x)
        nz = float((coords_2d[idx, 1] - mean_z) * pca_scale_z)
        nx = max(-95.0, min(95.0, nx))
        nz = max(-95.0, min(95.0, nz))
        doc_coordinates[did] = {"x": round(nx, 2), "z": round(nz, 2)}
    print(f"[3D Engine] Computed 2D PCA positions for {len(doc_coordinates)} documents.")


def project_to_3d(vec):
    """Chiếu vector mới (truy vấn hoặc tài liệu vừa thêm) lên mặt phẳng PCA hiện có."""
    if pca_model is None:
        return 0.0, 0.0
    try:
        c = pca_model.transform([vec])[0]
        nx = float(c[0] * pca_scale_x)
        nz = float((c[1] if len(c) > 1 else 0.0) * pca_scale_z)
        return round(max(-95.0, min(95.0, nx)), 2), round(max(-95.0, min(95.0, nz)), 2)
    except Exception as e:
        print(f"[3D Warning] PCA transform error: {e}")
        return 0.0, 0.0


MAX_BEAM_ANIMATION_STEPS = 120  # số bước tầng 0 tối đa đưa vào mô phỏng


def build_animation_steps(hnsw_res, query_x, query_z):
    trace = hnsw_res.get("trace", {})
    layers_trace = trace.get("layers", [])
    steps = []
    step_num = 1
    
    global_ep = trace.get("global_enter_point")
    max_lvl = trace.get("max_level", 0)
    
    if global_ep is not None:
        doc_info = store.get(global_ep)
        title = doc_info["title"] if doc_info else f"Node #{global_ep}"
        steps.append({
            "step": step_num,
            "type": "entry",
            "layer": max_lvl,
            "node_id": global_ep,
            "desc": f"Điểm vào toàn cục (Global Entry Node) tại Layer {max_lvl}: Node #{global_ep} - '{title[:32]}...'"
        })
        step_num += 1

    curr_node = global_ep
    
    for l_info in layers_trace:
        lvl = l_info["level"]
        if lvl > 0:
            best = l_info.get("best_node", curr_node)
            path = l_info.get("path") or ([curr_node, best] if best != curr_node else [curr_node])
            hop_dist = {h["node_id"]: h.get("distance") for h in l_info.get("hops", []) if "node_id" in h}
            for a, b in zip(path, path[1:]):
                b_doc = store.get(b)
                b_title = b_doc["title"] if b_doc else f"Node #{b}"
                steps.append({
                    "step": step_num,
                    "type": "jump",
                    "layer": lvl,
                    "from_node": a,
                    "to_node": b,
                    "dist": hop_dist.get(b, l_info.get("best_dist", 0.0)),
                    "desc": f"Layer {lvl} Nhảy tham lam: Node #{a} -> Node #{b} ('{b_title[:28]}...')"
                })
                step_num += 1
            curr_node = best
            
            steps.append({
                "step": step_num,
                "type": "drop",
                "from_layer": lvl,
                "to_layer": lvl - 1,
                "node_id": curr_node,
                "desc": f"Hạ độ cao dọc theo cột gióng đứng từ Layer {lvl} xuống Layer {lvl - 1} tại Node #{curr_node}"
            })
            step_num += 1
        else:
            visits = [h for h in l_info.get("hops", []) if h.get("action") == "visit_neighbor"]
            total_visits = trace.get("layer0_visited_count", len(visits))
            for h in visits[:MAX_BEAM_ANIMATION_STEPS]:
                nid = h.get("node_id")
                n_doc = store.get(nid)
                n_title = n_doc["title"] if n_doc else f"Node #{nid}"
                act = h.get("action", "")
                
                if act == "visit_neighbor":
                    steps.append({
                        "step": step_num,
                        "type": "beam_visit",
                        "layer": 0,
                        "from_node": h.get("from_node"),
                        "node_id": nid,
                        "dist": h.get("distance"),
                        "accepted": h.get("accepted", False),
                        "desc": f"Layer 0 Beam Search: Khám phá Node #{nid} ('{n_title[:28]}...')" + (" [✓ Chấp nhận vào beam]" if h.get("accepted") else " [✕ Loại bỏ]")
                    })
                    step_num += 1
            shown = min(len(visits), MAX_BEAM_ANIMATION_STEPS)
            if total_visits > shown:
                steps.append({
                    "step": step_num,
                    "type": "beam_truncated",
                    "layer": 0,
                    "desc": f"Mô phỏng hiển thị {shown}/{total_visits} lượt khám phá ở Layer 0; các lượt còn lại được bỏ qua cho gọn (kết quả vẫn tính trên toàn bộ)."
                })
                step_num += 1

    top_results = hnsw_res.get("results", [])
    top_ids = [r["id"] for r in top_results]
    steps.append({
        "step": step_num,
        "type": "winners",
        "layer": 0,
        "winner_ids": top_ids,
        "desc": f"Hoàn tất! Top {len(top_ids)} tài liệu tương đồng nhất được đánh dấu nổi bật tại Layer 0."
    })
    return steps


def build_insertion_animation_steps(insert_trace, node_id, node_level, nx, nz, doc_title):
    raw_steps = insert_trace.get("steps", [])
    steps = []
    step_num = 1
    
    # 1. Level assignment & initial spawn
    u_val = insert_trace.get("random_u", 0.5)
    mL_val = insert_trace.get("mL", 0.36)
    steps.append({
        "step": step_num,
        "mode": "insert",
        "type": "insert_spawn",
        "node_id": node_id,
        "layer": node_level,
        "x": nx,
        "z": nz,
        "title": doc_title,
        "u": u_val,
        "mL": mL_val,
        "desc": f"[Khởi tạo Tầng] Node #{node_id} ('{doc_title[:30]}...') xuất hiện tại Layer {node_level} (u={u_val}, mL={mL_val})."
    })
    step_num += 1

    for raw in raw_steps:
        stype = raw.get("type")
        if stype == "initial_entry":
            steps.append({
                "step": step_num,
                "mode": "insert",
                "type": "insert_ep",
                "node_id": node_id,
                "layer": node_level,
                "desc": f"[Entry Point Đầu Tiên] Node #{node_id} trở thành Điểm Vào Toàn Cục đầu tiên tại Layer {node_level}."
            })
            step_num += 1
        elif stype == "greedy_jump":
            steps.append({
                "step": step_num,
                "mode": "insert",
                "type": "insert_jump",
                "layer": raw.get("layer"),
                "from_node": raw.get("from_node"),
                "to_node": raw.get("to_node"),
                "dist": raw.get("dist"),
                "desc": f"[Layer {raw.get('layer')} Nhảy tham lam] Từ Node #{raw.get('from_node')} → Node #{raw.get('to_node')} (khoảng cách = {raw.get('dist')})."
            })
            step_num += 1
        elif stype == "layer_drop":
            steps.append({
                "step": step_num,
                "mode": "insert",
                "type": "insert_drop",
                "from_layer": raw.get("from_layer"),
                "to_layer": raw.get("to_layer"),
                "node_id": raw.get("node_id"),
                "desc": f"[Hạ tầng] Cực tiểu địa phương Layer {raw.get('from_layer')} → Hạ độ cao xuống Layer {raw.get('to_layer')} tại Node #{raw.get('node_id')}."
            })
            step_num += 1
        elif stype == "search_candidates":
            c_count = raw.get("candidates_count", 0)
            c_sample = [c["id"] for c in raw.get("candidates", [])[:8]]
            steps.append({
                "step": step_num,
                "mode": "insert",
                "type": "insert_beam",
                "layer": raw.get("layer"),
                "node_id": node_id,
                "candidates": c_sample,
                "desc": f"[Layer {raw.get('layer')} Quét chùm tia] efConstruction={insert_trace.get('ef_construction')}: Tìm được {c_count} ứng viên: {c_sample}."
            })
            step_num += 1
        elif stype == "heuristic_rng":
            lvl = raw.get("layer")
            sel = raw.get("selected_neighbors", [])
            decisions = raw.get("decisions", [])
            pruned_count = len([d for d in decisions if d.get("status") == "pruned_by_heuristic"])
            steps.append({
                "step": step_num,
                "mode": "insert",
                "type": "insert_heuristic",
                "layer": lvl,
                "node_id": node_id,
                "selected": sel,
                "decisions": decisions[:15],
                "desc": f"[Layer {lvl} Heuristic Empty-Lune] Chọn {len(sel)} láng giềng đa dạng hướng: {sel} (Loại {pruned_count} nút trong nón che khuất)."
            })
            step_num += 1
        elif stype == "connect_edges":
            lvl = raw.get("layer")
            neighs = raw.get("neighbors", [])
            steps.append({
                "step": step_num,
                "mode": "insert",
                "type": "insert_connect",
                "layer": lvl,
                "node_id": node_id,
                "neighbors": neighs,
                "desc": f"[Layer {lvl} Thiết lập liên kết] Nối cạnh 2 chiều giữa Node #{node_id} và các láng giềng: {neighs}."
            })
            step_num += 1
        elif stype == "prune_overflow":
            steps.append({
                "step": step_num,
                "mode": "insert",
                "type": "insert_prune",
                "layer": raw.get("layer"),
                "neighbor_id": raw.get("neighbor_id"),
                "removed_nodes": raw.get("removed_nodes"),
                "desc": f"[Layer {raw.get('layer')} Cắt tỉa bậc] Node #{raw.get('neighbor_id')} vượt quá M_max. Heuristic đã cắt cạnh tới {raw.get('removed_nodes')}."
            })
            step_num += 1
        elif stype == "update_entry_point":
            steps.append({
                "step": step_num,
                "mode": "insert",
                "type": "insert_ep",
                "node_id": raw.get("new_entry_point"),
                "layer": raw.get("new_max_level"),
                "desc": f"[Nâng cấp Entry Point] Node #{raw.get('new_entry_point')} (Layer {raw.get('new_max_level')}) cao hơn L_max cũ → Trở thành Điểm Vào Toàn Cục mới!"
            })
            step_num += 1

    steps.append({
        "step": step_num,
        "mode": "insert",
        "type": "insert_finish",
        "node_id": node_id,
        "layer": node_level,
        "desc": f"Hoàn tất chèn Node #{node_id} vào đồ thị HNSW! Tất cả các tầng từ Layer {node_level} xuống Layer 0 đã được kết nối tối ưu."
    })
    return steps


def initialize_engine():
    global store, embedder, hnsw_index, cached_embeddings
    print("[Engine] Initializing Semantic Document Search Engine...")
    
    store = DocumentStore(storage_path=CORPUS_FILE)
    embedder = SemanticEmbedder()
    
    # Check cache for existing embeddings
    doc_items = store.get_texts_for_embedding()
    doc_ids = [d[0] for d in doc_items]
    texts = [d[1] for d in doc_items]

    loaded_from_cache = False
    if os.path.exists(CACHE_FILE):
        try:
            data = np.load(CACHE_FILE)
            cached_ids = set(data["ids"])
            if set(doc_ids).issubset(cached_ids):
                id_to_vec = {int(i): data["vectors"][idx] for idx, i in enumerate(data["ids"])}
                cached_embeddings = {did: id_to_vec[did] for did in doc_ids}
                loaded_from_cache = True
                print(f"[Engine] Loaded {len(cached_embeddings)} embeddings directly from cache!")
        except Exception as e:
            print(f"[Engine Warning] Cache load failed ({e}), recomputing...")

    if not loaded_from_cache:
        print(f"[Engine] Computing embeddings for {len(texts)} documents on {embedder.device.upper()}...")
        t0 = time.time()
        vectors = embedder.embed_texts(texts)
        cached_embeddings = {doc_ids[i]: vectors[i] for i in range(len(doc_ids))}
        # Save cache
        try:
            np.savez_compressed(
                CACHE_FILE,
                ids=np.array(doc_ids, dtype=np.int32),
                vectors=np.array([cached_embeddings[i] for i in doc_ids], dtype=np.float32)
            )
            print(f"[Engine] Embeddings saved to cache in {time.time() - t0:.2f}s")
        except Exception as e:
            print(f"[Engine Warning] Failed to save embeddings cache: {e}")

    # Build HNSW Index
    build_hnsw_index(M=16, ef_construction=100, ef_search=50)


def build_hnsw_index(M=16, ef_construction=100, ef_search=50):
    global hnsw_index, cached_embeddings
    t0 = time.time()
    dim = embedder.dim if embedder else 384
    hnsw_index = HNSWIndex(
        d=dim,
        M=M,
        ef_construction=ef_construction,
        ef_search=ef_search,
        metric="cosine"
    )
    for doc_id, vec in cached_embeddings.items():
        hnsw_index.add_item(doc_id, vec)
    
    # Compute 3D PCA coordinates
    update_3d_coordinates()
    
    elapsed = time.time() - t0
    stats = hnsw_index.get_stats()
    print(f"[Engine] HNSW Index constructed in {elapsed*1000:.1f}ms. Total nodes: {stats['total_nodes']}, Max Level: {stats['max_level']}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats", methods=["GET"])
def get_stats():
    stats = hnsw_index.get_stats()
    stats["device"] = embedder.device.upper() if embedder else "CPU"
    stats["model_name"] = embedder.model_name if embedder else "TF-IDF"
    stats["total_corpus_docs"] = len(store.documents)
    return jsonify(stats)


@app.route("/api/3d-graph", methods=["GET"])
def get_3d_graph():
    if not doc_coordinates:
        update_3d_coordinates()

    nodes_list = []
    for did, coords in doc_coordinates.items():
        doc = store.get(did)
        if doc:
            lvl = hnsw_index.node_levels.get(did, 0)
            nodes_list.append({
                "id": did,
                "title": doc["title"],
                "category": doc["category"],
                "tags": doc.get("tags", []),
                "summary": doc.get("summary", ""),
                "level": lvl,
                "x": coords["x"],
                "z": coords["z"]
            })

    # Layer heights: L0 at 0, L1 at 70, L2 at 140...
    layer_heights = [l * 75.0 for l in range(max(1, hnsw_index.max_level + 1))]
    layer_colors = ["#10b981", "#3b82f6", "#ef4444", "#f59e0b", "#8b5cf6"]

    layers_data = []
    for l in range(hnsw_index.max_level + 1):
        edges_map = {}
        if l < len(hnsw_index.layers):
            for u_id, neighbors in hnsw_index.layers[l].items():
                edges_map[str(u_id)] = neighbors

        layers_data.append({
            "level": l,
            "y": layer_heights[l] if l < len(layer_heights) else l * 75.0,
            "color": layer_colors[l % len(layer_colors)],
            "edges": edges_map,
            "node_count": len(hnsw_index.layers[l]) if l < len(hnsw_index.layers) else 0
        })

    return jsonify({
        "nodes": nodes_list,
        "max_level": hnsw_index.max_level,
        "enter_point": hnsw_index.enter_point,
        "layers": layers_data,
        "total_nodes": len(hnsw_index.nodes),
        "dimension": hnsw_index.d
    })


@app.route("/api/search", methods=["POST"])
def search():
    data = request.get_json() or {}
    query_text = data.get("query", "").strip()
    top_k = int(data.get("top_k", 5))
    ef_search = int(data.get("ef_search", 50))
    compare_gt = bool(data.get("compare_gt", True))

    if not query_text:
        return jsonify({"error": "Empty query"}), 400

    # 1. Embed query
    t_emb_start = time.perf_counter()
    q_vec = embedder.embed_single(query_text)
    t_emb_ms = (time.perf_counter() - t_emb_start) * 1000

    # 2. HNSW Query: lần 1 không ghi vết để đo độ trễ và số phép tính khoảng cách,
    #    lần 2 ghi vết để mô phỏng (kết quả hai lần giống nhau vì thuật toán tất định)
    timed_res = hnsw_index.query(query_vec=q_vec, top_k=top_k, ef_search=ef_search, collect_trace=False)
    hnsw_res = hnsw_index.query(query_vec=q_vec, top_k=top_k, ef_search=ef_search, collect_trace=True)
    hnsw_res["latency_ms"] = timed_res["latency_ms"]
    hnsw_res["dist_comps"] = timed_res["dist_comps"]

    # 3. Enrich HNSW results with document metadata and snippets
    enriched_results = []
    for rank, r in enumerate(hnsw_res["results"], start=1):
        doc = store.get(r["id"])
        if doc:
            enriched_results.append({
                "rank": rank,
                "id": doc["id"],
                "title": doc["title"],
                "category": doc["category"],
                "tags": doc["tags"],
                "summary": doc["summary"],
                "content": doc["content"],
                "snippet": store.highlight_snippet(doc["content"], query_text),
                "distance": r["distance"],
                "similarity": r["similarity"],
                "score_pct": r["score_pct"]
            })

    response = {
        "query": query_text,
        "results": enriched_results,
        "trace": hnsw_res["trace"],
        "metrics": {
            "hnsw_latency_ms": hnsw_res["latency_ms"],
            "embedding_latency_ms": round(t_emb_ms, 2),
            "total_latency_ms": round(hnsw_res["latency_ms"] + t_emb_ms, 2),
            "hnsw_dist_comps": hnsw_res["dist_comps"],
            "total_nodes": len(hnsw_index.nodes)
        }
    }

    # 4. Ground-truth exact comparison (if requested)
    if compare_gt:
        gt_res = hnsw_index.brute_force_query(q_vec, top_k=top_k)
        hnsw_ids = [r["id"] for r in hnsw_res["results"]]
        gt_ids = [r["id"] for r in gt_res["results"]]
        recall = hnsw_index.evaluate_recall(hnsw_ids, gt_ids)
        calc_savings = round((1.0 - (hnsw_res["dist_comps"] / max(1, len(hnsw_index.nodes)))) * 100, 1)

        response["ground_truth"] = {
            "gt_latency_ms": gt_res["latency_ms"],
            "gt_dist_comps": gt_res["dist_comps"],
            "recall": recall,
            "recall_pct": round(recall * 100, 1),
            "calc_savings_pct": max(0.0, calc_savings),
            "top_gt_ids": gt_ids
        }

    # 5. Compute 3D Query Projection & Animation Steps
    query_x, query_z = project_to_3d(q_vec)

    animation_steps = build_animation_steps(hnsw_res, query_x, query_z)

    response["query_3d"] = {
        "x": query_x,
        "z": query_z,
        "animation_steps": animation_steps
    }

    return jsonify(response)


@app.route("/api/documents", methods=["GET"])
def list_documents():
    docs = store.get_all()
    # Enrich with level in HNSW
    for d in docs:
        d["level"] = hnsw_index.node_levels.get(d["id"], 0)
    return jsonify(docs)


@app.route("/api/visualize-insertion", methods=["POST"])
@app.route("/api/documents", methods=["POST"])
def add_document():
    global cached_embeddings, doc_coordinates
    data = request.get_json() or {}
    title = data.get("title", "").strip()
    content = data.get("content", "").strip()
    category = data.get("category", "Tùy chỉnh").strip() or "Tùy chỉnh"
    tags = data.get("tags", ["User Document"])
    summary = data.get("summary", "").strip()
    collect_trace = bool(data.get("collect_trace", True))

    if not title or not content:
        return jsonify({"error": "Tiêu đề và Nội dung là bắt buộc"}), 400

    doc = store.add_document(title=title, content=content, category=category, tags=tags, summary=summary)
    
    # Compute vector
    text_to_embed = f"Title: {doc['title']}. Category: {doc['category']}. Summary: {doc['summary']}. Content: {doc['content']}"
    vec = embedder.embed_single(text_to_embed)
    cached_embeddings[doc["id"]] = vec

    # Project 3D PCA coordinates for new node
    nx, nz = project_to_3d(vec)
    doc_coordinates[doc["id"]] = {"x": nx, "z": nz}

    # Insert into HNSW graph dynamically with trace collection
    insert_trace = hnsw_index.add_item(doc["id"], vec, collect_trace=collect_trace)
    doc["level"] = hnsw_index.node_levels.get(doc["id"], 0)

    # Build animation steps if trace requested
    anim_steps = []
    if collect_trace and insert_trace:
        anim_steps = build_insertion_animation_steps(
            insert_trace, doc["id"], doc["level"],
            doc_coordinates[doc["id"]]["x"], doc_coordinates[doc["id"]]["z"],
            doc["title"]
        )

    # Update cache file
    try:
        doc_ids = list(cached_embeddings.keys())
        vectors = np.array([cached_embeddings[i] for i in doc_ids], dtype=np.float32)
        np.savez_compressed(CACHE_FILE, ids=np.array(doc_ids, dtype=np.int32), vectors=vectors)
    except Exception as e:
        print(f"[Warning] Failed to update embeddings cache: {e}")

    return jsonify({
        "success": True,
        "document": doc,
        "node_3d": {
            "id": doc["id"],
            "title": doc["title"],
            "category": doc["category"],
            "level": doc["level"],
            "x": doc_coordinates[doc["id"]]["x"],
            "z": doc_coordinates[doc["id"]]["z"]
        },
        "animation_steps": anim_steps,
        "trace": insert_trace,
        "stats": hnsw_index.get_stats()
    })


@app.route("/api/documents/<int:doc_id>", methods=["DELETE"])
def delete_document(doc_id: int):
    global cached_embeddings
    if store.delete_document(doc_id):
        if doc_id in cached_embeddings:
            del cached_embeddings[doc_id]
        # Rebuild graph
        build_hnsw_index(M=hnsw_index.M, ef_construction=hnsw_index.ef_construction, ef_search=hnsw_index.ef_search)
        return jsonify({"success": True, "stats": hnsw_index.get_stats()})
    return jsonify({"error": "Document not found"}), 404


@app.route("/api/upload-pdf", methods=["POST"])
def upload_pdf():
    global cached_embeddings, doc_coordinates
    chunk_mode = request.form.get("chunk_mode", "page")
    category = request.form.get("category", "Tài liệu PDF").strip() or "Tài liệu PDF"

    # Support both direct file upload and preset file path
    file = request.files.get("file")
    preset_path = None

    if not file and request.is_json:
        data = request.get_json() or {}
        preset_path = data.get("file_path")
        chunk_mode = data.get("chunk_mode", "page")
        category = data.get("category", "Tài liệu PDF")

    if not file and not preset_path:
        return jsonify({"error": "Vui lòng chọn hoặc tải lên một file PDF"}), 400

    try:
        if file:
            filename = file.filename
            file_bytes = file.read()
            chunks_data = extract_chunks_from_pdf(
                file_bytes, filename=filename, chunk_mode=chunk_mode, category=category
            )
        else:
            filename = os.path.basename(preset_path)
            full_path = os.path.join(os.path.dirname(BASE_DIR), preset_path)
            if not os.path.exists(full_path):
                full_path = preset_path
            chunks_data = extract_chunks_from_pdf(
                full_path, filename=filename, chunk_mode=chunk_mode, category=category
            )

        if not chunks_data:
            return jsonify({"error": "Không trích xuất được nội dung hợp lệ từ file PDF này"}), 400

        print(f"[PDF Engine] Extracted {len(chunks_data)} chunks from '{filename}'. Vectorizing on {embedder.device.upper()}...")
        
        # 1. Add chunks to DocumentStore
        created_docs = []
        texts_to_embed = []
        for c in chunks_data:
            doc = store.add_document(
                title=c["title"],
                content=c["content"],
                category=c["category"],
                tags=c.get("tags", ["PDF"]),
                summary=c.get("summary", "")
            )
            created_docs.append(doc)
            text_for_vec = f"Title: {doc['title']}. Category: {doc['category']}. Summary: {doc['summary']}. Content: {doc['content']}"
            texts_to_embed.append((doc["id"], text_for_vec))

        # 2. Batch embedding on PyTorch GPU
        t0 = time.time()
        just_texts = [t[1] for t in texts_to_embed]
        vectors = embedder.embed_texts(just_texts, batch_size=32)
        print(f"[PDF Engine] Batch embedded {len(vectors)} vectors in {time.time() - t0:.2f}s")

        # 3. Dynamic insertion into HNSW
        for idx, (doc_id, _) in enumerate(texts_to_embed):
            vec = vectors[idx]
            cached_embeddings[doc_id] = vec
            nx, nz = project_to_3d(vec)
            doc_coordinates[doc_id] = {"x": nx, "z": nz}
            hnsw_index.add_item(doc_id, vec)
            # Record level for UI
            created_docs[idx]["level"] = hnsw_index.node_levels.get(doc_id, 0)

        # 4. Save cache to disk
        try:
            doc_ids = list(cached_embeddings.keys())
            vec_array = np.array([cached_embeddings[i] for i in doc_ids], dtype=np.float32)
            np.savez_compressed(CACHE_FILE, ids=np.array(doc_ids, dtype=np.int32), vectors=vec_array)
        except Exception as e:
            print(f"[Warning] Failed to update embeddings cache: {e}")

        return jsonify({
            "success": True,
            "filename": filename,
            "total_chunks": len(created_docs),
            "chunks": created_docs[:10], # sample preview
            "stats": hnsw_index.get_stats()
        })

    except Exception as e:
        print(f"[PDF Error] {e}")
        return jsonify({"error": f"Lỗi xử lý file PDF: {str(e)}"}), 500


@app.route("/api/reindex", methods=["POST"])
def reindex():
    data = request.get_json() or {}
    M = int(data.get("M", 16))
    ef_construction = int(data.get("ef_construction", 100))
    ef_search = int(data.get("ef_search", 50))
    
    build_hnsw_index(M=M, ef_construction=ef_construction, ef_search=ef_search)
    return jsonify({"success": True, "stats": hnsw_index.get_stats()})


@app.route("/api/benchmark", methods=["POST"])
def run_benchmark():
    test_queries = [
        "Cơ chế Self-Attention trong Transformer",
        "Thuật toán HNSW và vùng lưỡi liềm trống",
        "Định lý CAP và tính nhất quán phân tán",
        "AlphaFold dự đoán cấu trúc 3D protein",
        "Nguyên lý chồng chập lượng tử Qubit",
        "Phở truyền thống và nghệ thuật ninh nước dùng"
    ]
    ef_values = [4, 8, 16, 32, 64]
    
    results_by_ef = []
    for ef in ef_values:
        recalls = []
        hnsw_latencies = []
        gt_latencies = []
        hnsw_calcs = []
        
        for q in test_queries:
            q_vec = embedder.embed_single(q)
            h_res = hnsw_index.query(q_vec, top_k=5, ef_search=ef, collect_trace=False)
            g_res = hnsw_index.brute_force_query(q_vec, top_k=5)
            
            h_ids = [r["id"] for r in h_res["results"]]
            g_ids = [r["id"] for r in g_res["results"]]
            recalls.append(hnsw_index.evaluate_recall(h_ids, g_ids))
            hnsw_latencies.append(h_res["latency_ms"])
            gt_latencies.append(g_res["latency_ms"])
            hnsw_calcs.append(h_res["dist_comps"])

        results_by_ef.append({
            "ef_search": ef,
            "avg_recall_pct": round(float(np.mean(recalls)) * 100, 1),
            "avg_hnsw_latency_ms": round(float(np.mean(hnsw_latencies)), 2),
            "avg_gt_latency_ms": round(float(np.mean(gt_latencies)), 2),
            "avg_hnsw_calcs": round(float(np.mean(hnsw_calcs)), 1),
            "total_nodes": len(hnsw_index.nodes)
        })

    return jsonify({"benchmark_results": results_by_ef})


if __name__ == "__main__":
    initialize_engine()
    print("\n" + "="*70)
    print("🚀 HNSW SEMANTIC DOCUMENT SEARCH ENGINE IS READY!")
    print("🌐 Access Web UI at: http://localhost:5000 (or http://127.0.0.1:5000)")
    print("="*70 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
