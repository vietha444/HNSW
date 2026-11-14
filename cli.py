"""
Command-Line Interface (CLI) for HNSW Semantic Document Search
==============================================================
Allows quick searching and testing directly from the terminal.
Usage:
    python cli.py --query "Cơ chế Attention trong mạng học sâu"
    python cli.py --benchmark
    python cli.py --stats
"""

import sys
import os
import argparse
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hnsw import HNSWIndex
from embedder import SemanticEmbedder
from corpus import DocumentStore


def main():
    parser = argparse.ArgumentParser(description="HNSW Semantic Document Search CLI")
    parser.add_argument("--query", "-q", type=str, help="Search query string")
    parser.add_argument("--top_k", "-k", type=int, default=5, help="Number of results to return (default: 5)")
    parser.add_argument("--ef", type=int, default=50, help="efSearch beam width parameter (default: 50)")
    parser.add_argument("--stats", action="store_true", help="Print HNSW graph topology statistics")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark against Ground Truth")
    args = parser.parse_args()

    print("=" * 65)
    print("  HNSW SEMANTIC DOCUMENT SEARCH ENGINE (CLI)")
    print("=" * 65)

    store = DocumentStore()
    embedder = SemanticEmbedder()
    
    docs = store.get_texts_for_embedding()
    doc_ids = [d[0] for d in docs]
    texts = [d[1] for d in docs]

    print(f"\n[1] Vectorizing {len(texts)} documents with {embedder.device.upper()}...")
    vectors = embedder.embed_texts(texts)

    print("[2] Constructing HNSW Multi-Layer Graph Index...")
    dim = vectors.shape[1]
    hnsw = HNSWIndex(d=dim, M=16, ef_construction=100, ef_search=args.ef)
    for doc_id, vec in zip(doc_ids, vectors):
        hnsw.add_item(doc_id, vec)

    if args.stats or (not args.query and not args.benchmark):
        stats = hnsw.get_stats()
        print("\n--- HNSW GRAPH STATISTICS ---")
        print(f"Total Nodes: {stats['total_nodes']}")
        print(f"Max Level: Layer {stats['max_level']}")
        print(f"Global Entry Point: Node #{stats['enter_point']}")
        print(f"Nodes per Layer: {stats['nodes_per_level']}")
        print(f"Hyperparameters: M={stats['hyperparameters']['M']}, efConstruction={stats['hyperparameters']['ef_construction']}")
        print(f"Memory Footprint: {stats['memory_kb']} KB")
        if not args.query and not args.benchmark:
            return

    if args.benchmark:
        print("\n--- RUNNING BENCHMARK (HNSW vs EXACT SCAN) ---")
        test_queries = [
            "Cơ chế Attention trong mạng Transformer",
            "Thuật toán HNSW và đồ thị Small World",
            "Định lý CAP trong hệ thống phân tán",
            "AlphaFold và gấp cuộn protein",
            "Qubit và máy tính lượng tử"
        ]
        for q in test_queries:
            q_vec = embedder.embed_single(q)
            h_res = hnsw.query(q_vec, top_k=args.top_k, ef_search=args.ef)
            gt_res = hnsw.brute_force_query(q_vec, top_k=args.top_k)
            recall = hnsw.evaluate_recall([r['id'] for r in h_res['results']], [r['id'] for r in gt_res['results']])
            print(f"Query: '{q[:35]}...'")
            print(f"  Recall@{args.top_k}: {recall*100:.1f}% | HNSW: {h_res['latency_ms']:.2f}ms ({h_res['dist_comps']} calcs) | Exact: {gt_res['latency_ms']:.2f}ms ({gt_res['dist_comps']} calcs)")

    if args.query:
        print(f"\n--- SEARCH QUERY: '{args.query}' ---")
        t0 = time.perf_counter()
        q_vec = embedder.embed_single(args.query)
        res = hnsw.query(q_vec, top_k=args.top_k, ef_search=args.ef)
        gt = hnsw.brute_force_query(q_vec, top_k=args.top_k)
        recall = hnsw.evaluate_recall([r['id'] for r in res['results']], [r['id'] for r in gt['results']])

        print(f"Search Completed in {res['latency_ms']} ms | Recall@{args.top_k}: {recall*100:.1f}% | Distance Computations: {res['dist_comps']}/{len(hnsw.nodes)}")
        print("\nTOP MATCHING DOCUMENTS:")
        for rank, r in enumerate(res['results'], 1):
            doc = store.get(r['id'])
            print(f"\n[{rank}] Score: {r['score_pct']}% | Dist: {r['distance']} | Category: {doc['category']}")
            print(f"    Title: {doc['title']}")
            print(f"    Summary: {doc['summary']}")


if __name__ == "__main__":
    main()
