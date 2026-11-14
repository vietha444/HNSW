"""
Launcher Script for HNSW Semantic Document Search Engine
========================================================
Run: python run_semantic_search.py
"""

import os
import sys
import webbrowser
import threading
import time

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Add semantic_search folder to Python path
APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "semantic_search")
sys.path.insert(0, APP_DIR)

from app import app, initialize_engine


def open_browser():
    time.sleep(1.8)
    url = "http://localhost:5000"
    print(f"[Launcher] Opening web browser at {url} ...")
    webbrowser.open(url)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  🚀 KHỞI ĐỘNG HỆ THỐNG TÌM KIẾM TÀI LIỆU NGỮ NGHĨA HNSW")
    print("  Hierarchical Navigable Small World (Malkov & Yashunin, 2018)")
    print("=" * 70)

    # 1. Initialize Embedder, Corpus, and HNSW Graph
    initialize_engine()

    # 2. Open browser automatically in background thread
    threading.Thread(target=open_browser, daemon=True).start()

    # 3. Start Flask Web Server
    print("\n[Server] Dang lang nghe tai cong http://localhost:5000 (nhan Ctrl+C de dung)...")
    app.run(host="0.0.0.0", port=5000, debug=False)
