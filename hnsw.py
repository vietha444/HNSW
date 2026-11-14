"""
HNSW (Hierarchical Navigable Small World) Algorithm Implementation
==================================================================
Fully compliant with Malkov & Yashunin (IEEE TPAMI 2018).
Includes:
- Multi-layer skip-list graph hierarchy
- Geometric decay layer assignment: m_L = 1 / ln(M)
- Priority queue beam search (Algorithm 2)
- Heuristic Neighbor Selection with Empty-Lune (RNG) condition (Algorithm 4)
- Bidirectional link maintenance and degree pruning
- Query routing trace capture for UI visualization
- Exact brute-force comparator and Recall@K evaluation
"""

import math
import heapq
import time
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any


class HNSWIndex:
    """
    Hierarchical Navigable Small World Index for Vector Similarity Search.
    """
    TRACE_HOPS_CAP = 400  # số sự kiện tối đa ở tầng 0 gửi về giao diện để mô phỏng

    def __init__(
        self,
        d: int = 384,
        M: int = 16,
        ef_construction: int = 100,
        ef_search: int = 50,
        M0: Optional[int] = None,
        mL: Optional[float] = None,
        metric: str = "cosine"
    ):
        self.d = d
        self.M = M
        self.M0 = M0 if M0 is not None else 2 * M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.mL = mL if mL is not None else (1.0 / math.log(M))
        self.metric = metric.lower()

        # Graph storage
        # self.layers[level][node_id] = list of neighbor node_ids
        self.layers: List[Dict[int, List[int]]] = []
        self.nodes: Dict[int, np.ndarray] = {}
        self.node_levels: Dict[int, int] = {}
        self.enter_point: Optional[int] = None
        self.max_level: int = -1

        # Diagnostic counters
        self.dist_computations_count: int = 0

    def _dist(self, u: np.ndarray, v: np.ndarray) -> float:
        """
        Calculate distance between vectors u and v.
        Tracks computation counter.
        """
        self.dist_computations_count += 1
        if self.metric == "cosine":
            # For normalized vectors: cosine distance = 1.0 - dot_product
            # Clamped for numerical stability [0.0, 2.0]
            dot = float(np.dot(u, v))
            return max(0.0, min(2.0, 1.0 - dot))
        elif self.metric == "l2":
            diff = u - v
            return float(np.dot(diff, diff) ** 0.5)
        else:
            dot = float(np.dot(u, v))
            return -dot

    @staticmethod
    def _greedy_path(hops: List[Dict[str, Any]], start: int, best: int) -> List[int]:
        """
        Khôi phục đường đi thật của tìm kiếm tham lam (ef = 1) từ nhật ký hops:
        mỗi láng giềng được chấp nhận có cha là nút đang được mở rộng. Lần ngược từ
        nút tốt nhất về nút xuất phát để mỗi bước nhảy đi đúng theo một cạnh.
        """
        parent: Dict[int, int] = {}
        for h in hops:
            if h.get("action") == "visit_neighbor" and h.get("accepted"):
                parent.setdefault(h["node_id"], h["from_node"])
        path = [best]
        while path[-1] != start and path[-1] in parent and len(path) <= len(parent) + 1:
            path.append(parent[path[-1]])
        if path[-1] != start:
            return [start, best]
        return list(reversed(path))

    def _random_level(self) -> Tuple[int, float]:
        """
        Generate random level according to exponential distribution:
        l = floor(-ln(uniform(0, 1)) * m_L)
        Returns (level, uniform_value)
        """
        unif = max(1e-10, float(np.random.uniform(0.0, 1.0)))
        return int(math.floor(-math.log(unif) * self.mL)), unif

    def search_layer(
        self,
        query_vec: np.ndarray,
        entry_points: List[int],
        ef: int,
        level: int,
        collect_hops: bool = False,
        entry_dists: Optional[Dict[int, float]] = None
    ) -> Tuple[List[Tuple[float, int]], List[Dict[str, Any]]]:
        """
        Algorithm 2: SEARCH-LAYER(q, ep, ef, lc)
        Greedy beam search at a specific layer.

        entry_dists: khoảng cách đã biết của các điểm vào (từ tầng trên) để không tính lại.

        Returns:
            - Sorted list of (distance, node_id) of nearest candidates (size up to ef)
            - Optional list of hop exploration events for visualization
        """
        entry_dists = entry_dists or {}
        visited: Set[int] = set(entry_points)
        
        # C: min-heap of candidates to explore (ordered by distance ascending)
        # Stored as (dist, node_id)
        candidates: List[Tuple[float, int]] = []
        
        # W: max-heap of current nearest elements found so far (ordered by distance descending)
        # In Python heapq is min-heap, so we store (-dist, node_id)
        w_furthest_nearest: List[Tuple[float, int]] = []
        
        hops_log: List[Dict[str, Any]] = []

        for ep in entry_points:
            d = entry_dists[ep] if ep in entry_dists else self._dist(query_vec, self.nodes[ep])
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(w_furthest_nearest, (-d, ep))
            if collect_hops:
                hops_log.append({
                    "action": "init_entry",
                    "node_id": ep,
                    "distance": round(d, 4),
                    "level": level
                })

        while len(candidates) > 0:
            # c = extract-nearest element from C
            c_dist, c_id = heapq.heappop(candidates)
            
            # f = get-furthest element from W
            furthest_dist = -w_furthest_nearest[0][0]
            
            # If candidate is further than the furthest in W, cannot improve
            if c_dist > furthest_dist:
                break
                
            neighbors = self.layers[level].get(c_id, [])
            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    d_neighbor = self._dist(query_vec, self.nodes[neighbor_id])
                    furthest_dist = -w_furthest_nearest[0][0]
                    
                    if d_neighbor < furthest_dist or len(w_furthest_nearest) < ef:
                        heapq.heappush(candidates, (d_neighbor, neighbor_id))
                        heapq.heappush(w_furthest_nearest, (-d_neighbor, neighbor_id))
                        
                        if len(w_furthest_nearest) > ef:
                            popped = heapq.heappop(w_furthest_nearest)
                            if collect_hops:
                                hops_log.append({
                                    "action": "evict_furthest",
                                    "node_id": popped[1],
                                    "distance": round(-popped[0], 4),
                                    "level": level
                                })
                        
                        if collect_hops:
                            hops_log.append({
                                "action": "visit_neighbor",
                                "from_node": c_id,
                                "node_id": neighbor_id,
                                "distance": round(d_neighbor, 4),
                                "level": level,
                                "accepted": True
                            })
                    else:
                        if collect_hops:
                            hops_log.append({
                                "action": "visit_neighbor",
                                "from_node": c_id,
                                "node_id": neighbor_id,
                                "distance": round(d_neighbor, 4),
                                "level": level,
                                "accepted": False
                            })

        # Convert W from (-dist, id) to sorted (dist, id) ascending
        result = [(-neg_d, nid) for neg_d, nid in w_furthest_nearest]
        result.sort(key=lambda x: x[0])
        return result, hops_log

    def select_neighbors_heuristic(
        self,
        base_vec: np.ndarray,
        candidate_ids: List[int],
        M_max: int,
        level: int,
        extend_candidates: bool = False,
        keep_pruned_connections: bool = False,
        collect_decisions: bool = False
    ) -> Any:
        """
        Algorithm 4: SELECT-NEIGHBORS-HEURISTIC(q, C, M, lc, extendCandidates, keepPrunedConnections)
        Uses the Relative Neighborhood Graph (RNG) / Empty-Lune condition to ensure
        high directional diversity and prevent redundant short-distance triangles.
        Returns List[int] if collect_decisions=False, or (List[int], List[Dict]) if collect_decisions=True.
        """
        # Candidate set C
        W: Set[int] = set(candidate_ids)
        
        # Extend candidates with their 1-hop neighbors
        if extend_candidates:
            extended: Set[int] = set(W)
            for cid in candidate_ids:
                if cid in self.layers[level]:
                    for neighbor_id in self.layers[level][cid]:
                        extended.add(neighbor_id)
            W = extended

        # Calculate distances from base_vec to all candidates
        # Min-heap ordered by distance ascending
        candidate_heap: List[Tuple[float, int]] = []
        for cid in W:
            d = self._dist(base_vec, self.nodes[cid])
            candidate_heap.append((d, cid))
        candidate_heap.sort(key=lambda x: x[0])

        result_R: List[int] = []
        discarded_Wd: List[Tuple[float, int]] = []
        decisions: List[Dict[str, Any]] = []

        for e_dist, e_id in candidate_heap:
            if len(result_R) >= M_max:
                if collect_decisions:
                    decisions.append({
                        "node_id": e_id,
                        "dist_to_base": round(float(e_dist), 4),
                        "status": "pruned_by_capacity",
                        "reason": f"Đã đạt giới hạn dung lượng M_max={M_max}"
                    })
                continue
            
            # Check empty-lune condition against already selected neighbors in R
            # Candidate e is acceptable if it is closer to base_vec than to any r in R
            is_diverse = True
            occluding_r = None
            occluding_dist = 0.0
            e_vec = self.nodes[e_id]
            for r_id in result_R:
                r_vec = self.nodes[r_id]
                dist_e_r = self._dist(e_vec, r_vec)
                if dist_e_r < e_dist:
                    is_diverse = False
                    occluding_r = r_id
                    occluding_dist = dist_e_r
                    break
            
            if is_diverse:
                result_R.append(e_id)
                if collect_decisions:
                    decisions.append({
                        "node_id": e_id,
                        "dist_to_base": round(float(e_dist), 4),
                        "status": "accepted",
                        "reason": "Thỏa mãn điều kiện Empty-Lune (hướng phân tán tốt, không bị che khuất)"
                    })
            else:
                discarded_Wd.append((e_dist, e_id))
                if collect_decisions:
                    decisions.append({
                        "node_id": e_id,
                        "dist_to_base": round(float(e_dist), 4),
                        "status": "pruned_by_heuristic",
                        "occluding_node": occluding_r,
                        "dist_to_occluding": round(float(occluding_dist), 4),
                        "reason": f"Bị che khuất bởi Node #{occluding_r}: d(e, r)={occluding_dist:.4f} < d(e, q)={e_dist:.4f}"
                    })

        # If keep_pruned_connections is True and we haven't reached M_max,
        # backfill from discarded candidates
        if keep_pruned_connections and len(result_R) < M_max:
            for d, cid in discarded_Wd:
                if len(result_R) >= M_max:
                    break
                result_R.append(cid)
                if collect_decisions:
                    for dec in decisions:
                        if dec["node_id"] == cid:
                            dec["status"] = "backfilled"
                            dec["reason"] = f"Bổ sung lại từ tập loại bỏ để duy trì đủ bậc kết nối M_max={M_max}"
                            break

        if collect_decisions:
            return result_R, decisions
        return result_R

    def add_item(self, node_id: int, vector: np.ndarray, collect_trace: bool = False) -> Dict[str, Any]:
        """
        Algorithm 1: INSERT(hnsw, q, M, M0, efConstruction, mL)
        Inserts a new vector with unique node_id into the HNSW graph.
        Returns a rich trace dictionary if collect_trace=True.
        """
        # Ensure vector is float32 normalized
        norm = np.linalg.norm(vector)
        if norm > 1e-9:
            vector = vector / norm
        self.nodes[node_id] = vector

        # Assign random level
        node_level, unif = self._random_level()
        self.node_levels[node_id] = node_level

        trace: Dict[str, Any] = {
            "node_id": node_id,
            "assigned_level": node_level,
            "random_u": round(unif, 5),
            "mL": round(self.mL, 4),
            "M": self.M,
            "M0": self.M0,
            "ef_construction": self.ef_construction,
            "steps": [],
            "layer_connections": {}
        } if collect_trace else {}

        if collect_trace:
            trace["steps"].append({
                "type": "level_assigned",
                "node_id": node_id,
                "level": node_level,
                "u": round(unif, 5),
                "mL": round(self.mL, 4),
                "desc": f"Gán tầng ngẫu nhiên: u={unif:.4f} → l = ⌊-ln({unif:.4f}) · {self.mL:.3f}⌋ = Layer {node_level}"
            })

        # If this is the very first node in the graph
        if self.enter_point is None:
            self.max_level = node_level
            self.enter_point = node_id
            while len(self.layers) <= node_level:
                self.layers.append({})
            for l in range(node_level + 1):
                self.layers[l][node_id] = []
            if collect_trace:
                trace["steps"].append({
                    "type": "initial_entry",
                    "node_id": node_id,
                    "level": node_level,
                    "desc": f"Đồ thị ban đầu rỗng. Node #{node_id} trở thành Điểm Vào Toàn Cục (Global Entry Point) đầu tiên tại Layer {node_level}."
                })
            return trace

        # Ensure layers list is deep enough
        while len(self.layers) <= max(self.max_level, node_level):
            self.layers.append({})

        curr_obj = self.enter_point
        top_level = self.max_level
        curr_dist = self._dist(vector, self.nodes[curr_obj])

        if collect_trace:
            trace["initial_entry_point"] = curr_obj
            trace["initial_max_level"] = top_level

        # Phase 1: Top-down greedy 1-NN search through sparse layers above node_level
        for l in range(top_level, node_level, -1):
            candidates, hops = self.search_layer(vector, [curr_obj], ef=1, level=l, collect_hops=collect_trace,
                                                 entry_dists={curr_obj: curr_dist})
            if len(candidates) > 0:
                best_dist, best_node = candidates[0]
                if best_node != curr_obj:
                    if collect_trace:
                        hop_dist = {h["node_id"]: h["distance"] for h in hops if "node_id" in h}
                        path = self._greedy_path(hops, curr_obj, best_node)
                        for a, b in zip(path, path[1:]):
                            d_ab = hop_dist.get(b, best_dist)
                            trace["steps"].append({
                                "type": "greedy_jump",
                                "layer": l,
                                "from_node": a,
                                "to_node": b,
                                "dist": round(d_ab, 4),
                                "desc": f"Layer {l} Nhảy tham lam (1-NN): Node #{a} → Node #{b} (khoảng cách = {d_ab:.4f})"
                            })
                    curr_obj, curr_dist = best_node, best_dist
            
            if collect_trace:
                trace["steps"].append({
                    "type": "layer_drop",
                    "from_layer": l,
                    "to_layer": l - 1,
                    "node_id": curr_obj,
                    "desc": f"Đạt cực tiểu địa phương tại Layer {l}. Hạ độ cao xuống Layer {l - 1} tại Node #{curr_obj}"
                })

        # Phase 2: From min(top_level, node_level) down to Layer 0
        ep_list = [curr_obj]
        ep_dists = {curr_obj: curr_dist}
        for l in range(min(top_level, node_level), -1, -1):
            # Dynamic candidate search with ef = ef_construction
            candidates, hops = self.search_layer(vector, ep_list, ef=self.ef_construction, level=l, collect_hops=collect_trace,
                                                 entry_dists=ep_dists)
            candidate_ids = [c_id for _, c_id in candidates]

            if collect_trace:
                trace["steps"].append({
                    "type": "search_candidates",
                    "layer": l,
                    "candidates_count": len(candidates),
                    "candidates": [{"id": cid, "dist": round(d, 4)} for d, cid in candidates[:12]],
                    "desc": f"Layer {l} Tìm kiếm ứng viên (efConstruction={self.ef_construction}): Tìm được {len(candidates)} ứng viên lân cận."
                })

            # Select diverse neighbors using heuristic: chọn tối đa M láng giềng ở mọi tầng
            # (bài báo gốc); M_curr chỉ là giới hạn bậc khi tỉa (M0 = 2M ở tầng 0)
            M_curr = self.M0 if l == 0 else self.M
            if collect_trace:
                neighbors, decisions = self.select_neighbors_heuristic(
                    vector, candidate_ids, self.M, level=l, collect_decisions=True
                )
                trace["steps"].append({
                    "type": "heuristic_rng",
                    "layer": l,
                    "M_curr": self.M,
                    "selected_neighbors": neighbors,
                    "decisions": decisions[:20],
                    "desc": f"Layer {l} Heuristic đa dạng hướng (Empty-Lune RNG): Đã chọn {len(neighbors)} láng giềng (tối đa M={self.M})."
                })
            else:
                neighbors = self.select_neighbors_heuristic(
                    vector, candidate_ids, self.M, level=l, collect_decisions=False
                )
            
            # Initialize adjacency list for this node at level l
            if node_id not in self.layers[l]:
                self.layers[l][node_id] = []
            self.layers[l][node_id] = list(neighbors)

            if collect_trace:
                trace["layer_connections"][l] = list(neighbors)
                trace["steps"].append({
                    "type": "connect_edges",
                    "layer": l,
                    "node_id": node_id,
                    "neighbors": list(neighbors),
                    "desc": f"Layer {l} Thiết lập liên kết 2 chiều giữa Node #{node_id} và {len(neighbors)} láng giềng: {neighbors}"
                })

            # Bidirectional connection and degree pruning
            for neighbor_id in neighbors:
                if neighbor_id not in self.layers[l]:
                    self.layers[l][neighbor_id] = []
                
                if node_id not in self.layers[l][neighbor_id]:
                    self.layers[l][neighbor_id].append(node_id)
                
                # Check degree overflow
                if len(self.layers[l][neighbor_id]) > M_curr:
                    before_prune = list(self.layers[l][neighbor_id])
                    pruned_neighbors = self.select_neighbors_heuristic(
                        self.nodes[neighbor_id],
                        self.layers[l][neighbor_id],
                        M_curr,
                        level=l,
                        collect_decisions=False
                    )
                    self.layers[l][neighbor_id] = pruned_neighbors
                    
                    if collect_trace:
                        removed_ids = [n for n in before_prune if n not in pruned_neighbors]
                        if removed_ids:
                            trace["steps"].append({
                                "type": "prune_overflow",
                                "layer": l,
                                "neighbor_id": neighbor_id,
                                "removed_nodes": removed_ids,
                                "new_degree": len(pruned_neighbors),
                                "desc": f"Layer {l} Cắt tỉa bậc: Bậc Node #{neighbor_id} > M_max={M_curr}. Heuristic RNG đã cắt cạnh tới {removed_ids}."
                            })

            # Next level entry point is the best candidate found
            if len(candidates) > 0:
                ep_list = [candidates[0][1]]
                ep_dists = {candidates[0][1]: candidates[0][0]}

        # If new node has level higher than current max_level, update global enter_point
        if node_level > self.max_level:
            for l in range(self.max_level + 1, node_level + 1):
                self.layers[l][node_id] = []
            old_top = self.max_level
            old_ep = self.enter_point
            self.max_level = node_level
            self.enter_point = node_id
            if collect_trace:
                trace["steps"].append({
                    "type": "update_entry_point",
                    "new_entry_point": node_id,
                    "old_entry_point": old_ep,
                    "new_max_level": node_level,
                    "old_max_level": old_top,
                    "desc": f"Nâng cấp Entry Point: Node #{node_id} (Layer {node_level}) cao hơn L_max cũ ({old_top}) → Trở thành Điểm Vào Toàn Cục mới!"
                })

        if collect_trace:
            trace["final_max_level"] = self.max_level
            trace["final_entry_point"] = self.enter_point
            trace["steps"].append({
                "type": "insertion_complete",
                "node_id": node_id,
                "level": node_level,
                "desc": f"Hoàn tất chèn Node #{node_id} vào chỉ mục HNSW! Đồ thị đã cập nhật đầy đủ các tầng từ Layer {node_level} xuống Layer 0."
            })

        return trace

    def query(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        ef_search: Optional[int] = None,
        collect_trace: bool = True
    ) -> Dict[str, Any]:
        """
        Algorithm 5: K-NN-SEARCH(hnsw, q, K, efSearch)
        Queries the HNSW graph with comprehensive routing trace collection.
        """
        if len(self.nodes) == 0:
            return {"results": [], "trace": {}, "latency_ms": 0.0, "dist_comps": 0}

        t_start = time.perf_counter()
        dist_start = self.dist_computations_count

        ef = ef_search if ef_search is not None else self.ef_search
        ef = max(ef, top_k)

        # Normalize query vector
        norm = np.linalg.norm(query_vec)
        if norm > 1e-9:
            query_vec = query_vec / norm

        curr_obj = self.enter_point
        top_level = self.max_level
        # Khoảng cách tới điểm vào chỉ tính một lần rồi dùng lại ở các tầng dưới
        curr_dist = self._dist(query_vec, self.nodes[curr_obj])

        trace_layers: List[Dict[str, Any]] = []

        # Phase 1: Greedy routing through sparse upper layers (L_max down to 1)
        for l in range(top_level, 0, -1):
            step_info: Dict[str, Any] = {
                "level": l,
                "entry_point": curr_obj,
                "entry_dist": round(curr_dist, 4),
                "type": "greedy_1nn"
            }
            candidates, hops = self.search_layer(
                query_vec, [curr_obj], ef=1, level=l, collect_hops=collect_trace,
                entry_dists={curr_obj: curr_dist}
            )
            if len(candidates) > 0:
                best_dist, best_node = candidates[0]
                if collect_trace:
                    step_info["path"] = self._greedy_path(hops, curr_obj, best_node)
                curr_obj, curr_dist = best_node, best_dist
                step_info["best_node"] = curr_obj
                step_info["best_dist"] = round(best_dist, 4)
                step_info["hops"] = hops
            trace_layers.append(step_info)

        # Phase 2: Beam search at bottom Layer 0 with ef = ef_search
        layer0_info: Dict[str, Any] = {
            "level": 0,
            "entry_point": curr_obj,
            "entry_dist": round(curr_dist, 4),
            "type": f"beam_search_ef_{ef}"
        }
        candidates, layer0_hops = self.search_layer(
            query_vec, [curr_obj], ef=ef, level=0, collect_hops=collect_trace,
            entry_dists={curr_obj: curr_dist}
        )
        layer0_info["hops_count"] = len(layer0_hops)
        layer0_info["hops"] = layer0_hops[:self.TRACE_HOPS_CAP] if collect_trace else []  # giới hạn kích thước phản hồi
        trace_layers.append(layer0_info)

        # Extract Top-K nearest
        top_candidates = candidates[:top_k]
        results = [
            {
                "id": node_id,
                "distance": round(float(dist), 4),
                "similarity": round(float(max(0.0, 1.0 - dist)), 4),
                "score_pct": round(float(max(0.0, 1.0 - dist) * 100), 1)
            }
            for dist, node_id in top_candidates
        ]

        t_end = time.perf_counter()
        dist_end = self.dist_computations_count

        return {
            "results": results,
            "trace": {
                "max_level": self.max_level,
                "global_enter_point": self.enter_point,
                "layers": trace_layers,
                "ef_used": ef,
                "layer0_visited_count": len([h for h in layer0_hops if h.get("action") == "visit_neighbor"])
            },
            "latency_ms": round((t_end - t_start) * 1000, 2),
            "dist_comps": dist_end - dist_start
        }

    def brute_force_query(self, query_vec: np.ndarray, top_k: int = 10) -> Dict[str, Any]:
        """
        Exact ground-truth baseline via Linear Scan (O(N) distance calculations).
        """
        t_start = time.perf_counter()
        norm = np.linalg.norm(query_vec)
        if norm > 1e-9:
            query_vec = query_vec / norm

        scores: List[Tuple[float, int]] = []
        for node_id, vec in self.nodes.items():
            dot = float(np.dot(query_vec, vec))
            dist = max(0.0, min(2.0, 1.0 - dot))
            scores.append((dist, node_id))

        scores.sort(key=lambda x: x[0])
        top = scores[:top_k]

        t_end = time.perf_counter()
        results = [
            {
                "id": node_id,
                "distance": round(float(dist), 4),
                "similarity": round(float(max(0.0, 1.0 - dist)), 4),
                "score_pct": round(float(max(0.0, 1.0 - dist) * 100), 1)
            }
            for dist, node_id in top
        ]

        return {
            "results": results,
            "latency_ms": round((t_end - t_start) * 1000, 2),
            "dist_comps": len(self.nodes)
        }

    def evaluate_recall(self, hnsw_ids: List[int], ground_truth_ids: List[int]) -> float:
        """
        Computes Recall@K: |HNSW_K ∩ GT_K| / K
        """
        if not ground_truth_ids:
            return 1.0
        k = len(ground_truth_ids)
        intersect = set(hnsw_ids[:k]).intersection(set(ground_truth_ids))
        return round(len(intersect) / float(k), 4)

    def get_stats(self) -> Dict[str, Any]:
        """
        Provides structural topology statistics of the HNSW index.
        """
        nodes_per_level = [len(layer) for layer in self.layers]
        degree_per_level = []
        for l, layer in enumerate(self.layers):
            if len(layer) > 0:
                degrees = [len(neighbors) for neighbors in layer.values()]
                undirected = {(min(u, v), max(u, v)) for u, nbs in layer.items() for v in nbs}
                degree_per_level.append({
                    "level": l,
                    "avg_degree": round(sum(degrees) / len(degrees), 1),
                    "max_degree": max(degrees) if degrees else 0,
                    "min_degree": min(degrees) if degrees else 0,
                    "total_edges": len(undirected)
                })
            else:
                degree_per_level.append({
                    "level": l, "avg_degree": 0, "max_degree": 0, "min_degree": 0, "total_edges": 0
                })

        # Memory footprint estimation
        total_vectors_bytes = len(self.nodes) * self.d * 4
        total_edges_count = sum(d["total_edges"] for d in degree_per_level)
        total_edges_bytes = total_edges_count * 2 * 8 # 8-byte pointers/ints

        return {
            "total_nodes": len(self.nodes),
            "dimension": self.d,
            "max_level": self.max_level,
            "enter_point": self.enter_point,
            "hyperparameters": {
                "M": self.M,
                "M0": self.M0,
                "ef_construction": self.ef_construction,
                "ef_search": self.ef_search,
                "mL": round(self.mL, 4)
            },
            "nodes_per_level": nodes_per_level,
            "degree_per_level": degree_per_level,
            "memory_kb": round((total_vectors_bytes + total_edges_bytes) / 1024, 2)
        }
