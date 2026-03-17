"""
De Quiénes — Graph Explorer Server
=====================================
FastAPI backend with word-based search, graph expansion,
source documents, and optimized DuckDB queries.

Run:  python server/app.py
Open: http://localhost:8000
"""
import duckdb
import re
import unicodedata
import os
from contextlib import contextmanager
from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
from typing import Optional

# ── Configuration ───────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_DB   = os.path.join(BASE_DIR, "graph.duckdb")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# ── DuckDB Connection Pool ──────────────────────────────────────────────────
_db = duckdb.connect(GRAPH_DB, read_only=True)
_db.execute("PRAGMA threads=2")

_tables = [t[0] for t in _db.execute("SHOW TABLES").fetchall()]
_HAS_AGG     = "graph_edges_agg"  in _tables
_HAS_DETAILS = "company_details"  in _tables
print(f"  Tables: {_tables}")
print(f"  graph_edges_agg: {'YES' if _HAS_AGG else 'NO'}")
print(f"  company_details: {'YES' if _HAS_DETAILS else 'NO'}")


@contextmanager
def get_cursor():
    cur = _db.cursor()
    try:
        yield cur
    finally:
        cur.close()


# ── Pre-compiled patterns ───────────────────────────────────────────────────
_RE_DIGITS    = re.compile(r'[.\s]')
_RE_IS_RUT    = re.compile(r'^\d{5,}')
_RE_NON_DIGIT = re.compile(r'[^0-9]')

# ── Tramo labels ────────────────────────────────────────────────────────────
TRAMO_LABELS = {
    0: "Sin ventas",              1: "Micro 1 (< 800 UF)",
    2: "Micro 2 (800-2.400 UF)", 3: "Micro 3 (2.400-5.000 UF)",
    4: "Pequeña 1 (5.000-10.000 UF)",  5: "Pequeña 2 (10.000-25.000 UF)",
    6: "Pequeña 3 (25.000-50.000 UF)", 7: "Mediana 1 (50.000-100.000 UF)",
    8: "Mediana 2 (100.000-200.000 UF)", 9: "Grande 1 (200.000-600.000 UF)",
    10: "Grande 2 (600.000-1M UF)",    11: "Grande 3 (> 1.000.000 UF)",
}


# ── Text helpers ────────────────────────────────────────────────────────────

def strip_accents(s: str) -> str:
    """Lowercase + remove diacritics. Must match norm_name() in the ETL."""
    nfkd = unicodedata.normalize('NFKD', s.lower())
    return ''.join(c for c in nfkd if unicodedata.category(c) != 'Mn')


def _parse_documents(raw_docs) -> list:
    """
    Safely convert DuckDB LIST<STRUCT> to plain dicts.
    Handles both dict-like Row objects and native dicts across DuckDB versions.
    """
    result = []
    if not raw_docs:
        return result
    for d in raw_docs:
        try:
            # Works for both dict and DuckDB Row / Struct objects
            url  = d["url"]  if isinstance(d, dict) else d[2]
            cve  = d["cve"]  if isinstance(d, dict) else d[0]
            date = d["date"] if isinstance(d, dict) else d[1]
            if url:
                result.append({"cve": cve, "date": str(date) if date else None, "url": url})
        except Exception:
            pass
    return result


def _clean_rut_display(rut: Optional[str]) -> Optional[str]:
    """
    Return None for internal fallback ids (COMP|...) that should never
    be shown to the user, pass real RUTs through unchanged.
    """
    if rut is None:
        return None
    if rut.startswith("COMP|") or "|" in rut:
        return None
    return rut


# ── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(title="De Quiénes", docs_url="/docs")


@app.get("/api/search")
def search(q: str = Query(..., min_length=2), consolidate: bool = True):
    """
    Word-based search with optional consolidation.
    If consolidate=True, nodes with identical name_search and rut are grouped.
    """
    q_clean   = q.strip()
    q_digits  = _RE_DIGITS.sub('', q_clean)

    with get_cursor() as cur:
        if _RE_IS_RUT.match(q_digits):
            sql = """
                SELECT node_id, name, rut, node_type, name_search
                FROM graph_nodes
                WHERE rut_digits LIKE ? OR replace(replace(rut, '.', ''), '-', '') LIKE ?
                ORDER BY LENGTH(name)
                LIMIT 100
            """
            rows = cur.execute(sql, [f"%{q_digits}%", f"%{q_digits}%"]).fetchall()
        else:
            q_norm = strip_accents(q_clean)
            words  = q_norm.split()
            if not words: return []
            where  = " AND ".join(["name_search LIKE ?"] * len(words))
            params = [f"%{w}%" for w in words]
            sql = f"""
                SELECT node_id, name, rut, node_type, name_search
                FROM graph_nodes
                WHERE {where}
                ORDER BY CASE WHEN name_search LIKE ? THEN 0 ELSE 1 END, LENGTH(name)
                LIMIT 100
            """
            rows = cur.execute(sql, params + [f"{q_norm}%"]).fetchall()

    if not consolidate:
        return [
            {"id": str(r[0]), "name": r[1], "rut": _clean_rut_display(r[2]), "type": r[3]}
            for r in rows[:20]
        ]
    
    # Pass 1: Create name-to-RUT mapping for all people found in results
    # This helps bridge people with no RUT to their RUT-carrying counterparts.
    p_name_to_rut = {}
    for r_id, r_name, r_rut, r_type, r_norm in rows:
        if r_type == "person" and r_rut:
            p_name_to_rut[r_norm] = r_rut

    # Pass 2: Group by identity
    grouped = {}
    for r_id, r_name, r_rut, r_type, r_norm in rows:
        if r_type == "person":
            # Identity = RUT if found anywhere in the result set for this name, else name
            ident_rut = r_rut or p_name_to_rut.get(r_norm)
            key = (ident_rut, "p_rut") if ident_rut else (r_norm, "p_name")
        else:
            # Companies ONLY group together if they have a known RUT. Otherwise they remain distinct.
            key = ("c_rut", r_rut) if r_rut else ("c_id", r_id)

        if key not in grouped:
            grouped[key] = {
                "id_list": [str(r_id)],
                "name": r_name,
                "rut": r_rut,
                "type": r_type,
                "occurrences": [r_name]
            }
        else:
            grouped[key]["id_list"].append(str(r_id))
            if r_name not in grouped[key]["occurrences"]:
                grouped[key]["occurrences"].append(r_name)
            # Update RUT if we found one for this identity
            if not grouped[key]["rut"] and r_rut:
                grouped[key]["rut"] = r_rut
            # If a person has a RUT elsewhere, prioritize that for the group
            if r_type == "person" and not grouped[key]["rut"]:
                ident_rut = p_name_to_rut.get(r_norm)
                if ident_rut: grouped[key]["rut"] = ident_rut

    results = []
    for g in grouped.values():
        results.append({
            "id": ",".join(g["id_list"]),
            "name": g["name"],
            "rut": _clean_rut_display(g["rut"]),
            "type": g["type"],
            "count": len(g["id_list"]),
            "names": sorted(g["occurrences"])
        })
    
    return results[:20]


def _build_subgraph(cur, node_ids_raw: str, consolidate: bool = True) -> dict:
    """
    Build a subgraph for one or more node_ids.
    If consolidate=True, neighboring nodes with same name/rut are also merged.
    """
    try:
        node_ids = [int(x) for x in node_ids_raw.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid node_id format")
    
    if not node_ids:
        raise HTTPException(status_code=404, detail="No nodes specified")

    # Fetch center nodes
    ph = ",".join(["?"] * len(node_ids))
    centers = cur.execute(
        f"SELECT node_id, name, rut, node_type, name_search FROM graph_nodes WHERE node_id IN ({ph})",
        node_ids
    ).fetchall()
    
    if not centers:
        raise HTTPException(status_code=404, detail="Nodes not found")

    center_data = {
        "id":   node_ids_raw,
        "name": centers[0][1],
        "rut":  _clean_rut_display(centers[0][2]),
        "type": centers[0][3],
        "names": sorted(list(set(c[1] for c in centers)))
    }

    # Fetch all edges for all center IDs
    edge_rows = []
    if _HAS_AGG:
        edges = cur.execute(f"""
            SELECT source_id, target_id, relation, doc_count, documents, porcentaje_participacion
            FROM graph_edges_agg
            WHERE source_id IN ({ph}) OR target_id IN ({ph})
        """, node_ids + node_ids).fetchall()

        for e in edges:
            edge_rows.append({
                "source":    e[0],
                "target":    e[1],
                "relation":  e[2],
                "count":     e[3],
                "documents": _parse_documents(e[4]),
                "percentage": e[5],
            })
    else:
        edges = cur.execute(f"""
            SELECT source_id, target_id, relation, cve, date, url
            FROM graph_edges
            WHERE source_id IN ({ph}) OR target_id IN ({ph})
            LIMIT 1000
        """, node_ids + node_ids).fetchall()
        for e in edges:
            edge_rows.append({
                "source": e[0], "target": e[1], "relation": e[2],
                "count": 1, "documents": [{"cve": e[3], "date": str(e[4]), "url": e[5]}],
                "percentage": None
            })

    # Neighbor details
    all_node_ids = set()
    for e in edge_rows:
        all_node_ids.add(e["source"])
        all_node_ids.add(e["target"])
    
    center_ids_set = set(node_ids)
    neighbor_ids = all_node_ids - center_ids_set

    neighbor_rows = []
    if neighbor_ids:
        ph_n = ",".join(["?"] * len(neighbor_ids))
        neighbor_rows = cur.execute(f"""
            SELECT node_id, name, rut, node_type, name_search 
            FROM graph_nodes WHERE node_id IN ({ph_n})
        """, list(neighbor_ids)).fetchall()

    # Consolidation Map
    all_nodes_rows = centers + neighbor_rows
    groups = {} 
    
    if not consolidate:
        # No consolidation: every node is its own group
        id_map = {r[0]: str(r[0]) for r in all_nodes_rows}
        consolidated_nodes = []
        for r in all_nodes_rows:
            consolidated_nodes.append({
                "id": str(r[0]),
                "name": r[1],
                "rut": _clean_rut_display(r[2]),
                "type": r[3],
                "names": [r[1]]
            })
    else:
        # We use a Union-Find approach to group nodes dynamically based on relations
        parent = {r[0]: r[0] for r in all_nodes_rows}
        def find(i):
            if parent.get(i, i) == i: return i
            parent[i] = find(parent[i])
            return parent[i]
        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        # 1. Map Adjacency for shared neighbor check
        adj = {}
        for e in edge_rows:
            adj.setdefault(e["source"], set()).add(e["target"])
            adj.setdefault(e["target"], set()).add(e["source"])

        # 2. Gather attributes
        p_name_to_rut = {}
        for r in all_nodes_rows:
            if r[3] == "person" and r[2]:
                p_name_to_rut[r[4]] = r[2]

        comp_by_rut = {}
        pers_by_rut = {}
        pers_by_name = {}

        # Pre-clean company names for fast fuzzy matching
        all_comps_clean = []
        comp_ruts = {}
        
        for r in all_nodes_rows:
            n_id, n_name, n_rut, n_type, n_norm = r
            if n_type == "person":
                ident_rut = n_rut or p_name_to_rut.get(n_norm)
                if ident_rut:
                    pers_by_rut.setdefault(ident_rut, []).append(n_id)
                else:
                    pers_by_name.setdefault(n_norm, []).append(n_id)
            else:
                if n_rut:
                    comp_by_rut.setdefault(n_rut, []).append(n_id)
                comp_ruts[n_id] = n_rut
                
                # Cleaning logic for company names
                import unicodedata, re
                s = ''.join(c for c in unicodedata.normalize('NFKD', str(n_norm).lower()) if unicodedata.category(c) != 'Mn')
                s = re.sub(r'[^\w\s]', ' ', s)
                words = s.split()
                stops = {"spa", "limitada", "ltda", "sa", "s a", "eirl", "sociedad", "anonima", "inversiones", "comercial", "constructora", "y", "cia", "empresas", "grupo", "los", "las", "el", "la", "de"}
                core = [w for w in words if w not in stops]
                clean_str = " ".join(core) if core else " ".join(words)
                all_comps_clean.append((n_id, clean_str))

        # 3. Apply Unions for exact matches (People and RUT-based Companies)
        for id_list in pers_by_rut.values():
            for i in range(1, len(id_list)): union(id_list[0], id_list[i])
            
        for id_list in pers_by_name.values():
            for i in range(1, len(id_list)): union(id_list[0], id_list[i])
        
        for id_list in comp_by_rut.values():
            for i in range(1, len(id_list)): union(id_list[0], id_list[i])
                
        # Companies with highly similar names AND connected within 3 hops
        import difflib
        def get_distance(start_node, target_node, max_depth=3):
            if start_node == target_node: return 0
            visited = {start_node}
            queue = [(start_node, 0)]
            while queue:
                curr, depth = queue.pop(0)
                if curr == target_node: return depth
                if depth < max_depth:
                    for nxt in adj.get(curr, []):
                        if nxt not in visited:
                            visited.add(nxt)
                            queue.append((nxt, depth + 1))
            return float('inf')

        for i in range(len(all_comps_clean)):
            for j in range(i+1, len(all_comps_clean)):
                id1, c1 = all_comps_clean[i]
                id2, c2 = all_comps_clean[j]
                
                root1, root2 = find(id1), find(id2)
                if root1 == root2: continue
                
                # Check RUT conflict: if both have different RUTs, NEVER merge
                rut1, rut2 = comp_ruts[id1], comp_ruts[id2]
                if rut1 and rut2 and rut1 != rut2: continue
                
                # Fuzzy match >= 85%
                match = (c1 == c2)
                if not match and abs(len(c1) - len(c2)) <= 15:
                    match = difflib.SequenceMatcher(None, c1, c2).ratio() >= 0.85
                
                if match:
                    if get_distance(id1, id2, max_depth=3) <= 3:
                        union(id1, id2)

        # 4. Group by root
        groups = {}
        for r in all_nodes_rows:
            n_id, n_name, n_rut, n_type, n_norm = r
            root = find(n_id)
            if root not in groups:
                groups[root] = {"ids": [], "name": n_name, "rut": n_rut, "type": n_type, "names": set()}
            
            groups[root]["ids"].append(n_id)
            groups[root]["names"].add(n_name)
            
            # Prefer explicit RUT assignment
            if not groups[root]["rut"] and n_rut:
                groups[root]["rut"] = n_rut
            # If person and no rut, try fallback
            if n_type == "person" and not groups[root]["rut"]:
                groups[root]["rut"] = p_name_to_rut.get(n_norm)

        id_map = {}
        consolidated_nodes = []
        for root, g in groups.items():
            # Check if this group contains any of our starting identities
            is_center = any(cid in center_ids_set for cid in g["ids"])
            
            # If we are starting from a specific set of IDs, we keep that set as the center ID
            cons_id = node_ids_raw if is_center else ",".join(str(x) for x in sorted(g["ids"]))
            
            for nid in g["ids"]:
                id_map[nid] = cons_id
                
            consolidated_nodes.append({
                "id": cons_id,
                "name": g["name"],
                "rut": _clean_rut_display(g["rut"]),
                "type": g["type"],
                "names": sorted(list(g["names"]))
            })

    cons_edges = {}
    for e in edge_rows:
        src_cons = id_map.get(e["source"])
        dst_cons = id_map.get(e["target"])
        if not src_cons or not dst_cons or src_cons == dst_cons: continue
        
        ekey = (src_cons, dst_cons, e["relation"])
        if ekey not in cons_edges:
            cons_edges[ekey] = {
                "source": src_cons,
                "target": dst_cons,
                "relation": e["relation"],
                "count": e["count"],
                "documents": e["documents"],
                "percentage": e["percentage"]
            }
        else:
            exist = cons_edges[ekey]
            exist["count"] += e["count"]
            urls = set(d["url"] for d in exist["documents"])
            for d in e["documents"]:
                if d["url"] not in urls:
                    exist["documents"].append(d)
                    urls.add(d["url"])

    return {"center": center_data, "nodes": consolidated_nodes, "edges": list(cons_edges.values())}


@app.get("/api/graph/{node_id}")
def get_graph(node_id: str, consolidate: bool = True):
    with get_cursor() as cur:
        return _build_subgraph(cur, node_id, consolidate=consolidate)


@app.get("/api/expand/{node_id}")
def expand_node(node_id: str, consolidate: bool = True):
    with get_cursor() as cur:
        return _build_subgraph(cur, node_id, consolidate=consolidate)


@app.get("/api/company/{rut:path}")
def get_company_details(rut: str):
    """
    Get SII company details by RUT.
    Accepts both clean (76019328-3) and dotted (76.019.328-3) formats.
    company_details.rut is stored clean (no dots) matching graph_nodes.rut.
    """
    if not _HAS_DETAILS:
        return {"found": False}

    # Normalise to clean format for the primary lookup
    rut_clean  = rut.replace(".", "").strip().upper()
    rut_digits = _RE_NON_DIGIT.sub('', rut_clean)

    with get_cursor() as cur:
        # Primary: clean RUT match
        row = cur.execute("""
            SELECT node_id, name, rut, tramo_ventas, n_trabajadores,
                   actividad, tramo_capital, fecha_inicio, anno_comercial
            FROM company_details
            WHERE replace(rut, '.', '') = ?
        """, [rut_clean]).fetchone()

        if not row:
            # Fallback: digits-only match via graph_nodes join
            row = cur.execute("""
                SELECT cd.node_id, cd.name, cd.rut, cd.tramo_ventas,
                       cd.n_trabajadores, cd.actividad, cd.tramo_capital,
                       cd.fecha_inicio, cd.anno_comercial
                FROM company_details cd
                JOIN graph_nodes gn ON gn.node_id = cd.node_id
                WHERE gn.rut_digits = ?
                LIMIT 1
            """, [rut_digits]).fetchone()

        if not row:
            return {"found": False}

    return {
        "found":         True,
        "node_id":       row[0],
        "name":          row[1],
        "rut":           row[2],
        "tramo_ventas":  TRAMO_LABELS.get(row[3], f"Tramo {row[3]}") if row[3] is not None else None,
        "n_trabajadores": row[4],
        "actividad":     row[5],
        "tramo_capital": row[6],
        "fecha_inicio":  row[7],
        "anno_comercial": row[8],
    }


# ── Static Files ────────────────────────────────────────────────────────────
os.makedirs(STATIC_DIR, exist_ok=True)


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    print("De Quiénes Graph Explorer")
    print(f"  DB: {GRAPH_DB}")
    print(f"  Open: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)