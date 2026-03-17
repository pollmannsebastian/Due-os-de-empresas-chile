"""
ETL: Build Graph Database
===========================
Vectorized pipeline using Polars DataFrame and DuckDB for final storage.
Creates graph.duckdb from extraction results + external RUT data.

Focus: Person-to-Person links derived from shared document (CVE).
- People are NEVER consolidated across CVEs, even if they share a RUT.
- Companies are ONLY consolidated globally if they share a known RUT.
- Companies without a known RUT are scoped to the document (CVE) to prevent false merging.

Run:  python etl/build_graph_db.py
"""
import duckdb
import time
import os
import sys
import polars as pl

# ── Paths (relative to project root) ───────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACTION_DB   = os.path.join(BASE_DIR, "extraction_results_v7.duckdb")
GRAPH_DB        = os.path.join(BASE_DIR, "graph.duckdb")
RUTS_PEOPLE     = os.path.join(BASE_DIR, "data", "Ruts.parquet")
SII_DATA        = os.path.join(BASE_DIR, "data", "data_tramos_sii.parquet")
COMPOSICION_CSV = os.path.join(BASE_DIR, "data", "composicion_sociedades.csv")

# ── UDF Functions ───────────────────────────────────────────────────────────

def compute_dv(rut_int: int) -> str:
    """Compute Chilean RUT verification digit (mod 11)."""
    s = str(rut_int)
    total, mul = 0, 2
    for ch in reversed(s):
        total += int(ch) * mul
        mul = mul + 1 if mul < 7 else 2
    rest = 11 - (total % 11)
    return '0' if rest == 11 else ('K' if rest == 10 else str(rest))


def format_rut_with_dv(rut_int):
    """77181977 → 77.181.977-K"""
    if rut_int is None:
        return None
    try:
        val = int(rut_int)
        body = str(val)
        dv = compute_dv(val)
        formatted = '.'.join(
            [body[::-1][i:i+3] for i in range(0, len(body), 3)]
        )[::-1]
        return f"{formatted}-{dv}"
    except (ValueError, TypeError):
        return None

def clean_rut(expr):
    cleaned = expr.str.to_uppercase().str.strip_chars().str.replace_all(r"\.", "").str.replace_all(r" ", "")
    return pl.when(cleaned.str.contains(r"^\d+-[\dK]$")).then(cleaned).otherwise(pl.lit(None))

def norm_name(expr):
    """
    General name normalization: lowercase, strip accents, remove punctuation,
    collapse whitespace.
    """
    return (
        expr.str.to_lowercase()
        .str.strip_chars()
        # Accents & ñ
        .str.replace_all(r"[áàäâã]", "a")
        .str.replace_all(r"[éèëê]", "e")
        .str.replace_all(r"[íìïî]", "i")
        .str.replace_all(r"[óòöôõ]", "o")
        .str.replace_all(r"[úùüû]", "u")
        .str.replace_all(r"ñ", "n")
        # Punctuation (including dots that break company joins)
        .str.replace_all(r"[.,;:()\[\]{}\"'`\-_/\\]+", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )

def norm_name_company(expr):
    """
    Normalization for company name matching against SII.
    Applies general norm_name first, then collapses common Chilean legal-form
    abbreviations so e.g. 'Inversiones S.A.' and 'Inversiones SA' match.
    """
    base = norm_name(expr)
    return (
        base
        # Ampersand → y (keep as a word, don't strip it)
        .str.replace_all(r"&", "y")
        # Legal suffixes only — order matters (longer / more specific first)
        .str.replace_all(r"\beirl$",      "eirl")
        .str.replace_all(r"\bsociedad anonima$", "sa")
        .str.replace_all(r"\bsociedad anonima\b", "sa")
        .str.replace_all(r"\bs a$",       "sa")
        .str.replace_all(r"\bspa$",       "spa")
        .str.replace_all(r"\bltda$",      "ltda")
        .str.replace_all(r"\bltd$",       "ltda")
        .str.replace_all(r"\blimitada$",  "ltda")
        .str.replace_all(r"\by cia ltda$","y cia ltda")
        .str.replace_all(r"\by cia$",     "y cia")
        .str.replace_all(r"\bsa$",        "sa")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )

def extract_rut_digits(expr):
    cleaned = expr.str.to_uppercase().str.strip_chars().str.replace_all(r"\.", "").str.replace_all(r" ", "")
    return cleaned.str.extract(r"^(\d+)-[\dK]$", 1)


# ── Logging ─────────────────────────────────────────────────────────────────

def log(step, total, msg, t0=None):
    elapsed = f" ({time.time() - t0:.1f}s)" if t0 else ""
    print(f"  [{step}/{total}] {msg}{elapsed}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    STEPS = 9
    t0 = time.time()

    print("=" * 60)
    print("  BUILD GRAPH DATABASE — Strict Person Scoping Mode + Composicion")
    print("=" * 60)
    print(f"  Source : {EXTRACTION_DB}")
    print(f"  Output : {GRAPH_DB}")
    print()

    # Validate inputs
    for path, label in [(EXTRACTION_DB, "Extraction DB"), 
                         (RUTS_PEOPLE, "People RUTs"), (SII_DATA, "SII Data"),
                         (COMPOSICION_CSV, "Composicion Data")]:
        if not os.path.exists(path):
            print(f"  ERROR: {label} not found at {path}")
            sys.exit(1)

    # ────────────────────────────────────────────────────────────────────────
    # 1. Load Extraction Results (PERSON-PERSON Co-occurrence)
    # ────────────────────────────────────────────────────────────────────────
    t1 = time.time()
    log(1, STEPS, "Loading extraction results (Linking people by CVE)...")
    
    with duckdb.connect(EXTRACTION_DB, read_only=True) as src_con:
        raw_df = src_con.execute("""
            SELECT 
                cve,
                source_name,
                source_rut,
                relation,
                target_name,
                target_rut,
                date,
                url
            FROM relationships
            WHERE source_name IS NOT NULL AND target_name IS NOT NULL
        """).pl()
    
    invalid_names = [
        "dr torres boonen",
        "directores",
        "felipe andres peroti diaz",
        "de las entidades"
    ]

    raw = raw_df.with_columns(
        norm_name(pl.col("source_name")).alias("src_name"),
        clean_rut(pl.col("source_rut")).alias("src_rut"),
        norm_name(pl.col("target_name")).alias("tgt_name"),
        clean_rut(pl.col("target_rut")).alias("tgt_rut")
    ).filter(
        ~pl.col("src_name").is_in(invalid_names) &
        ~pl.col("tgt_name").is_in(invalid_names)
    ).drop(["source_name", "source_rut", "target_name", "target_rut"])

    n = len(raw)
    log(1, STEPS, f"  → {n:,} raw edges loaded (Direct Relationships)", t1)

    # ────────────────────────────────────────────────────────────────────────
    # 2. Extract Unique Companies from SII DATA
    # ────────────────────────────────────────────────────────────────────────
    t2 = time.time()
    log(2, STEPS, "Loading unique company names from SII_DATA...")

    sii = (
        pl.scan_parquet(SII_DATA)
        .select(["RUT", "Razón social"])
        .drop_nulls()
        .group_by("Razón social")
        .agg(pl.col("RUT").n_unique().alias("n_ruts"), pl.col("RUT").first().alias("RUT"))
        .filter(pl.col("n_ruts") == 1)
        .drop("n_ruts")
        .with_columns(
            norm_name_company(pl.col("Razón social")).alias("name_norm"),
            (pl.col("RUT").cast(pl.Utf8) + "-" + pl.col("RUT").map_elements(
                lambda r: compute_dv(int(r)) if r else None, return_dtype=pl.Utf8
            )).alias("rut"),
            pl.col("RUT").cast(pl.Utf8).alias("rut_digits")
        )
        .collect()
    )

    n = len(sii)
    log(2, STEPS, f"  → {n:,} unique companies found from SII", t2)

    # ────────────────────────────────────────────────────────────────────────
    # 3. Load External People RUTs
    # ────────────────────────────────────────────────────────────────────────
    t3 = time.time()
    log(3, STEPS, "Loading people RUTs...")
    
    ppl_parquet = (
        pl.scan_parquet(RUTS_PEOPLE)
        .filter(pl.col("RUT").is_not_null() & pl.col("DV").is_not_null() & pl.col("NOMBRE").is_not_null())
        .with_columns(
            norm_name(pl.col("NOMBRE")).alias("name_norm"),
            (pl.col("RUT").cast(pl.Utf8) + "-" + pl.col("DV").str.strip_chars().str.to_uppercase()).alias("rut")
        )
    )

    ppl = (
        ppl_parquet
        .group_by("name_norm")
        .agg(pl.col("rut").first())
        .collect()
    )
    
    ppl_names = (
        ppl_parquet
        .group_by("rut")
        .agg(pl.col("NOMBRE").first().alias("ppl_name"))
        .collect()
    )
    
    n = len(ppl)
    log(3, STEPS, f"  → {n:,} people listed", t3)

    # ────────────────────────────────────────────────────────────────────────
    # 4. Load Composicion Sociedades
    # ────────────────────────────────────────────────────────────────────────
    t_comp = time.time()
    log(4, STEPS, "Loading composicion_sociedades.csv...")
    
    comp = (
        pl.scan_csv(COMPOSICION_CSV, separator=";", infer_schema_length=0)
        .filter(pl.col("Rut Sociedad").is_not_null() & (pl.col("Rut Sociedad") != ""))
        .with_columns([
            (pl.col("Rut Sociedad") + "-" + pl.col("DV Sociedad").str.to_uppercase()).alias("rut_sociedad"),
            pl.when(pl.col("RUT Socio").is_not_null() & (pl.col("RUT Socio") != ""))
              .then(pl.col("RUT Socio") + "-" + pl.col("DV Socio").str.to_uppercase())
              .otherwise(pl.lit(None))
              .alias("rut_socio"),
            pl.col("Participación").cast(pl.Float64, strict=False).alias("porcentaje")
        ])
        .filter(pl.col("rut_socio").is_not_null())
        .select(["rut_sociedad", "rut_socio", "porcentaje"])
        .collect()
    )
    n_comp = len(comp)
    log(4, STEPS, f"  → {n_comp:,} ownership edges from composicion", t_comp)

    # ────────────────────────────────────────────────────────────────────────
    # 5. Build Enriched Entities (STRICT CVE Scope for People)
    #
    #    global_id strategy:
    #      • Person          → global_id = name||cve
    #        (NEVER consolidates same person across CVEs, even with RUT)
    #      • Company         → global_id = RUT (if known) or name||cve
    #        (Collapses companies globally ONLY when RUT is known)
    # ────────────────────────────────────────────────────────────────────────
    t4 = time.time()
    log(5, STEPS, "Building and enriching entities (Strict Isolation)...")

    src = raw.select([
        pl.col("src_name").alias("name"),
        pl.col("src_rut").alias("rut"),
        pl.lit(0).alias("is_company"),
        pl.col("cve"),
    ])
    tgt = raw.select([
        pl.col("tgt_name").alias("name"),
        pl.col("tgt_rut").alias("rut"),
        pl.lit(1).alias("is_company"),
        pl.col("cve"),
    ])

    agg = (
        pl.concat([src, tgt])
        .group_by(["name", "cve"])
        .agg([
            pl.col("rut").max().alias("extracted_rut"),
            pl.col("is_company").max(),
        ])
    )

    # Assign global_id for entities WITH an extracted RUT
    has_rut = agg.filter(pl.col("extracted_rut").is_not_null()).with_columns(
        pl.when(pl.col("is_company") == 1)
        .then(pl.col("extracted_rut"))                                        # Global consolidation for companies
        .otherwise(pl.concat_str([pl.col("name"), pl.col("cve")], separator="||")) # Strict isolation for people
        .alias("global_id")
    )

    needs_lookup = agg.filter(pl.col("extracted_rut").is_null())

    # Apply company normalization to extracted names before the SII join
    needs_lookup = needs_lookup.with_columns(
        norm_name_company(pl.col("name")).alias("name_company_norm")
    )

    # Assign global_id for entities WITHOUT an extracted RUT
    # WITHOUT INFERRING IT FROM EXACT NAME MATCHES (to prevent false positives)
    looked_up = (
        needs_lookup
        .with_columns([
            pl.lit(None).cast(pl.Utf8).alias("rut"),
            pl.concat_str([pl.col("name"), pl.col("cve")], separator="||").alias("global_id")
        ])
        .select(["name", "cve", "is_company", "rut", "global_id"])
    )

    enriched = pl.concat([
        looked_up, 
        has_rut.select(["name", "cve", "is_company", pl.col("extracted_rut").alias("rut"), "global_id"])
    ])

    cnt_enriched = (enriched["rut"].is_not_null()).sum()
    log(5, STEPS, f"  → {cnt_enriched:,} entities enriched with RUTs (No cross-CVE person merging)", t4)

    # ────────────────────────────────────────────────────────────────────────
    # 6. Build Unified Graph Nodes
    # ────────────────────────────────────────────────────────────────────────
    t5 = time.time()
    log(6, STEPS, "Building graph nodes (combining CVE and Composicion)...")

    enriched_agg = (
        enriched
        .group_by("global_id")
        .agg([
            pl.col("name").max(),
            pl.col("rut").max(),
            pl.col("is_company").max()
        ])
    )

    comp_sociedades = comp.select([
        pl.lit(None).cast(pl.Utf8).alias("name"),
        pl.col("rut_sociedad").alias("rut"),
        pl.lit(1).cast(pl.Int32).alias("is_company"),
    ]).unique("rut")

    comp_socios = comp.select([
        pl.lit(None).cast(pl.Utf8).alias("name"),
        pl.col("rut_socio").alias("rut"),
        pl.lit(0).cast(pl.Int32).alias("is_company"), 
    ]).unique("rut")

    comp_entities = (
        pl.concat([comp_sociedades, comp_socios])
        .group_by("rut")
        .agg([
            pl.col("name").max(),
            pl.col("is_company").max()
        ])
        .with_columns(pl.col("rut").alias("global_id"))
    )

    combined_nodes = (
        pl.concat([enriched_agg, comp_entities], how="diagonal_relaxed")
        .group_by("global_id")
        .agg([
            pl.col("name").max(),
            pl.col("rut").max(),
            pl.col("is_company").max()
        ])
    )

    # rut → latest company name (by most recent document date)
    cve_dates = raw.select(["cve", "date"]).unique()
    company_latest_name = (
        enriched
        .filter((pl.col("is_company") == 1) & pl.col("rut").is_not_null())
        .join(cve_dates, on="cve", how="left")
        .sort("date", descending=True, nulls_last=True)
        .group_by("rut")
        .agg(pl.col("name").first().alias("latest_name"))
    )

    sii_names = sii.select([pl.col("rut"), pl.col("Razón social").alias("sii_name")])

    graph_nodes = (
        combined_nodes
        .join(company_latest_name, on="rut", how="left")
        .join(sii_names, on="rut", how="left")
        .join(ppl_names, on="rut", how="left")
        .with_columns(
            pl.coalesce([
                # 1. Names from external data cross by RUT (Prioritize data over extraction)
                pl.when(pl.col("is_company") == 1).then(pl.col("sii_name")).otherwise(pl.col("ppl_name")),
                # 2. Extraction Results (Companies latest name)
                pl.when(pl.col("is_company") == 1).then(pl.col("latest_name")).otherwise(pl.lit(None)),
                # 3. Aggregated name (From document raw extraction)
                pl.col("name"),
                # 4. Fallbacks
                pl.col("sii_name"),
                pl.col("ppl_name"),
                pl.lit("Unknown Entity")
            ]).alias("name")
        )
        .drop(["latest_name", "sii_name", "ppl_name"])
        .with_columns(
            pl.int_range(1, pl.len() + 1).alias("node_id"),
            extract_rut_digits(pl.col("rut")).alias("rut_digits"),
            pl.when(pl.col("is_company") == 1).then(pl.lit("company")).otherwise(pl.lit("person")).alias("node_type"),
            norm_name(pl.col("name")).alias("name_search")
        )
        .select(["node_id", "name", "rut", "rut_digits", "node_type", "name_search", "global_id"])
    )

    cnt = len(graph_nodes)
    log(6, STEPS, f"  → {cnt:,} total nodes across graph", t5)

    # ────────────────────────────────────────────────────────────────────────
    # 7. Build Mapping
    # ────────────────────────────────────────────────────────────────────────
    t6 = time.time()
    log(7, STEPS, "Building node mapping...")

    node_mapping = (
        enriched.select(["name", "cve", "global_id"])
        .join(graph_nodes.select(["global_id", "node_id"]), on="global_id")
        .select(["name", "cve", "node_id"])
        .unique(subset=["name", "cve"], keep="first")
    )

    n_map = len(node_mapping)
    log(7, STEPS, f"  → {n_map:,} mappings derived", t6)

    # ────────────────────────────────────────────────────────────────────────
    # 8. Pre-aggregate edges for fast graph queries
    # ────────────────────────────────────────────────────────────────────────
    t7 = time.time()
    log(8, STEPS, "Building aggregated edges directly...")

    m1 = node_mapping.rename({"name": "src_name", "node_id": "source_id"})
    m2 = node_mapping.rename({"name": "tgt_name", "node_id": "target_id"})

    edges_join = (
        raw
        .join(m1, on=["src_name", "cve"], how="inner")
        .join(m2, on=["tgt_name", "cve"], how="inner")
        .sort("date", descending=True)
    )

    graph_edges_agg_cve = (
        edges_join
        .group_by(["source_id", "target_id"])
        .agg([
            pl.col("relation").drop_nulls().mode().first().alias("relation"),
            pl.len().cast(pl.UInt32).alias("doc_count"),
            pl.struct(["cve", "date", "url"]).head(5).alias("documents")
        ])
        .with_columns([
            pl.lit(None).cast(pl.Float64).alias("porcentaje_participacion"),
            pl.lit("CVE").alias("source")
        ])
    )

    # Map rut_socio -> source_id, rut_sociedad -> target_id for COMPOSICION
    rut_to_node = graph_nodes.filter(pl.col("rut").is_not_null()).select(["rut", "node_id"]).unique("rut")

    comp_edges = (
        comp
        .join(rut_to_node.rename({"rut": "rut_socio", "node_id": "source_id"}), on="rut_socio", how="inner")
        .join(rut_to_node.rename({"rut": "rut_sociedad", "node_id": "target_id"}), on="rut_sociedad", how="inner")
        .select([
            pl.col("source_id"),
            pl.col("target_id"),
            pl.lit("PARTNER_OF").alias("relation"),
            pl.lit(1).cast(pl.UInt32).alias("doc_count"),
            pl.lit(None).alias("documents"),
            pl.col("porcentaje").alias("porcentaje_participacion"),
            pl.lit("COMPOSICION").alias("source")
        ])
    )

    graph_edges_agg = pl.concat([graph_edges_agg_cve, comp_edges], how="diagonal_relaxed")
    
    na = len(graph_edges_agg)
    log(8, STEPS, f"  → {na:,} aggregated edges (CVE + Composicion)", t7)

    # ────────────────────────────────────────────────────────────────────────
    # 9. Company Details (SII) - Optional fallback
    # ────────────────────────────────────────────────────────────────────────
    t8 = time.time()
    log(9, STEPS, "Building entity details (SII)...")

    company_nodes_df = graph_nodes.filter(pl.col("node_type") == 'company').select(["node_id", "rut_digits"])

    company_details = (
        pl.scan_parquet(SII_DATA)
        .filter(pl.col("RUT").is_not_null())
        .with_columns([
            norm_name(pl.col("Razón social")).alias("name"),
            (pl.col("RUT").cast(pl.Utf8) + "-" + pl.col("RUT").map_elements(
                lambda r: compute_dv(int(r)), return_dtype=pl.Utf8
            )).alias("rut"),
            pl.col("RUT").cast(pl.Utf8).alias("rut_digits")
        ])
        .join(company_nodes_df.lazy(), on="rut_digits", how="inner")
        .sort("Año comercial", descending=True)
        .unique(subset=["RUT"], keep="first")
        .select([
            pl.col("node_id"),
            pl.col("name"),
            pl.col("rut"),
            pl.col("Tramo según ventas").alias("tramo_ventas"),
            pl.col("Número de trabajadores dependie").alias("n_trabajadores"),
            pl.col("Actividad económica").alias("actividad"),
            pl.col("Tramo capital propio").alias("tramo_capital"),
            pl.col("Fecha inicio de actividades vige").alias("fecha_inicio"),
            pl.col("Año comercial").alias("anno_comercial")
        ])
        .collect()
    )

    nc = len(company_details)
    log(9, STEPS, f"  → {nc:,} entities matched with SII data", t8)

    # ────────────────────────────────────────────────────────────────────────
    # DuckDB Output & Indexes
    # ────────────────────────────────────────────────────────────────────────
    print("\n  Writing to DuckDB and creating indexes...")
    if os.path.exists(GRAPH_DB):
        try:
            os.remove(GRAPH_DB)
        except PermissionError:
            print(f"  ERROR: Cannot delete {GRAPH_DB} — is it open?")
            sys.exit(1)

    con = duckdb.connect(GRAPH_DB)
    # Write frames directly
    con.register("graph_nodes_df", graph_nodes)
    con.register("graph_edges_agg_df", graph_edges_agg)
    con.register("company_details_df", company_details)
    
    con.execute("CREATE TABLE graph_nodes AS SELECT * FROM graph_nodes_df")
    con.execute("CREATE TABLE graph_edges_agg AS SELECT * FROM graph_edges_agg_df")
    con.execute("CREATE TABLE company_details AS SELECT * FROM company_details_df")

    con.execute("""
        CREATE VIEW edge_audit AS
        SELECT
            e.source_id,
            e.target_id,
            ns.name  AS source_name,
            ns.rut   AS source_rut,
            nt.name  AS target_name,
            nt.rut   AS target_rut,
            e.relation,
            e.doc_count,
            e.porcentaje_participacion,
            e.source AS data_source,
            doc.cve,
            doc.date,
            doc.url
        FROM graph_edges_agg e
        JOIN graph_nodes ns ON ns.node_id = e.source_id
        JOIN graph_nodes nt ON nt.node_id = e.target_id
        LEFT JOIN UNNEST(e.documents) AS t(doc) ON true
    """)
    
    con.execute("CREATE INDEX idx_nodes_id      ON graph_nodes(node_id)")
    con.execute("CREATE INDEX idx_nodes_search   ON graph_nodes(name_search)")
    con.execute("CREATE INDEX idx_nodes_rut      ON graph_nodes(rut)")
    con.execute("CREATE INDEX idx_nodes_digits   ON graph_nodes(rut_digits)")
    con.execute("CREATE INDEX idx_edges_src      ON graph_edges_agg(source_id)")
    con.execute("CREATE INDEX idx_edges_tgt      ON graph_edges_agg(target_id)")
    con.execute("CREATE INDEX idx_details_rut    ON company_details(rut)")
    con.execute("CREATE INDEX idx_details_nodeid ON company_details(node_id)")

    con.close()

    elapsed = time.time() - t0
    size_mb = os.path.getsize(GRAPH_DB) / (1024 * 1024)
    print()
    print(f"  ✓ DONE in {elapsed:.1f}s — {size_mb:.0f} MB")
    print(f"  ✓ {cnt:,} nodes · {na:,} agg edges · {nc:,} SII")
    print("=" * 60)

if __name__ == "__main__":
    main()