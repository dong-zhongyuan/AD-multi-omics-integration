#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Drug target mining (multi-database) for a gene list.

✅ Jupyter/VScode Notebook friendly:
- Will NOT crash on Jupyter-injected args like: --f=...kernel.json
- In notebook: just Shift+Enter to run.
- Optional CLI usage still supported.

Outputs:
- ./drug_mining_out/drug_mining_all.csv
- ./drug_mining_out/drug_mining_ranked.csv
- ./drug_mining_out/drug_mining_summary.json

Optional:
- --ion_priors PATH: annotate "real ion channel" (contains 'channel' in ion_group_hits or gene_group)
- --gene_list PATH: path to CSV file with gene list (default: output/step5_gene_classification/therapeutic_targets.csv)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd


# -----------------------------
# 0) Gene list - Load from CSV file
# -----------------------------
# Default path: Use config_loader to get project paths
sys.path.insert(0, '')
from tools.config_loader import get_config
config = get_config()

DEFAULT_GENE_LIST_PATH = str(config.get_path("paths.output_dir")) + "/step5_gene_classification/therapeutic_targets.csv"

def load_gene_list_from_csv(csv_path: str) -> List[str]:
    """
    Load gene list from CSV file.
    Expected format: CSV with 'gene' or 'symbol' column.
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Find gene column
        gene_col = None
        for col in ['gene', 'Gene', 'symbol', 'Symbol', 'SYMBOL']:
            if col in df.columns:
                gene_col = col
                break
        
        if gene_col is None:
            raise ValueError(f"No gene column found in {csv_path}. Available columns: {df.columns.tolist()}")
        
        genes = df[gene_col].dropna().astype(str).tolist()
        genes = [normalize_symbol(g) for g in genes]
        genes = [g for g in genes if g]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_genes = []
        for g in genes:
            if g not in seen:
                seen.add(g)
                unique_genes.append(g)
        
        return unique_genes
    except Exception as e:
        raise RuntimeError(f"Failed to load gene list from {csv_path}: {e}")


HGNC_FETCH_SYMBOL = "https://rest.genenames.org/fetch/symbol/{}"
OPENTARGETS_GQL = "https://api.platform.opentargets.org/api/v4/graphql"
DGIDB_GQL = "https://dgidb.org/api/graphql"
CHEMBL_TARGET_SEARCH = "https://www.ebi.ac.uk/chembl/api/data/target/search.json"
CHEMBL_DRUG_MECH = "https://www.ebi.ac.uk/chembl/api/data/drug_mechanism.json"


# -----------------------------
# 2) Utilities
# -----------------------------
def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def normalize_symbol(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = re.split(r"[,\s]+", s)[0].strip()
    return s.upper()


def parse_gene_list_text(txt: str) -> List[str]:
    raw = re.split(r"[,\s]+", txt.strip())
    genes = [normalize_symbol(x) for x in raw]
    genes = [g for g in genes if g]
    seen = set()
    out = []
    for g in genes:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def safe_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def safe_read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def cache_key(prefix: str, key: str) -> str:
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
    return f"{prefix}__{key}.json"


def requests_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "drug-target-mining/1.2",
        "Accept": "application/json"
    })
    return s


def request_with_retry(
    sess: requests.Session,
    method: str,
    url: str,
    *,
    json_body: Optional[dict] = None,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: int = 30,
    max_retries: int = 5,
    sleep_base: float = 0.6
) -> requests.Response:
    last_err = None
    for i in range(max_retries):
        try:
            resp = sess.request(
                method=method,
                url=url,
                json=json_body,
                params=params,
                headers=headers,
                timeout=timeout,
            )
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(sleep_base * (2 ** i))
                continue
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_err = e
            time.sleep(sleep_base * (2 ** i))
    raise RuntimeError(f"Request failed after retries: {url} ({last_err})")


def is_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        if ip is None:
            return False
        # Jupyter / VSCode notebook kernels typically have this
        return "IPKernelApp" in ip.config
    except Exception:
        return False


# -----------------------------
# 3) Parse args safely in Jupyter
# -----------------------------
def parse_args_safely(argv: List[str]) -> argparse.Namespace:
    """
    Jupyter injects arguments like:
      --f=c:\\Users\\...\\kernel-xxxx.json
    We must ignore unknown args to avoid SystemExit.
    """
    ap = argparse.ArgumentParser(add_help=not is_notebook())
    ap.add_argument("--gene_list", type=str, default="", help="Path to gene list CSV (default: output/step5_gene_classification/therapeutic_targets.csv)")
    ap.add_argument("--out_dir", type=str, default="", help="Output dir (default: ./output/step5_clinical_validation/drug_mining)")
    ap.add_argument("--cache_dir", type=str, default="", help="Cache dir (default: ./drug_mining_cache)")
    ap.add_argument("--sleep", type=float, default=0.15, help="Sleep between requests")
    ap.add_argument("--max_genes", type=int, default=0, help="Limit gene count (0 = no limit)")
    ap.add_argument("--dgidb_batch", type=int, default=80, help="DGIdb batch size")
    ap.add_argument("--ion_priors", type=str, default="", help="Optional TSV/CSV: symbol + (ion_group_hits or gene_group)")

    # parse_known_args: ignore unknowns (e.g., --f=kernel.json)
    args, unknown = ap.parse_known_args(argv)
    # If you want to see what got ignored:
    if unknown and is_notebook():
        print(f"[INFO] Ignored notebook args: {unknown}")
    return args


# -----------------------------
# 4) Ion priors loader (optional)
# -----------------------------
def load_ion_priors(path: Path) -> pd.DataFrame:
    """
    Accepts:
      - target_class_priors.tsv (recommended)
      - your HGNC extracted table (symbol + gene_group)
    We try to infer ion_group_hits column.
    """
    if not path.exists():
        raise FileNotFoundError(f"ion priors file not found: {path}")

    # Try TSV first, then CSV
    try:
        df = pd.read_csv(path, sep="\t", dtype=str)
    except Exception:
        df = pd.read_csv(path, sep=",", dtype=str)

    df.columns = [c.strip() for c in df.columns]

    if "symbol" not in df.columns:
        for alt in ["Symbol", "SYMBOL", "gene", "Gene"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "symbol"})
                break

    if "symbol" not in df.columns:
        raise ValueError(f"ion priors file must contain symbol column. got columns={df.columns.tolist()}")

    if "ion_group_hits" not in df.columns:
        if "gene_group" in df.columns:
            df["ion_group_hits"] = df["gene_group"].fillna("")
        else:
            df["ion_group_hits"] = ""

    df["symbol"] = df["symbol"].fillna("").map(normalize_symbol)
    df["ion_group_hits"] = df["ion_group_hits"].fillna("").astype(str)

    # Your "real ion channel" rule
    df["is_real_ion_channel"] = df["ion_group_hits"].str.lower().str.contains("channel").astype(int)

    df = df.drop_duplicates(subset=["symbol"], keep="first")
    return df[["symbol", "ion_group_hits", "is_real_ion_channel"]]


# -----------------------------
# 5) HGNC mapping
# -----------------------------
def hgnc_fetch(sess: requests.Session, cache_dir: Path, symbol: str, sleep_s: float) -> Dict[str, Any]:
    ck = cache_dir / "hgnc" / cache_key("hgnc_symbol", symbol)
    cached = safe_read_json(ck)
    if cached is not None:
        return cached

    url = HGNC_FETCH_SYMBOL.format(symbol)
    resp = request_with_retry(sess, "GET", url, headers={"Accept": "application/json"})
    data = resp.json()

    doc = None
    try:
        docs = data.get("response", {}).get("docs", [])
        if docs:
            doc = docs[0]
    except Exception:
        doc = None

    out = {
        "symbol": symbol,
        "found": bool(doc),
        "hgnc_id": doc.get("hgnc_id") if doc else None,
        "ensembl_gene_id": doc.get("ensembl_gene_id") if doc else None,
        "entrez_id": doc.get("entrez_id") if doc else None,
        "uniprot_ids": doc.get("uniprot_ids") if doc else None,
        "locus_group": doc.get("locus_group") if doc else None,
        "locus_type": doc.get("locus_type") if doc else None,
        "name": doc.get("name") if doc else None,
        "status": doc.get("status") if doc else None,
        "alias_symbol": doc.get("alias_symbol") if doc else None,
        "prev_symbol": doc.get("prev_symbol") if doc else None,
        "timestamp": now_iso(),
    }
    safe_write_json(ck, out)
    time.sleep(sleep_s)
    return out


# -----------------------------
# 6) OpenTargets (known drugs + phase + tractability)
# -----------------------------
OT_QUERY = """
query target($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    biotype
    tractability {
      label
      value
    }
    drugAndClinicalCandidates {
      count
      rows {
        id
        maxClinicalStage
        drug {
          id
          name
        }
      }
    }
  }
}
"""

TRACT_LABEL_SCORE = [
    (re.compile(r"(approved|clinical)", re.I), 1.0),
    (re.compile(r"phase", re.I), 0.8),
    (re.compile(r"(structure|high)", re.I), 0.6),
    (re.compile(r"(medium|predicted)", re.I), 0.4),
    (re.compile(r"low", re.I), 0.2),
]


def _tract_score_from_labels(labels: List[str]) -> Tuple[float, str]:
    best = 0.0
    basis = ""
    for lab in labels:
        for pat, sc in TRACT_LABEL_SCORE:
            if pat.search(lab or ""):
                if sc > best:
                    best = sc
                    basis = lab
    return best, basis


def opentargets_fetch(sess: requests.Session, cache_dir: Path, ensembl_id: str, sleep_s: float) -> Dict[str, Any]:
    ck = cache_dir / "opentargets" / cache_key("ot_target", ensembl_id)
    cached = safe_read_json(ck)
    if cached is not None:
        return cached

    body = {"query": OT_QUERY, "variables": {"ensemblId": ensembl_id}}
    try:
        resp = request_with_retry(sess, "POST", OPENTARGETS_GQL, json_body=body)
        data = resp.json()
    except Exception as e:
        out = {
            "ensembl_id": ensembl_id,
            "found": False,
            "error": str(e),
            "approvedSymbol": None,
            "biotype": None,
            "known_drugs_n": 0,
            "approved_drugs_n": 0,
            "max_phase": 0,
            "drug_names": [],
            "tract_labels": [],
            "tract_score": 0.0,
            "tract_basis": "",
            "timestamp": now_iso(),
        }
        safe_write_json(ck, out)
        time.sleep(sleep_s)
        return out

    tgt = (data.get("data") or {}).get("target")
    if not tgt:
        out = {
            "ensembl_id": ensembl_id,
            "found": False,
            "error": None,
            "approvedSymbol": None,
            "biotype": None,
            "known_drugs_n": 0,
            "approved_drugs_n": 0,
            "max_phase": 0,
            "drug_names": [],
            "tract_labels": [],
            "tract_score": 0.0,
            "tract_basis": "",
            "timestamp": now_iso(),
        }
        safe_write_json(ck, out)
        time.sleep(sleep_s)
        return out

    known = tgt.get("drugAndClinicalCandidates") or {}
    rows = known.get("rows") or []
    drug_names = []
    max_phase = 0
    approved_n = 0

    # maxClinicalStage 格式: "PHASE_1", "PHASE_2", "PHASE_3", "PHASE_4", "APPROVED"
    for r in rows:
        stage = r.get("maxClinicalStage") or ""
        
        # 提取phase数字
        if "PHASE_" in stage:
            try:
                ph = int(stage.split("_")[1])
                max_phase = max(max_phase, ph)
            except Exception:
                pass
        
        # APPROVED 或 PHASE_4 算作已批准
        if stage == "APPROVED" or stage == "PHASE_4":
            approved_n += 1
            max_phase = max(max_phase, 4)

        d = r.get("drug") or {}
        dn = d.get("name")
        if dn:
            drug_names.append(str(dn))

    tract = tgt.get("tractability") or []
    labels = []
    for t in tract:
        lab = t.get("label")
        if lab:
            labels.append(str(lab))
    tract_score, tract_basis = _tract_score_from_labels(labels)

    out = {
        "ensembl_id": ensembl_id,
        "found": True,
        "error": None,
        "approvedSymbol": tgt.get("approvedSymbol"),
        "biotype": tgt.get("biotype"),
        "known_drugs_n": int(known.get("count") or len(rows) or 0),
        "approved_drugs_n": int(approved_n),
        "max_phase": int(max_phase),
        "drug_names": sorted(set(drug_names)),
        "tract_labels": labels,
        "tract_score": float(tract_score),
        "tract_basis": tract_basis,
        "timestamp": now_iso(),
    }
    safe_write_json(ck, out)
    time.sleep(sleep_s)
    return out


# -----------------------------
# 7) ChEMBL (mechanism evidence)
# -----------------------------
# --- 2. ChEMBL Fix: 使用 UniProt ID 且修正 API Endpoint ---
CHEMBL_TARGET_URL = "https://www.ebi.ac.uk/chembl/api/data/target.json"
CHEMBL_MECH_URL   = "https://www.ebi.ac.uk/chembl/api/data/mechanism.json"

def chembl_fetch(sess: requests.Session, cache_dir: Path, symbol: str, uniprot_string: str, sleep_s: float) -> Dict[str, Any]:
    # 缓存 Key 用 symbol 即可，内容基于 UniProt 查找
    ck = cache_dir / "chembl" / cache_key("chembl_u", symbol)
    cached = safe_read_json(ck)
    if cached is not None:
        return cached

    # 提取第一个 UniProt ID (HGNC 可能返回 P12345|Q67890)
    uid = None
    if uniprot_string:
        parts = uniprot_string.split("|")
        if parts:
            uid = parts[0].strip()

    if not uid:
        # 如果没有 UniProt ID，直接返回空，不再尝试模糊搜索
        out = {
            "symbol": symbol, "found": False, "error": "No UniProt ID",
            "mechanisms_n": 0, "unique_molecules_n": 0, "molecule_names": []
        }
        safe_write_json(ck, out)
        return out

    # 1. 查找 Target (通过 target_components__accession)
    params = {
        "target_components__accession": uid,
        "target_type": "SINGLE PROTEIN",
        "format": "json"
    }
    
    target_chembl_id = None
    pref_name = None
    
    try:
        resp = request_with_retry(sess, "GET", CHEMBL_TARGET_URL, params=params)
        data = resp.json()
        targets = data.get("targets") or []
        if targets:
            # 优先取人 (Homo sapiens)
            best = targets[0]
            for t in targets:
                if "homo sapiens" in str(t.get("organism", "")).lower():
                    best = t
                    break
            target_chembl_id = best.get("target_chembl_id")
            pref_name = best.get("pref_name")
    except Exception as e:
        pass

    # 2. 查找 Mechanism
    mechs = []
    mol_names = set()
    if target_chembl_id:
        try:
            # 注意：旧版 drug_mechanism.json 已废弃，现用 mechanism.json
            p2 = {"target_chembl_id": target_chembl_id, "limit": 1000, "format": "json"}
            r2 = request_with_retry(sess, "GET", CHEMBL_MECH_URL, params=p2)
            d2 = r2.json()
            mechs = d2.get("mechanisms") or [] # 返回字段通常是 mechanisms
            for m in mechs:
                mn = m.get("molecule_chembl_id") # 或者 molecule_name
                # 有些版本 API 返回结构不同，尝试获取名称
                name_guess = m.get("molecule_name") or m.get("molecule_chembl_id")
                if name_guess:
                    mol_names.add(str(name_guess))
        except Exception:
            pass

    out = {
        "symbol": symbol,
        "found": bool(target_chembl_id),
        "target_chembl_id": target_chembl_id,
        "pref_name": pref_name,
        "mechanisms_n": len(mechs),
        "unique_molecules_n": len(mol_names),
        "molecule_names": sorted(list(mol_names)),
        "timestamp": now_iso(),
    }
    safe_write_json(ck, out)
    time.sleep(sleep_s)
    return out


# -----------------------------
# 8) DGIdb (best-effort)
# -----------------------------
# --- 1. DGIdb Fix: 适配 v5 GraphQL Schema (增加 nodes 层级) ---
DGIDB_QUERY_GUESS = """
query Interactions($genes: [String!]!) {
  genes(names: $genes) {
    nodes {
      name
      interactions {
        drug { name }
      }
    }
  }
}
"""
def dgidb_fetch_batch(sess: requests.Session, cache_dir: Path, genes: List[str], sleep_s: float) -> Dict[str, Any]:
    # 生成缓存 Key
    key = "batch_" + str(abs(hash("|".join(genes))))[:10]
    ck = cache_dir / "dgidb" / cache_key("dgidb", key)
    cached = safe_read_json(ck)
    if cached is not None:
        return cached

    # 发送 GraphQL 请求
    body = {"query": DGIDB_QUERY_GUESS, "variables": {"genes": genes}}
    try:
        resp = request_with_retry(sess, "POST", DGIDB_GQL, json_body=body)
        data = resp.json()
        # 简单检查是否有 errors
        if "errors" in data:
            out = {"ok": False, "timestamp": now_iso(), "error": str(data["errors"]), "raw": None}
        else:
            out = {"ok": True, "timestamp": now_iso(), "raw": data}
    except Exception as e:
        out = {"ok": False, "timestamp": now_iso(), "error": str(e), "raw": None}

    safe_write_json(ck, out)
    time.sleep(sleep_s)
    return out
def dgidb_parse_interactions(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    if not raw or not raw.get("ok"):
        return result

    # 注意这里多了 .get("nodes")
    data = (raw.get("raw") or {}).get("data") or {}
    genes_data = (data.get("genes") or {}).get("nodes") or []
    
    for g in genes_data:
        name = normalize_symbol(g.get("name") or "")
        inter = g.get("interactions") or []
        drug_names = []
        for it in inter:
            d = it.get("drug") or {}
            dn = d.get("name")
            if dn:
                drug_names.append(str(dn))
        result[name] = {
            "interactions_n": int(len(inter)),
            "unique_drugs_n": int(len(set(drug_names))),
            "drug_names": sorted(set(drug_names)),
        }
    return result


# -----------------------------
# 9) Scoring (0-1)
# -----------------------------
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def log_norm(x: int, cap: int) -> float:
    x = max(0, int(x))
    return clamp01(math.log1p(x) / math.log1p(cap))


def compute_drug_score(
    approved_drugs_n: int,
    max_phase: int,
    tract_score: float,
    dgidb_unique_drugs_n: int,
    chembl_mechanisms_n: int,
    chembl_unique_molecules_n: int,
) -> Tuple[float, Dict[str, float], str]:
    """
    Teacher's Strategy: 50% Phase + 25% Structure + 25% Evidence
    """
    # 1. Phase Score (50%)
    phase_score = 1.0 if approved_drugs_n > 0 else clamp01((max_phase or 0) / 4.0)
    
    # 2. Tractability Score (25%)
    tract = clamp01(float(tract_score or 0.0))
    
    # 3. Evidence Score (25%)
    dgidb_score = log_norm(dgidb_unique_drugs_n, cap=30)
    chembl_score = max(log_norm(chembl_mechanisms_n, cap=30), log_norm(chembl_unique_molecules_n, cap=30))

    # Formula
    score = (
        0.50 * phase_score +
        0.25 * tract +
        0.15 * dgidb_score +
        0.10 * chembl_score
    )
    score = clamp01(score)

    # Tiers
    if approved_drugs_n > 0:
        tier = "ApprovedTarget"
    elif (max_phase or 0) >= 3:
        tier = "LateClinicalTarget"
    elif (max_phase or 0) >= 1:
        tier = "EarlyClinicalTarget"
    elif tract >= 0.6:
        tier = "Tractable(Structure/High)"
    elif tract > 0:
        tier = "Tractable(Predicted)"
    else:
        tier = "Unknown"

    parts = {
        "phase_score": float(phase_score),
        "tract_score": float(tract),
        "dgidb_score": float(dgidb_score),
        "chembl_score": float(chembl_score),
    }
    return score, parts, tier


# -----------------------------
# 10) Main
# -----------------------------
def run():
    # parse safely in notebooks
    args = parse_args_safely(sys.argv[1:])

    # Output dirs: Use project-wide paths from config
    project_root = Path(config.get_path("paths.project_root"))
    output_base = Path(config.get_path("paths.output_dir"))
    
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (output_base / "step5_clinical_validation/drug_mining")
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else (output_base / "step5_clinical_validation/drug_mining_cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load gene list from CSV
    gene_list_path = args.gene_list if args.gene_list else DEFAULT_GENE_LIST_PATH
    if not Path(gene_list_path).is_absolute():
        gene_list_path = str(project_root / gene_list_path)
    
    print(f"[INFO] Loading gene list from: {gene_list_path}")
    genes = load_gene_list_from_csv(gene_list_path)
    if args.max_genes and args.max_genes > 0:
        genes = genes[: args.max_genes]

    if len(genes) == 0:
        print("[ERROR] No genes parsed from GENE_LIST_TEXT.")
        return

    print(f"[INFO] Notebook mode: {is_notebook()}")
    print(f"[INFO] Genes loaded: {len(genes)}")
    print(f"[INFO] Output dir: {out_dir}")
    print(f"[INFO] Cache dir : {cache_dir}")

    # optional ion priors
    ion_map: Dict[str, Dict[str, Any]] = {}
    if args.ion_priors:
        ion_path = Path(args.ion_priors).resolve()
        ion_df = load_ion_priors(ion_path)
        ion_map = {
            r["symbol"]: {
                "ion_group_hits": r["ion_group_hits"],
                "is_real_ion_channel": int(r["is_real_ion_channel"]),
            }
            for _, r in ion_df.iterrows()
        }
        print(f"[INFO] Ion priors loaded: {len(ion_map)} symbols")
        print(f"[INFO] Ion channels (real, contains 'channel'): {sum(v['is_real_ion_channel'] for v in ion_map.values())}")
    else:
        print("[INFO] No ion priors provided. (Run with --ion_priors PATH if needed)")

    sess = requests_session()

    rows = []
    # Per gene: HGNC + OpenTargets + ChEMBL
    for i, sym in enumerate(genes, 1):
        if i % 25 == 0 or i == 1:
            print(f"[INFO] Progress {i}/{len(genes)} ...")

        h = hgnc_fetch(sess, cache_dir, sym, args.sleep)
        ensg = h.get("ensembl_gene_id")

        if ensg:
            ot = opentargets_fetch(sess, cache_dir, ensg, args.sleep)
        else:
            ot = {
                "ensembl_id": None,
                "found": False,
                "error": None,
                "approvedSymbol": None,
                "biotype": None,
                "known_drugs_n": 0,
                "approved_drugs_n": 0,
                "max_phase": 0,
                "drug_names": [],
                "tract_labels": [],
                "tract_score": 0.0,
                "tract_basis": "",
            }

                # 原代码: ch = chembl_fetch(sess, cache_dir, sym, args.sleep)
        # 修改为: ↓
        uids = "|".join(h.get("uniprot_ids") or []) if isinstance(h.get("uniprot_ids"), list) else (h.get("uniprot_ids") or "")
        ch = chembl_fetch(sess, cache_dir, sym, uids, args.sleep)

        ion = ion_map.get(sym, {})
        rows.append({
            "symbol": sym,

            # ion annotation (optional)
            "is_real_ion_channel": int(ion.get("is_real_ion_channel", 0)) if ion_map else 0,
            "ion_group_hits": ion.get("ion_group_hits", "") if ion_map else "",

            # HGNC
            "hgnc_found": bool(h.get("found", False)),
            "hgnc_id": h.get("hgnc_id"),
            "ensembl_gene_id": ensg,
            "entrez_id": h.get("entrez_id"),
            "uniprot_ids": "|".join(h.get("uniprot_ids") or []) if isinstance(h.get("uniprot_ids"), list) else (h.get("uniprot_ids") or ""),
            "locus_group": h.get("locus_group"),
            "locus_type": h.get("locus_type"),
            "hgnc_name": h.get("name"),

            # OpenTargets
            "ot_found": bool(ot.get("found", False)),
            "ot_error": ot.get("error"),
            "ot_known_drugs_n": int(ot.get("known_drugs_n", 0) or 0),
            "ot_approved_drugs_n": int(ot.get("approved_drugs_n", 0) or 0),
            "ot_max_phase": int(ot.get("max_phase", 0) or 0),
            "ot_drug_names": "|".join(ot.get("drug_names") or []),
            "ot_tract_score": float(ot.get("tract_score", 0.0) or 0.0),
            "ot_tract_basis": ot.get("tract_basis", ""),
            "ot_tract_labels": "|".join(ot.get("tract_labels") or []),

            # ChEMBL
            "chembl_found": bool(ch.get("found", False)),
            "chembl_error": ch.get("error"),
            "chembl_target_chembl_id": ch.get("target_chembl_id"),
            "chembl_pref_name": ch.get("pref_name"),
            "chembl_target_type": ch.get("target_type"),
            "chembl_mechanisms_n": int(ch.get("mechanisms_n", 0) or 0),
            "chembl_unique_molecules_n": int(ch.get("unique_molecules_n", 0) or 0),
            "chembl_molecule_names": "|".join(ch.get("molecule_names") or []),
        })

    df = pd.DataFrame(rows)

    # DGIdb in batches
    dgidb_map: Dict[str, Dict[str, Any]] = {}
    for start in range(0, len(genes), args.dgidb_batch):
        batch = genes[start:start + args.dgidb_batch]
        raw = dgidb_fetch_batch(sess, cache_dir, batch, args.sleep)
        parsed = dgidb_parse_interactions(raw)
        dgidb_map.update(parsed)

    df["dgidb_interactions_n"] = df["symbol"].map(lambda s: dgidb_map.get(s, {}).get("interactions_n", 0))
    df["dgidb_unique_drugs_n"] = df["symbol"].map(lambda s: dgidb_map.get(s, {}).get("unique_drugs_n", 0))
    df["dgidb_drug_names"] = df["symbol"].map(lambda s: "|".join(dgidb_map.get(s, {}).get("drug_names", [])))

    # Score + Tier
        # Score + Tier (Updated for Ladder Logic)
        # Score + Tier
    # --- Modified to match Teacher's Logic unpacking ---
    scores, tiers = [], []
    sp, st, sd, sc = [], [], [], []

    for _, r in df.iterrows():
        score, parts, tier = compute_drug_score(
            approved_drugs_n=int(r.get("ot_approved_drugs_n") or 0),
            max_phase=int(r.get("ot_max_phase") or 0),
            tract_score=float(r.get("ot_tract_score") or 0.0),
            dgidb_unique_drugs_n=int(r.get("dgidb_unique_drugs_n") or 0),
            chembl_mechanisms_n=int(r.get("chembl_mechanisms_n") or 0),
            chembl_unique_molecules_n=int(r.get("chembl_unique_molecules_n") or 0),
        )
        scores.append(score)
        tiers.append(tier)
        
        # Unpack the 4 parts
        sp.append(parts["phase_score"])
        st.append(parts["tract_score"])
        sd.append(parts["dgidb_score"])
        sc.append(parts["chembl_score"])

    # 写入 DataFrame
    df["DrugEvidenceScore"] = scores
    df["TargetTier"] = tiers
    df["score_phase"] = sp
    df["score_tractability"] = st
    df["score_dgidb"] = sd
    df["score_chembl"] = sc
    # Ranked
    ranked = df.sort_values(
        by=[
            "ot_approved_drugs_n",
            "ot_max_phase",
            "DrugEvidenceScore",
            "ot_tract_score",
            "dgidb_unique_drugs_n",
            "chembl_mechanisms_n",
            "is_real_ion_channel",  # tie-breaker only; does not inflate score
        ],
        ascending=[False, False, False, False, False, False, False]
    ).reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)

    # Save
    out_all = out_dir / "drug_mining_all.csv"
    out_ranked = out_dir / "drug_mining_ranked.csv"
    df.to_csv(out_all, index=False, encoding="utf-8-sig")
    ranked.to_csv(out_ranked, index=False, encoding="utf-8-sig")

    summary = {
        "timestamp": now_iso(),
        "n_genes": int(len(ranked)),
        "n_approved_targets": int((ranked["TargetTier"] == "ApprovedTarget").sum()),
        "n_late_clinical": int((ranked["TargetTier"] == "LateClinicalTarget").sum()),
        "n_early_clinical": int((ranked["TargetTier"] == "EarlyClinicalTarget").sum()),
        "n_real_ion_channels_in_list": int(ranked["is_real_ion_channel"].sum()) if "is_real_ion_channel" in ranked.columns else 0,
        "outputs": {
            "all": str(out_all),
            "ranked": str(out_ranked),
        }
    }
    safe_write_json(out_dir / "drug_mining_summary.json", summary)

    print("\n[DONE] Outputs:")
    print(f"  - {out_all}")
    print(f"  - {out_ranked}")
    print(f"  - {out_dir / 'drug_mining_summary.json'}")

    print("\n[TOP 15]")
    show_cols = ["rank", "symbol", "TargetTier", "DrugEvidenceScore", "ot_approved_drugs_n", "ot_max_phase", "ot_tract_score"]
    if "is_real_ion_channel" in ranked.columns:
        show_cols += ["is_real_ion_channel", "ion_group_hits"]
    print(ranked.head(15)[show_cols].to_string(index=False))


# -----------------------------
# 11) Entry
# -----------------------------
if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    run()
else:
    # If pasted into a notebook cell, it will still run automatically once the cell executes.
    # Comment out the next line if you prefer to call run() manually.
    run()
