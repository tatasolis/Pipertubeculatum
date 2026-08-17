# -*- coding: utf-8 -*-
"""
pipeline_lib.py — Funciones del pipeline in silico para tesis
=============================================================
Aceite esencial de Piper tuberculatum: identificación de compuestos,
predicción de dianas, enriquecimiento de vías e integración con
resultados in vitro (HCT-116, MTT, DCFDA/ROS).

Organización:
  §1  Configuración y constantes
  §2  Identidad química (PubChem)
  §3  Bioactividad experimental (ChEMBL)
  §4  Predicción de dianas (SwissTargetPrediction)
  §5  Estandarización de símbolos génicos
  §6  Relevancia en cáncer colorrectal (OpenTargets)
  §7  Enriquecimiento de vías (gseapy / Enrichr)
  §8  Red de interacción proteína-proteína (STRING)
  §9  Integración y priorización
  §10 Visualización
"""

from dataclasses import dataclass, field
from typing import Optional
import json
import os
import time
import warnings

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["figure.dpi"] = 150
matplotlib.rcParams["savefig.dpi"] = 300
matplotlib.rcParams["font.size"] = 9

import networkx as nx
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore", category=FutureWarning)


# ════════════════════════════════════════════════════════════════
# §1  CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """Parámetros centralizados — modifica aquí, no en el código."""

    # --- Compuestos (GC-MS) ---
    # Fuente: Tabla 5 de Zavala Lalama & Haro López (2026),
    # "Evaluación del potencial antibacteriano del aceite esencial
    # de Piper tuberculatum Jacq.", UPS Guayaquil.
    # Extracción: hidrodestilación con trampa de Clevenger.
    # Material: hojas secas, recolectadas en Vía a la Costa,
    # Guayaquil (2°09'51.6"S 79°58'31.8"W).
    # Total identificado: 93.56% del aceite.
    #
    # CID verificado manualmente.  None = resolver por nombre.
    #
    # ⚠ ALERTA: La Figura 9 y Anexo 17 de la tesis fuente muestran
    # para "β-cis-ocimene" una estructura con anillo bencénico y
    # grupos metilenodioxi + metoxi (3 oxígenos).  Eso NO corresponde
    # a β-cis-ocimene (C10H16, acíclico, sin oxígeno).  La estructura
    # dibujada se asemeja más a croweacin u otro fenilpropanoide.
    # Si la identificación GC-MS es correcta, la figura es un error
    # editorial.  Si la figura es correcta, el compuesto está mal
    # identificado y los SMILES/CID de ocimene serían incorrectos.
    # → VERIFICAR CON LOS AUTORES antes de interpretar resultados
    #   de predicción de targets para este compuesto.
    compounds: list = field(default_factory=lambda: [
        {"name": "beta-cis-Ocimene",
         "gcms_percent": 28.22,
         "verified_cid": 5281553,              # (Z)-β-ocimene — VERIFICAR
         "note": "CID 5281553 = (Z)-beta-ocimene (C10H16, acíclico). "
                 "ALERTA: la Fig.9 del paper fuente muestra una estructura "
                 "con O que NO coincide con ocimene. Verificar identidad "
                 "con autores/tutor. IR exp=1166, IR ref=1173."},
        {"name": "Germacra-4(15),5,10(14)-trien-1-alpha-ol",
         "gcms_percent": 15.60,     # Tabla 5 del paper: 15.6%
         "verified_cid": 13304977,
         "note": "CID verificado manualmente. IR exp=1381, IR ref=1392."},
        {"name": "Croweacin",
         "gcms_percent": 15.64,
         "verified_cid": None,
         "note": "Fenilpropanoide metilenodioxi. IR exp=1182, IR ref=1191."},
        {"name": "Caryophyllene oxide",
         "gcms_percent": 10.50,     # Tabla 5 del paper: 10.5%
         "verified_cid": None,
         "note": "Sesquiterpeno oxigenado. IR exp=1556, IR ref=1565."},
    ])

    # --- SwissTargetPrediction ---
    # Mapeo archivo CSV → compuesto (los CSVs se descargan manualmente
    # de http://www.swisstargetprediction.ch/)
    swisstarget_files: dict = field(default_factory=lambda: {
        "SwissTargetPrediction (2).csv": "beta-cis-Ocimene",
        "SwissTargetPrediction (3).csv": "Croweacin",
        "SwissTargetPrediction (4).csv": "Caryophyllene oxide",
        "SwissTargetPrediction (5).csv": "Germacra-4(15),5,10(14)-trien-1-alpha-ol",
    })

    # Filtrado de predicciones
    swisstarget_prob_high: float = 0.10    # umbral de alta confianza
    swisstarget_prob_low: float  = 0.05    # umbral mínimo de inclusión
    min_known_actives: int       = 5       # mínimo de ligandos de ref.

    # --- ChEMBL ---
    chembl_organism: str = "Homo sapiens"

    # --- CRC (OpenTargets) ---
    # EFO/MONDO IDs para cáncer colorrectal
    crc_disease_ids: list = field(default_factory=lambda: [
        "EFO_0005842",    # colorectal carcinoma
        "MONDO_0005575",  # colorectal cancer
        "EFO_0004142",    # colorectal neoplasm
    ])

    # --- Enrichment ---
    enrichment_gene_sets: list = field(default_factory=lambda: [
        "KEGG_2021_Human",
        "Reactome_2022",
        "GO_Biological_Process_2023",
    ])
    enrichment_fdr_threshold: float = 0.05

    # --- STRING PPI ---
    string_species: int     = 9606   # Homo sapiens
    string_score_threshold: int = 400  # combined score ≥ 0.4

    # --- Directorios ---
    output_dir: str = "outputs"

    # --- API delays (segundos) ---
    api_delay: float = 0.35   # pausa entre requests para evitar throttle


# ════════════════════════════════════════════════════════════════
# §2  IDENTIDAD QUÍMICA (PubChem)
# ════════════════════════════════════════════════════════════════

def resolve_compounds(compound_data: list[dict],
                      delay: float = 0.3) -> pd.DataFrame:
    """
    Resuelve identidad química vía PubChem.

    Para cada compuesto:
      - Si tiene verified_cid, busca por CID (identidad confirmada).
      - Si no, busca por nombre y toma el primer hit.

    Retorna DataFrame con: query_name, gcms_percent, status,
    cid, smiles, molecular_formula, iupac_name, note.
    """
    import pubchempy as pcp

    rows = []
    for entry in compound_data:
        name = entry["name"]
        note = entry.get("note", "")
        try:
            if entry.get("verified_cid") is not None:
                hits = pcp.get_compounds(entry["verified_cid"], "cid")
                status_ok = "FOUND (CID verificado manual)"
            else:
                hits = pcp.get_compounds(name, "name")
                status_ok = "FOUND" if len(hits) == 1 else "MULTIPLE HITS"

            if not hits:
                rows.append(dict(
                    query_name=name, gcms_percent=entry["gcms_percent"],
                    status="NOT FOUND", cid=None, smiles=None,
                    molecular_formula=None, iupac_name=None, note=note))
                continue

            c = hits[0]
            rows.append(dict(
                query_name=name,
                gcms_percent=entry["gcms_percent"],
                status=status_ok,
                cid=c.cid,
                smiles=c.isomeric_smiles,
                molecular_formula=c.molecular_formula,
                iupac_name=c.iupac_name,
                note=note))

        except Exception as e:
            rows.append(dict(
                query_name=name, gcms_percent=entry["gcms_percent"],
                status=f"ERROR: {e}", cid=None, smiles=None,
                molecular_formula=None, iupac_name=None, note=note))

        time.sleep(delay)

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════
# §3  BIOACTIVIDAD EXPERIMENTAL (ChEMBL)
# ════════════════════════════════════════════════════════════════
# Usa la API REST directamente.  La librería
# chembl_webresource_client depende de un endpoint /spore que
# frecuentemente devuelve 500 en los servidores de EBI.  El REST
# directo es más robusto y no tiene esa dependencia de schema.

_CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"


def _chembl_rest_get(endpoint: str, params: dict = None,
                     delay: float = 0.4) -> dict | None:
    """
    GET genérico al REST API de ChEMBL con reintentos.
    Retorna el JSON decodificado o None si falla.
    """
    url = f"{_CHEMBL_API}/{endpoint}.json"
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            # 500/503 → reintentar
            print(f"    [ChEMBL HTTP {r.status_code}] Reintento "
                  f"{attempt+1}/3 para {endpoint}...")
            time.sleep(delay * (attempt + 1))
        except requests.RequestException as e:
            print(f"    [ChEMBL CONN ERROR] {e} — reintento "
                  f"{attempt+1}/3")
            time.sleep(delay * (attempt + 1))
    return None


def _chembl_search_by_smiles(smiles: str,
                             delay: float = 0.4) -> str | None:
    """
    Busca un compuesto en ChEMBL por SMILES (flexmatch).
    Retorna molecule_chembl_id o None.
    """
    data = _chembl_rest_get("molecule", params={
        "molecule_structures__canonical_smiles__flexmatch": smiles,
        "limit": 1,
    }, delay=delay)
    if data and data.get("molecules"):
        return data["molecules"][0].get("molecule_chembl_id")
    return None


def _chembl_get_activities(chembl_id: str,
                           organism: str = "Homo sapiens",
                           delay: float = 0.4) -> list[dict]:
    """
    Obtiene todas las actividades para un molecule_chembl_id,
    paginando si es necesario.
    """
    all_activities = []
    params = {
        "molecule_chembl_id": chembl_id,
        "target_organism": organism,
        "limit": 100,
        "offset": 0,
    }
    while True:
        data = _chembl_rest_get("activity", params=params, delay=delay)
        if data is None:
            break
        activities = data.get("activities", [])
        all_activities.extend(activities)
        # Paginación
        if data.get("page_meta", {}).get("next"):
            params["offset"] = params["offset"] + params["limit"]
        else:
            break
    return all_activities


def _chembl_get_target_type(target_chembl_id: str,
                            delay: float = 0.3) -> str | None:
    """Obtiene target_type para un target_chembl_id."""
    data = _chembl_rest_get(f"target/{target_chembl_id}", delay=delay)
    if data:
        return data.get("target_type")
    return None


def query_chembl_bioactivity(df_identity: pd.DataFrame,
                             organism: str = "Homo sapiens",
                             delay: float = 0.5) -> pd.DataFrame:
    """
    Busca cada compuesto en ChEMBL por SMILES (REST API directa)
    y cuenta actividades reportadas en el organismo indicado.

    Retorna DataFrame con: compound, chembl_id, chembl_status,
    n_activities.

    NOTA: Usa REST directo en lugar de chembl_webresource_client
    para evitar la dependencia del endpoint /spore que frecuentemente
    falla con HTTP 500.
    """
    rows = []
    for _, row in df_identity.iterrows():
        name = row["query_name"]
        smiles = row["smiles"]

        if smiles is None or pd.isna(smiles):
            rows.append(dict(compound=name, chembl_id=None,
                             chembl_status="SIN SMILES", n_activities=0))
            continue

        print(f"  Buscando {name} en ChEMBL...")
        chembl_id = _chembl_search_by_smiles(smiles, delay=delay)

        if chembl_id is None:
            rows.append(dict(compound=name, chembl_id=None,
                             chembl_status="NO ENCONTRADO", n_activities=0))
            time.sleep(delay)
            continue

        activities = _chembl_get_activities(chembl_id, organism, delay)
        status = ("CON ACTIVIDAD" if activities
                  else "SIN ACTIVIDAD EN HUMANO")
        rows.append(dict(compound=name, chembl_id=chembl_id,
                         chembl_status=status,
                         n_activities=len(activities)))
        print(f"    → {chembl_id}: {len(activities)} actividades")
        time.sleep(delay)

    return pd.DataFrame(rows)


def get_chembl_activity_details(chembl_id: str,
                                organism: str = "Homo sapiens",
                                delay: float = 0.4
                                ) -> pd.DataFrame:
    """
    Obtiene detalle de actividades para un compuesto ChEMBL.
    Distingue entre SINGLE PROTEIN y otros target types.
    Usa REST API directa.
    """
    activities = _chembl_get_activities(chembl_id, organism, delay)

    if not activities:
        return pd.DataFrame()

    # Obtener target_type para cada diana única
    unique_targets = {a.get("target_chembl_id") for a in activities
                      if a.get("target_chembl_id")}
    target_types = {}
    for tid in unique_targets:
        target_types[tid] = _chembl_get_target_type(tid, delay=delay / 2)

    rows = []
    for act in activities:
        tid = act.get("target_chembl_id")
        rows.append(dict(
            target_name=act.get("target_pref_name"),
            target_chembl_id=tid,
            target_type=target_types.get(tid),
            standard_type=act.get("standard_type"),
            standard_value=act.get("standard_value"),
            standard_units=act.get("standard_units"),
            assay_description=act.get("assay_description"),
        ))

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════
# §4  PREDICCIÓN DE DIANAS (SwissTargetPrediction)
# ════════════════════════════════════════════════════════════════

def load_swisstarget_predictions(file_mapping: dict) -> pd.DataFrame:
    """
    Carga CSVs de SwissTargetPrediction y los etiqueta con el
    compuesto de origen.

    Limpia la columna 'Known actives (3D/2D)' separándola en
    known_actives_3d y known_actives_2d.
    """
    frames = []
    for filename, compound in file_mapping.items():
        if not os.path.isfile(filename):
            print(f"  [WARN] Archivo no encontrado: {filename}")
            continue
        df = pd.read_csv(filename)
        df["compound"] = compound
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No se encontró ningún archivo de SwissTargetPrediction. "
            "Verifica los nombres en config.swisstarget_files.")

    df_raw = pd.concat(frames, ignore_index=True)

    # Parsear columna de known actives
    def _parse_ka(val):
        try:
            parts = str(val).split("/")
            return pd.Series([int(parts[0].strip()), int(parts[1].strip())])
        except Exception:
            return pd.Series([0, 0])

    df_raw[["known_actives_3d", "known_actives_2d"]] = \
        df_raw["Known actives (3D/2D)"].apply(_parse_ka)
    df_raw["known_actives_total"] = (df_raw["known_actives_3d"]
                                     + df_raw["known_actives_2d"])

    return df_raw


def filter_predictions(df_raw: pd.DataFrame,
                       prob_threshold: float = 0.05,
                       prob_high: float = 0.10,
                       min_known_actives: int = 5) -> pd.DataFrame:
    """
    Filtra predicciones de SwissTP por probabilidad y marca confianza.

    Criterios:
      - Retiene toda predicción con probability >= prob_threshold
      - Marca 'HIGH' si probability >= prob_high
      - Marca 'LOW_REFERENCE' si known_actives_total < min_known_actives

    Justificación del threshold:
      SwissTP reporta 'Probability*' como la probabilidad de que
      la molécula sea activa contra esa diana, calibrada contra
      su espacio de entrenamiento.  Un corte de 0.05 retiene
      predicciones con señal mínima; 0.10 filtra a alta confianza.
      A diferencia de un top-N fijo, esto es independiente del
      número de predicciones por compuesto y evita incluir hits
      con P~0 solo para llenar cupo.
    """
    # Estandarizar nombre de columna de probabilidad
    prob_col = "Probability*"
    if prob_col not in df_raw.columns:
        # Intentar variantes
        for c in df_raw.columns:
            if "probab" in c.lower():
                prob_col = c
                break

    df = df_raw[df_raw[prob_col] >= prob_threshold].copy()

    # Marcas de confianza
    df["confidence"] = "MEDIUM"
    df.loc[df[prob_col] >= prob_high, "confidence"] = "HIGH"
    df.loc[df["known_actives_total"] < min_known_actives, "confidence"] = \
        "LOW_REFERENCE"

    # Renombrar columnas a formato limpio
    rename_map = {
        "Common name": "gene_symbol",
        "Target": "protein_name",
        "Uniprot ID": "uniprot_id",
        "ChEMBL ID": "chembl_id_swisstp",
        "Target Class": "target_class",
        prob_col: "probability",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items()
                            if k in df.columns})

    keep_cols = ["compound", "gene_symbol", "protein_name", "uniprot_id",
                 "chembl_id_swisstp", "target_class", "probability",
                 "known_actives_3d", "known_actives_2d",
                 "known_actives_total", "confidence"]
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols].sort_values(
        ["compound", "probability"], ascending=[True, False]
    ).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════
# §5  ESTANDARIZACIÓN DE SÍMBOLOS GÉNICOS
# ════════════════════════════════════════════════════════════════

def _uniprot_to_gene(uniprot_id: str, delay: float = 0.35) -> Optional[str]:
    """Consulta UniProt REST para obtener el gene symbol oficial."""
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        r = requests.get(url, timeout=15,
                         headers={"Accept": "application/json"})
        r.raise_for_status()
        data = r.json()
        gene_name = data["genes"][0]["geneName"]["value"]
        time.sleep(delay)
        return gene_name
    except Exception:
        time.sleep(delay)
        return None


def standardize_gene_symbols(df: pd.DataFrame,
                             delay: float = 0.35) -> pd.DataFrame:
    """
    1. Separa filas con gene_symbol = 'N/A' (complejos multiproteicos).
    2. Desagrega complejos por subunidad (split en '&' de uniprot_id).
    3. Resuelve gene symbols vía UniProt REST API.
    4. Recombina y elimina filas sin símbolo resuelto.
    5. Intenta verificar símbolos existentes vía mygene si disponible.
    """
    df_single = df[df["gene_symbol"] != "N/A"].copy()
    df_complex = df[df["gene_symbol"] == "N/A"].copy()

    # Desagregar complejos
    exploded = []
    for _, row in df_complex.iterrows():
        if pd.isna(row.get("uniprot_id")):
            continue
        for uid in str(row["uniprot_id"]).split("&"):
            new_row = row.copy()
            new_row["uniprot_id"] = uid.strip()
            new_row["gene_symbol"] = None
            new_row["from_complex"] = True
            exploded.append(new_row)

    df_single["from_complex"] = False

    if exploded:
        df_exploded = pd.DataFrame(exploded)
        # Resolver gene symbols
        print(f"  Resolviendo {len(df_exploded)} subunidades vía UniProt...")
        df_exploded["gene_symbol"] = df_exploded["uniprot_id"].apply(
            lambda uid: _uniprot_to_gene(uid, delay=delay))
        df_all = pd.concat([df_single, df_exploded], ignore_index=True)
    else:
        df_all = df_single.copy()

    # Eliminar filas sin símbolo resuelto
    n_before = len(df_all)
    df_all = df_all.dropna(subset=["gene_symbol"])
    n_dropped = n_before - len(df_all)
    if n_dropped:
        print(f"  Eliminadas {n_dropped} filas sin gene_symbol resuelto.")

    # Intentar verificación con mygene (opcional)
    try:
        import mygene
        mg = mygene.MyGeneInfo()
        symbols = df_all["gene_symbol"].unique().tolist()
        result = mg.querymany(symbols, scopes="symbol", fields="symbol",
                              species="human", returnall=True)
        alias_map = {}
        for hit in result.get("out", []):
            query = hit.get("query")
            official = hit.get("symbol")
            if query and official and query != official:
                alias_map[query] = official

        if alias_map:
            print(f"  Corregidos {len(alias_map)} alias → símbolo oficial:")
            for old, new in alias_map.items():
                print(f"    {old} → {new}")
            df_all["gene_symbol"] = df_all["gene_symbol"].replace(alias_map)
    except ImportError:
        print("  [INFO] mygene no instalado — omitiendo verificación HGNC. "
              "Instalar con: pip install mygene")

    return df_all.reset_index(drop=True)


# ════════════════════════════════════════════════════════════════
# §6  RELEVANCIA EN CÁNCER COLORRECTAL (OpenTargets)
# ════════════════════════════════════════════════════════════════

_OT_GRAPHQL = "https://api.platform.opentargets.org/api/v4/graphql"


def _ot_gene_to_ensembl(gene_symbol: str) -> tuple[Optional[str], Optional[str]]:
    """
    Mapea gene symbol → (Ensembl ID, approved symbol) vía OpenTargets.
    Retorna (None, None) si no se encuentra.
    """
    query = """
    query($q: String!) {
      search(queryString: $q, entityNames: ["target"], page: {index: 0, size: 3}) {
        hits { id object { ... on Target { approvedSymbol } } }
      }
    }"""
    try:
        r = requests.post(_OT_GRAPHQL, json={
            "query": query, "variables": {"q": gene_symbol}},
            timeout=15)
        r.raise_for_status()
        data = r.json()
        hits = data.get("data", {}).get("search", {}).get("hits", [])
        if not hits:
            return None, None

        # Preferir hit cuyo approvedSymbol coincida exactamente
        for h in hits:
            approved = (h.get("object") or {}).get("approvedSymbol", "")
            if approved.upper() == gene_symbol.upper():
                return h["id"], approved

        # Si no hay match exacto, tomar el primero
        approved = (hits[0].get("object") or {}).get("approvedSymbol")
        return hits[0]["id"], approved
    except Exception:
        return None, None


def _ot_disease_association(ensembl_id: str,
                            disease_ids: list[str]) -> dict:
    """
    Consulta TODAS las asociaciones de un target en OpenTargets,
    luego filtra en Python por los disease IDs de interés (CRC).

    Esta estrategia evita depender de filtros GraphQL cuya sintaxis
    cambia entre versiones del API.
    """
    # Query sin filtro de enfermedad — obtener todas las asociaciones
    query = """
    query($ensemblId: String!) {
      target(ensemblId: $ensemblId) {
        approvedSymbol
        associatedDiseases(page: { index: 0, size: 500 }) {
          count
          rows {
            disease { id name }
            score
          }
        }
      }
    }"""
    try:
        r = requests.post(_OT_GRAPHQL, json={
            "query": query,
            "variables": {"ensemblId": ensembl_id}},
            timeout=20)
        r.raise_for_status()
        resp = r.json()

        # Verificar errores GraphQL
        if "errors" in resp:
            err_msg = resp["errors"][0].get("message", "Unknown GraphQL error")
            return {"ensembl_id": ensembl_id, "ot_crc_score": np.nan,
                    "ot_crc_diseases": f"GQL_ERROR: {err_msg}",
                    "ot_approved_symbol": None}

        data = resp.get("data", {}).get("target")
        if not data:
            return {"ensembl_id": ensembl_id, "ot_crc_score": 0,
                    "ot_crc_diseases": "", "ot_approved_symbol": None}

        approved = data.get("approvedSymbol")
        all_rows = data.get("associatedDiseases", {}).get("rows", [])

        # Filtrar por disease IDs de CRC
        # Comparar con y sin prefijo (EFO_0005842 vs EFO:0005842)
        disease_ids_normalized = set()
        for did in disease_ids:
            disease_ids_normalized.add(did)
            disease_ids_normalized.add(did.replace("_", ":"))
            disease_ids_normalized.add(did.replace(":", "_"))

        crc_rows = [row for row in all_rows
                    if row["disease"]["id"] in disease_ids_normalized]

        if not crc_rows:
            return {"ensembl_id": ensembl_id, "ot_crc_score": 0,
                    "ot_crc_diseases": "",
                    "ot_approved_symbol": approved}

        best = max(crc_rows, key=lambda x: x["score"])
        diseases = "; ".join(
            f"{r['disease']['name']} ({r['score']:.3f})"
            for r in crc_rows)

        return {"ensembl_id": ensembl_id,
                "ot_crc_score": best["score"],
                "ot_crc_diseases": diseases,
                "ot_approved_symbol": approved}

    except Exception as e:
        return {"ensembl_id": ensembl_id, "ot_crc_score": np.nan,
                "ot_crc_diseases": f"ERROR: {e}",
                "ot_approved_symbol": None}


def query_crc_relevance(gene_symbols: list[str],
                        disease_ids: list[str],
                        delay: float = 0.4,
                        cache_path: Optional[str] = None
                        ) -> pd.DataFrame:
    """
    Para cada gen, consulta OpenTargets para determinar su asociación
    con cáncer colorrectal.

    Retorna DataFrame con: gene_symbol, ensembl_id, ot_crc_score,
    ot_crc_diseases, ot_approved_symbol, crc_relevance_flag.

    crc_relevance_flag:
      - 'STRONG':   score >= 0.5
      - 'MODERATE': 0.1 <= score < 0.5
      - 'WEAK':     0 < score < 0.1
      - 'NONE':     score == 0 o no encontrado
      - 'ERROR':    fallo en la consulta
    """
    # Intentar cargar cache
    if cache_path and os.path.isfile(cache_path):
        print(f"  Cargando cache CRC: {cache_path}")
        return pd.read_csv(cache_path)

    unique_genes = sorted(set(gene_symbols))
    print(f"  Consultando OpenTargets para {len(unique_genes)} genes...")

    results = []
    n_found = 0
    n_with_crc = 0
    for i, gene in enumerate(unique_genes):
        if (i + 1) % 10 == 0:
            print(f"    ... {i+1}/{len(unique_genes)} "
                  f"(encontrados: {n_found}, con CRC: {n_with_crc})")

        ens_id, approved = _ot_gene_to_ensembl(gene)
        time.sleep(delay / 2)

        if ens_id is None:
            results.append({"gene_symbol": gene, "ensembl_id": None,
                            "ot_crc_score": 0.0,
                            "ot_crc_diseases": "",
                            "ot_approved_symbol": None})
            continue

        n_found += 1
        assoc = _ot_disease_association(ens_id, disease_ids)
        assoc["gene_symbol"] = gene
        if not pd.isna(assoc.get("ot_crc_score", 0)) and assoc.get("ot_crc_score", 0) > 0:
            n_with_crc += 1
        results.append(assoc)
        time.sleep(delay)

    df = pd.DataFrame(results)

    # Flag de relevancia
    def _flag(score):
        if pd.isna(score):
            return "ERROR"
        if score >= 0.5:
            return "STRONG"
        if score >= 0.1:
            return "MODERATE"
        if score > 0:
            return "WEAK"
        return "NONE"

    df["crc_relevance"] = df["ot_crc_score"].apply(_flag)

    if cache_path:
        df.to_csv(cache_path, index=False)
        print(f"  Cache guardada: {cache_path}")

    return df


# ════════════════════════════════════════════════════════════════
# §7  ENRIQUECIMIENTO DE VÍAS
# ════════════════════════════════════════════════════════════════

def run_pathway_enrichment(gene_list: list[str],
                           gene_sets: list[str],
                           fdr_threshold: float = 0.05,
                           output_dir: str = "outputs"
                           ) -> pd.DataFrame:
    """
    Over-Representation Analysis (ORA) con gseapy/Enrichr.

    Usa corrección Benjamini-Hochberg (FDR) y retorna solo
    términos con FDR < fdr_threshold.

    gene_sets: nombres de bibliotecas Enrichr, e.g.
      ['KEGG_2021_Human', 'Reactome_2022', 'GO_Biological_Process_2023']
    """
    import gseapy as gp

    all_results = []

    for gs in gene_sets:
        print(f"  Enrichment: {gs} ...")
        try:
            enr = gp.enrich(
                gene_list=gene_list,
                gene_sets=gs,
                outdir=None,         # no guardar automáticamente
                no_plot=True,
                cutoff=fdr_threshold,
            )
            df = enr.results.copy()
            if not df.empty:
                df["gene_set_library"] = gs
                all_results.append(df)
                print(f"    {len(df)} términos significativos (FDR < {fdr_threshold})")
            else:
                print(f"    Sin resultados significativos.")
        except Exception as e:
            print(f"    [ERROR] {gs}: {e}")

    if not all_results:
        print("  [WARN] No se encontraron términos enriquecidos en ninguna base.")
        return pd.DataFrame()

    df_all = pd.concat(all_results, ignore_index=True)

    # Estandarizar nombres de columnas de gseapy
    col_map = {
        "Term": "term",
        "Overlap": "overlap",
        "P-value": "pvalue",
        "Adjusted P-value": "fdr",
        "Odds Ratio": "odds_ratio",
        "Combined Score": "combined_score",
        "Genes": "genes",
    }
    df_all = df_all.rename(columns={k: v for k, v in col_map.items()
                                     if k in df_all.columns})

    # Guardar
    path = os.path.join(output_dir, "pathway_enrichment.csv")
    df_all.to_csv(path, index=False)
    print(f"  Guardado: {path}")

    return df_all


# ════════════════════════════════════════════════════════════════
# §8  RED PPI (STRING)
# ════════════════════════════════════════════════════════════════

_STRING_API = "https://string-db.org/api"


def build_ppi_network(gene_list: list[str],
                      species: int = 9606,
                      score_threshold: int = 400,
                      output_dir: str = "outputs"
                      ) -> tuple[nx.Graph, pd.DataFrame]:
    """
    Consulta STRING para obtener interacciones y construye un grafo
    con networkx.

    Retorna (grafo, DataFrame de interacciones).
    """
    # STRING acepta hasta ~2000 proteínas por request
    identifiers = "%0d".join(gene_list)

    # Obtener interacciones
    url = f"{_STRING_API}/json/network"
    params = {
        "identifiers": identifiers,
        "species": species,
        "required_score": score_threshold,
        "caller_identity": "thesis_pipeline",
    }

    print(f"  Consultando STRING ({len(gene_list)} genes, "
          f"score ≥ {score_threshold})...")

    try:
        r = requests.post(url, data=params, timeout=60)
        r.raise_for_status()
        interactions = r.json()
    except Exception as e:
        print(f"  [ERROR] STRING API: {e}")
        return nx.Graph(), pd.DataFrame()

    if not interactions:
        print("  Sin interacciones encontradas.")
        return nx.Graph(), pd.DataFrame()

    # Construir DataFrame
    rows = []
    for edge in interactions:
        rows.append({
            "node1": edge["preferredName_A"],
            "node2": edge["preferredName_B"],
            "combined_score": edge["score"],
            "nscore": edge.get("nscore", 0),
            "escore": edge.get("escore", 0),
            "dscore": edge.get("dscore", 0),
        })

    df_ppi = pd.DataFrame(rows)

    # Construir grafo
    G = nx.Graph()
    G.add_nodes_from(gene_list)
    for _, row in df_ppi.iterrows():
        G.add_edge(row["node1"], row["node2"],
                    weight=row["combined_score"])

    # Remover nodos aislados que no estaban en las interacciones
    # (mantenerlos es informativo, pero para el análisis de red los
    # reportamos aparte)
    isolated = list(nx.isolates(G))

    print(f"  Nodos: {G.number_of_nodes()} "
          f"({len(isolated)} aislados)")
    print(f"  Interacciones: {G.number_of_edges()}")

    # Guardar
    path = os.path.join(output_dir, "ppi_interactions.csv")
    df_ppi.to_csv(path, index=False)

    return G, df_ppi


def get_ppi_metrics(G: nx.Graph) -> pd.DataFrame:
    """
    Calcula métricas de centralidad para cada nodo del grafo PPI.
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame()

    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)

    # Detectar clusters (Louvain si disponible, sino connected components)
    try:
        from community import community_louvain
        partition = community_louvain.best_partition(G)
    except ImportError:
        # Fallback: connected components como pseudo-clusters
        partition = {}
        for i, component in enumerate(nx.connected_components(G)):
            for node in component:
                partition[node] = i

    df = pd.DataFrame({
        "gene_symbol": list(degree.keys()),
        "ppi_degree": list(degree.values()),
        "ppi_betweenness": [betweenness.get(n, 0) for n in degree],
        "ppi_closeness": [closeness.get(n, 0) for n in degree],
        "ppi_cluster": [partition.get(n, -1) for n in degree],
    })

    return df.sort_values("ppi_degree", ascending=False).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════
# §9  INTEGRACIÓN Y PRIORIZACIÓN
# ════════════════════════════════════════════════════════════════

def build_integration_table(df_targets: pd.DataFrame,
                            df_crc: pd.DataFrame,
                            df_enrichment: pd.DataFrame,
                            df_ppi_metrics: pd.DataFrame,
                            ) -> pd.DataFrame:
    """
    Une todas las capas de evidencia en una tabla maestra por gen.
    """
    # Convergencia: en cuántos compuestos aparece cada gen
    convergence = (df_targets.groupby("gene_symbol")["compound"]
                   .nunique().reset_index()
                   .rename(columns={"compound": "n_compounds"}))

    # Probabilidad media y máxima por gen
    prob_stats = (df_targets.groupby("gene_symbol")["probability"]
                  .agg(["mean", "max"]).reset_index()
                  .rename(columns={"mean": "prob_mean", "max": "prob_max"}))

    # Compuestos que predicen cada gen
    compounds_per_gene = (df_targets.groupby("gene_symbol")["compound"]
                          .apply(lambda x: "; ".join(sorted(x.unique())))
                          .reset_index()
                          .rename(columns={"compound": "predicted_by"}))

    # Target class (tomar la más frecuente)
    if "target_class" in df_targets.columns:
        tclass = (df_targets.groupby("gene_symbol")["target_class"]
                  .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0
                       else "Unknown")
                  .reset_index())
    else:
        tclass = pd.DataFrame({"gene_symbol": convergence["gene_symbol"],
                                "target_class": "Unknown"})

    # Unir todo
    df = convergence.copy()
    df = df.merge(prob_stats, on="gene_symbol", how="left")
    df = df.merge(compounds_per_gene, on="gene_symbol", how="left")
    df = df.merge(tclass, on="gene_symbol", how="left")

    # CRC
    if not df_crc.empty:
        crc_cols = ["gene_symbol", "ot_crc_score", "crc_relevance",
                    "ot_crc_diseases"]
        crc_cols = [c for c in crc_cols if c in df_crc.columns]
        df = df.merge(df_crc[crc_cols], on="gene_symbol", how="left")
    else:
        df["ot_crc_score"] = np.nan
        df["crc_relevance"] = "NOT_QUERIED"

    # Pathways enriquecidos que contienen cada gen
    if not df_enrichment.empty and "genes" in df_enrichment.columns:
        gene_pathways = {}
        for _, row in df_enrichment.iterrows():
            genes_in_term = str(row["genes"]).split(";")
            for g in genes_in_term:
                g = g.strip()
                if g not in gene_pathways:
                    gene_pathways[g] = []
                gene_pathways[g].append(row.get("term", ""))

        pathway_df = pd.DataFrame([
            {"gene_symbol": g,
             "n_enriched_pathways": len(terms),
             "enriched_pathways_list": "; ".join(terms[:5])  # max 5
             }
            for g, terms in gene_pathways.items()
        ])
        df = df.merge(pathway_df, on="gene_symbol", how="left")
    else:
        df["n_enriched_pathways"] = 0

    df["n_enriched_pathways"] = df["n_enriched_pathways"].fillna(0).astype(int)

    # PPI
    if not df_ppi_metrics.empty:
        ppi_cols = ["gene_symbol", "ppi_degree", "ppi_betweenness",
                    "ppi_cluster"]
        ppi_cols = [c for c in ppi_cols if c in df_ppi_metrics.columns]
        df = df.merge(df_ppi_metrics[ppi_cols], on="gene_symbol", how="left")
    else:
        df["ppi_degree"] = 0
        df["ppi_betweenness"] = 0.0

    df["ppi_degree"] = df["ppi_degree"].fillna(0).astype(int)
    df["ppi_betweenness"] = df["ppi_betweenness"].fillna(0.0)

    return df


def calculate_priority_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula un score compuesto de priorización para cada gen candidato.

    Componentes (todos normalizados a [0, 1] por min-max dentro del dataset):
      1. prob_max           — máxima probabilidad SwissTP       (peso 0.25)
      2. ot_crc_score       — score OpenTargets para CRC        (peso 0.30)
      3. n_enriched_pathways — participación en vías enriquecidas (peso 0.20)
      4. ppi_degree         — centralidad en red PPI            (peso 0.15)
      5. n_compounds        — convergencia entre compuestos     (peso 0.10)

    Justificación de pesos:
      - CRC relevance tiene el peso más alto porque es la pregunta
        biológica central de la tesis.
      - La probabilidad de target prediction es el input primario
        pero tiene sesgo del modelo (SwissTP favorece kinasas).
      - Pathway y PPI aportan contexto funcional.
      - Convergencia es informativa pero potencialmente artefactual
        (sesgo de SwissTP), por eso tiene el peso más bajo.

    Este score es una heurística para priorización.  NO es una
    medida de probabilidad ni reemplaza validación experimental.
    """
    df = df.copy()

    weights = {
        "prob_max": 0.25,
        "ot_crc_score": 0.30,
        "n_enriched_pathways": 0.20,
        "ppi_degree": 0.15,
        "n_compounds": 0.10,
    }

    # Normalizar min-max
    for col in weights:
        if col not in df.columns:
            df[col] = 0
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            df[f"{col}_norm"] = (vals - vmin) / (vmax - vmin)
        else:
            df[f"{col}_norm"] = 0.0

    # Score compuesto
    df["priority_score"] = sum(
        weights[col] * df[f"{col}_norm"] for col in weights
    )

    # Rank
    df = df.sort_values("priority_score", ascending=False).reset_index(drop=True)
    df["priority_rank"] = range(1, len(df) + 1)

    return df


def create_phenotype_bridge(df_prioritized: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la estructura de integración:
      compound → predicted targets → CRC evidence → enriched pathways
      → PPI centrality → experimental phenotype

    Los campos de fenotipos experimentales (MTT IC50, ROS fold-change)
    se dejan como NaN para que el usuario los complete con sus datos.
    """
    df = df_prioritized.copy()

    # Placeholder para datos experimentales
    df["mtt_ic50_ugml"] = np.nan       # IC50 del aceite completo (µg/mL)
    df["ros_fold_change"] = np.nan     # fold-change DCFDA vs control
    df["phenotype_note"] = ""          # notas del investigador

    # Clasificación mecanística automática basada en target_class
    def _mech_hint(row):
        tc = str(row.get("target_class", "")).lower()
        pathways = str(row.get("enriched_pathways_list", "")).lower()
        hints = []
        if any(k in tc for k in ["kinase", "phosphatase"]):
            hints.append("señalización")
        if any(k in tc for k in ["protease", "caspase"]):
            hints.append("apoptosis/proteólisis")
        if any(k in tc for k in ["oxidoreductase", "cytochrome"]):
            hints.append("estrés oxidativo")
        if any(k in tc for k in ["nuclear receptor"]):
            hints.append("regulación transcripcional")
        if any(k in tc for k in ["gpcr", "g protein"]):
            hints.append("señalización GPCR")
        if any(k in pathways for k in ["apoptosis", "death"]):
            hints.append("apoptosis")
        if any(k in pathways for k in ["cell cycle", "proliferation"]):
            hints.append("ciclo celular")
        if any(k in pathways for k in ["oxidative", "ros", "reactive oxygen"]):
            hints.append("ROS")
        if any(k in pathways for k in ["wnt", "beta-catenin"]):
            hints.append("Wnt/β-catenina")
        if any(k in pathways for k in ["pi3k", "akt", "mtor"]):
            hints.append("PI3K-AKT")
        if any(k in pathways for k in ["mapk", "erk", "ras"]):
            hints.append("MAPK")
        return "; ".join(sorted(set(hints))) if hints else ""

    df["mechanism_hint"] = df.apply(_mech_hint, axis=1)

    return df


# ════════════════════════════════════════════════════════════════
# §10  VISUALIZACIÓN
# ════════════════════════════════════════════════════════════════

def plot_enrichment_barplot(df_enr: pd.DataFrame,
                            top_n: int = 25,
                            output_dir: str = "outputs"):
    """
    Barplot horizontal de los top-N términos enriquecidos,
    coloreado por base de datos, con -log10(FDR) en el eje X.
    """
    if df_enr.empty:
        print("  Sin datos de enrichment para graficar.")
        return

    df = df_enr.copy()
    df["neg_log_fdr"] = -np.log10(df["fdr"].clip(lower=1e-50))
    df = df.nlargest(top_n, "neg_log_fdr")

    # Truncar nombres largos
    df["term_short"] = df["term"].apply(
        lambda x: (x[:60] + "...") if len(str(x)) > 63 else x)

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.3)))

    colors = {"KEGG_2021_Human": "#2196F3",
              "Reactome_2022": "#FF9800",
              "GO_Biological_Process_2023": "#4CAF50"}

    for lib in df["gene_set_library"].unique():
        mask = df["gene_set_library"] == lib
        ax.barh(df.loc[mask, "term_short"],
                df.loc[mask, "neg_log_fdr"],
                color=colors.get(lib, "#9E9E9E"),
                label=lib, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("−log₁₀(FDR)")
    ax.set_title(f"Top-{top_n} términos enriquecidos")
    ax.legend(fontsize=7, loc="lower right")
    ax.invert_yaxis()
    plt.tight_layout()

    path = os.path.join(output_dir, "enrichment_barplot.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"  Guardado: {path}")
    plt.show()


def plot_ppi_network(G: nx.Graph,
                     df_crc: Optional[pd.DataFrame] = None,
                     output_dir: str = "outputs"):
    """
    Visualización de la red PPI con tamaño de nodo proporcional al
    degree y color según relevancia CRC.
    """
    if G.number_of_edges() == 0:
        print("  Sin interacciones para graficar.")
        return

    # Subgrafo sin nodos aislados para mejor visualización
    nodes_with_edges = [n for n in G.nodes() if G.degree(n) > 0]
    H = G.subgraph(nodes_with_edges)

    fig, ax = plt.subplots(figsize=(10, 10))

    pos = nx.spring_layout(H, k=1.5 / np.sqrt(H.number_of_nodes()),
                           seed=42, iterations=50)

    degrees = dict(H.degree())
    node_sizes = [max(100, degrees[n] * 80) for n in H.nodes()]

    # Colorear por CRC relevance si disponible
    if df_crc is not None and not df_crc.empty:
        crc_scores = df_crc.set_index("gene_symbol")["ot_crc_score"]
        node_colors = [crc_scores.get(n, 0) if not pd.isna(crc_scores.get(n, 0))
                       else 0 for n in H.nodes()]
        nc = ax.scatter([], [], c=[], cmap="YlOrRd", vmin=0, vmax=1)
        nx.draw_networkx_nodes(H, pos, node_size=node_sizes,
                               node_color=node_colors,
                               cmap=plt.cm.YlOrRd, vmin=0, vmax=1,
                               edgecolors="black", linewidths=0.5, ax=ax)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd,
                                    norm=plt.Normalize(0, 1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.6, label="CRC relevance (OpenTargets)")
    else:
        nx.draw_networkx_nodes(H, pos, node_size=node_sizes,
                               node_color="#4FC3F7",
                               edgecolors="black", linewidths=0.5, ax=ax)

    nx.draw_networkx_edges(H, pos, alpha=0.3, width=0.5, ax=ax)

    # Labels solo para top hubs
    top_hubs = sorted(degrees, key=degrees.get, reverse=True)[:20]
    labels = {n: n for n in top_hubs}
    nx.draw_networkx_labels(H, pos, labels, font_size=7, ax=ax)

    ax.set_title("Red PPI (STRING)", fontsize=12)
    ax.axis("off")
    plt.tight_layout()

    path = os.path.join(output_dir, "ppi_network.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"  Guardado: {path}")
    plt.show()


def plot_compound_target_heatmap(df_targets: pd.DataFrame,
                                 top_n: int = 30,
                                 output_dir: str = "outputs"):
    """
    Heatmap de probabilidad: compuestos (columnas) × genes (filas).
    """
    # Seleccionar top genes por probabilidad máxima
    top_genes = (df_targets.groupby("gene_symbol")["probability"]
                 .max().nlargest(top_n).index)
    sub = df_targets[df_targets["gene_symbol"].isin(top_genes)]

    pivot = sub.pivot_table(index="gene_symbol", columns="compound",
                            values="probability", aggfunc="max",
                            fill_value=0)

    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.25)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd",
                    vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Probability (SwissTP)")
    ax.set_title(f"Predicción de dianas (top-{top_n} genes)")
    plt.tight_layout()

    path = os.path.join(output_dir, "compound_target_heatmap.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"  Guardado: {path}")
    plt.show()
