# -*- coding: utf-8 -*-
"""
run_pipeline.py — Ejecución del pipeline in silico
===================================================
Tesis: Potencial anticancerígeno del aceite esencial de
Piper tuberculatum (HCT-116, MTT, DCFDA/ROS).

Cada sección (# %%) corresponde a una celda de notebook.
Para usar en Colab/Jupyter, copiar cada sección como celda
o convertir con jupytext.

Requisitos:
  pip install pubchempy rdkit chembl_webresource_client
  pip install gseapy mygene networkx matplotlib pandas requests
  pip install python-louvain   # opcional, para community detection
"""

# %% [markdown]
# # Pipeline In Silico — Piper tuberculatum vs HCT-116
# **Tesis de grado**
#
# Pipeline: GC-MS → identidad química → bioactividad (ChEMBL) →
# predicción de dianas (SwissTP) → relevancia CRC (OpenTargets) →
# pathway enrichment → PPI → integración → priorización

# %% ============================================================
# CELDA 0: Instalación de dependencias (solo Colab)
# ============================================================

# Descomentar en Google Colab:
# !pip install pubchempy rdkit-pypi chembl_webresource_client
# !pip install gseapy mygene networkx python-louvain

# %% ============================================================
# CELDA 1: Imports y configuración
# ============================================================

import os
import sys
import pandas as pd

# Si pipeline_lib.py está en el mismo directorio:
from pipeline_lib import (
    PipelineConfig,
    resolve_compounds,
    query_chembl_bioactivity,
    get_chembl_activity_details,
    load_swisstarget_predictions,
    filter_predictions,
    standardize_gene_symbols,
    query_crc_relevance,
    run_pathway_enrichment,
    build_ppi_network,
    get_ppi_metrics,
    build_integration_table,
    calculate_priority_score,
    create_phenotype_bridge,
    plot_enrichment_barplot,
    plot_ppi_network,
    plot_compound_target_heatmap,
)

# ---- Configuración central ----
cfg = PipelineConfig()

# Puedes modificar parámetros aquí:
# cfg.swisstarget_prob_low = 0.03
# cfg.enrichment_fdr_threshold = 0.1
# cfg.string_score_threshold = 700

os.makedirs(cfg.output_dir, exist_ok=True)

print("Pipeline configurado.")
print(f"  Compuestos: {len(cfg.compounds)}")
print(f"  SwissTP threshold: ≥{cfg.swisstarget_prob_low} "
      f"(alta confianza: ≥{cfg.swisstarget_prob_high})")
print(f"  Enrichment FDR: <{cfg.enrichment_fdr_threshold}")
print(f"  STRING score: ≥{cfg.string_score_threshold}")

# %% ============================================================
# CELDA 2: PASO 1 — Identidad química (PubChem)
# ============================================================
# Pregunta: ¿Cuáles son exactamente los compuestos mayoritarios
# del aceite esencial y cuál es su estructura?

print("\n" + "="*60)
print("PASO 1: Resolución de identidad química (PubChem)")
print("="*60)

df_identity = resolve_compounds(cfg.compounds, delay=cfg.api_delay)

print("\n--- Tabla de identidad ---")
print(df_identity[["query_name", "gcms_percent", "status", "cid",
                    "molecular_formula"]].to_string(index=False))

# NOTA para beta-cis-Ocimene:
ocimene = df_identity[df_identity["query_name"] == "beta-cis-Ocimene"]
if not ocimene.empty:
    print(f"\n⚠ VERIFICACIÓN REQUERIDA para beta-cis-Ocimene:")
    print(f"  CID usado: {ocimene.iloc[0]['cid']}")
    print(f"  SMILES: {ocimene.iloc[0]['smiles']}")
    print(f"  Nota: {ocimene.iloc[0]['note']}")
    print("  → Confirmar que este CID corresponde al isómero de tu GC-MS")
    print("    comparando con el índice de retención reportado.")

df_identity.to_csv(os.path.join(cfg.output_dir, "01_identity.csv"),
                    index=False)

# %% ============================================================
# CELDA 3: PASO 2 — Bioactividad experimental (ChEMBL)
# ============================================================
# Pregunta: ¿Alguno de estos compuestos tiene actividad biológica
# reportada experimentalmente contra dianas humanas?

print("\n" + "="*60)
print("PASO 2: Bioactividad experimental (ChEMBL)")
print("="*60)

# ChEMBL usa la API REST del EBI que puede estar temporalmente caída.
# Si falla, el pipeline continúa — ChEMBL aporta evidencia adicional
# pero no es bloqueante para el análisis de dianas.
_chembl_ok = True
try:
    df_chembl = query_chembl_bioactivity(df_identity,
                                         organism=cfg.chembl_organism,
                                         delay=cfg.api_delay)

    print("\n--- Resumen ChEMBL ---")
    print(df_chembl.to_string(index=False))

    # Tabla maestra: identidad + ChEMBL
    df_master = df_identity.merge(
        df_chembl, left_on="query_name", right_on="compound", how="left"
    )
    df_master = df_master.drop(columns=["compound"], errors="ignore")

    df_master.to_csv(os.path.join(cfg.output_dir, "02_master_identity.csv"),
                      index=False)
    print(f"\nTabla maestra guardada ({len(df_master)} compuestos).")

except Exception as e:
    _chembl_ok = False
    print(f"\n  ⚠ ChEMBL API no disponible: {type(e).__name__}: {e}")
    print("  Esto es un error del servidor de EBI, no de tu código.")
    print("  El pipeline continúa sin datos de ChEMBL.")
    print("  Reintenta más tarde o continúa — este paso es complementario.")
    df_chembl = pd.DataFrame(columns=["compound", "chembl_id",
                                       "chembl_status", "n_activities"])
    df_master = df_identity.copy()
    df_master.to_csv(os.path.join(cfg.output_dir, "02_master_identity.csv"),
                      index=False)

# %% ============================================================
# CELDA 4: PASO 2b — Detalle de actividades ChEMBL (por compuesto)
# ============================================================
# Pregunta: Para compuestos con actividad reportada,
# ¿contra qué dianas específicas actúan?

print("\n" + "="*60)
print("PASO 2b: Detalle de actividades por compuesto")
print("="*60)

chembl_details_all = []
df_chembl_detail = pd.DataFrame()

if _chembl_ok and not df_chembl.empty:
    for _, row in df_chembl.iterrows():
        if row.get("chembl_id") and row.get("n_activities", 0) > 0:
            print(f"\n  {row['compound']} ({row['chembl_id']}): "
                  f"{row['n_activities']} actividades")
            try:
                df_detail = get_chembl_activity_details(
                    row["chembl_id"], organism=cfg.chembl_organism)
                if not df_detail.empty:
                    df_detail["compound"] = row["compound"]
                    chembl_details_all.append(df_detail)

                    single = df_detail[df_detail["target_type"] == "SINGLE PROTEIN"]
                    if not single.empty:
                        print(f"    Proteínas individuales: {len(single)}")
                        print(single[["target_name", "standard_type",
                                      "standard_value", "standard_units"]]
                              .head(10).to_string(index=False))
            except Exception as e:
                print(f"    [ERROR detalle] {e}")

    if chembl_details_all:
        df_chembl_detail = pd.concat(chembl_details_all, ignore_index=True)
        df_chembl_detail.to_csv(
            os.path.join(cfg.output_dir, "02b_chembl_details.csv"), index=False)
        print(f"\nDetalle ChEMBL guardado ({len(df_chembl_detail)} registros).")
    else:
        print("\nSin actividades detalladas disponibles.")
else:
    print("  Omitido: ChEMBL no disponible o sin resultados.")

# %% ============================================================
# CELDA 5: PASO 3 — Predicción de dianas (SwissTargetPrediction)
# ============================================================
# Pregunta: ¿Qué dianas moleculares podrían ser afectadas
# por estos compuestos según similaridad estructural?
#
# NOTA: SwissTP no tiene API pública.  Los CSVs se descargan
# manualmente de http://www.swisstargetprediction.ch/
# ingresando el SMILES de cada compuesto.

print("\n" + "="*60)
print("PASO 3: Predicción de dianas (SwissTargetPrediction)")
print("="*60)

try:
    df_swisstp_raw = load_swisstarget_predictions(cfg.swisstarget_files)
    print(f"  Predicciones cargadas: {len(df_swisstp_raw)}")
    print(f"  Compuestos: {df_swisstp_raw['compound'].nunique()}")

    df_targets = filter_predictions(
        df_swisstp_raw,
        prob_threshold=cfg.swisstarget_prob_low,
        prob_high=cfg.swisstarget_prob_high,
        min_known_actives=cfg.min_known_actives,
    )

    print(f"\n  Predicciones retenidas (P ≥ {cfg.swisstarget_prob_low}): "
          f"{len(df_targets)}")
    print(f"  De alta confianza (P ≥ {cfg.swisstarget_prob_high}): "
          f"{(df_targets['confidence'] == 'HIGH').sum()}")
    print(f"  De baja referencia: "
          f"{(df_targets['confidence'] == 'LOW_REFERENCE').sum()}")

    print("\n  Por compuesto:")
    print(df_targets.groupby("compound").size().to_string())

    df_targets.to_csv(
        os.path.join(cfg.output_dir, "03_swisstarget_filtered.csv"),
        index=False)

except FileNotFoundError as e:
    print(f"  [ERROR] {e}")
    print("  Asegúrate de que los CSVs de SwissTP estén en el directorio.")
    print("  Archivos esperados:")
    for f in cfg.swisstarget_files:
        print(f"    {f}")
    df_targets = pd.DataFrame()

# %% ============================================================
# CELDA 6: PASO 4 — Estandarización de símbolos génicos
# ============================================================
# Pregunta: ¿Los identificadores de genes son correctos y
# corresponden a símbolos oficiales HGNC?

print("\n" + "="*60)
print("PASO 4: Estandarización de símbolos génicos")
print("="*60)

if not df_targets.empty:
    df_targets_std = standardize_gene_symbols(df_targets,
                                              delay=cfg.api_delay)

    # Análisis de convergencia
    convergence = (df_targets_std.groupby("gene_symbol")["compound"]
                   .nunique().reset_index()
                   .rename(columns={"compound": "n_compounds"})
                   .sort_values("n_compounds", ascending=False))

    multi = convergence[convergence["n_compounds"] > 1]
    print(f"\n  Total genes únicos: {len(convergence)}")
    print(f"  Genes en ≥2 compuestos: {len(multi)}")
    if not multi.empty:
        print("\n  Genes convergentes:")
        print(multi.to_string(index=False))

    df_targets_std.to_csv(
        os.path.join(cfg.output_dir, "04_targets_standardized.csv"),
        index=False)
    convergence.to_csv(
        os.path.join(cfg.output_dir, "04_gene_convergence.csv"),
        index=False)
else:
    df_targets_std = pd.DataFrame()
    convergence = pd.DataFrame()
    print("  Omitido: sin predicciones de targets.")

# %% ============================================================
# CELDA 7: PASO 5 — Relevancia en cáncer colorrectal (OpenTargets)
# ============================================================
# Pregunta: ¿Cuáles de las dianas predichas tienen evidencia
# de asociación con cáncer colorrectal en la literatura?

print("\n" + "="*60)
print("PASO 5: Relevancia CRC (OpenTargets)")
print("="*60)

if not df_targets_std.empty:
    gene_list = df_targets_std["gene_symbol"].unique().tolist()

    # Si existe un cache previo con errores, borrarlo para re-consultar
    cache_file = os.path.join(cfg.output_dir, "05_crc_relevance_cache.csv")
    if os.path.isfile(cache_file):
        try:
            _cache_check = pd.read_csv(cache_file)
            _n_errors = (_cache_check.get("crc_relevance", pd.Series()) == "ERROR").sum()
            if _n_errors > len(_cache_check) * 0.5:
                print(f"  Cache anterior tiene {_n_errors}/{len(_cache_check)} errores. "
                      "Borrando para re-consultar...")
                os.remove(cache_file)
        except Exception:
            pass

    df_crc = query_crc_relevance(
        gene_list,
        disease_ids=cfg.crc_disease_ids,
        delay=cfg.api_delay,
        cache_path=cache_file
    )

    print(f"\n  Distribución de relevancia CRC:")
    print(df_crc["crc_relevance"].value_counts().to_string())

    df_crc.to_csv(os.path.join(cfg.output_dir, "05_crc_relevance.csv"),
                   index=False)
else:
    df_crc = pd.DataFrame()
    print("  Omitido: sin genes para consultar.")

# %% ============================================================
# CELDA 8: PASO 6 — Enriquecimiento de vías
# ============================================================
# Pregunta: ¿Las dianas predichas convergen en vías biológicas
# relevantes para cáncer (apoptosis, ciclo celular, PI3K-AKT,
# Wnt, ROS, MAPK)?

print("\n" + "="*60)
print("PASO 6: Enriquecimiento de vías (ORA)")
print("="*60)

if not df_targets_std.empty:
    gene_list_enr = df_targets_std["gene_symbol"].unique().tolist()
    print(f"  Genes en input: {len(gene_list_enr)}")

    df_enrichment = run_pathway_enrichment(
        gene_list_enr,
        gene_sets=cfg.enrichment_gene_sets,
        fdr_threshold=cfg.enrichment_fdr_threshold,
        output_dir=cfg.output_dir,
    )

    if not df_enrichment.empty:
        print(f"\n  Total términos significativos: {len(df_enrichment)}")
        print(f"\n  Top 15 por combined score:")
        top = df_enrichment.nlargest(15, "combined_score")
        print(top[["gene_set_library", "term", "fdr",
                    "combined_score"]].to_string(index=False))

        # Buscar vías de interés para la tesis
        keywords_of_interest = [
            "apoptosis", "cell cycle", "pi3k", "akt", "wnt",
            "beta-catenin", "oxidative", "reactive oxygen", "ros",
            "mapk", "cancer", "colorectal", "proliferat",
            "death", "survival", "nf-kb", "p53", "tnf",
        ]
        mask = df_enrichment["term"].str.lower().apply(
            lambda t: any(k in t for k in keywords_of_interest))
        cancer_related = df_enrichment[mask]

        if not cancer_related.empty:
            print(f"\n  Vías potencialmente relevantes para cáncer/ROS "
                  f"({len(cancer_related)}):")
            print(cancer_related[["gene_set_library", "term", "fdr"]]
                  .to_string(index=False))
        else:
            print("\n  Ninguna vía directamente nombrada con keywords "
                  "de cáncer/ROS. Esto NO significa ausencia de "
                  "relevancia — revisar términos manualmente.")
else:
    df_enrichment = pd.DataFrame()
    print("  Omitido: sin genes para enrichment.")

# %% ============================================================
# CELDA 9: PASO 7 — Red PPI (STRING)
# ============================================================
# Pregunta: ¿Las dianas predichas forman módulos funcionales
# interconectados o son hits dispersos sin relación?

print("\n" + "="*60)
print("PASO 7: Red PPI (STRING)")
print("="*60)

if not df_targets_std.empty:
    gene_list_ppi = df_targets_std["gene_symbol"].unique().tolist()

    G_ppi, df_ppi = build_ppi_network(
        gene_list_ppi,
        species=cfg.string_species,
        score_threshold=cfg.string_score_threshold,
        output_dir=cfg.output_dir,
    )

    df_ppi_metrics = get_ppi_metrics(G_ppi)

    if not df_ppi_metrics.empty:
        print(f"\n  Top 10 hubs (por degree):")
        print(df_ppi_metrics.head(10)[
            ["gene_symbol", "ppi_degree", "ppi_betweenness", "ppi_cluster"]
        ].to_string(index=False))

        n_clusters = df_ppi_metrics["ppi_cluster"].nunique()
        print(f"\n  Clusters detectados: {n_clusters}")

        df_ppi_metrics.to_csv(
            os.path.join(cfg.output_dir, "07_ppi_metrics.csv"), index=False)
else:
    G_ppi = None
    df_ppi_metrics = pd.DataFrame()
    print("  Omitido: sin genes para PPI.")

# %% ============================================================
# CELDA 10: PASO 8 — Integración y priorización
# ============================================================
# Pregunta: ¿Qué dianas candidatas acumulan mayor evidencia
# desde múltiples perspectivas?

print("\n" + "="*60)
print("PASO 8: Integración y priorización")
print("="*60)

if not df_targets_std.empty:
    df_integrated = build_integration_table(
        df_targets_std, df_crc, df_enrichment, df_ppi_metrics)

    df_prioritized = calculate_priority_score(df_integrated)

    print(f"\n  Total genes candidatos: {len(df_prioritized)}")
    print(f"\n  Top 20 por priority score:")
    top_cols = ["priority_rank", "gene_symbol", "n_compounds",
                "prob_max", "target_class"]
    if "ot_crc_score" in df_prioritized.columns:
        top_cols += ["ot_crc_score", "crc_relevance"]
    top_cols += ["n_enriched_pathways", "ppi_degree", "priority_score"]
    top_cols = [c for c in top_cols if c in df_prioritized.columns]

    print(df_prioritized.head(20)[top_cols].to_string(index=False))

    # Score breakdown
    print("\n  Componentes del score de priorización:")
    print("    prob_max (SwissTP):        peso 0.25")
    print("    ot_crc_score (OpenTargets): peso 0.30")
    print("    n_enriched_pathways:       peso 0.20")
    print("    ppi_degree:                peso 0.15")
    print("    n_compounds (convergencia): peso 0.10")

    df_prioritized.to_csv(
        os.path.join(cfg.output_dir, "08_prioritized.csv"), index=False)

    # Bridge con fenotipos experimentales
    df_bridge = create_phenotype_bridge(df_prioritized)
    df_bridge.to_csv(
        os.path.join(cfg.output_dir, "08_phenotype_bridge.csv"), index=False)

    print("\n  Tabla de integración con fenotipos guardada.")
    print("  → Completar columnas 'mtt_ic50_ugml' y 'ros_fold_change' "
          "con tus datos experimentales.")
else:
    df_prioritized = pd.DataFrame()
    df_bridge = pd.DataFrame()
    print("  Omitido: sin datos para integrar.")

# %% ============================================================
# CELDA 11: PASO 9 — Visualizaciones
# ============================================================

print("\n" + "="*60)
print("PASO 9: Visualizaciones")
print("="*60)

# 9a: Enrichment barplot
if not df_enrichment.empty:
    plot_enrichment_barplot(df_enrichment, top_n=25,
                            output_dir=cfg.output_dir)

# 9b: PPI network
if G_ppi is not None and G_ppi.number_of_edges() > 0:
    plot_ppi_network(G_ppi, df_crc=df_crc, output_dir=cfg.output_dir)

# 9c: Compound-target heatmap
if not df_targets_std.empty:
    plot_compound_target_heatmap(df_targets_std, top_n=30,
                                 output_dir=cfg.output_dir)

# %% ============================================================
# CELDA 12: RESUMEN DEL PIPELINE
# ============================================================

print("\n" + "="*60)
print("RESUMEN DEL PIPELINE")
print("="*60)

print(f"""
Archivos generados en '{cfg.output_dir}/':
  01_identity.csv                  — Identidad química (PubChem)
  02_master_identity.csv           — Identidad + ChEMBL
  02b_chembl_details.csv           — Detalle de actividades ChEMBL
  03_swisstarget_filtered.csv      — Predicciones SwissTP filtradas
  04_targets_standardized.csv      — Genes estandarizados
  04_gene_convergence.csv          — Convergencia de genes
  05_crc_relevance.csv             — Relevancia CRC (OpenTargets)
  pathway_enrichment.csv           — Vías enriquecidas
  ppi_interactions.csv             — Interacciones PPI (STRING)
  07_ppi_metrics.csv               — Métricas de centralidad PPI
  08_prioritized.csv               — Tabla maestra priorizada
  08_phenotype_bridge.csv          — Bridge con fenotipos (completar)
  enrichment_barplot.png           — Figura: enrichment
  ppi_network.png                  — Figura: red PPI
  compound_target_heatmap.png      — Figura: heatmap compuestos×genes

Parámetros usados:
  SwissTP prob threshold:  ≥{cfg.swisstarget_prob_low}
  SwissTP prob high:       ≥{cfg.swisstarget_prob_high}
  Min known actives:       {cfg.min_known_actives}
  Enrichment FDR:          <{cfg.enrichment_fdr_threshold}
  STRING combined score:   ≥{cfg.string_score_threshold}
  CRC disease IDs:         {cfg.crc_disease_ids}
""")

# %% ============================================================
# CELDA 13: PRÓXIMOS PASOS
# ============================================================

print("""
═══════════════════════════════════════════════════════════════
PRÓXIMOS PASOS
═══════════════════════════════════════════════════════════════

1. COMPLETAR datos experimentales en 08_phenotype_bridge.csv:
   - mtt_ic50_ugml:    IC50 del aceite/extracto en HCT-116
   - ros_fold_change:  fold-change DCFDA vs control
   - phenotype_note:   observaciones relevantes

2. VERIFICAR identidad de beta-cis-Ocimene:
   - Comparar CID 5281553 con tu índice de retención GC-MS
   - Si no coincide, actualizar verified_cid en la configuración

3. REVISAR vías enriquecidas:
   - ¿Son consistentes con tus observaciones de citotoxicidad/ROS?
   - ¿Aparecen vías de apoptosis, estrés oxidativo, ciclo celular?

4. SELECCIONAR 2-3 targets para docking molecular:
   - Priorizar targets con:
     a) Alto priority_score
     b) Estructura cristalográfica en PDB (resolución ≤ 2.5 Å)
     c) Relevancia CRC STRONG o MODERATE
   - Preparar receptor + ligando para AutoDock Vina

5. REDACTAR la discusión conectando:
   compound → targets → pathways → phenotype experimental
""")
