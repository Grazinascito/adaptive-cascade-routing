"""BenchmarkHarness — orquestra os 3 cenários e gera relatórios e gráficos de métricas.

Métricas coletadas: latência p50/p95/p99, tokens decodificados, acurácia e APGR.
Componente G do design doc (docs/adaptive-cascade-design-doc.md §5.7);
stories correspondentes em docs/system2-evaluator-spec.md (Stories 2 e 3).
"""
