# Stage 2: Telco Data Ingestion & Low-Level Hygiene

## Overview
Welcome to the production-grade **Stage 2: Data Ingestion & Hygiene** layer configured specifically for the **IBM Telco Customer Churn Dataset**. This module establishes an enterprise ingestion framework that handles:
1. **Cryptographic Checksum & File Integrity:** Validates SHA-256 signatures and executes zero-byte corruption checks prior to reading payloads.
2. **Multi-Source Extractions:** Ingests subscriber records across flat batch files (CSV, Parquet, JSON), relational databases (SQL pagination), rate-limited REST APIs, and Kafka event streams.
3. **Data Contract Enforcement:** Enforces YAML-defined schema rules across all 21 Telco features and automatically routes corrupted or out-of-bounds records into a Dead-Letter Queue (Quarantine).
4. **Low-Level Memory Optimization:** Downcasts high-precision numerical types (int64 -> int8/int16, float64 -> float32) to minimize RAM footprint.

---

## Directory Structure

```text
Day1/O2_Telco_Data_Ingestion/
├── README.md                                  # Ingestion module documentation
├── config/
│   └── O01_telco_schema_contract.yaml        # Schema contract for all 21 Telco attributes
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv          # Landing zone (Bronze layer)
│   └── quarantine/                           # Dead-letter queue for invalid records
└── src/
    ├── hygiene/
    │   ├── O02_checksum_validator.py         # SHA-256 hash & zero-byte check
    │   └── O07_schema_enforcer.py            # Contract validation & quarantine DLQ
    ├── extractors/
    │   ├── O03_file_batch_extractor.py       # Batch parsing (CSV, Parquet, JSON)
    │   ├── O04_db_extractor.py               # SQL chunking & connection pooling
    │   ├── O05_api_scraper.py                # REST API scraper with retry backoff
    │   └── O06_kafka_consumer.py             # Real-time event streaming consumer
    ├── optimization/
    │   └── O08_memory_optimizer.py           # Numerical downcasting & RAM hygiene
    ├── O09_orchestrator.py                   # Master batch pipeline runner
    └── O09_orchestrator_all.py               # Master multi-source pipeline runner
```

---

## Quickstart Execution

### Run Single-Source File Batch Pipeline:
```powershell
python Day1/O2_Telco_Data_Ingestion/src/O09_orchestrator.py
```

### Run Multi-Source (File + DB + API + Kafka) Pipeline:
```powershell
python Day1/O2_Telco_Data_Ingestion/src/O09_orchestrator_all.py
```
