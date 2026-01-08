"""
BigQuery metrics module.
"""
import logging
import os
import time
from typing import List, Dict, Any
from google.cloud import bigquery
from app.config import get_settings

logger = logging.getLogger(__name__)

_bq_client = None
_dataset_ref = None
_table_ref = None
_initialized = False

def init_bigquery():
    """Initialize BigQuery client"""
    global _bq_client, _dataset_ref, _table_ref, _initialized
    
    settings = get_settings()
    if not settings.ENABLE_BQ_METRICS:
        return False
        
    try:
        _bq_client = bigquery.Client(project=settings.GCP_PROJECT, location=settings.GCP_LOCATION)
        _dataset_ref = _bq_client.dataset(settings.BQ_DATASET)
        _table_ref = _dataset_ref.table(settings.BQ_TABLE)
        
        # Verify dataset exists (optional, could just let insert fail if missing)
        try:
            _bq_client.get_dataset(_dataset_ref)
        except Exception:
            logger.warning(f"BigQuery dataset {settings.BQ_DATASET} not found or not accessible")
            
        _initialized = True
        logger.info(f"BigQuery metrics initialized. Target: {settings.BQ_DATASET}.{settings.BQ_TABLE}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize BigQuery client: {e}")
        return False

def insert_metrics(metrics: List[Dict[str, Any]]):
    """
    Insert metrics into BigQuery.
    
    Expected schema:
    - timestamp: TIMESTAMP
    - metric_name: STRING
    - value: FLOAT
    - metric_type: STRING
    - tags: RECORD REPEATED (key: STRING, value: STRING)
    - service: STRING
    - env: STRING
    """
    if not _initialized or not _bq_client:
        return

    settings = get_settings()
    rows_to_insert = []
    current_time = time.time()
    
    for m in metrics:
        # Transform tags ["key:value"] -> [{"key": "key", "value": "value"}]
        bq_tags = []
        for tag in m.get('tags', []):
            if ':' in tag:
                k, v = tag.split(':', 1)
                bq_tags.append({'key': k, 'value': v})
            else:
                bq_tags.append({'key': tag, 'value': ''})
                
        row = {
            "timestamp": current_time,
            "metric_name": m['metric'],
            "value": float(m['value']),
            "metric_type": m.get('type', 'gauge'),
            "tags": bq_tags,
            "service": settings.DD_SERVICE,
            "env": settings.DD_ENV
        }
        rows_to_insert.append(row)
        
    if not rows_to_insert:
        return
        
    try:
        errors = _bq_client.insert_rows_json(_table_ref, rows_to_insert)
        if errors:
            logger.error(f"BigQuery insert errors: {errors}")
        else:
            logger.debug(f"Inserted {len(rows_to_insert)} metrics to BigQuery")
    except Exception as e:
        logger.error(f"Failed to insert metrics to BigQuery: {e}")
