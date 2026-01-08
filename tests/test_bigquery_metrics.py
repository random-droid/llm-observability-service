"""
Tests for BigQuery metrics module.
"""
import unittest
from unittest.mock import patch, MagicMock
from app import bigquery_metrics
from app.config import Settings

class TestBigQueryMetrics(unittest.TestCase):
    
    def setUp(self):
        # Reset module state
        bigquery_metrics._bq_client = None
        bigquery_metrics._initialized = False
        
    @patch('app.bigquery_metrics.bigquery.Client')
    @patch('app.bigquery_metrics.get_settings')
    def test_init_bigquery_success(self, mock_settings, mock_bq_client):
        """Test successful initialization"""
        settings = MagicMock()
        settings.ENABLE_BQ_METRICS = True
        settings.GCP_PROJECT = "test-project"
        settings.GCP_LOCATION = "us-central1"
        settings.BQ_DATASET = "test_dataset"
        settings.BQ_TABLE = "test_table"
        mock_settings.return_value = settings
        
        # Setup mock client
        client_instance = MagicMock()
        mock_bq_client.return_value = client_instance
        
        result = bigquery_metrics.init_bigquery()
        
        self.assertTrue(result)
        self.assertTrue(bigquery_metrics._initialized)
        mock_bq_client.assert_called_with(project="test-project", location="us-central1")
        client_instance.dataset.assert_called_with("test_dataset")
        
    @patch('app.bigquery_metrics.get_settings')
    def test_init_bigquery_disabled(self, mock_settings):
        """Test initialization when disabled"""
        settings = MagicMock()
        settings.ENABLE_BQ_METRICS = False
        mock_settings.return_value = settings
        
        result = bigquery_metrics.init_bigquery()
        
        self.assertFalse(result)
        self.assertFalse(bigquery_metrics._initialized)

    @patch('app.bigquery_metrics.get_settings')
    def test_insert_metrics(self, mock_settings):
        """Test inserting metrics"""
        settings = MagicMock()
        settings.DD_SERVICE = "test-service"
        settings.DD_ENV = "test-env"
        mock_settings.return_value = settings
        
        # Manually set initialized state
        bigquery_metrics._initialized = True
        mock_client = MagicMock()
        bigquery_metrics._bq_client = mock_client
        bigquery_metrics._table_ref = "test_table_ref"
        
        metrics = [
            {
                'metric': 'test.metric',
                'value': 1.0,
                'type': 'gauge',
                'tags': ['repo:test', 'keyless']
            }
        ]
        
        bigquery_metrics.insert_metrics(metrics)
        
        mock_client.insert_rows_json.assert_called_once()
        call_args = mock_client.insert_rows_json.call_args
        table_ref, rows = call_args[0]
        
        self.assertEqual(table_ref, "test_table_ref")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['metric_name'], 'test.metric')
        self.assertEqual(rows[0]['value'], 1.0)
        # Check tag transformation
        expected_tags = [
            {'key': 'repo', 'value': 'test'},
            {'key': 'keyless', 'value': ''}
        ]
        self.assertEqual(rows[0]['tags'], expected_tags)
        self.assertEqual(rows[0]['service'], 'test-service')

    def test_insert_metrics_not_initialized(self):
        """Test insert when not initialized should do nothing"""
        bigquery_metrics._initialized = False
        bigquery_metrics._bq_client = MagicMock()
        
        bigquery_metrics.insert_metrics([{'metric': 'foo', 'value': 1}])
        
        bigquery_metrics._bq_client.insert_rows_json.assert_not_called()

if __name__ == '__main__':
    unittest.main()
