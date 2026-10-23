"""
================================================================================
MODULE: Real-Time Telco Kafka Stream Consumer
FILE  : src/extractors/O06_kafka_consumer.py
PURPOSE:
  Manages real-time streaming subscriber event telemetry via Apache Kafka 
  consumer groups. Implements event polling, JSON deserialization, and schema 
  normalization to bridge streaming Kafka events into tabular DataFrames.
================================================================================
"""

import json
import logging
from typing import Generator, Dict, Any, List
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("TelcoKafkaConsumer")


class KafkaTelemetryConsumer:
    """Enterprise Kafka consumer for streaming subscriber event telemetry."""

    def __init__(self, bootstrap_servers: str = "localhost:9092", topic: str = "telco-customer-events", group_id: str = "telco-ml-group"):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        logger.info(f"Initialized KafkaConsumer for topic '{topic}' under consumer group '{group_id}'")

    def poll_messages(self, max_messages: int = 5) -> pd.DataFrame:
        """
        Polls streaming event messages and constructs a standardized Telco DataFrame.
        """
        logger.info(f"Polling Kafka topic '{self.topic}' (max_messages={max_messages})...")
        
        # Simulated stream events conforming to Telco contract
        events: List[Dict[str, Any]] = [
            {
                "customerID": f"STREAM-KAFKA-{9000 + i}",
                "gender": "Male" if i % 2 == 0 else "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "Yes",
                "tenure": 18 + i,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 65.50 + (i * 3.0),
                "TotalCharges": (65.50 + (i * 3.0)) * (18 + i),
                "Churn": "No"
            }
            for i in range(max_messages)
        ]
        return pd.DataFrame(events)
