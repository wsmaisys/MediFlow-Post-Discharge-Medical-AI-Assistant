"""
Tests for public FastAPI routes that do not require LLM startup.
"""

import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app as mediflow_app


def test_api_patients_returns_sanitized_json():
    client = TestClient(mediflow_app.app)

    response = client.get("/api/patients")

    assert response.status_code == 200
    payload = response.json()
    assert "patients" in payload
    assert payload["count"] == len(payload["patients"])
    assert payload["patients"]
    first_patient = payload["patients"][0]
    assert "patient_name" in first_patient
    assert "discharge_date" in first_patient
    assert "medications" not in first_patient
    assert "warning_signs" not in first_patient


def test_patients_page_serves_html():
    client = TestClient(mediflow_app.app)

    response = client.get("/patients")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Demo Patients" in response.text
