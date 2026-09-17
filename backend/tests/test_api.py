from datetime import datetime, timedelta
from pathlib import Path


TEST_DB = Path(__file__).parent / "test_parking.db"
if TEST_DB.exists():
    TEST_DB.unlink()

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import func, select  # noqa: E402

from app.database import SessionLocal  # noqa: E402
from app.database.init_db import initialize_database  # noqa: E402
from app.main import app  # noqa: E402
from app.models import ParkingSession, ParkingSlot, Payment, Setting  # noqa: E402
from app.services.parking import calculate_fee  # noqa: E402


def test_phase_one_workflow():
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}

        slots = client.get("/api/slots").json()
        assert len(slots) == 20
        assert all(slot["status"] == "EMPTY" for slot in slots)

        settings = client.get("/api/settings").json()
        assert settings["price_per_hour"] == "10000.00"
        assert settings["minimum_hours"] == 1

        preflight = client.options(
            "/api/slots",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == "http://localhost:5173"

        slot_a2 = next(slot for slot in slots if slot["code"] == "A2")
        disabled = client.patch(f"/api/slots/{slot_a2['id']}", json={"is_active": False})
        assert disabled.status_code == 200
        assert disabled.json()["status"] == "DISABLED"
        reactivated = client.patch(f"/api/slots/{slot_a2['id']}", json={"is_active": True})
        assert reactivated.status_code == 200
        assert reactivated.json()["status"] == "EMPTY"

        slot_a1 = next(slot for slot in slots if slot["code"] == "A1")
        check_in = client.post(
            "/api/parking/check-in",
            json={"plate_number": "29-H1 999.99", "slot_id": slot_a1["id"]},
        )
        assert check_in.status_code == 201
        assert check_in.json()["plate_number"] == "29H199999"

        refreshed_a1 = next(slot for slot in client.get("/api/slots").json() if slot["code"] == "A1")
        assert refreshed_a1["status"] == "OCCUPIED"
        assert refreshed_a1["current_parking"]["plate_number"] == "29H199999"
        cannot_disable = client.patch(f"/api/slots/{slot_a1['id']}", json={"is_active": False})
        assert cannot_disable.status_code == 409

        duplicate = client.post(
            "/api/parking/check-in",
            json={"plate_number": "29H199999", "slot_id": next(s for s in slots if s["code"] == "A2")["id"]},
        )
        assert duplicate.status_code == 409

        occupied = client.post(
            "/api/parking/check-in",
            json={"plate_number": "30A12345", "slot_id": slot_a1["id"]},
        )
        assert occupied.status_code == 409

        before = client.get("/api/slots").json()
        preview = client.post(
            "/api/parking/checkout-preview", json={"plate_number": "29H199999"}
        )
        assert preview.status_code == 200
        assert preview.json()["billed_hours"] == 1
        assert preview.json()["amount"] == "10000.00"
        after = client.get("/api/slots").json()
        assert before == after

        checkout = client.post(
            "/api/parking/checkout",
            json={"plate_number": "29H199999", "payment_method": "CASH"},
        )
        assert checkout.status_code == 200
        assert checkout.json()["payment_status"] == "PAID"
        assert next(s for s in client.get("/api/slots").json() if s["code"] == "A1")["status"] == "EMPTY"

        transfer_in = client.post(
            "/api/parking/check-in",
            json={"plate_number": "51-F1 222.33", "slot_id": slot_a1["id"]},
        )
        assert transfer_in.status_code == 201
        transfer_out = client.post(
            "/api/parking/checkout",
            json={"plate_number": "51F122233", "payment_method": "TRANSFER"},
        )
        assert transfer_out.status_code == 200
        assert transfer_out.json()["payment_method"] == "TRANSFER"

        invalid_method = client.post(
            "/api/parking/checkout",
            json={"plate_number": "51F122233", "payment_method": "CARD"},
        )
        assert invalid_method.status_code == 400

        dashboard = client.get("/api/dashboard").json()
        assert dashboard["total_slots"] == 20
        assert dashboard["occupied_slots"] == 0
        assert dashboard["empty_slots"] == 20
        assert dashboard["vehicles_today"] == 2
        assert dashboard["revenue_today"] == "20000.00"

        history = client.get("/api/history", params={"plate_number": "29-H1 999.99"}).json()
        assert len(history) == 1
        assert history[0]["payment_method"] == "CASH"

    with SessionLocal() as db:
        initialize_database()
        assert db.scalar(select(func.count(Setting.id))) == 1
        assert db.scalar(select(func.count(Payment.id))) == 2
        assert db.scalar(
            select(func.count(ParkingSession.id)).where(ParkingSession.status == "COMPLETED")
        ) == 2


def test_fee_rounding_examples():
    with SessionLocal() as db:
        settings = db.scalar(select(Setting).limit(1))
        now = datetime.utcnow()
        cases = [(15, 1), (59, 1), (65, 2), (140, 3)]
        for minutes, expected in cases:
            fee = calculate_fee(now - timedelta(minutes=minutes), settings, now)
            assert fee.billed_hours == expected


def teardown_module():
    from app.database import engine

    engine.dispose()
    if TEST_DB.exists():
        TEST_DB.unlink()
