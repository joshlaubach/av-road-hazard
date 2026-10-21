from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_analyze_index_query_records_flow(frame_bytes: bytes) -> None:
    rec = client.post("/analyze", files={"file": ("f.png", frame_bytes, "image/png")}).json()
    assert rec["objects"] and rec["defects"]

    got = client.get(f"/records/{rec['frame_id']}")
    assert got.status_code == 200 and got.json() == rec

    idx = client.post("/index", files=[("files", ("f.png", frame_bytes, "image/png"))])
    assert idx.json()["indexed"] == [rec["frame_id"]]

    q = client.post("/query", json={"text": "pothole near pedestrian", "top_k": 3})
    assert q.status_code == 200
    assert q.json()["records"][0]["frame_id"] == rec["frame_id"]


def test_rejects_non_image_and_bad_content_type(frame_bytes: bytes) -> None:
    assert (
        client.post("/analyze", files={"file": ("f.txt", b"hi", "text/plain")}).status_code == 415
    )
    assert (
        client.post("/analyze", files={"file": ("f.png", b"nope", "image/png")}).status_code == 422
    )
    assert client.get("/records/missing").status_code == 404
    assert client.post("/query", json={"top_k": 2}).status_code == 422
