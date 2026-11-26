"""End-to-end API checks for cut planning and pouch evaluation.

Run with:  python3 -I scripts/test_dissector_api.py
Uses a temporary catalog seeded with a synthetic strip scan plus Grip marks.
"""

import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/opt/pointwise-env/site-packages")

import numpy as np
from PIL import Image

from backend.server import create_app
from backend.values import image_bytes


DPI = 120.0
WIDTH, HEIGHT = 1180, 200


def strip_image():
    array = np.full((HEIGHT, WIDTH, 3), 0.35, dtype=np.float32)
    array[:, :] = (0.75, 0.6, 0.45)
    for x0 in (40, 340, 640, 940):
        array[60:140, x0:x0 + 220] = (0.2, 0.22, 0.25)
    image = Image.fromarray(np.clip(array * 255, 0, 255).astype(np.uint8), mode="RGB")
    return image_bytes(image, dpi=(DPI, DPI))


def check(name, condition, detail=""):
    print(f"[{'PASS' if condition else 'FAIL'}] {name} {detail}")
    if not condition:
        raise SystemExit(1)


def run(client):
    roll = client.post("/api/rolls", json={
        "title": "Grip strip", "code": "GRIP-1", "stock": "35 mm",
        "location": "Bench", "notes": "", "tags": [],
    }).get_json()
    data = io.BytesIO(strip_image())
    scan = client.post(
        f"/api/rolls/{roll['id']}/scans",
        data={"metadata": '{"role":"scan","medium":"negative","dpi":120}',
              "file": (data, "strip.png")},
        content_type="multipart/form-data",
    ).get_json()
    scan_id = scan["id"]

    def mark(kind, label, geometry):
        return client.post(f"/api/scans/{scan_id}/marks",
                           json={"kind": kind, "label": label, "geometry": geometry, "notes": ""}).get_json()

    for index, x0 in enumerate((40, 340, 640, 940), start=1):
        mark("frame", f"Frame {index}", {"x": x0, "y": 60, "w": 220, "h": 80})
    grip_geoms = [
        (8, 10, 70, 18), (1102, 10, 70, 18), (266, 10, 26, 18), (308, 10, 26, 18),
        (566, 10, 26, 18), (608, 10, 26, 18), (866, 10, 26, 18), (908, 10, 26, 18),
    ]
    grips = [mark("grip", f"Grip {i}", {"x": x, "y": y, "w": w, "h": h})
             for i, (x, y, w, h) in enumerate(grip_geoms, start=1)]

    plan_payload = {
        "kind": "cut-plan",
        "cut": {"guide": {"x": 0, "y": 0, "w": WIDTH, "h": HEIGHT},
                "kerf_mm": 0.16, "reserve_mm": 0.0, "dpi": DPI, "retained": "both"},
        "plan": {
            "min_grip_mm": 2.0, "reach_mm": 80.0,
            "bag_min_mm": 0.0, "bag_max_mm": 400.0, "bags": [],
            "start_grips": [grips[0]["id"], grips[1]["id"]],
            "preferred_frame": 2,
        },
    }
    response = client.post(f"/api/scans/{scan_id}/analysis", json=plan_payload)
    check("plan endpoint responds", response.status_code == 201, str(response.data[:200]))
    job = response.get_json()
    result = job["result"]
    check("three cuts planned", result["cut_count"] == 3, f"moves={result['clamp_moves']}")
    check("sequence has grip steps", len(result["steps"]) == 3)
    check("leader sacrifice reported", result["leader"] is not None)
    check("preview image attached", any(f["name"].endswith(".png") for f in job["files"]))

    observed = []
    for cut in result["cuts"]:
        column = cut["column"]
        observed.append({
            "boundary": cut["boundary"],
            "planned": cut["line"],
            "actual": {"start": {"x": column + 1, "y": 0}, "end": {"x": column + 1, "y": HEIGHT - 1}},
        })
    pouch_payload = {
        "kind": "pouch-check",
        "cut": plan_payload["cut"],
        "tolerance": 0.5,
        "pouch": {"bag_min_mm": 0, "bag_max_mm": 400, "bags": []},
        "observed": observed,
    }
    response = client.post(f"/api/scans/{scan_id}/analysis", json=pouch_payload)
    check("pouch endpoint responds", response.status_code == 201, str(response.data[:200]))
    pouch = response.get_json()["result"]
    interior = [row for row in pouch["frames"] if row["frame"] in (2, 3)]
    check("interior cells confirmed", all(row["status"] == "confirmed" for row in interior),
          str([(r["frame"], r["status"]) for r in pouch["frames"]]))
    check("outer cells still pending",
          all(row["status"] == "pending" for row in pouch["frames"] if row["frame"] in (1, 4)))

    narrow = dict(pouch_payload)
    narrow["pouch"] = {"bag_min_mm": 0, "bag_max_mm": interior[0]["min_width_mm"] - 0.5, "bags": []}
    held = client.post(f"/api/scans/{scan_id}/analysis", json=narrow).get_json()["result"]
    check("over-width cell held", held["status"] == "hold" and interior[0]["frame"] in held["held_frames"])

    bad = dict(plan_payload)
    bad["plan"] = {**plan_payload["plan"], "start_grips": [grips[0]["id"], grips[0]["id"]]}
    response = client.post(f"/api/scans/{scan_id}/analysis", json=bad)
    check("duplicate start grips rejected", response.status_code == 422, str(response.get_json()))

    print("all API checks passed")


def main():
    with tempfile.TemporaryDirectory() as storage:
        app = create_app(storage=Path(storage), seed_demo=False)
        run(app.test_client())


if __name__ == "__main__":
    main()
