#!/usr/bin/env python3
"""
E2WRS - Environmental Encroachment Early Warning & Response System
Prototype pipeline: two satellite dates -> NDVI change -> GIS boundary check
-> permit check -> alerts + case records in SQLite.

Usage
-----
  python e2wrs.py demo                      # synthetic data, runs end to end
  python e2wrs.py run \
      --wetland-name "Muko Wetland" --boundary boundary.geojson \
      --before-red B04_2025.tif --before-nir B08_2025.tif --before-date 2025-06-15 \
      --after-red  B04_2026.tif --after-nir  B08_2026.tif --after-date  2026-06-20 \
      [--permits permits.geojson]

Inputs must share one projected CRS (UTM, metres), e.g. Sentinel-2 L2A bands
B04 (red) and B08 (NIR) at 10 m, clipped to the same area. The boundary and
permit GeoJSON files must be in that CRS too (reproject first with ogr2ogr).

A detection is only ever a *potential* encroachment. It becomes a case for a
human officer to verify; the program never decides that anyone is guilty.
"""
import argparse
import json
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform
from shapely.geometry import Polygon, box, mapping, shape
from shapely.ops import unary_union

HERE = Path(__file__).parent
DB_PATH = HERE / "e2wrs.db"

# ---- tunable thresholds ----------------------------------------------------
NDVI_DROP = 0.25          # minimum fall in NDVI to count as change
MIN_PIXELS = 12           # smallest cluster kept (12 px at 10 m = 0.12 ha)
HIGH_PRIORITY_HA = 0.5    # unauthorized change at least this big = high priority
PERMIT_OVERLAP = 0.5      # share of the change inside a permit footprint


# ---- 1. preprocessing / change detection ----------------------------------
def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    red = red.astype("float32")
    nir = nir.astype("float32")
    denom = nir + red
    out = np.full(red.shape, np.nan, dtype="float32")
    ok = denom > 0
    out[ok] = (nir[ok] - red[ok]) / denom[ok]
    return out


def detect_change(ndvi_before, ndvi_after, transform, boundary,
                  drop=NDVI_DROP, min_pixels=MIN_PIXELS):
    """Return a list of dicts, one per changed patch inside the boundary."""
    delta = ndvi_after - ndvi_before                       # negative = loss
    inside = features.geometry_mask([mapping(boundary)], delta.shape,
                                    transform, invert=True)
    changed = (delta <= -drop) & inside & np.isfinite(delta)

    # sieve removes isolated specks (sensor noise, shadows)
    cleaned = features.sieve(changed.astype("uint8"), size=min_pixels)

    pixel_area_m2 = abs(transform.a * transform.e)
    patches = []
    for geom, val in features.shapes(cleaned, mask=cleaned.astype(bool),
                                     transform=transform):
        if val != 1:
            continue
        poly = shape(geom)
        m = features.geometry_mask([geom], delta.shape, transform, invert=True)
        n_pix = int(m.sum())
        if n_pix < min_pixels:
            continue
        mean_delta = float(np.nanmean(delta[m]))
        mean_after = float(np.nanmean(ndvi_after[m]))
        patches.append({
            "polygon": poly,
            "area_ha": n_pix * pixel_area_m2 / 10_000,
            "mean_delta_ndvi": mean_delta,
            "change_type": ("bare_ground_or_building" if mean_after < 0.20
                            else "vegetation_loss"),
        })
    return patches


def confidence(mean_delta: float, area_ha: float) -> float:
    drop_score = min(1.0, abs(mean_delta) / 0.5)
    size_score = min(1.0, area_ha / 0.5)
    return round(0.6 * drop_score + 0.4 * size_score, 2)


# ---- 2. encroachment engine (boundary + permit logic) ----------------------
def find_permit(poly: Polygon, permits: list, on: str):
    """Return the permit covering >= PERMIT_OVERLAP of the patch, or None."""
    for p in permits:
        if p["valid_from"] and on < p["valid_from"]:
            continue
        if p["valid_to"] and on > p["valid_to"]:
            continue
        share = poly.intersection(p["footprint"]).area / poly.area
        if share >= PERMIT_OVERLAP:
            return p
    return None


def prioritise(authorized: bool, area_ha: float) -> str:
    if authorized:
        return "low"
    return "high" if area_ha >= HIGH_PRIORITY_HA else "medium"


# ---- 3. alert and case management -----------------------------------------
def init_db(path=DB_PATH):
    con = sqlite3.connect(path)
    con.executescript((HERE / "schema.sql").read_text())
    return con


def log_event(con, alert_id, actor, from_s, to_s, note):
    con.execute("INSERT INTO case_events (alert_id, at, actor, from_status, "
                "to_status, note) VALUES (?,?,?,?,?,?)",
                (alert_id, date.today().isoformat(), actor, from_s, to_s, note))


def save_alerts(con, wetland_id, patches, permits, crs, d_before, d_after):
    created = 0
    for pt in patches:
        poly = pt["polygon"]
        permit = find_permit(poly, permits, d_after)
        authorized = permit is not None
        cx, cy = poly.centroid.x, poly.centroid.y
        lon, lat = warp_transform(crs, "EPSG:4326", [cx], [cy])
        cur = con.execute(
            """INSERT INTO alerts (wetland_id, detected_on, image_before,
               image_after, lon, lat, geometry_wkt, change_type, area_ha,
               mean_delta_ndvi, confidence, authorized, priority, status)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (wetland_id, date.today().isoformat(), d_before, d_after,
             lon[0], lat[0], poly.wkt, pt["change_type"],
             round(pt["area_ha"], 3), round(pt["mean_delta_ndvi"], 3),
             confidence(pt["mean_delta_ndvi"], pt["area_ha"]),
             int(authorized), prioritise(authorized, pt["area_ha"]),
             "closed" if authorized else "new"))
        aid = cur.lastrowid
        if authorized:
            log_event(con, aid, "system", None, "closed",
                      f"Matched permit #{permit['id']} ({permit['holder']})")
        else:
            log_event(con, aid, "system", None, "new",
                      "No authorization found. Needs field verification.")
            created += 1
    con.commit()
    return created


# Allowed moves in the case workflow
TRANSITIONS = {
    "new": {"assigned"},
    "assigned": {"field_verification"},
    "field_verification": {"verified", "not_verified"},
    "verified": {"action_taken"},
    "action_taken": {"closed"},
    "not_verified": {"closed"},
}


def advance_case(con, alert_id, to_status, actor, note=""):
    (cur_status,) = con.execute("SELECT status FROM alerts WHERE id=?",
                                (alert_id,)).fetchone()
    if to_status not in TRANSITIONS.get(cur_status, set()):
        raise ValueError(f"Cannot move {cur_status} -> {to_status}")
    con.execute("UPDATE alerts SET status=? WHERE id=?", (to_status, alert_id))
    if to_status == "assigned":
        con.execute("UPDATE alerts SET assigned_to=? WHERE id=?",
                    (actor, alert_id))
    if to_status in ("verified", "not_verified"):
        con.execute("UPDATE alerts SET verification_result=? WHERE id=?",
                    (to_status, alert_id))
    log_event(con, alert_id, actor, cur_status, to_status, note)
    con.commit()


# ---- run on real GeoTIFFs --------------------------------------------------
def load_geojson_geoms(path):
    data = json.loads(Path(path).read_text())
    feats = data["features"] if data["type"] == "FeatureCollection" else [data]
    return [(shape(f["geometry"]), f.get("properties", {})) for f in feats]


def run(args):
    def read(p):
        with rasterio.open(p) as src:
            return src.read(1), src.transform, src.crs
    b_red, tr, crs = read(args.before_red)
    b_nir, _, _ = read(args.before_nir)
    a_red, _, _ = read(args.after_red)
    a_nir, _, _ = read(args.after_nir)

    boundary = unary_union([g for g, _ in load_geojson_geoms(args.boundary)])
    permits = []
    if args.permits:
        for i, (g, props) in enumerate(load_geojson_geoms(args.permits), 1):
            permits.append({"id": props.get("id", i),
                            "holder": props.get("holder", "unknown"),
                            "valid_from": props.get("valid_from"),
                            "valid_to": props.get("valid_to"),
                            "footprint": g})

    patches = detect_change(ndvi(b_red, b_nir), ndvi(a_red, a_nir), tr, boundary)
    con = init_db()
    cur = con.execute("INSERT INTO wetlands (name, boundary_wkt, area_ha) "
                      "VALUES (?,?,?)",
                      (args.wetland_name, boundary.wkt, boundary.area / 10_000))
    n = save_alerts(con, cur.lastrowid, patches, permits, crs,
                    args.before_date, args.after_date)
    print(f"{len(patches)} changed patches, {n} new cases opened -> {DB_PATH}")


# ---- demo with synthetic data ---------------------------------------------
def demo():
    """Synthetic 4 km x 4 km wetland, 10 m pixels. No real data is used."""
    rng = np.random.default_rng(42)
    size, res = 400, 10.0
    x0, y0 = 833_000.0, 9_862_000.0
    tr = from_origin(x0, y0, res, res)
    crs = "EPSG:32735"

    def px_box(c0, r0, c1, r1):          # pixel box -> map-coords polygon
        return box(x0 + c0 * res, y0 - r1 * res, x0 + c1 * res, y0 - r0 * res)

    boundary = px_box(40, 40, 360, 360)

    base = np.clip(rng.normal(0.62, 0.05, (size, size)), 0.3, 0.9)
    before_ndvi, after_ndvi = base.copy(), base.copy()
    # three simulated disturbances inside the wetland
    after_ndvi[100:110, 90:100] = 0.08    # 100 px = 1.0 ha, building / bare ground
    after_ndvi[220:228, 250:260] = 0.30   # 80 px = 0.8 ha, clearing (permit below)
    after_ndvi[300:303, 120:123] = 0.05   # 9 px = 0.09 ha, too small, ignored
    after_ndvi[60:66, 300:306] = 0.15     # 36 px = 0.36 ha, bare ground
    after_ndvi += rng.normal(0, 0.02, after_ndvi.shape)   # sensor noise

    # convert NDVI back to fake red/nir so the same code path is exercised
    def to_bands(n):
        nir = np.full(n.shape, 3000.0)
        red = nir * (1 - n) / (1 + n)
        return red, nir
    permits = [{"id": 1, "holder": "Demo Farmers Co-op",
                "valid_from": "2025-01-01", "valid_to": "2027-12-31",
                "footprint": px_box(245, 215, 265, 235)}]

    nd_b = ndvi(*to_bands(before_ndvi))
    nd_a = ndvi(*to_bands(after_ndvi))
    patches = detect_change(nd_b, nd_a, tr, boundary)

    if DB_PATH.exists():
        DB_PATH.unlink()
    con = init_db()
    cur = con.execute("INSERT INTO wetlands (name, district, boundary_wkt, area_ha)"
                      " VALUES (?,?,?,?)", ("Demo Wetland (synthetic)", "Demo",
                      boundary.wkt, boundary.area / 10_000))
    n = save_alerts(con, cur.lastrowid, patches, permits, crs,
                    "2025-06-15", "2026-06-20")
    print(f"Detected {len(patches)} patches; {n} need field verification.\n")
    for row in con.execute("SELECT id, change_type, area_ha, confidence, "
                           "priority, status FROM alerts ORDER BY id"):
        print(row)
    # walk the first open case through the workflow
    first = con.execute("SELECT id FROM alerts WHERE status='new' "
                        "ORDER BY area_ha DESC").fetchone()[0]
    advance_case(con, first, "assigned", "Officer A")
    advance_case(con, first, "field_verification", "Officer A", "Site visit booked")
    advance_case(con, first, "verified", "Officer A", "Structure confirmed on site")
    print(f"\nCase {first} moved to 'verified'. Audit trail:")
    for ev in con.execute("SELECT at, actor, from_status, to_status, note "
                          "FROM case_events WHERE alert_id=?", (first,)):
        print("  ", ev)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="run on synthetic data")
    r = sub.add_parser("run", help="run on real GeoTIFFs")
    r.add_argument("--wetland-name", required=True)
    r.add_argument("--boundary", required=True, help="GeoJSON, same CRS as rasters")
    r.add_argument("--permits", help="GeoJSON with id, holder, valid_from, valid_to")
    for k in ("before", "after"):
        r.add_argument(f"--{k}-red", required=True)
        r.add_argument(f"--{k}-nir", required=True)
        r.add_argument(f"--{k}-date", required=True, help="YYYY-MM-DD")
    a = ap.parse_args()
    demo() if a.cmd == "demo" else run(a)
