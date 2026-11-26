# ReelDesk

ReelDesk is a local workbench for cataloguing transparent film scans, checking repeats and keeping readings with their original scan coordinates. The practice roll is synthesized on first launch. Its scenes, dust, carrier holes, coverage gaps and scanner controls are simulated.

## Start

Use Docker Compose with Python 3.11 in the image.

```sh
docker compose up --build
```

Open [the local desk](http://127.0.0.1:18541/). The Compose service binds to loopback and stores originals, marks and reviews in its named catalog volume. Ctrl+C stops the foreground service. Restarting or recreating the container with that volume retains the catalog. `REELDESK_BIND_PORT` changes the host port. `REELDESK_DEMO=0` starts a new catalog without the practice roll.

For native development, create an environment with Python 3.11, install `requirements.txt`, and use `./start.sh`. Native mode opens port 8080 by default and uses the local `data` directory. Set `REELDESK_PORT`, `REELDESK_DATA` and `REELDESK_PYTHON` to select a port, data folder and interpreter.

## Use the desk

- Open a roll and import each original in its source format. Use the Control image roles for a backlight flat and a shutter-dark exposure made at the same dimensions as the scan. Original files are kept byte for byte.
- Draw a region, line or point on the main image. Select a frame from the index to load its saved rectangle. Add a mark to record a frame, a reference, a protected detail or a proposed cut.
- **Inspect** shows the negative, positive, calibrated density or clipping overlay. Use a clear base patch to measure film-base transmission. Measure optical density by region, source point or line. Use the scanner checks for illumination, coverage, dust, grain, slanted-edge focus and certified wedge patches. Retouch is opt-in and writes a separate region preview.
- **Repeat** aligns a second scan from the same roll. Draw a textured primary patch to search for a translation or use matched landmarks to set a shift, similarity or affine transform. Read paired density at a clicked point, inspect differences over a region, or double-check a selected dust finding against the repeat.
- **Strip** checks perforation pitch and carrier edges, suggests frame boxes and checks physical cut clearance with a recorded DPI. Set a planned cut from a clear position, draw the observed cut and verify its offset and reserve. Native frame masters retain oriented source pixels and include a frame-coordinate manifest.
- **Archive** makes an annotated proof, a printable contact sheet or a 16-bit optical density master. Save a working result to store its exact settings and evidence. Set a review decision or download its packet. The packet verifier checks the file hashes, reading CSV and catalogue source matches.

Saved records and original image downloads survive container restart. Working results are held in memory for up to 20 minutes. Current catalog revisions prevent saving a result against changed marks or source metadata. Deletion is blocked when a saved record references that scan.

## Input and measurement limits

The desk accepts single-image PNG, TIFF, JPEG and WebP originals up to 36 MiB and 12 million pixels. Measurements use upright source pixel coordinates after interpreting orientation. Rectangle extents are half-open. Integer 16-bit grayscale input preserves measurement precision. Multi-channel formats are measured through Pillow's decoded RGB representation, which may be eight bits per channel. Image processing assumes scanner code values proportional to light for calibrated density. The local software does not measure an unknown scanner transfer function automatically.

Flat and dark controls use the same raster coordinates as the original. Transparent pixels are unavailable for measurement. The coverage map identifies low and high clipping when those samples are excluded. Registration uses local texture or user-supplied landmarks. Strong repetition or severe curvature can require additional landmarks. The focus check uses a straight, slightly tilted light-to-dark edge and reports an approximate local MTF. For meaningful physical distances, enter verified scanner DPI. Review suggested cuts against the real film and carrier.

The 16-bit density ZIP stores three unsigned 16-bit PNG channels, three 8-bit quality-flag PNGs and a JSON scale manifest. A code of zero is unavailable. Codes 1 through 65535 use the finite range specified by the manifest. Each archive is a locally generated evidence package. The ZIP checksums test integrity relative to its manifest.

## Source layout

`backend/server.py` serves the catalog and workbench operations. `backend/catalog.py` manages SQLite originals, marks and persisted reviews. The imaging modules use NumPy, Pillow and SciPy. `static/` contains the workbench interface, served directly by Flask. The Python-only source count can be reproduced with `python3 scripts/count_backend.py --project-root . --source-root backend`.
