# OPS

2026-07-07T15:30:00Z Initialized copy/export too-large messaging slice. Prediction: bounded frontend handling is sufficient because the server already returns discriminating evidence (`status=413`, `max_bytes`) for oversized transcript export; the risk is misclassifying unrelated 413/upload errors or weakening the export guard.
