#!/usr/bin/env python3
"""Exercise the real codoxear.file_upload.remove_session_uploads symlink branch
inside the container against a symlink entry whose target is OUTSIDE uploads.
Proves the link is unlinked itself and the outside target survives.
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from codoxear.file_upload import remove_session_uploads

HOME = Path(os.environ["HOME"])
root = HOME / ".local/share/codoxear/uploads"
root.mkdir(parents=True, exist_ok=True)

# Outside target.
outside = HOME / "symlink-target-outside.txt"
outside.write_bytes(b"precious outside bytes - must survive unlink of the link")

# Symlink entry inside uploads masquerading as a session upload dir.
sid = "cert-symlinksess"
link = root / sid
if link.exists() or link.is_symlink():
    link.unlink()
link.symlink_to(outside)

before = {
    "link_exists": link.is_symlink(),
    "link_target": os.readlink(str(link)) if link.is_symlink() else None,
    "target_exists": outside.exists(),
    "target_size": outside.stat().st_size if outside.exists() else None,
    "root_entries": sorted(p.name for p in root.iterdir()),
}

removed = remove_session_uploads(root, sid)

after = {
    "link_exists": link.exists() or link.is_symlink(),
    "target_exists": outside.exists(),
    "target_size": outside.stat().st_size if outside.exists() else None,
    "target_content": outside.read_text() if outside.exists() else None,
    "root_entries": sorted(p.name for p in root.iterdir()),
}

print(json.dumps({"sid": sid, "removed_returned": removed, "before": before, "after": after}, indent=2))
# Assertions for exit code semantics.
assert removed is True, "remove_session_uploads should return True (entry existed)"
assert not (link.exists() or link.is_symlink()), "symlink entry must be unlinked"
assert outside.exists() and outside.read_text() == "precious outside bytes - must survive unlink of the link", \
    "outside target must survive (link was NOT followed)"
print("SYMLINK_BRANCH_PASS")
