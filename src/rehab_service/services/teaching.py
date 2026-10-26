"""Versioned public teaching assets, with review information per exact version."""

import copy
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

from rehab_service.storage.database import require, now
from rehab_service.catalog import ACTIONS
from rehab_service.settings import MEDIA_DIR


class TeachingService:
    def __init__(self, store, root=None):
        self.store = store
        self.root = Path(root) if root else MEDIA_DIR
        self.items = []
        for original in ACTIONS:
            item = copy.deepcopy(original)
            files = {}
            stamp = 0
            for kind, extension in (("video", "mp4"), ("poster", "jpg"), ("subtitle", "srt")):
                path = self.root / (item["key"] + "." + extension)
                if path.is_file():
                    files[kind] = {"url": "/media/" + path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                   "bytes": path.stat().st_size}
                    stamp = max(stamp, path.stat().st_mtime)
            encoded = json.dumps({"content": original, "files": files}, sort_keys=True, ensure_ascii=False)
            item.update({"content_version": hashlib.sha256(encoded.encode()).hexdigest()[:16], "files": files,
                         "updated_at": datetime.fromtimestamp(stamp, timezone.utc).isoformat() if stamp else None})
            self.items.append(item)

    def catalog(self):
        items = copy.deepcopy(self.items)
        with self.store.db() as db:
            for item in items:
                review = db.execute("SELECT * FROM teaching_reviews WHERE action=? AND version=?",
                                    (item["key"], item["content_version"])).fetchone()
                item["review"] = dict(review) if review else {"state": "pending", "reviewer": None, "reviewed_at": None, "note": ""}
        return items

    def review(self, key, payload, author):
        item = next((i for i in self.items if i["key"] == key), None)
        require(item is not None, "动作不存在", 404)
        require(payload.get("content_version") == item["content_version"], "教学内容已更新，请刷新后审核", 409)
        state = payload.get("state")
        require(state in ("pending", "approved", "rejected"), "审核状态错误")
        reviewer, note = payload.get("reviewer", ""), payload.get("note", "")
        require(isinstance(reviewer, str) and len(reviewer.strip()) <= 60 and (state == "pending" or reviewer.strip()), "请填写审核人（最多 60 字）")
        require(isinstance(note, str) and len(note) <= 500, "审核备注最多 500 字")
        with self.store.db() as db:
            db.execute("INSERT OR REPLACE INTO teaching_reviews VALUES (?,?,?,?,?,?,?)",
                       (key, item["content_version"], state, reviewer.strip() or None,
                        None if state == "pending" else now(), note, author))
        return next(i for i in self.catalog() if i["key"] == key)
