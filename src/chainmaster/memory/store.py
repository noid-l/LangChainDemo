"""SQLite 持久化会话存储。

将 InMemoryChatMessageHistory 升级为 SQLite 后端，
支持跨会话恢复和 FTS5 全文检索。

展示了 LangChain ChatMessageHistory 的自定义实现模式。
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from ..logging_utils import get_logger

logger = get_logger(__name__)

_DEFAULT_DB_DIR = ".cache"
_DEFAULT_DB_NAME = "chat_history.db"

_CREATE_MESSAGES_TABLE = """
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}'
)
"""

_CREATE_FTS_INDEX = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
USING fts5(content, content='messages', content_rowid='rowid', tokenize='trigram')
"""

_CREATE_FTS_TRIGGER_INSERT = """
CREATE TRIGGER IF NOT EXISTS messages_fts_insert
AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
END
"""

_CREATE_FTS_TRIGGER_DELETE = """
CREATE TRIGGER IF NOT EXISTS messages_fts_delete
AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content)
        VALUES ('delete', old.rowid, old.content);
END
"""

_CREATE_FTS_TRIGGER_UPDATE = """
CREATE TRIGGER IF NOT EXISTS messages_fts_update
AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content)
        VALUES ('delete', old.rowid, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
END
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_messages_session
ON messages(session_id, created_at)
"""


def _resolve_db_path(project_root: Path, db_path: str | None = None) -> Path:
    if db_path:
        return Path(db_path)
    return project_root / _DEFAULT_DB_DIR / _DEFAULT_DB_NAME


class ChatHistoryStore(BaseChatMessageHistory):
    """SQLite 持久化会话存储。

    实现了 LangChain 的 BaseChatMessageHistory 接口，
    可直接替换 InMemoryChatMessageHistory。

    用法::

        store = ChatHistoryStore(session_id="user-1", db_path=Path(".cache/chat.db"))
        store.add_user_message("你好")
        store.add_ai_message("你好！有什么可以帮你的？")
        print(store.messages)  # 从 SQLite 读取
    """

    def __init__(
        self,
        session_id: str = "default",
        db_path: str | Path | None = None,
        project_root: Path | None = None,
    ) -> None:
        self.session_id = session_id
        if db_path:
            self._db_path = Path(db_path)
        elif project_root:
            self._db_path = _resolve_db_path(project_root)
        else:
            self._db_path = _resolve_db_path(Path.cwd())

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_MESSAGES_TABLE)
            conn.execute(_CREATE_INDEX)
            self._ensure_fts(conn)
            conn.commit()

    def _ensure_fts(self, conn: sqlite3.Connection) -> None:
        """初始化 FTS5 全文检索索引。"""
        try:
            conn.execute(_CREATE_FTS_INDEX)
            conn.execute(_CREATE_FTS_TRIGGER_INSERT)
            conn.execute(_CREATE_FTS_TRIGGER_DELETE)
            conn.execute(_CREATE_FTS_TRIGGER_UPDATE)
            conn.commit()
            logger.debug("FTS5 索引已就绪。")
        except sqlite3.OperationalError as e:
            logger.warning("FTS5 索引创建失败（SQLite 可能未编译 FTS5）: %s", e)

    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, metadata FROM messages "
                "WHERE session_id = ? ORDER BY created_at",
                (self.session_id,),
            ).fetchall()

        result: list[BaseMessage] = []
        for row in rows:
            msg = _row_to_message(row["role"], row["content"])
            if msg is not None:
                result.append(msg)
        return result

    @messages.setter
    def messages(self, value: list[BaseMessage]) -> None:
        raise AttributeError("ChatHistoryStore.messages 不支持直接赋值，请使用 add_message()")

    def add_message(self, message: BaseMessage) -> None:
        role = _message_type_to_role(message)
        content = message.content if isinstance(message.content, str) else json.dumps(message.content, ensure_ascii=False)
        msg_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (id, session_id, role, content, created_at, metadata) "
                "VALUES (?, ?, ?, ?, ?, '{}')",
                (msg_id, self.session_id, role, content, now),
            )
            conn.commit()

    def add_user_message(self, message: str) -> None:
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.add_message(AIMessage(content=message))

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (self.session_id,))
            conn.commit()
        logger.info("会话 [%s] 历史已清除。", self.session_id)

    def remove_older_than(self, keep_last: int) -> int:
        """删除旧消息，仅保留最近 N 条。返回删除数量。"""
        with self._connect() as conn:
            result = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()
            total = result[0] if result else 0

            if total <= keep_last:
                return 0

            deleted = conn.execute(
                "DELETE FROM messages WHERE session_id = ? AND id NOT IN ("
                "  SELECT id FROM messages WHERE session_id = ? ORDER BY created_at DESC LIMIT ?"
                ")",
                (self.session_id, self.session_id, keep_last),
            ).rowcount
            conn.commit()
            logger.info("会话 [%s] 清理旧消息: 删除 %d 条，保留 %d 条", self.session_id, deleted, keep_last)
            return deleted

    def prepend_message(self, message: BaseMessage) -> None:
        """在会话开头插入一条消息（用于注入摘要）。"""
        role = _message_type_to_role(message)
        content = message.content if isinstance(message.content, str) else json.dumps(message.content, ensure_ascii=False)
        msg_id = str(uuid.uuid4())
        early_time = "1970-01-01T00:00:00+00:00"

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (id, session_id, role, content, created_at, metadata) "
                "VALUES (?, ?, ?, ?, ?, '{}')",
                (msg_id, self.session_id, role, content, early_time),
            )
            conn.commit()

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """FTS5 全文检索历史消息。

        trigram 分词器要求查询 >= 3 字符，短查询自动回退到 LIKE。
        """
        with self._connect() as conn:
            if len(query) >= 3:
                try:
                    rows = conn.execute(
                        "SELECT m.role, m.content, m.created_at, "
                        "snippet(messages_fts, -1, '«', '»', '...', 32) AS snippet "
                        "FROM messages_fts fts "
                        "JOIN messages m ON m.rowid = fts.rowid "
                        "WHERE m.session_id = ? AND messages_fts MATCH ? "
                        "ORDER BY rank LIMIT ?",
                        (self.session_id, query, limit),
                    ).fetchall()
                    if rows:
                        return self._rows_to_search_results(rows)
                except sqlite3.OperationalError:
                    pass

            rows = conn.execute(
                "SELECT role, content, created_at, content AS snippet "
                "FROM messages WHERE session_id = ? AND content LIKE ? "
                "ORDER BY created_at DESC LIMIT ?",
                (self.session_id, f"%{query}%", limit),
            ).fetchall()
            return self._rows_to_search_results(rows)

    def _rows_to_search_results(self, rows) -> list[dict]:
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
                "snippet": row["snippet"],
            }
            for row in rows
        ]

    def message_count(self) -> int:
        with self._connect() as conn:
            result = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()
            return result[0] if result else 0

    def total_chars(self) -> int:
        """估算总字符数（作为 Token 的粗略近似）。"""
        with self._connect() as conn:
            result = conn.execute(
                "SELECT COALESCE(SUM(LENGTH(content)), 0) FROM messages WHERE session_id = ?",
                (self.session_id,),
            ).fetchone()
            return result[0] if result else 0


def _message_type_to_role(message: BaseMessage) -> str:
    if isinstance(message, HumanMessage):
        return "human"
    if isinstance(message, AIMessage):
        return "ai"
    if isinstance(message, SystemMessage):
        return "system"
    return "unknown"


def _row_to_message(role: str, content: str) -> BaseMessage | None:
    if role == "human":
        return HumanMessage(content=content)
    if role == "ai":
        return AIMessage(content=content)
    if role == "system":
        return SystemMessage(content=content)
    return None


class StoreManager:
    """管理多个 ChatHistoryStore 实例。

    替代 agent.py 中的 `_sessions: dict[str, InMemoryChatMessageHistory]`。
    """

    def __init__(self, db_path: str | Path | None = None, project_root: Path | None = None) -> None:
        self._db_path = db_path
        self._project_root = project_root
        self._stores: dict[str, ChatHistoryStore] = {}

    def get(self, session_id: str = "default") -> ChatHistoryStore:
        if session_id not in self._stores:
            self._stores[session_id] = ChatHistoryStore(
                session_id=session_id,
                db_path=self._db_path,
                project_root=self._project_root,
            )
        return self._stores[session_id]

    def list_sessions(self) -> list[str]:
        if self._db_path:
            db = Path(self._db_path)
        elif self._project_root:
            db = _resolve_db_path(self._project_root)
        else:
            db = _resolve_db_path(Path.cwd())

        if not db.is_file():
            return list(self._stores.keys())

        with sqlite3.connect(str(db)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT DISTINCT session_id FROM messages").fetchall()
        db_sessions = [row["session_id"] for row in rows]

        merged = set(db_sessions) | set(self._stores.keys())
        return sorted(merged)

    def clear_session(self, session_id: str) -> bool:
        if session_id in self._stores:
            self._stores[session_id].clear()
            return True
        return False

    def search_all(self, query: str, limit: int = 20) -> list[dict]:
        """跨所有会话搜索。长查询用 FTS5 trigram，短查询回退 LIKE。"""
        if self._db_path:
            db = Path(self._db_path)
        elif self._project_root:
            db = _resolve_db_path(self._project_root)
        else:
            db = _resolve_db_path(Path.cwd())

        if not db.is_file():
            return []

        with sqlite3.connect(str(db)) as conn:
            conn.row_factory = sqlite3.Row
            if len(query) >= 3:
                try:
                    rows = conn.execute(
                        "SELECT m.session_id, m.role, m.content, m.created_at "
                        "FROM messages_fts fts "
                        "JOIN messages m ON m.rowid = fts.rowid "
                        "WHERE messages_fts MATCH ? "
                        "ORDER BY rank LIMIT ?",
                        (query, limit),
                    ).fetchall()
                    if rows:
                        return self._format_search_results(rows)
                except sqlite3.OperationalError:
                    pass

            rows = conn.execute(
                "SELECT session_id, role, content, created_at "
                "FROM messages WHERE content LIKE ? "
                "ORDER BY created_at DESC LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()
            return self._format_search_results(rows)

    def _format_search_results(self, rows) -> list[dict]:
        return [
            {
                "session_id": row["session_id"],
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
