import aiosqlite
from src.config import settings
from src.models import Post

SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL
);
"""


async def init_db(db_path: str | None = None) -> None:
    """Initialize SQLite database schema."""
    target_db = db_path or settings.database_url
    async with aiosqlite.connect(target_db) as db:
        await db.execute(SCHEMA)
        await db.commit()


async def save_post(post: Post, db_path: str | None = None) -> None:
    """Persist a single Post record into SQLite (upsert pattern)."""
    target_db = db_path or settings.database_url
    await init_db(target_db)
    
    async with aiosqlite.connect(target_db) as db:
        await db.execute(
            """
            INSERT INTO posts (id, user_id, title, body)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                user_id = excluded.user_id,
                title = excluded.title,
                body = excluded.body
            """,
            (post.id, post.user_id, post.title, post.body),
        )
        await db.commit()


async def get_post(post_id: int, db_path: str | None = None) -> Post | None:
    """Retrieve a Post record from SQLite by ID."""
    target_db = db_path or settings.database_url
    async with aiosqlite.connect(target_db) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT id, user_id, title, body FROM posts WHERE id = ?", (post_id,)) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return Post(
                id=row["id"],
                user_id=row["user_id"],
                title=row["title"],
                body=row["body"],
            )
