import sqlite3
import os

DB_PATH = "database/image_enhancer.db"

def get_connection():
    # Create database folder if not exists
    os.makedirs("database", exist_ok=True)
    return sqlite3.connect(DB_PATH)

def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path TEXT,
        quality TEXT,
        caption TEXT,
        suggestions TEXT,
        enhanced_image_path TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    conn.commit()
    conn.close()
    print("Tables created successfully!")

def insert_log(image_path, quality, caption, suggestions, enhanced_image_path):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO logs (image_path, quality, caption, suggestions, enhanced_image_path)
    VALUES (?, ?, ?, ?, ?)
    """, (image_path, quality, caption, suggestions, enhanced_image_path))

    conn.commit()
    conn.close()

def fetch_logs():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM logs")
    rows = cursor.fetchall()

    conn.close()
    return rows
if __name__ == "__main__":
    create_tables()