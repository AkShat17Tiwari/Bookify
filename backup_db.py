"""
BOOKIFY — Database Backup Utility
====================================
Automated SQLite backup with:
  • Timestamped copies
  • Retention policy (configurable days)
  • Safe online backup using sqlite3 backup API

Usage:
    python backup_db.py                  # Run once
    crontab: 0 3 * * * cd /path/to/app && python backup_db.py   # Daily at 3 AM

SECURITY: Regular backups protect against data loss from
accidental deletion, corruption, or ransomware attacks.
"""

import os
import sqlite3
import shutil
from datetime import datetime, timedelta

# Configuration
DB_PATH      = os.path.join(os.path.dirname(__file__), 'users.db')
BACKUP_DIR   = os.path.join(os.path.dirname(__file__), 'backups')
RETENTION_DAYS = 7  # Keep backups for 7 days


def create_backup():
    """
    Create a timestamped backup of the SQLite database.
    Uses sqlite3's online backup API for a consistent copy,
    even if the database is being written to.
    """
    os.makedirs(BACKUP_DIR, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(BACKUP_DIR, f'users_backup_{timestamp}.db')

    # SECURITY: Use sqlite3 backup API for a consistent, safe copy
    src = sqlite3.connect(DB_PATH)
    dst = sqlite3.connect(backup_path)
    src.backup(dst)
    dst.close()
    src.close()

    size_kb = os.path.getsize(backup_path) / 1024
    print(f"✅ Backup created: {backup_path} ({size_kb:.1f} KB)")
    return backup_path


def cleanup_old_backups():
    """
    Remove backups older than RETENTION_DAYS.
    SECURITY: Prevents disk exhaustion from accumulated backups.
    """
    if not os.path.exists(BACKUP_DIR):
        return

    cutoff = datetime.now() - timedelta(days=RETENTION_DAYS)
    removed = 0

    for fname in os.listdir(BACKUP_DIR):
        if not fname.startswith('users_backup_') or not fname.endswith('.db'):
            continue
        fpath = os.path.join(BACKUP_DIR, fname)
        # Parse timestamp from filename
        try:
            ts_str = fname.replace('users_backup_', '').replace('.db', '')
            file_time = datetime.strptime(ts_str, '%Y%m%d_%H%M%S')
            if file_time < cutoff:
                os.remove(fpath)
                removed += 1
        except (ValueError, OSError):
            continue

    if removed:
        print(f"🗑️  Removed {removed} old backup(s) (>{RETENTION_DAYS} days)")


if __name__ == '__main__':
    print("━━━ BOOKIFY Database Backup ━━━")
    if not os.path.exists(DB_PATH):
        print("❌ Database not found:", DB_PATH)
    else:
        create_backup()
        cleanup_old_backups()
        print("━━━ Done ━━━")
