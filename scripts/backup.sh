#!/bin/bash
# =============================================================================
# MindFu Backup Script
# =============================================================================
set -e

echo "💾 Starting MindFu backup..."

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Load environment variables
if [ -f .env ]; then
    source .env
fi

# Configuration
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_DIR/backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="$BACKUP_DIR/$TIMESTAMP"

POSTGRES_USER="${POSTGRES_USER:-mindfu}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-mindfu_secret}"
POSTGRES_DB="${POSTGRES_DB:-mindfu}"

echo "Backup directory: $BACKUP_PATH"
mkdir -p "$BACKUP_PATH"

# Backup PostgreSQL
echo "📊 Backing up PostgreSQL..."
docker compose exec -T postgres pg_dump \
    -U "$POSTGRES_USER" \
    -d "$POSTGRES_DB" \
    --format=custom \
    > "$BACKUP_PATH/postgres.dump"

if [ -f "$BACKUP_PATH/postgres.dump" ]; then
    echo "✅ PostgreSQL backup complete"
else
    echo "❌ PostgreSQL backup failed"
fi

# Backup Qdrant snapshots
echo "🔍 Creating Qdrant snapshot..."
COLLECTIONS=$(curl -s http://localhost:6333/collections | jq -r '.result.collections[].name')

for collection in $COLLECTIONS; do
    echo "  - Snapshotting collection: $collection"
    curl -s -X POST "http://localhost:6333/collections/$collection/snapshots" > /dev/null
done

# Copy Qdrant snapshots
echo "📁 Copying Qdrant data..."
docker compose cp qdrant:/qdrant/snapshots "$BACKUP_PATH/qdrant-snapshots" 2>/dev/null || true

# Backup conversations
echo "💬 Backing up conversations..."
if [ -d "data/conversations" ]; then
    cp -r data/conversations "$BACKUP_PATH/"
    echo "✅ Conversations backup complete"
fi

# Backup documents
echo "📄 Backing up documents..."
if [ -d "data/documents" ]; then
    cp -r data/documents "$BACKUP_PATH/"
    echo "✅ Documents backup complete"
fi

# Create backup manifest
cat > "$BACKUP_PATH/manifest.json" << EOF
{
    "timestamp": "$TIMESTAMP",
    "created_at": "$(date -Iseconds)",
    "components": {
        "postgres": true,
        "qdrant": true,
        "conversations": $([ -d "$BACKUP_PATH/conversations" ] && echo true || echo false),
        "documents": $([ -d "$BACKUP_PATH/documents" ] && echo true || echo false)
    }
}
EOF

# Compress backup
echo "🗜️  Compressing backup..."
cd "$BACKUP_DIR"
tar -czf "$TIMESTAMP.tar.gz" "$TIMESTAMP"
rm -rf "$TIMESTAMP"

BACKUP_SIZE=$(du -h "$TIMESTAMP.tar.gz" | cut -f1)

echo ""
echo "✅ Backup complete!"
echo "   Location: $BACKUP_DIR/$TIMESTAMP.tar.gz"
echo "   Size: $BACKUP_SIZE"
echo ""
echo "To restore, run: ./scripts/restore.sh $TIMESTAMP.tar.gz"

# Cleanup old backups (keep last 7)
echo ""
echo "🧹 Cleaning up old backups..."
ls -t "$BACKUP_DIR"/*.tar.gz 2>/dev/null | tail -n +8 | xargs -r rm -f
echo "✅ Cleanup complete"
