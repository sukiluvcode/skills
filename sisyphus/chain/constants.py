RECORD_LOCATION = 'record'
RECORD_NAME = 'extract_record.sqlite'
FAILED = 'failed'

# Filesystem layout for SQLite databases built by the indexing step and read
# by the chain. Kept here (rather than in sisyphus.index) so helper_functions
# can reference it without dragging in the heavy index loaders.
DEFAULT_DB_DIR = 'db'