"""
Document Tracker for Incremental Indexing

Tracks which documents have been embedded to avoid re-processing unchanged files.
Uses MD5 hashing to detect file changes.
Also tracks embedding mode to prevent dimension mismatches.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Set, Optional
from datetime import datetime


class DocumentTracker:
    """Tracks document hashes for incremental indexing"""
    
    def __init__(self, tracker_path: str = "./vectorstore/document_tracker.json"):
        self.tracker_path = Path(tracker_path)
        self.documents: Dict[str, dict] = {}
        self.embedding_mode: Optional[str] = None
        self._load()
    
    def _load(self):
        """Load tracker from disk"""
        if self.tracker_path.exists():
            try:
                with open(self.tracker_path, "r") as f:
                    data = json.load(f)
                    self.documents = data.get("documents", {})
                    self.embedding_mode = data.get("embedding_mode")
            except Exception as e:
                print(f"⚠️  Warning: Could not load document tracker: {e}")
                self.documents = {}
                self.embedding_mode = None
        else:
            self.documents = {}
            self.embedding_mode = None
    
    def save(self):
        """Save tracker to disk"""
        # Ensure directory exists
        self.tracker_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "documents": self.documents,
            "embedding_mode": self.embedding_mode,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(self.tracker_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def compute_hash(self, file_path: Path) -> str:
        """Compute MD5 hash of a file"""
        md5_hash = hashlib.md5()
        
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b""):
                md5_hash.update(chunk)
        
        return md5_hash.hexdigest()
    
    def check_embedding_mode_compatibility(self, current_mode: str) -> bool:
        """
        Check if current embedding mode is compatible with tracked mode.
        Returns False if we need to rebuild the index.
        """
        if self.embedding_mode is None:
            # No previous mode, compatible
            return True
        
        if self.embedding_mode == current_mode:
            # Same mode, compatible
            return True
        
        # Different modes = incompatible (different dimensions)
        return False
    
    def set_embedding_mode(self, mode: str):
        """Set the embedding mode"""
        self.embedding_mode = mode
    
    def reset(self):
        """Reset tracker (used when rebuilding index)"""
        self.documents = {}
        self.embedding_mode = None
    
    def is_changed(self, file_path: Path) -> bool:
        """
        Check if a file has changed since last indexing
        
        Returns:
            True if file is new or changed, False if unchanged
        """
        file_str = str(file_path.absolute())
        current_hash = self.compute_hash(file_path)
        
        if file_str not in self.documents:
            # New file
            return True
        
        previous_hash = self.documents[file_str].get("hash")
        return current_hash != previous_hash
    
    def update(self, file_path: Path, num_chunks: int = 0):
        """Update tracker with file information"""
        file_str = str(file_path.absolute())
        current_hash = self.compute_hash(file_path)
        
        self.documents[file_str] = {
            "hash": current_hash,
            "num_chunks": num_chunks,
            "last_indexed": datetime.now().isoformat(),
            "filename": file_path.name
        }
    
    def remove(self, file_path: Path):
        """Remove a file from tracking (e.g., if deleted)"""
        file_str = str(file_path.absolute())
        if file_str in self.documents:
            del self.documents[file_str]
    
    def get_all_tracked_files(self) -> Set[str]:
        """Get set of all tracked file paths"""
        return set(self.documents.keys())
    
    def get_stats(self) -> dict:
        """Get statistics about tracked documents"""
        total_chunks = sum(doc.get("num_chunks", 0) for doc in self.documents.values())
        return {
            "total_files": len(self.documents),
            "total_chunks": total_chunks,
        }
    
    def clean_missing_files(self, existing_files: Set[Path]):
        """Remove entries for files that no longer exist"""
        existing_str = {str(f.absolute()) for f in existing_files}
        tracked = set(self.documents.keys())
        
        missing = tracked - existing_str
        for file_str in missing:
            del self.documents[file_str]
        
        return len(missing)
    
    def __repr__(self):
        stats = self.get_stats()
        mode_str = f", mode={self.embedding_mode}" if self.embedding_mode else ""
        return f"DocumentTracker({stats['total_files']} files, {stats['total_chunks']} chunks{mode_str})"

