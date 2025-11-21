"""
Configuration Manager for Fleet Insights RAG

Centralized configuration management that:
- Loads environment variables and config file
- Provides embedding factory method
- Manages environment-based model selection
- Validates configuration
"""

import os
from pathlib import Path
from typing import Literal, Optional
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

EmbeddingMode = Literal["auto", "openai", "openai-legacy", "local"]
Environment = Literal["development", "production"]


class ConfigManager:
    """Centralized configuration manager"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config = self._load_config()
        self.embedding_mode = self._determine_embedding_mode()
        
    def _load_config(self) -> dict:
        """Load configuration from YAML file"""
        if not Path(self.config_path).exists():
            # Return defaults if config doesn't exist
            return self._get_default_config()
        
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Merge with defaults for any missing keys
        defaults = self._get_default_config()
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
        
        return config
    
    def _get_default_config(self) -> dict:
        """Default configuration values"""
        return {
            "chunk_size": 800,
            "chunk_overlap": 120,
            "top_k": 4,
            "index_path": "./vectorstore/faiss_index",
            "embedding": {
                "mode": "auto",
                "batch_size": 100,
            },
            "incremental": {
                "enabled": True,
                "tracker_path": "./vectorstore/document_tracker.json",
            }
        }
    
    def _determine_embedding_mode(self) -> str:
        """Determine which embedding mode to use based on environment"""
        # Check environment variable first
        env_mode = os.getenv("EMBEDDING_MODE", "").lower()
        
        if env_mode in ["openai", "openai-legacy", "local"]:
            return env_mode
        
        # Auto mode: determine based on ENVIRONMENT
        if env_mode == "auto" or not env_mode:
            environment = os.getenv("ENVIRONMENT", "development").lower()
            
            if environment == "production":
                # Production: use OpenAI if API key available
                if os.getenv("OPENAI_API_KEY"):
                    return "openai"
                else:
                    print("⚠️  Warning: No OPENAI_API_KEY found, falling back to local embeddings")
                    return "local"
            else:
                # Development: use local embeddings (free, fast)
                return "local"
        
        # Fallback to config file
        config_mode = self._config.get("embedding", {}).get("mode", "auto")
        if config_mode in ["openai", "openai-legacy", "local"]:
            return config_mode
        
        # Final fallback
        return "local"
    
    def get_embeddings(self):
        """
        Factory method to get the appropriate embeddings instance
        based on current configuration
        """
        mode = self.embedding_mode
        
        if mode == "openai":
            from langchain_openai import OpenAIEmbeddings
            print(f"🔢 Using OpenAI embeddings (text-embedding-3-small)")
            return OpenAIEmbeddings(model="text-embedding-3-small")
        
        elif mode == "openai-legacy":
            from langchain_openai import OpenAIEmbeddings
            print(f"🔢 Using OpenAI embeddings (text-embedding-ada-002)")
            return OpenAIEmbeddings(model="text-embedding-ada-002")
        
        elif mode == "local":
            from langchain_huggingface import HuggingFaceEmbeddings
            print(f"🔢 Using local embeddings (all-MiniLM-L6-v2)")
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        
        else:
            raise ValueError(f"Invalid embedding mode: {mode}")
    
    def get(self, key: str, default=None):
        """Get configuration value by key"""
        return self._config.get(key, default)
    
    def is_incremental_enabled(self) -> bool:
        """Check if incremental indexing is enabled"""
        return self._config.get("incremental", {}).get("enabled", True)
    
    def get_tracker_path(self) -> str:
        """Get path to document tracker file"""
        return self._config.get("incremental", {}).get("tracker_path", 
                                                        "./vectorstore/document_tracker.json")
    
    def get_index_path(self) -> str:
        """Get path to FAISS index"""
        return self._config.get("index_path", "./vectorstore/faiss_index")
    
    def print_config_summary(self):
        """Print current configuration for debugging"""
        print("=" * 60)
        print("⚙️  Configuration Summary")
        print("=" * 60)
        print(f"Embedding mode:         {self.embedding_mode}")
        print(f"Environment:            {os.getenv('ENVIRONMENT', 'development')}")
        print(f"Incremental indexing:   {self.is_incremental_enabled()}")
        print(f"Chunk size:             {self.get('chunk_size')}")
        print(f"Chunk overlap:          {self.get('chunk_overlap')}")
        print(f"Top-k retrieval:        {self.get('top_k')}")
        print(f"Index path:             {self.get_index_path()}")
        if self.is_incremental_enabled():
            print(f"Tracker path:           {self.get_tracker_path()}")
        print("=" * 60)
        print()


# Global instance for easy access
_global_config: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get or create global configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = ConfigManager()
    return _global_config


def get_embeddings():
    """Convenience function to get embeddings instance"""
    return get_config().get_embeddings()


# Backwards compatibility
def load_config(path: str = "config.yaml") -> dict:
    """Legacy function for backwards compatibility"""
    config = ConfigManager(path)
    return config._config
