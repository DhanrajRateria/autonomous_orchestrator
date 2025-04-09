import os
import gc
import logging
import asyncio
import time
import resource
import psutil
import numpy as np
import weakref
from typing import Dict, Set, Optional, Any, Callable
from dataclasses import dataclass
import threading
import tracemalloc
from functools import wraps

@dataclass
class MemoryStats:
    """Memory usage statistics"""
    current_bytes: int = 0
    peak_bytes: int = 0
    system_total: int = 0
    system_available: int = 0
    process_rss: int = 0
    last_collection_time: float = 0.0
    collection_count: int = 0

class MemoryManager:
    """
    Memory management optimizations for the security orchestrator.
    Provides monitoring, limiting and optimization of memory usage.
    """
    
    def __init__(self, memory_limit_mb: int = 8192,
                 threshold_pct: float = 0.85,
                 monitoring_interval_sec: int = 60,
                 enable_profiling: bool = False):
        """
        Initialize the memory manager.
        
        Args:
            memory_limit_mb: Memory limit in MB
            threshold_pct: Memory threshold percentage for cleanup
            monitoring_interval_sec: Interval for memory monitoring
            enable_profiling: Whether to enable memory profiling
        """
        self.logger = logging.getLogger("system.memory_manager")
        self.memory_limit = memory_limit_mb * 1024 * 1024  # Convert to bytes
        self.threshold = threshold_pct
        self.monitoring_interval = monitoring_interval_sec
        self.enable_profiling = enable_profiling
        
        # Memory stats
        self.stats = MemoryStats(
            system_total=psutil.virtual_memory().total,
            system_available=psutil.virtual_memory().available
        )
        
        # Object tracking
        self._large_objects = {}  # weak references to large objects
        self._object_sizes = {}   # size estimates for tracked objects
        
        # Tracemalloc for memory profiling
        if self.enable_profiling:
            tracemalloc.start()
        
        # Set resource limits in OS
        self._set_resource_limit()
        
        # Start monitoring task
        self.monitoring_task = None
        self.is_monitoring = False
        
        self.logger.info(f"Memory manager initialized with {memory_limit_mb}MB limit "
                       f"({threshold_pct*100}% threshold)")
    
    def _set_resource_limit(self):
        """Set resource limits using the OS facilities"""
        try:
            # Set soft and hard limits
            resource.setrlimit(resource.RLIMIT_AS, (self.memory_limit, self.memory_limit))
            self.logger.info(f"Set memory resource limit to {self.memory_limit/1024/1024:.2f}MB")
        except (ValueError, resource.error) as e:
            self.logger.warning(f"Could not set memory limit: {e}")
    
    def track_object(self, obj: Any, name: str = None) -> None:
        """
        Track a large object for potential cleanup.
        
        Args:
            obj: Object to track
            name: Optional name for the object
        """
        if obj is None:
            return
        
        # Get object size
        size = self._get_object_size(obj)
        
        # Only track relatively large objects
        if size > 1024 * 1024:  # 1MB
            obj_id = id(obj)
            self._large_objects[obj_id] = weakref.ref(obj)
            self._object_sizes[obj_id] = size
            
            obj_name = name or obj.__class__.__name__
            self.logger.debug(f"Tracking large object {obj_name} ({size/1024/1024:.2f}MB)")
    
    def _get_object_size(self, obj: Any) -> int:
        """
        Estimate memory size of an object.
        
        Args:
            obj: Object to measure
            
        Returns:
            Estimated size in bytes
        """
        if isinstance(obj, (np.ndarray, list, tuple, dict, set, str, bytes)):
            try:
                return obj.__sizeof__()
            except:
                pass
        
        # Fall back to rough estimate
        import sys
        return sys.getsizeof(obj)
    
    def limit_memory_usage(self, decorator_limit_mb: Optional[int] = None):
        """
        Decorator to limit memory usage of a function.
        
        Args:
            decorator_limit_mb: Optional memory limit specific to the function
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Get current memory usage
                process = psutil.Process(os.getpid())
                start_memory = process.memory_info().rss
                
                # Execute the function
                result = await func(*args, **kwargs)
                
                # Check memory usage after execution
                end_memory = process.memory_info().rss
                memory_used = end_memory - start_memory
                
                # Log memory usage if significant
                if memory_used > 10 * 1024 * 1024:  # More than 10MB
                    self.logger.info(
                        f"Function {func.__name__} used {memory_used/1024/1024:.2f}MB of memory"
                    )
                    
                    # Force garbage collection if memory usage is high
                    limit = decorator_limit_mb * 1024 * 1024 if decorator_limit_mb else self.memory_limit
                    if end_memory > limit * self.threshold:
                        self.force_garbage_collection()
                
                return result
            
            return wrapper
        
        return decorator
    
    def force_garbage_collection(self) -> int:
        """
        Force garbage collection to free memory.
        
        Returns:
            Number of objects collected
        """
        start_time = time.time()
        
        # Disable garbage collector during manual collection
        gc_enabled = gc.isenabled()
        if gc_enabled:
            gc.disable()
        
        # Force a full garbage collection
        collected = gc.collect(2)  # Full collection
        
        # Re-enable garbage collector if it was enabled
        if gc_enabled:
            gc.enable()
        
        self.stats.collection_count += 1
        self.stats.last_collection_time = time.time()
        collection_time = time.time() - start_time
        
        self.logger.info(f"Garbage collection freed {collected} objects in {collection_time:.2f}s")
        return collected
    
    def memory_profile(self, top_n: int = 10) -> Dict[str, Any]:
        """
        Get memory profiling information.
        
        Args:
            top_n: Number of top memory consumers to return
            
        Returns:
            Dictionary with memory profiling information
        """
        if not self.enable_profiling or not tracemalloc.is_tracing():
            return {"error": "Memory profiling not enabled"}
        
        snapshot = tracemalloc.take_snapshot()
        
        # Group by filename and line number
        stats = snapshot.statistics('filename')
        
        # Format results
        top_stats = [{
            "file": stat.traceback[0].filename,
            "line": stat.traceback[0].lineno,
            "size": stat.size,
            "count": stat.count
        } for stat in stats[:top_n]]
        
        return {
            "top_allocations": top_stats,
            "total_tracked": sum(stat.size for stat in stats)
        }
    
    async def start_monitoring(self) -> None:
        """Start periodic memory monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.logger.info("Starting memory monitoring")
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop memory monitoring"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Memory monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Background task to monitor memory usage"""
        while self.is_monitoring:
            try:
                await self._check_memory_usage()
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in memory monitoring: {e}")
                await asyncio.sleep(self.monitoring_interval * 2)  # Longer delay on error
    
    async def _check_memory_usage(self) -> None:
        """Check current memory usage and take action if needed"""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        system_mem = psutil.virtual_memory()
        
        # Update stats
        self.stats.current_bytes = mem_info.rss
        self.stats.peak_bytes = max(self.stats.peak_bytes, mem_info.rss)
        self.stats.system_total = system_mem.total
        self.stats.system_available = system_mem.available
        self.stats.process_rss = mem_info.rss
        
        # Calculate usage percentage
        usage_pct = mem_info.rss / self.memory_limit
        
        # Log memory usage
        self.logger.debug(
            f"Memory usage: {mem_info.rss/1024/1024:.2f}MB "
            f"({usage_pct*100:.1f}% of limit)"
        )
        
        # Check if we need to take action
        if usage_pct > self.threshold:
            self.logger.warning(
                f"Memory usage above threshold: {usage_pct*100:.1f}% > {self.threshold*100}%"
            )
            collected = self.force_garbage_collection()
            
            # If garbage collection didn't help enough, take more aggressive actions
            if mem_info.rss / self.memory_limit > self.threshold:
                self.logger.warning("Memory still high after garbage collection, clearing caches")
                self._clear_caches()
    
    def _clear_caches(self) -> None:
        """Clear internal caches to free memory"""
        # Clear any cached data in the tracking dictionaries
        self._object_sizes = {k: v for k, v in self._object_sizes.items() if k in self._large_objects}
        
        # Remove references to objects that no longer exist
        for obj_id, weak_ref in list(self._large_objects.items()):
            if weak_ref() is None:
                del self._large_objects[obj_id]
                if obj_id in self._object_sizes:
                    del self._object_sizes[obj_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current memory statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        # Update process stats
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        system_mem = psutil.virtual_memory()
        
        self.stats.process_rss = mem_info.rss
        self.stats.system_available = system_mem.available
        
        # Calculate additional metrics
        return {
            "current_usage_mb": self.stats.process_rss / 1024 / 1024,
            "peak_usage_mb": self.stats.peak_bytes / 1024 / 1024,
            "limit_mb": self.memory_limit / 1024 / 1024,
            "usage_percent": (self.stats.process_rss / self.memory_limit) * 100,
            "system_percent": (self.stats.process_rss / self.stats.system_total) * 100,
            "system_available_mb": self.stats.system_available / 1024 / 1024,
            "gc_collections": self.stats.collection_count,
            "last_collection": time.time() - self.stats.last_collection_time if self.stats.last_collection_time else None,
            "tracked_objects": len(self._large_objects)
        }