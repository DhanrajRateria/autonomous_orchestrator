import asyncio
import concurrent.futures
import logging
from typing import Any, Callable, Coroutine, List, TypeVar, Optional, Dict, Set, Union
from functools import partial
import time
import threading
import queue
import os
import signal
import traceback
from dataclasses import dataclass, field

T = TypeVar('T')  # Type variable for task results

@dataclass
class TaskStats:
    """Statistics for executed tasks"""
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    total_execution_time: float = 0.0
    
    @property
    def avg_execution_time(self) -> float:
        """Calculate average execution time"""
        completed = self.completed or 1  # Avoid division by zero
        return self.total_execution_time / completed

class ParallelExecutor:
    """
    Efficient parallel task executor optimized for system programming.
    Uses thread and process pools for parallel execution of CPU and I/O tasks.
    """
    
    def __init__(self, max_workers: int = None, task_queue_size: int = 1000):
        """
        Initialize the parallel executor.
        
        Args:
            max_workers: Maximum number of worker threads/processes
            task_queue_size: Maximum size of the task queue
        """
        self.logger = logging.getLogger("system.parallel_executor")
        
        # If max_workers is None, use CPU count for optimal performance
        self.max_workers = max_workers or os.cpu_count()
        self.task_queue_size = task_queue_size
        
        # Thread pool for I/O-bound tasks
        self.thread_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="security-thread"
        )
        
        # Process pool for CPU-bound tasks
        # Note: This takes more resources to create, so initialize it on-demand
        self._process_executor = None
        self._process_executor_lock = threading.Lock()
        
        # Task queues and tracking
        self.task_queue = asyncio.Queue(maxsize=task_queue_size)
        self.active_tasks: Set[asyncio.Task] = set()
        self.task_stats = TaskStats()
        self._stats_lock = threading.Lock()
        
        # Shutdown flag
        self.shutting_down = False
        
        self.logger.info(f"Parallel executor initialized with {self.max_workers} workers")
    
    @property
    def process_executor(self):
        """Lazy initialization of process executor"""
        if self._process_executor is None:
            with self._process_executor_lock:
                if self._process_executor is None:
                    self._process_executor = concurrent.futures.ProcessPoolExecutor(
                        max_workers=max(1, self.max_workers // 2)  # Use fewer processes than threads
                    )
        return self._process_executor
    
    async def submit(self, func: Callable, *args, 
                    cpu_bound: bool = False, 
                    timeout: Optional[float] = None, 
                    **kwargs) -> Any:
        """
        Submit a task for parallel execution.
        
        Args:
            func: Function to execute
            *args: Arguments to pass to the function
            cpu_bound: Whether this task is CPU-bound (use process pool instead of thread pool)
            timeout: Maximum execution time in seconds
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the function execution
        """
        if self.shutting_down:
            raise RuntimeError("Executor is shutting down, not accepting new tasks")
        
        # Update statistics
        with self._stats_lock:
            self.task_stats.submitted += 1
        
        # Select the appropriate executor based on task type
        executor = self.process_executor if cpu_bound else self.thread_executor
        
        # Execute in thread/process pool and return result
        loop = asyncio.get_running_loop()
        start_time = time.time()
        
        try:
            if timeout:
                # Handle timeouts using asyncio.wait_for
                result = await asyncio.wait_for(
                    loop.run_in_executor(executor, partial(func, *args, **kwargs)),
                    timeout=timeout
                )
            else:
                # Execute without timeout
                result = await loop.run_in_executor(
                    executor, partial(func, *args, **kwargs)
                )
            
            # Update statistics for successful execution
            execution_time = time.time() - start_time
            with self._stats_lock:
                self.task_stats.completed += 1
                self.task_stats.total_execution_time += execution_time
            
            return result
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            self.logger.warning(
                f"Task {func.__name__} timed out after {execution_time:.2f}s (limit: {timeout}s)"
            )
            with self._stats_lock:
                self.task_stats.failed += 1
            
            raise  # Re-raise timeout error
            
        except asyncio.CancelledError:
            with self._stats_lock:
                self.task_stats.cancelled += 1
            raise  # Re-raise cancellation
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(
                f"Task {func.__name__} failed after {execution_time:.2f}s: {e}",
                exc_info=True
            )
            with self._stats_lock:
                self.task_stats.failed += 1
            raise  # Re-raise exception to caller
    
    async def map(self, func: Callable[[T], Any], items: List[T], 
                 cpu_bound: bool = False, 
                 max_concurrency: Optional[int] = None,
                 timeout: Optional[float] = None) -> List[Any]:
        """
        Execute a function on each item in parallel.
        
        Args:
            func: Function to execute
            items: List of items to process
            cpu_bound: Whether this task is CPU-bound
            max_concurrency: Maximum number of concurrent executions
            timeout: Maximum execution time in seconds per item
            
        Returns:
            List of results in the same order as the input items
        """
        if not items:
            return []
        
        # Limit concurrency to avoid overloading the system
        sem_size = max_concurrency or self.max_workers
        semaphore = asyncio.Semaphore(sem_size)
        
        async def process_item(item):
            async with semaphore:
                return await self.submit(func, item, cpu_bound=cpu_bound, timeout=timeout)
        
        # Create tasks for each item
        tasks = [asyncio.create_task(process_item(item)) for item in items]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Error processing item {i}: {result}")
        
        return results
    
    async def shutdown(self, wait: bool = True, timeout: float = 30.0):
        """
        Shutdown the executor.
        
        Args:
            wait: Whether to wait for pending tasks to complete
            timeout: Maximum time to wait for tasks to complete
        """
        if self.shutting_down:
            return
        
        self.shutting_down = True
        self.logger.info("Shutting down parallel executor")
        
        # If not waiting, cancel all active tasks
        if not wait:
            for task in self.active_tasks:
                if not task.done():
                    task.cancel()
        
        # Wait for tasks to complete with timeout
        if wait and self.active_tasks:
            try:
                pending = [t for t in self.active_tasks if not t.done()]
                if pending:
                    self.logger.info(f"Waiting for {len(pending)} tasks to complete")
                    done, pending = await asyncio.wait(
                        pending, 
                        timeout=timeout, 
                        return_when=asyncio.ALL_COMPLETED
                    )
                    
                    if pending:
                        self.logger.warning(f"{len(pending)} tasks did not complete within timeout")
                        for task in pending:
                            task.cancel()
            except Exception as e:
                self.logger.error(f"Error while waiting for tasks: {e}")
        
        # Shutdown thread executor
        self.thread_executor.shutdown(wait=wait)
        
        # Shutdown process executor if it was initialized
        if self._process_executor is not None:
            self._process_executor.shutdown(wait=wait)
        
        self.logger.info("Parallel executor shutdown complete")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics"""
        with self._stats_lock:
            return {
                "submitted": self.task_stats.submitted,
                "completed": self.task_stats.completed,
                "failed": self.task_stats.failed,
                "cancelled": self.task_stats.cancelled,
                "avg_execution_time": self.task_stats.avg_execution_time,
                "active_tasks": len(self.active_tasks),
                "max_workers": self.max_workers,
                "queue_size": self.task_queue.qsize()
            }
    
    async def batch_process(self, items: List[T], batch_size: int, 
                          processor: Callable[[List[T]], Any],
                          cpu_bound: bool = False) -> List[Any]:
        """
        Process items in batches to optimize throughput.
        
        Args:
            items: List of items to process
            batch_size: Size of each batch
            processor: Function to process each batch
            cpu_bound: Whether processing is CPU-bound
            
        Returns:
            List of results from batch processing
        """
        # Split items into batches
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        # Process each batch
        results = await self.map(processor, batches, cpu_bound=cpu_bound)
        
        # Flatten results if needed
        flattened = []
        for batch_result in results:
            if isinstance(batch_result, list):
                flattened.extend(batch_result)
            else:
                flattened.append(batch_result)
        
        return flattened