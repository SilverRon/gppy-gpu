import concurrent.futures
import multiprocessing
import cupy as cp
import threading
import psutil
import time
import signal
from queue import Queue, Empty
from typing import Callable, Any, Optional, List, Tuple, Dict
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
import gc
from concurrent.futures import ProcessPoolExecutor

class classmethodproperty:
    def __init__(self, func):
        self.func = classmethod(func)
    
    def __get__(self, instance, owner):
        return self.func.__get__(instance, owner)()

class Priority(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3  # New priority level for essential tasks

class MemoryState(Enum):
    HEALTHY = ("healthy", "continue")
    WARNING = ("warning", "cleanup")
    CRITICAL = ("critical", "pause")
    EMERGENCY = ("emergency", "emergency")
    
    def __init__(self, state: str, action: str):
        self.state = state
        self.action = action

class MemoryAction(Enum):
    CONTINUE = "continue"
    PAUSE = "pause"
    CLEANUP = "cleanup"
    EMERGENCY = "emergency"

@dataclass
class MemoryThresholds:
    warning: float
    critical: float
    emergency: float
    recovery: float  # New threshold for resuming normal operation

@dataclass
class Task:
    id: str
    func: Callable
    args: tuple
    kwargs: dict
    priority: Priority
    gpu: bool
    timeout: float
    timestamp: datetime
    task_name: str
    device: int

class TaskResult:
    def __init__(self, task_id:str, task_name:str, success: bool, result: Any = None, error: Exception = None):
        self.task_id = task_id
        self.task_name = task_name
        self.success = success
        self.result = result
        self.error = error
        self.timestamp = datetime.now()

class QueueManager:
    def __init__(
        self, 
        max_workers: Optional[int] = None, 
        max_gpu_workers: Optional[int] = None,
        peak_memory_percent: float = 80.0,
        check_interval: float = 1.0,
        memory_history_size: int = 100,
        logger: Optional[logging.Logger] = None,
        auto_cleanup: bool = True,
        task_timeout: float = 1800.0,  # 30 minutes default timeout
        recovery_threshold: float = 60.0  # Memory % to resume processing
    ):
        # Initialize logging
        if logger:
            self.logger = logger
        else:
            from .logging import Logger
            self.logger = Logger()
        
        # Basic initialization
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.devices = self._initialize_gpu_devices()
        self.max_gpu_workers = max_gpu_workers or len(self.devices)
        self.peak_memory_percent = peak_memory_percent
        
        # Task queues with priorities
        self.task_queues: Dict[Priority, Queue] = {
            priority: Queue() for priority in Priority
        }
        
        # Results and error handling
        self.results: List[TaskResult] = []
        self.errors: List[Dict] = []
        self.lock = threading.Lock()
        
        # Memory thresholds
        self.thresholds = MemoryThresholds(
            warning=peak_memory_percent * 0.7,
            critical=peak_memory_percent * 0.85,
            emergency=peak_memory_percent,
            recovery=recovery_threshold
        )
        
        # Control flags and timing
        self.check_interval = check_interval
        self.task_timeout = task_timeout
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._processing_thread = None
        self._monitor_thread = None
        
        # Memory monitoring
        self.memory_history: List[Dict] = []
        self.memory_history_size = memory_history_size
        self.auto_cleanup = auto_cleanup
        self.current_memory_state = MemoryState.HEALTHY
        self.warning_threshold = peak_memory_percent * 0.8 
        
        # Memory tracking
        self.process = psutil.Process()
        self.initial_memory = self.current_memory["used"]
        self.peak_memory = self.initial_memory
        self.peak_memory_timestamp = datetime.now()
        
        # Initial GPU memory tracking
        self.initial_gpu_memory = self.current_gpu_memory
        self.peak_gpu_memory = {}  # Track maximum GPU memory per device
        for device, stats in self.initial_gpu_memory.items():
            # Initialize max GPU memory for each device
            self.peak_gpu_memory[device] = {
                'used': stats['used'],
                'timestamp': datetime.now()
            }
        
        # Performance metrics
        self.performance_metrics = self._initialize_metrics()
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        
        # Process pool for memory operations
        self._process_pool = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())
        self._memory_queue = multiprocessing.Queue()

    def _handle_shutdown(self):
        """Handle graceful shutdown."""
        if self.logger:
            self.logger.info("Shutdown signal received, stopping processing...")
        self.stop_processing()

    def __enter__(self):
        """Context manager entry."""
        self.start_processing()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_processing()
        if exc_type:
            self.logger.error(f"Error during execution: {exc_val}")
            return False
        return True

    @classmethodproperty
    def current_memory(cls): 
        used = psutil.Process().memory_info().rss / 1024 / 1024
        total = psutil.virtual_memory().total / 1024 / 1024 
        return {
            'total': total,
            'used': used,
            'free': total - used,
            'percent': (used / total) *100
        }

    @classmethodproperty
    def current_gpu_memory(cls) -> Dict:
        """Get GPU memory statistics for all available devices."""
        gpu_stats = {}
        if cp.cuda.runtime.getDeviceCount() > 0:
            for device in range(cp.cuda.runtime.getDeviceCount()):
                with cp.cuda.Device(device):
                    mem_info = cp.cuda.runtime.memGetInfo()
                    total = mem_info[1] / 1024 / 1024 # in MB
                    free = mem_info[0] / 1024 / 1024  # in MB
                    used = total - free
                    gpu_stats[f'device_{device}'] = {
                        'total': total,
                        'used': used,
                        'free': free,
                        'percent': (used / total) * 100
                    }
        return gpu_stats

    @classmethodproperty
    def log_memory_usage(cls):
        gpu_summary = []
        for device, stats in cls.current_gpu_memory.items():
            percent = stats['percent']
            gpu_summary.append(f"{device}: {percent:.2f}%")
        gpu_info = f", GPU [{', '.join(gpu_summary)}]"
        return f"System [{cls.current_memory['percent']:.2f}%] {gpu_info}"

    ######### Initialize #########

    def _initialize_gpu_devices(self) -> List[int]:
        """Safely initialize GPU devices."""
        try:
            return list(range(cp.cuda.runtime.getDeviceCount()))
        except Exception as e:
            self.logger.warning(f"GPU initialization failed: {e}. Falling back to CPU only.")
            return []

    def _initialize_metrics(self) -> Dict:
        """Initialize performance metrics."""
        return {
            'tasks_processed': 0,
            'tasks_failed': 0,
            'pauses': 0,
            'cleanup_actions': 0,
            'emergency_actions': 0,
            'start_time': time.time(),
            'total_processing_time': 0,
            'avg_task_duration': 0,
            'peak_memory_usage': 0,
            'memory_cleanup_time': 0
        }

    ######### Add task #########
    def add_task(self, 
                 func: Callable, 
                 args: tuple = (), 
                 kwargs: dict = None, 
                 priority: Priority = Priority.MEDIUM, 
                 gpu: bool = False,
                 timeout: Optional[float] = None,
                 device: int = None,
                 task_name: str = None) -> str:
        """
        Add a task to the queue with specified priority and execution target.
        
        Args:
            func: The function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            priority: Task priority level
            gpu: Whether to execute on GPU
            timeout: Optional specific timeout for this task
            name: Name of the task
            
        Returns:
            task_id: Unique identifier for the task
        """
        task_id = f"task_{time.time()}_{id(func)}"
        kwargs = kwargs or {}
        
        task = Task(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            gpu=gpu,
            device=device,
            timeout=timeout or self.task_timeout,
            timestamp=datetime.now(),
            task_name=task_name
        )
        
        self.task_queues[priority].put(task)
        self.logger.debug(f"Added task {task_name} (id: {task_id}) with priority {priority}")
        return task_id

    def add_cpu_task(self, 
                     func: Callable, 
                     *args, 
                     priority: Priority = Priority.MEDIUM, 
                     task_name: str = None,
                     **kwargs) -> str:
        """Convenience method to add a CPU task."""
        return self.add_task(func, args, kwargs, priority, gpu=False, task_name=task_name)

    def add_gpu_task(self, 
                     func: Callable, 
                     *args, 
                     priority: Priority = Priority.MEDIUM, 
                     task_name: str = None,
                     device: int = None,
                     **kwargs) -> str:
        """Convenience method to add a GPU task."""
        return self.add_task(func, args, kwargs, priority, gpu=True, device=device, task_name=task_name)

    def add_critical_task(self, 
                         func: Callable, 
                         *args, 
                         gpu: bool = False, 
                         task_name: str = None,
                         device: int = None,
                         **kwargs) -> str:
        """Add a critical priority task."""
        return self.add_task(func, args, kwargs, Priority.CRITICAL, gpu, device=device, task_name=task_name)

    def _choose_gpu_device(self):
        """Choose a GPU device randomly weighted by available memory."""
        import random

        devices = []
        total_free = 0
        # Gather free memory info for each GPU
        for dev_id in range(cp.cuda.runtime.getDeviceCount()):
            with cp.cuda.Device(dev_id):
                free, total = cp.cuda.runtime.memGetInfo()
                devices.append((dev_id, free))
                total_free += free
        
        # Randomly pick device based on free memory
        r = random.uniform(0, total_free)
        for dev_id, free in devices:
            if r < free:
                return dev_id
            r -= free
        return 0  # fallback

    @contextmanager
    def gpu_context(self, device: int = None):
        """Context manager for safe GPU operations."""
        if device is None:
            device = self._choose_gpu_device()
        try:
            with cp.cuda.Device(device):
                yield
        except Exception as e:
            self.logger.error(f"GPU operation failed on device {device}: {e}")
            raise

    def _handle_task_error(self, task: Task, error: Exception):
        """Handle task execution errors."""
        error_info = {
            'timestamp': datetime.now(),
            'task': str(task),
            'error': str(error),
            'traceback': getattr(error, '__traceback__', None)
        }
        with self.lock:
            self.errors.append(error_info)
            self.performance_metrics['tasks_failed'] += 1
        self.logger.error(f"Task error: {error_info['error']}")

    def _age_task_priorities(self):
        """
        Age tasks in queues to prevent starvation of lower priority tasks.
        Promotes tasks that have been waiting too long.
        """
        current_time = datetime.now()
        priority_aging_thresholds = {
            Priority.LOW: 300,     # 5 minutes
            Priority.MEDIUM: 600,  # 10 minutes
            Priority.HIGH: 1200    # 20 minutes
        }
        
        with self.lock:
            for priority in list(Priority)[:-1]:  # Exclude CRITICAL
                queue = self.task_queues[priority]
                if queue.empty():
                    continue
                
                # Check each task in the queue
                aged_tasks = []
                while not queue.empty():
                    task = queue.get()
                    wait_time = (current_time - task.timestamp).total_seconds()
                    
                    # Promote task if it has waited too long
                    if wait_time > priority_aging_thresholds[priority]:
                        new_priority = Priority(priority.value + 1)
                        self.logger.info(
                            f"Promoting task {task.id} from {priority.name} to {new_priority.name} "
                            f"after waiting {wait_time:.1f} seconds"
                        )
                        task.priority = new_priority
                        self.task_queues[new_priority].put(task)
                    else:
                        aged_tasks.append(task)
                
                # Put non-promoted tasks back
                for task in aged_tasks:
                    queue.put(task)

    def _process_tasks(self):
        """Process tasks with priority and memory management."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as cpu_executor, \
             concurrent.futures.ThreadPoolExecutor(max_workers=self.max_gpu_workers) as gpu_executor:
            
            last_aging_check = time.time()
            last_leak_check = time.time()
            
            while not self._stop_event.is_set():
                current_time = time.time()
                
                # Periodic checks
                if current_time - last_aging_check > 60:  # Check every minute
                    self._age_task_priorities()
                    last_aging_check = current_time
                
                if current_time - last_leak_check > 300:  # Check every 5 minutes
                    if self._detect_memory_leaks():
                        self.cleanup_memory(force=True)
                    last_leak_check = current_time
                
                # Memory check
                action, stats = self._check_memory_usage()
                self._handle_memory_action(action)
                
                if self._pause_event.is_set():
                    time.sleep(self.check_interval)
                    continue
                
                # Process tasks by priority
                for priority in reversed(list(Priority)):
                    while not self.task_queues[priority].empty():
                        try:
                            task = self.task_queues[priority].get_nowait()
                            executor = gpu_executor if task.gpu else cpu_executor
                            
                            if task.gpu:
                                device_index = task.device if task.device is not None else self.devices[len(self.results) % len(self.devices)]
                                future = executor.submit(
                                    self._execute_gpu_task,
                                    task,
                                    device_index,
                                    executor
                                )
                            else:
                                future = executor.submit(
                                    self._execute_task,
                                    task,
                                    executor
                                )
                            
                            self._handle_task_result(future, task)
                            
                        except Empty:
                            break
                        except Exception as e:
                            self._handle_task_error(task, e)
                
                if all(q.empty() for q in self.task_queues.values()):
                    time.sleep(0.1)

    def _execute_task(self, task: Task, executor) -> TaskResult:
        """Execute a task with timeout."""
        start_time = time.time()
        try:
            future = executor.submit(
                task.func,
                *task.args,
                **task.kwargs
            )
            result = future.result(timeout=task.timeout)
            
            execution_time = time.time() - start_time
            self._update_task_metrics(execution_time)
            return TaskResult(task_id = task.id, task_name=task.task_name, success=True, result=result)
            
        except concurrent.futures.TimeoutError:
            return TaskResult(
                task_id = task.id, task_name=task.task_name, 
                success=False, 
                error=TimeoutError(f"Task {task.id} exceeded timeout of {task.timeout}s")
            )
        except Exception as e:
            return TaskResult(task_id = task.id, task_name=task.task_name, success=False, error=e)

    def _execute_gpu_task(self, task: Task, device: int, executor) -> TaskResult:
        """Execute a task on GPU with timeout."""
        try:
            with self.gpu_context(device):
                return self._execute_task(task, executor)
        except Exception as e:
            return TaskResult(task_id = task.id, task_name=task.task_name, success=False, error=e)

    def _handle_task_result(self, future: concurrent.futures.Future, task: Task):
        """Handle task completion and results."""
        try:
            result = future.result(timeout=task.timeout)
            with self.lock:
                self.results.append(result)
                self.performance_metrics['tasks_processed'] += 1
                
        except Exception as e:
            self._handle_task_error(task, e)

    def _update_task_metrics(self, execution_time: float):
        """Update running metrics with new task execution time."""
        with self.lock:
            total_tasks = self.performance_metrics['tasks_processed']
            current_avg = self.performance_metrics['avg_task_duration']
            
            # Update running average
            self.performance_metrics['avg_task_duration'] = (
                (current_avg * total_tasks + execution_time) / (total_tasks + 1)
            )

    def get_task_status(self, task_id: str = None, task_name: str = None) -> Dict:
        """Get the status of a specific task."""
        for result in self.results:
            if (hasattr(result, 'task_id') and result.task_id == task_id) \
                or (hasattr(result, 'task_name') and result.task_name == task_name):
                return {
                    'task_id': result.task_id,
                    'task_name': result.task_name,
                    'status': 'completed' if result.success else 'failed',
                    'result': result.result if result.success else None,
                    'error': str(result.error) if result.error else None,
                    'timestamp': result.timestamp,
                }
            
        # Check if task is still in queues
        for priority in Priority:
            try:
                queue_items = self.task_queues[priority].queue
                for task in queue_items:
                    if task.id == task_id:
                        return {
                            'task_id': task_id,
                            'status': 'pending',
                            'priority': priority.name,
                            'queued_at': task.timestamp
                        }
            except Exception:
                continue
        
        return {'task_id': task_id, 'status': 'not_found'}

    ######### Processing actions #########
    def start_processing(self):
        """Start task processing and memory monitoring threads."""
        if self._processing_thread is None:
            # Log initial memory state before processing
            memory_percent = self.current_memory['percent']
            memory_state = self._get_memory_state(memory_percent)
            
            # Brief summary for info level
            gpu_summary = []
            gpu_stats = self.current_gpu_memory
            if gpu_stats:
                for device, stats in gpu_stats.items():
                    percent = stats['percent']
                    gpu_state = self._get_memory_state(percent)
                    gpu_summary.append(f"{device}: {percent:.1f}% ({gpu_state})")
            
            gpu_info = f", GPU [{', '.join(gpu_summary)}]" if gpu_summary else ""
            self.logger.info(f"Initial memory states: System {memory_percent:.1f}% ({memory_state}){gpu_info}")
            
            # Detailed report for debug level
            debug_message = ["Initial detailed memory report:"]
            debug_message.extend([
                "System Memory:",
                f"  Initial: {self.initial_memory:.1f} MB",
                f"  Current: {self.current_memory['used']:.1f} MB",
                f"  Total: {self.current_memory['total']:.1f} MB",
                f"  State: {memory_state}"
            ])
            
            if gpu_stats:
                debug_message.append("\nGPU Memory:")
                for device, stats in gpu_stats.items():
                    used_mb = stats['used']
                    total_mb = stats['total']
                    percent = stats['percent']
                    gpu_state = self._get_memory_state(percent)
                    initial_used = self.initial_gpu_memory[device]['used']
                    debug_message.extend([
                        f"  {device}:",
                        f"    Initial: {initial_used:.1f} MB",
                        f"    Current: {used_mb:.1f} MB",
                        f"    Total: {total_mb:.1f} MB",
                        f"    State: {gpu_state}"
                    ])
            
            self.logger.debug("\n".join(debug_message))
            
            self._processing_thread = threading.Thread(target=self._process_tasks)
            self._processing_thread.daemon = True
            self._processing_thread.start()
            
        if self._monitor_thread is None:
            self._monitor_thread = threading.Thread(target=self._monitor_memory)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()

    def pause_processing(self):
        """Temporarily pause task processing."""
        if not self._pause_event.is_set():
            self._pause_event.set()
            self.logger.info("Processing paused")
            self.performance_metrics['pauses'] += 1

    def resume_processing(self):
        """Resume task processing."""
        if self._pause_event.is_set():
            self._pause_event.clear()
            self.logger.info("Processing resumed")

    def is_processing(self) -> bool:
        """Check if processing is active."""
        return (self._processing_thread is not None and 
                self._processing_thread.is_alive() and 
                not self._pause_event.is_set())

    def stop_processing(self, timeout: float = 30.0):
        """
        Stop processing tasks gracefully.
        
        Args:
            timeout: Maximum time to wait for critical tasks to complete
        """
        self._stop_event.set()
        
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout)
            
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout)
        
        self._update_memory_stats("Processing stop")
        self.cleanup_memory(force=True)
        self.log_memory_report()

    def wait_until_all_tasks_complete(self, timeout: Optional[float] = None):
        """
        Wait until all queued tasks are completed.
        
        Args:
            timeout: Maximum time to wait in seconds. If None, wait indefinitely.
        
        Returns:
            bool: True if all tasks completed, False if timeout occurred
        """
        start_time = time.time()
        
        while True:
            # Check if any tasks remain in any queue
            all_queues_empty = all(q.empty() for q in self.task_queues.values())
            
            if all_queues_empty:
                self.logger.debug("All tasks completed successfully")
                return True
                
            # Check for timeout
            if timeout is not None and (time.time() - start_time) > timeout:
                remaining_tasks = sum(not q.empty() for q in self.task_queues.values())
                self.logger.warning(f"Timeout waiting for tasks. {remaining_tasks} tasks remaining")
                return False
                
            # Small sleep to prevent CPU spinning
            time.sleep(0.1)

    def wait_until_task_complete(self, task_id: str, timeout: float = None):
        """Wait until the specified task completes or until timeout."""
        
        start_time = time.time()
        while True:
            with self.lock:
                if any(r.task_id == task_id for r in self.results):
                    return  # Task completed
            if timeout is not None and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within timeout.")
            
            time.sleep(0.1)
            
    ######### Cleanup memory #########
    @classmethod
    def cleanup_memory(self, force: bool = False):
        """Enhanced memory cleanup."""
        cleanup_steps = [
            (self._cleanup_gpu_memory, "GPU"),
            (self._cleanup_python_memory, "Python"),
            (self._cleanup_system_memory, "System")
        ]
        if hasattr(self, "performance_metrics"):
            self.performance_metrics['cleanup_actions'] += 1
            for cleanup_func, cleanup_type in cleanup_steps:
                with self._cleanup_handler(cleanup_type):
                    cleanup_func(force)
        else:
            for cleanup_func, cleanup_type in cleanup_steps:
                cleanup_func(force)

    @contextmanager
    def _cleanup_handler(self, cleanup_type: str):
        """Context manager for handling cleanup operations."""
        start_time = time.time()
        try:
            yield
            self.logger.debug(f"{cleanup_type} cleanup completed")
        except Exception as e:
            self.logger.error(f"{cleanup_type} cleanup failed: {e}")
            raise
        finally:
            cleanup_time = time.time() - start_time
            self.performance_metrics['memory_cleanup_time'] += cleanup_time

    @classmethod
    def _cleanup_gpu_memory(self, force: bool = False):
        """Clean up GPU memory by clearing cache and garbage collecting unused tensors."""
        if cp.cuda.runtime.getDeviceCount() > 0:
            cp.get_default_memory_pool().free_all_blocks()
            
            if force:
                cp.cuda.stream.get_current_stream().synchronize()
                for device in range(cp.cuda.runtime.getDeviceCount()):
                    with cp.cuda.Device(device):
                        cp.get_default_memory_pool().free_all_blocks()
                        cp.get_default_pinned_memory_pool().free_all_blocks()
            
    @classmethod
    def _cleanup_python_memory(self, force: bool = False):
        """Clean up Python memory through garbage collection."""
        gc.collect()
        if force:
            for _ in range(3):
                gc.collect()
            gc.disable()
            gc.collect()
            gc.enable() 

    @staticmethod
    def _process_memory_info() -> Tuple[float, Dict]:
        """Process memory info in separate process."""
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_percent = process.memory_percent()
        return mem_percent, mem_info._asdict()

    @classmethod
    def _cleanup_system_memory(cls, force: bool = False) -> int:
        """Clean up system memory using OS-specific methods."""
        def cleanup_worker():
            process = psutil.Process()
            mem_before = process.memory_info()
            
            if force and hasattr(process, "memory_full_info"):
                process.memory_full_info()
            
            if hasattr(psutil, "Process"):
                process.memory_info()
            
            mem_after = process.memory_info()
            return mem_before.rss - mem_after.rss

        with ProcessPoolExecutor() as executor:
            memory_freed = executor.submit(cleanup_worker).result()
        return memory_freed

    ######### Memory related #########
    def log_memory_report(self):
        """Log memory usage report with multiprocessing."""
        if not self.logger:
            return

        # Run memory info collection in separate process
        future = self._process_pool.submit(self._process_memory_info)
        memory_percent, memory_info = future.result()
        
        memory_state = self._get_memory_state(memory_percent)
        
        # Brief summary for info level
        gpu_summary = []
        gpu_stats = self.current_gpu_memory  # GPU stats still in main process
        
        if gpu_stats:
            for device, stats in gpu_stats.items():
                used_mb = stats['used']
                percent = stats['percent']
                gpu_state = self._get_memory_state(percent)
                gpu_summary.append(f"{device}: {percent:.1f}% (used {used_mb:.1f} MB, {gpu_state})")
        
        gpu_info = f", GPU [{', '.join(gpu_summary)}]" if gpu_summary else ""
        self.logger.info(f"Memory usage report: System {memory_percent:.1f}% (used {self.current_memory['used']:.1f} MB, {memory_state}){gpu_info}")
        
        # Detailed report for debug level
        
        log_message = ["Detailed memory usage report:"]
        log_message.extend([
            "System Memory:",
            f"  Initial: {self.initial_memory:.1f} MB",
            f"  Peak: {self.peak_memory:.1f} MB",
            f"  Current: {self.current_memory['used']:.1f} MB ({memory_percent:.1f}%)",
            f"  Total: {self.current_memory['total']:.1f} MB",
            f"  State: {memory_state}"
        ])
        
        if gpu_stats:
            log_message.append("\nGPU Memory:")
            for device, stats in gpu_stats.items():
                log_message.extend([
                    f"  {device}:",
                    f"    Initial: {self.initial_gpu_memory[device]['used']:.1f} MB",
                    f"    Peak: {self.peak_gpu_memory[device]['used']:.1f} MB",
                    f"    Current: {stats['used']:.1f} MB ({stats['percent']} %)",
                    f"    Total: {stats['total']:.1f} MB",
                    f"    State: {self._get_memory_state(percent)}"
                ])
            
        self.logger.debug("\n".join(log_message))

    def _handle_memory_action(self, action: MemoryAction):
        """Handle memory action."""
        if action == MemoryAction.EMERGENCY:
            self._handle_emergency_state()
        elif action == MemoryAction.CLEANUP:
            self.cleanup_memory()
        elif action == MemoryAction.PAUSE:
            self.pause_processing()

    def _handle_emergency_state(self):
        """Handle emergency memory state - immediate aggressive action required."""
        self.logger.critical("EMERGENCY: Memory state critical - taking immediate action")
        try:
            # Force immediate garbage collection
            gc.collect(generation=2)
            
            # Aggressive memory cleanup
            self.cleanup_memory(force=True)
            
            # Clear GPU memory
            if cp.cuda.runtime.getDeviceCount() > 0:
                for device in range(cp.cuda.runtime.getDeviceCount()):
                    with cp.cuda.Device(device):
                        cp.get_default_memory_pool().free_all_blocks()
                        
            self._update_memory_stats("After emergency cleanup")
            
        except Exception as e:
            self.logger.error(f"Emergency state handling failed: {e}")
            raise

    def _monitor_memory(self):
        """Dedicated memory monitoring thread."""
        while not self._stop_event.is_set():
            try:
                memory_state, stats = self._check_memory_usage()
                
                if memory_state != self.current_memory_state:
                    self.logger.info(f"Memory state changed: {self.current_memory_state} -> {memory_state}")
                    self.current_memory_state = memory_state
                    self._handle_memory_state_change(memory_state, stats)
                
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")

    def _handle_memory_state_change(self, state: MemoryState, stats: Dict):
        """Handle memory state transitions."""
        if state == MemoryState.EMERGENCY:
            self._handle_emergency_state()
        elif state == MemoryState.CRITICAL:
            self._handle_critical_state()
        elif state == MemoryState.WARNING:
            self._handle_warning_state()

    def _detect_memory_leaks(self):
        """
        Detect potential memory leaks by analyzing memory usage patterns.
        Returns True if a leak is detected.
        """
        if len(self.memory_history) < 10:  # Need enough history for analysis
            return False
            
        # Calculate memory growth rate
        recent_memory = [entry['memory'] for entry in self.memory_history[-10:]]
        growth_rate = (recent_memory[-1] - recent_memory[0]) / len(recent_memory)
        
        # Check if memory consistently growing without cleanup effects
        is_consistent_growth = all(
            recent_memory[i] <= recent_memory[i+1] 
            for i in range(len(recent_memory)-1)
        )
        
        # Memory leak threshold: 5% growth rate with consistent increase
        if growth_rate > 0.05 * self.initial_memory and is_consistent_growth:
            self.logger.warning(f"Potential memory leak detected! Growth rate: {growth_rate:.2f} MB/sample")
            return True
        return False

    def _update_memory_stats(self, stage: str = None) -> float:
        """
        Update memory statistics and optionally log the current stage.
        
        Args:
            stage: Optional description of the current processing stage
            
        Returns:
            float: Current memory usage in MB
        """
        if self.current_memory["used"] > self.peak_memory:
            self.peak_memory = self.current_memory["used"]
            self.peak_memory_timestamp = datetime.now()
            
        if stage and self.logger:
            self.logger.debug(f"Memory at {stage}: {self.current_memory['used']:.2f} MB")
            
        return current

    def _check_memory_usage(self) -> Tuple[MemoryState, Dict]:
        """Enhanced memory check with action recommendations."""
        current_time = datetime.now()
        
        # Get current memory stats
        memory_percent = self.current_memory["percent"]
        
        # Get GPU memory stats
        gpu_stats = self.current_gpu_memory
        
        # Update peak GPU memory
        self._update_gpu_memory_stats(gpu_stats)
        
        # Calculate memory pressure
        memory_pressure = self._calculate_memory_pressure({
            'system_memory_percent': memory_percent,
            'gpu_memory': gpu_stats
        })
        
        if memory_pressure >= self.thresholds.emergency:
            return MemoryState.EMERGENCY, {
                'timestamp': current_time,
                'system': {'percent': memory_percent},
                'gpu': gpu_stats
            }
        elif memory_pressure >= self.thresholds.critical:
            return MemoryState.CRITICAL, {
                'timestamp': current_time,
                'system': {'percent': memory_percent},
                'gpu': gpu_stats
            }
        elif memory_pressure >= self.thresholds.warning:
            return MemoryState.WARNING, {
                'timestamp': current_time,
                'system': {'percent': memory_percent},
                'gpu': gpu_stats
            }
        else:
            return MemoryState.HEALTHY, {
                'timestamp': current_time,
                'system': {'percent': memory_percent},
                'gpu': gpu_stats
            }

    def _calculate_memory_pressure(self, stats: Dict) -> float:
        """
        Calculate overall memory pressure as a weighted score.
        
        Args:
            stats: Dictionary containing memory statistics
            
        Returns:
            float: Memory pressure score between 0 and 100
        """
        # Weights for different memory types
        weights = {
            'system': 0.5,    # System memory weight
            'gpu': 0.3,       # GPU memory weight
            'process': 0.2    # Process memory weight
        }
        
        # Calculate system memory pressure
        sys_pressure = stats['system_memory_percent']
        
        # Calculate GPU memory pressure (average across all devices)
        gpu_pressure = 0
        if stats['gpu_memory']:
            gpu_pressures = [device_stats['percent'] for device_stats in stats['gpu_memory'].values()]
            gpu_pressure = sum(gpu_pressures) / len(gpu_pressures)
            
        # Calculate process memory pressure
        proc_pressure = stats['system_memory_percent']
        
        # Calculate weighted average
        total_pressure = (
            sys_pressure * weights['system'] +
            gpu_pressure * weights['gpu'] +
            proc_pressure * weights['process']
        )
        
        return total_pressure

    def _update_memory_stats(self, event: str):
        """Update memory statistics."""
        current_memory = self.current_memory['used']
        
        # Update max memory if current usage is higher
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
            self.peak_memory_timestamp = datetime.now()
            
        # Update GPU memory stats
        gpu_stats = self.current_gpu_memory
        if gpu_stats:
            for device, stats in gpu_stats.items():
                used_mb = stats['used'] / 1024 / 1024
                if used_mb > self.peak_gpu_memory[device]['used']:
                    self.peak_gpu_memory[device].update({
                        'used': used_mb,
                        'timestamp': datetime.now()
                    })
        
        # Add to history
        self.memory_history.append({
            'timestamp': datetime.now(),
            'memory': current_memory,
            'event': event
        })
        
        # Trim history if needed
        if len(self.memory_history) > self.memory_history_size:
            self.memory_history.pop(0)

    def _update_gpu_memory_stats(self, stats: Dict) -> None:
        """
        Update GPU memory statistics and track peak usage.
        
        Args:
            stats: Dictionary containing GPU memory statistics
        """
        current_time = datetime.now()
        for device, device_stats in stats.items():
            # Convert to MB if not already
            used_memory = device_stats['used']
            
            # Initialize device if not exists
            if device not in self.peak_gpu_memory:
                self.peak_gpu_memory[device] = {
                    'used': used_memory,
                    'timestamp': current_time
                }
            # Update if current usage is higher
            elif used_memory > self.peak_gpu_memory[device]['used']:
                self.peak_gpu_memory[device]['used'] = used_memory
                self.peak_gpu_memory[device]['timestamp'] = current_time

    def _get_memory_state(self, percent: float) -> str:
        """Get memory state string based on percentage."""
        if percent >= self.thresholds.emergency:
            return "EMERGENCY"
        elif percent >= self.thresholds.critical:
            return "CRITICAL"
        elif percent >= self.thresholds.warning:
            return "WARNING"
        return "HEALTHY"

    ######### Output #########
    def get_task_statistics(self) -> Dict:
        """Get detailed task statistics."""
        with self.lock:
            total_tasks = self.performance_metrics['tasks_processed'] + self.performance_metrics['tasks_failed']
            success_rate = (self.performance_metrics['tasks_processed'] / total_tasks * 100 
                          if total_tasks > 0 else 0)
            
            return {
                'total_tasks': total_tasks,
                'successful_tasks': self.performance_metrics['tasks_processed'],
                'failed_tasks': self.performance_metrics['tasks_failed'],
                'success_rate': success_rate,
                'avg_task_duration': self.performance_metrics['avg_task_duration'],
                'peak_memory_usage': self.performance_metrics['peak_memory_usage'],
                'current_queue_size': sum(q.qsize() for q in self.task_queues.values())
            }

    def get_memory_health_report(self) -> Dict:
        """Generate comprehensive memory health report."""
        current_stats = self._check_memory_usage()[1]
        
        return {
            'current_state': self.current_memory_state.value,
            'system_memory': {
                'usage_percent': current_stats['system']['percent'],
                'available_gb': current_stats['system']['available'] / (1024 ** 3),
                'total_gb': current_stats['system']['total'] / (1024 ** 3)
            },
            'gpu_memory': {
                device: {
                    'usage_percent': stats['percent'],
                    'available_gb': stats['free'] / (1024 ** 3),
                    'total_gb': stats['total'] / (1024 ** 3)
                }
                for device, stats in current_stats['gpu'].items()
            },
            'memory_pressure': current_stats['pressure'],
            'cleanup_stats': {
                'total_cleanups': self.performance_metrics['cleanup_actions'],
                'total_cleanup_time': self.performance_metrics['memory_cleanup_time'],
                'avg_cleanup_time': (self.performance_metrics['memory_cleanup_time'] / 
                                   self.performance_metrics['cleanup_actions']
                                   if self.performance_metrics['cleanup_actions'] > 0 else 0)
            }
        }

    def __del__(self):
        """Cleanup process pool on deletion."""
        if hasattr(self, '_process_pool'):
            self._process_pool.shutdown()


