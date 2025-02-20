from multiprocessing.pool import Pool
import multiprocessing
from queue import Queue, PriorityQueue
import threading

import cupy as cp

import time
import signal
import os

from typing import Callable, Any, Optional, List, Dict, Union
from datetime import datetime
from gppy.logger import Logger

from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager

import itertools

from .memory import MemoryState, MemoryMonitor

signal.signal(signal.SIGINT, signal.SIG_IGN)

class Priority(Enum):
    """
    Task priority levels for workload management.

    Provides a hierarchical priority system to manage task execution:
    - LOW: Background or non-critical tasks
    - MEDIUM: Standard processing tasks
    - HIGH: Time-sensitive or critical tasks

    Allows fine-grained control over task scheduling and resource allocation.
    """

    LOW = 0
    MEDIUM = 1
    HIGH = 2


@dataclass(order=True)
class Task:
    """
    Represents a single task in the processing queue.

    Encapsulates all necessary information for task tracking, including:
    - Unique identification
    - Execution details
    - Priority
    - Resource requirements
    - Execution status
    - Performance metrics

    Attributes:
        id (str): Unique task identifier
        task_name (str): Descriptive name of the task
        func (Callable): Function to be executed
        args (tuple): Positional arguments for the function
        kwargs (dict): Keyword arguments for the function
        priority (Priority): Task priority level
        starttime (datetime): Task creation timestamp
        endtime (datetime, optional): Task completion timestamp
        gpu (bool): Whether the task requires GPU processing
        device (int): Specific GPU device for processing
        status (str): Current task status (pending, processing, completed, failed)
        result (Any): Task execution result
        error (Exception, optional): Error encountered during task execution
    """

    # Add a sort_index field for comparison
    sort_index: float = field(init=False, repr=False)

    # Original fields
    id: str = field(compare=False)
    task_name: str = field(compare=False)
    func: Callable = field(compare=False)
    args: tuple = field(compare=False)
    kwargs: dict = field(compare=False)
    priority: Priority = field(compare=False)
    starttime: datetime = field(compare=False)
    endtime: Optional[datetime] = field(default=None, compare=False)
    gpu: bool = field(default=False, compare=False)
    device: int = field(default=0, compare=False)
    status: str = field(default="pending", compare=False)
    result: Any = field(default=None, compare=False)
    error: Optional[Exception] = field(default=None, compare=False)
    
    def __post_init__(self):
        """
        Generate a sorting index for priority-based task ordering.

        The sort index combines:
        - Priority level (scaled to millions)
        - Microsecond-level timestamp

        This ensures that:
        - Higher priority tasks are processed first
        - Tasks within the same priority are ordered by submission time
        """
        # Create a sort index based on priority and timestamp
        # Priority values (0,1,2) become (1000000, 2000000, 3000000)
        # Then add the decimal part from the timestamp
        timestamp_part = time.time() % 1  # Get just the decimal part
        self.sort_index = (self.priority.value + 1) * 1000000 + timestamp_part


class AbruptStopException(Exception):
    """Custom exception to signal abrupt stop processing."""
    pass


class QueueManager:
    """
    Advanced task management system for parallel processing.

    Provides comprehensive task queuing, processing, and resource management
    capabilities. Supports both CPU and GPU task processing with:
    - Multi-priority task scheduling
    - Dynamic resource allocation
    - Memory state monitoring
    - Error tracking and logging

    Key Responsibilities:
    - Task submission and tracking
    - Parallel task execution
    - Resource allocation
    - Memory management
    - Error handling and logging

    Workflow:
    1. Initialize with configurable worker pool
    2. Submit tasks with priorities and resource requirements
    3. Automatically distribute tasks across CPU/GPU
    4. Monitor and manage system resources
    5. Track task status and results

    Args:
        max_workers (int, optional): Maximum number of CPU workers
        logger (Logger, optional): Custom logger instance
        **kwargs: Additional configuration parameters

    Example:
        >>> queue = QueueManager(max_workers=8)
        >>> queue.add_task(my_function, args=(param1, param2), priority=Priority.HIGH)
        >>> queue.wait_all_task_completion()
    """

    _id_counter = itertools.count(1)

    def __init__(
        self,
        max_workers: Optional[int] = None,
        logger: Optional[Logger] = None,
        **kwargs,
    ):
        # Initialize logging
        if logger:
            self.logger = logger
        else:
            self.logger = Logger()

        self.logger.debug(f"Initialize QueueManager.")

        self.memory_monitor = MemoryMonitor(logger=self.logger)

        # Default CPU allocation
        total_cpu = max_workers or multiprocessing.cpu_count() - 1
        self.cpu_pool = Pool(processes=total_cpu)

        # Single priority queue for CPU tasks
        self.cpu_task = PriorityQueue()
        # Create CPU worker thread
        self.cpu_thread = threading.Thread(target=self._cpu_worker, daemon=True)
        
        # Create GPU process queue
        self.gpu_task = Queue()
        self.gpu_thread = threading.Thread(target=self._gpu_worker, daemon=True)
        
        # Results and error handling
        self.tasks: List[Task] = []
        self.results: List[Any] = []
        self.errors: List[Dict] = []
        self.lock = threading.Lock()

        # Memory tracking
        self.memory_history: List[Dict] = []
        self._memory_history_size = kwargs.pop("memory_history_size", 100)
        self.current_memory_state = MemoryState.HEALTHY

        # Memory tracking
        self.initial_memory = self.memory_monitor.current_memory["used"]
        self.initial_gpu_memory = self.memory_monitor.current_gpu_memory

        self.peak_memory = {"CPU": self.initial_memory, "CPU_TIMESTAMP": datetime.now()}
        for device, stats in self.initial_gpu_memory.items():
            # Initialize max GPU memory for each device
            self.peak_memory[f"GPU_{device}"] = stats["used"]
            self.peak_memory[f"GPU_{device}_TIMESTAMP"] = datetime.now()

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._handle_keyboard_interrupt)
        signal.signal(signal.SIGINT, self._handle_keyboard_interrupt)

        # Optional: Jupyter notebook interrupt handling
        try:
            get_ipython  # Check if running in Jupyter
            from ipykernel.kernelbase import Kernel
            Kernel.raw_interrupt_handler = self._jupyter_interrupt_handler
        except (NameError, ImportError):
            pass

        # Abrupt stop flag
        self._abrupt_stop_requested = threading.Event()

        self.logger.debug(f"{self.memory_monitor.log_memory_usage}")
        self.memory_history.append(
            {
                "timestamp": datetime.now(),
                "memory": self.memory_monitor.current_memory["used"],
                "gpu_memory": [
                    device["used"]
                    for _, device in self.memory_monitor.current_gpu_memory.items()
                ],
                "event": "Initialization",
            }
        )

        self.cpu_thread.start()
        self.gpu_thread.start()
        
        self.logger.debug("QueueManager Initialization complete")

    def __exit__(self, exc_type, exc_val, _):
        """Context manager exit."""
        self.stop_processing()
        if exc_type:
            self.logger.error(f"Error during execution: {exc_val}")
            return False
        return True

    ######### Add task #########
    def add_task(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        priority: Priority = Priority.MEDIUM,
        gpu: bool = False,
        device: int = None,
        task_name: str = None,
    ) -> str:
        """
        Add a task to the processing queue with specified execution parameters.

        Allows flexible task submission with fine-grained control over:
        - Execution function
        - Arguments
        - Priority
        - Processing target (CPU/GPU)

        Args:
            func (Callable): Function to be executed
            args (tuple, optional): Positional arguments for the function
            kwargs (dict, optional): Keyword arguments for the function
            priority (Priority, optional): Task priority level. Defaults to MEDIUM.
            gpu (bool, optional): Execute on GPU. Defaults to False.
            device (int, optional): Specific GPU device. Defaults to 0.
            task_name (str, optional): Descriptive task name

        Returns:
            str: Unique task identifier for tracking

        Example:
            >>> queue.add_task(
            ...     process_data,
            ...     args=(dataset,),
            ...     priority=Priority.HIGH,
            ...     gpu=True
            ... )
        """

        if self._abrupt_stop_requested.is_set():
            self.logger.warning("Cannot add task. Abrupt stop is active.")
            return None

        # Generate a unique task ID
        task_id = f"t{next(self._id_counter)}"

        # Ensure kwargs is a dictionary
        kwargs = kwargs or {}

        # Create the task
        task = Task(
            id=task_id,
            task_name=task_name or func.__name__,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            starttime=datetime.now(),
            endtime=None,
            gpu=gpu,
            device=(
                (device or 0) if gpu else "cpu"
            ),  # Default to device 0 if not specified
        )

        # Add task to the task list
        self.tasks.append(task)

        # Put the task in the appropriate queue
        if gpu:
            self.gpu_task.put(task)
        else:
            self.cpu_task.put(task)

        self.logger.info(
            f"Added {'GPU' if gpu else 'CPU'} task {task.task_name} (id: {task_id}) with priority {priority}"
        )
        time.sleep(0.1)

        self.log_memory_stats(f"Task {task.task_name}(id: {task.id}) submitted")
        return task_id

    ######### Task processing #########
    def _cpu_worker(self):
        """Distribute tasks based on priority and available resources."""
        active_tasks = []
        while True:
            try:
                self._check_abrupt_stop()
                self.manage_memory_state()
                
                # Get the highest priority task from the queue
                active_tasks = [t for t in active_tasks if not t[0].ready()]

                while (self.current_memory_state != MemoryState.EMERGENCY) and (not self.cpu_task.empty()):

                    task = self.cpu_task.get()

                    try:

                        def make_callback(task_id):
                            return lambda result: self._task_callback(task_id, result)

                        # Submit task to appropriate pool
                        
                        # if type(task.args) != tuple:
                        #     task.args = (task.args,)

                        async_result = self.cpu_pool.apply_async(
                            task.func,
                            args=task.args,
                            kwds=task.kwargs,
                            callback=make_callback(task.id)
                        )

                        task.status = "processing"
                        active_tasks.append((async_result, task))
                        
                    except Exception as e:
                        self.logger.error(f"Error processing task {task.id}: {e}")
                        task.status = "failed"
                        task.error = e
                        self.errors.append(e)

                    finally:
                        self.cpu_task.task_done()
                        self.logger.debug(self.log_detailed_memory_report())
                
            except AbruptStopException:
                self.logger.info("CPU worker stopped by abrupt stop.")
                break
            except Exception as e:
                self.logger.error(f"Error in CPU worker: {e}")

            time.sleep(0.1)

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

    def _gpu_worker(self):
        """Worker thread for processing GPU tasks."""

        while True:
            try:
                self._check_abrupt_stop()
                self.manage_memory_state()
                # Get the task from the queue
                if self.current_memory_state != MemoryState.EMERGENCY:
                    task = self.gpu_task.get()
                    try:
                        # Execute the task
                        task.status = "processing"
                        with self.gpu_context(task.device):
                            result = task.func(*task.args, **task.kwargs)

                        self._task_callback(task.id, result)

                    except Exception as e:
                        # Update task with error details
                        task.endtime = datetime.now()
                        task.status = "failed"
                        task.result = None
                        task.error = e

                        # Log the error
                        self.logger.error(f"GPU task {task.id} failed: {e}")

                    finally:
                        self.gpu_task.task_done()
                        self.logger.debug(self.log_detailed_memory_report())

            except AbruptStopException:
                self.logger.info("GPU worker stopped by abrupt stop.")
                break
            except Exception as e:
                self.logger.error(f"Error in GPU worker: {e}")

            time.sleep(0.1)

    def _task_callback(self, task_id: str, result: Any):
        """Callback function for task completion."""
        try:
            with self.lock:
                for task in self.tasks:
                    if task.id == task_id:
                        # Update task status and result atomically
                        task.status = "completed"
                        task.result = result
                        task.endtime = datetime.now()
                        task.error = None
                        
                        break
                else:
                    # If no matching task is found, log a warning
                    self.logger.warning(
                        f"No matching task found for task_id: {task_id}"
                    )

                # Append result to results list
                self.results.append(result)

        except Exception as e:
            # Log and track any errors during task callback
            error_info = {"task_id": task_id, "error": e, "timestamp": datetime.now()}
            self.logger.error(f"Error in task callback for task {task_id}: {e}")
            self.errors.append(error_info)

    def wait_until_task_complete(
        self, task_id: Union[str, List[str]], timeout: Optional[float] = None
    ):
        """
        Wait until the specified task(s) complete or until timeout.

        Args:
            task_id: A single task ID, list of task IDs, or "all" to wait for all tasks
            timeout: Optional timeout in seconds

        Returns:
            True if all tasks completed, False if timeout occurred
        """
        start_time = time.time()

        # Special case for "all" keyword
        if task_id == "all":
            task_id = [task.id for task in self.tasks]

        # Convert single task_id to a list for consistent handling
        if isinstance(task_id, str):
            task_id = [task_id]

        # If no tasks to wait for, return immediately
        if not task_id:
            return True

        while task_id:
            # Check if timeout has occurred
            if timeout is not None and time.time() - start_time > timeout:
                return False

            # Find tasks that are not yet completed
            remaining_tasks = [
                tid for tid in task_id 
                if next((task for task in self.tasks if task.id == tid and task.status != "completed"), None) is not None
            ]

            # If all tasks are completed, exit
            if not remaining_tasks:
                return True

            # Small sleep to prevent tight loop
            if task_id == "all":
                time.sleep(10)
            else:
                time.sleep(1)

        return True

    def stop_processing(self, *args):
        """
        Gracefully stop all task processing.

        Can be used as a signal handler or manually called to halt processing.

        Args:
            *args: Signal arguments (ignored)

        Notes:
            - Closes and terminates all CPU process pool
            - Clears pending tasks in CPU and GPU queues
            - Logs memory usage and shutdown details
            - Provides a clean shutdown mechanism for the task processing system

        Raises:
            Exception: If an error occurs during the shutdown process
        """
        try:
            # Stop all CPU and GPU task processing
            self.cpu_pool.close()
            self.cpu_pool.terminate()

            # Clear task queues
            while not self.cpu_task.empty():
                self.cpu_task.get()
            while not self.gpu_task.empty():
                self.gpu_task.get()

            # Log shutdown details
            self.logger.info("Task processing stopped")
            self.log_memory_stats("Shutdown")

        except Exception as e:
            self.logger.error(f"Error during task processing shutdown: {e}")

    def _check_abrupt_stop(self):
        """
        Check if abrupt stop has been requested.
        
        Raises AbruptStopException if abrupt stop is active.
        Provides a mechanism to gracefully exit long-running tasks.
        """
        if self._abrupt_stop_requested.is_set():
            # Minimal logging for interruption
            self.logger.debug("Task interrupted by abrupt stop mechanism.")
            raise AbruptStopException("Task processing stopped by abrupt stop mechanism.")
    
    def abrupt_stop(self):
        """Abrupt stop processing mechanism."""
        if self._abrupt_stop_requested.is_set():
            os._exit(0)
            return  # Already in abrupt stop process
        
        self._abrupt_stop_requested.set()
        self.logger.warning("Abrupt stop initiated. Clearing task queues and stopping tasks...")

        try:
            # Clear CPU task queue
            while not self.cpu_task.empty():
                try:
                    task = self.cpu_task.get_nowait()
                    self.cpu_task.task_done()
                    if hasattr(task, 'id'):
                        self.logger.debug(f"Removed CPU task {task.id}")
                except Exception:
                    break

            # Clear GPU task queue
            while not self.gpu_task.empty():
                try:
                    task = self.gpu_task.get_nowait()
                    if hasattr(task, 'id'):
                        self.logger.debug(f"Removed GPU task {task.id}")
                except Exception:
                    break

            # Terminate all running task pool
            try:
                self.cpu_pool.terminate()
                self.cpu_pool.join() 
            except Exception as e:
                self.logger.debug(f"Error during pool termination: {e}")

        except Exception as e:
            self.logger.error(f"Error during abrupt stop: {e}")
        finally:
            # Always reset the stop flag
            self._abrupt_stop_requested.clear()
            self.logger.info("Abrupt stop completed. All tasks cleared.")
            os._exit(0)

    def _handle_keyboard_interrupt(self, signum, frame):
        """Handle keyboard interrupt with abrupt stop mechanism."""
        self.logger.warning("Keyboard interrupt detected. Initiating abrupt stop...")
        self.abrupt_stop()

    def _jupyter_interrupt_handler(self, kernel, signum, frame):
        """Custom interrupt handler for Jupyter notebook."""
        self.logger.warning("Jupyter notebook interrupt detected. Initiating abrupt stop...")
        self.abrupt_stop()
        raise KeyboardInterrupt()

    ######### Memory related #########
    def manage_memory_state(self) -> None:
        """Manage memory state considering both CPU and GPU memory."""

        new_state, trigger_source = self.memory_monitor.get_unified_state()

        if new_state != self.current_memory_state:
            self.logger.info(
                f"Memory state changed from {self.current_memory_state.state} to "
                f"{new_state.state} (triggered by {trigger_source})"
            )
            self.logger.warning(f"{self.memory_monitor.log_memory_usage}")
            self.current_memory_state = new_state

            if self.current_memory_state != MemoryState.HEALTHY:
                self.memory_monitor.handle_state(
                    trigger_source=trigger_source,
                    gpu_context=self.gpu_context,
                    stop_callback=self.stop_processing,
                )

    def log_detailed_memory_report(self):
        """Log memory usage report with multiprocessing."""
        # Detailed report for debug level
        log_message = ["Detailed memory usage report:"]
        log_message.extend(
            [
                "System Memory:",
                f"  Initial: {self.initial_memory:.1f} MB",
                f"  Peak: {self.peak_memory['CPU']:.1f} MB ({self.peak_memory['CPU_TIMESTAMP']})",
                f"  Current: {self.memory_monitor.current_memory['used']:.1f} MB ({self.memory_monitor.current_memory_percent:.1f}%)",
                f"  Total: {self.memory_monitor.current_memory['total']:.1f} MB",
            ]
        )

        gpu_stats = (
            self.memory_monitor.current_gpu_memory
        )  # GPU stats still in main process
        if gpu_stats:
            log_message.append("\nGPU Memory:")
            for device, stats in gpu_stats.items():
                log_message.extend(
                    [
                        f"  {device}:",
                        f"    Initial: {self.initial_gpu_memory[device]['used']:.1f} MB",
                        f"    Peak: {self.peak_memory[f'GPU_{device}']:.1f} MB ({self.peak_memory[f'GPU_{device}_TIMESTAMP']})",
                        f"    Current: {stats['used']:.1f} MB ({stats['percent']:.1f} %)",
                        f"    Total: {stats['total']:.1f} MB",
                    ]
                )

        self.logger.debug("\n".join(log_message))

    def log_memory_stats(self, stage: str = None) -> float:
        """
        Update memory statistics and optionally log the current stage.

        Args:
            stage: Optional description of the current processing stage

        Returns:
            float: Current memory usage in MB
        """
        current_memory = self.memory_monitor.current_memory["used"]
        if current_memory > self.peak_memory["CPU"]:
            self.peak_memory["CPU"] = current_memory
            self.peak_memory["CPU_TIMESTAMP"] = datetime.now()

        for device, stats in self.memory_monitor.current_gpu_memory.items():
            if stats["used"] > self.peak_memory[f"GPU_{device}"]:
                self.peak_memory[f"GPU_{device}"] = stats["used"]
                self.peak_memory[f"GPU_{device}_TIMESTAMP"] = datetime.now()

        if stage and self.logger:
            self.logger.debug(f"Memory at {stage}: {current_memory:.2f} MB")

        self.memory_history.append(
            {
                "timestamp": datetime.now(),
                "memory": current_memory,
                "gpu_memory": [
                    gpu["used"]
                    for _, gpu in self.memory_monitor.current_gpu_memory.items()
                ],
                "event": stage,
            }
        )

        if len(self.memory_history) > self._memory_history_size:
            self.memory_history.pop(0)

        self.logger.debug(f"Memory at {stage}: {self.memory_monitor.log_memory_usage}")

    def plot_memory_history(self):
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # Extract data
        data = self.memory_history

        timestamps = [entry["timestamp"] for entry in data]
        memory_usage = [entry["memory"] for entry in data]
        plt.plot(timestamps, memory_usage, label="CPU Memory (MB)", marker="o")
        for i in range(len(data[0]["gpu_memory"])):
            gpu_memory = [entry["gpu_memory"][i] for entry in data]
            plt.plot(
                timestamps, gpu_memory, label=f"GPU device_{i} Memory (MB)", marker="s"
            )

        # Format timestamps on x-axis
        plt.gca().xaxis.set_major_formatter(
            mdates.DateFormatter("%H:%M:%S")
        )  # Show HH:MM:SS format
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  # Auto spacing

        # Formatting
        plt.xlabel("Timestamp", fontsize=13)
        plt.ylabel("Memory Usage (MB)", fontsize=13)
        plt.title("Memory Usage Over Time", fontsize=13)
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True)

        # Show the plot
        plt.show()
