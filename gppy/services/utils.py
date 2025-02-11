import gc
import cupy as cp


class classmethodproperty:
    """
    A custom decorator that combines class method and property behaviors.

    Allows creating class-level properties that can be accessed 
    without instantiating the class, while maintaining the 
    flexibility of class methods.

    Typical use cases:
    - Generating computed class-level attributes
    - Providing dynamic class-level information
    - Implementing lazy-loaded class properties

    Attributes:
        func (classmethod): The underlying class method

    Example:
        class Example:
            @classmethodproperty
            def dynamic_property(cls):
                return compute_something_for_class()
    """

    def __init__(self, func):
        """
        Initialize the classmethodproperty decorator.

        Args:
            func (callable): The function to be converted to a class method property
        """
        self.func = classmethod(func)
    
    def __get__(self, instance, owner):
        """
        Retrieve the value of the class method property.

        Args:
            instance: The instance calling the property (ignored)
            owner: The class on which the property is defined

        Returns:
            The result of calling the class method
        """
        return self.func.__get__(instance, owner)()


def cleanup_memory() -> None:
    """
    Perform comprehensive memory cleanup across CPU and GPU.

    This function provides a robust mechanism for releasing 
    unused memory resources in both CPU and GPU contexts.

    Key Operations:
    - Trigger Python garbage collection
    - Free CuPy default memory pool blocks
    - Free CuPy pinned memory pool blocks
    - Perform a second garbage collection to ensure complete memory release

    Ideal for:
    - Preventing memory leaks
    - Managing memory in long-running scientific computing tasks
    - Preparing for memory-intensive operations

    Notes:
    - Calls garbage collection twice to ensure thorough cleanup
    - Uses CuPy's memory pool management for GPU memory
    - Minimal performance overhead

    Example:
        >>> # Before starting a memory-intensive task
        >>> cleanup_memory()
    """
    gc.collect()  # Initial garbage collection
    cp.get_default_memory_pool().free_all_blocks()  # Free GPU memory pool
    cp.get_default_pinned_memory_pool().free_all_blocks()  # Free pinned memory
    gc.collect()  # Ensure all GPU memory is freed