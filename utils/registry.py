from typing import Callable, Dict, TypeVar, Generic


class ConfigOutline:
    """base class for configs"""

    @classmethod
    def get_resolver(cls, name: str) -> Callable[[Dict[str, any]], Dict[str, any]]:

        # Resolve correct types for given config values.
        def config_resolver(config_values: Dict[str, any]) -> Dict[str, any]:
            # We choose to only pick out parameters listed in the config outline
            # and therefore silently ignore additional parameters in the given
            # config_values.
            wrapper_config = {}
            for _cls in cls.__mro__:
                if "__annotations__" not in _cls.__dict__:
                    continue
                for param in _cls.__annotations__.keys():
                    if param not in config_values:
                        raise Exception(
                            f"Did you forget to add '{param}' in the {name} YAML file? "
                        )
                    value = config_values[param]
                    try:
                        wrapper_config[param] = (
                            None
                            if value is None
                            else _cls.__annotations__[param](value)
                        )
                    except:
                        raise Exception(
                            f"Expected type {_cls.__annotations__[param]} for parameter {param} but got {value}."
                        )
            return wrapper_config

        return config_resolver


from typing import TypeVar, Generic, Callable, overload

T = TypeVar("T")


class _TypedProperty(Generic[T]):
    """A property descriptor that preserves return type for static analysis."""

    def __init__(self, func: Callable[..., T]) -> None:
        attr_name = f"_{func.__name__}"
        self.__doc__ = func.__doc__

        def getter(self_inner: object) -> T:
            return getattr(self_inner, attr_name)  # type: ignore[no-any-return]

        def setter(self_inner: object, value: T) -> None:
            setattr(self_inner, attr_name, value)

        self._prop = property(getter, setter, doc=func.__doc__)

    @overload
    def __get__(self, obj: None, objtype: type) -> "_TypedProperty[T]": ...
    @overload
    def __get__(self, obj: object, objtype: type) -> T: ...

    def __get__(self, obj: object | None, objtype: type) -> "T | _TypedProperty[T]":
        if obj is None:
            return self
        return self._prop.fget(obj)  # type: ignore[misc]

    def __set__(self, obj: object, value: T) -> None:
        self._prop.fset(obj, value)  # type: ignore[misc]


def auto_property(func: Callable[..., T]) -> _TypedProperty[T]:
    """Automatically create a typed getter and setter from a stub function."""
    return _TypedProperty(func)
