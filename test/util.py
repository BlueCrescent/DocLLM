from typing import Callable, List, ParamSpec, Tuple, TypeVar

Params = ParamSpec("Params")
Ret = TypeVar("Ret")


def extract_return_value(fct: Callable[Params, Ret]) -> Tuple[List[Ret], Callable[Params, Ret]]:
    result_list = []

    def new_func(*args: Params.args, **kwargs: Params.kwargs) -> Ret:
        res = fct(*args, **kwargs)
        result_list.append(res)
        return res

    return result_list, new_func
