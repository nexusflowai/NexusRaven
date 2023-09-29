def no_print(*_, **__) -> None:
    pass


__builtins__["print"] = no_print
