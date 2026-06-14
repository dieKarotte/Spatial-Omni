from __future__ import annotations


class TransformParams:
    """
    Placeholder for UFB transform parameters.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind

    @classmethod
    def RaisedSine(cls) -> "TransformParams":
        return cls("raised_sine")


__all__ = ["TransformParams"]
