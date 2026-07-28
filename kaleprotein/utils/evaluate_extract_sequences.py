"""Extract generated or reference sequences from metric inputs."""


def extract_sequences(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        if "sequences" in value:
            return list(value["sequences"])
        if "sequence" in value:
            return [value["sequence"]]
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(extract_sequences(item))
        return result
    return []


__all__ = ["extract_sequences"]
