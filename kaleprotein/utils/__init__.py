"""Step-scoped helper functions shared across KaleProtein operations."""

from importlib import import_module


_IMPORT_STRUCTURE = {
    "MetricUndefinedError": (
        ".evaluate_extract_binary_inputs",
        "MetricUndefinedError",
    ),
    "build_sequence_targets": (
        ".evaluate_build_sequence_targets",
        "build_sequence_targets",
    ),
    "extract_binary_inputs": (
        ".evaluate_extract_binary_inputs",
        "extract_binary_inputs",
    ),
    "extract_checkpoint_state_dict": (
        ".model_load_checkpoint_state_dict",
        "extract_checkpoint_state_dict",
    ),
    "extract_sequences": (".evaluate_extract_sequences", "extract_sequences"),
    "load_checkpoint_state_dict": (
        ".model_load_checkpoint_state_dict",
        "load_checkpoint_state_dict",
    ),
    "move_to_device": (".model_move_to_device", "move_to_device"),
    "parse_mmcif": (".loaddata_parse_mmcif", "parse_mmcif"),
    "parse_pdb": (".loaddata_parse_pdb", "parse_pdb"),
    "read_csv": (".loaddata_read_dict_rows", "read_csv"),
    "read_dict_rows": (".loaddata_read_dict_rows", "read_dict_rows"),
    "read_fasta": (".loaddata_read_fasta", "read_fasta"),
    "read_tsv": (".loaddata_read_dict_rows", "read_tsv"),
    "require_file": (".loaddata_require_file", "require_file"),
    "safe_path_part": (".loaddata_safe_path_part", "safe_path_part"),
    "validate_binary_inputs": (
        ".evaluate_extract_binary_inputs",
        "validate_binary_inputs",
    ),
    "verify_checksum": (".model_verify_checksum", "verify_checksum"),
}

__all__ = list(_IMPORT_STRUCTURE)


def __getattr__(name):
    try:
        module_name, object_name = _IMPORT_STRUCTURE[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), object_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted((*globals(), *__all__))
