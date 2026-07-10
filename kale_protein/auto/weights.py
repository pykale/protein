from hashlib import sha256
from pathlib import Path
from urllib.request import urlretrieve


def resolve_pretrained_weight(config, downloader=urlretrieve):
    block = config.get("pretrained", {})
    config_dir = Path(config.get("_config_dir", "."))
    local_dir = config_dir / block.get("local_dir", "weights")
    filename = block.get("filename")
    if not filename:
        raise ValueError(_missing_weight_message(config))

    weight_path = local_dir / filename
    if weight_path.exists():
        _verify_checksum(weight_path, block.get("sha256"))
        return weight_path

    url = block.get("url")
    if not _valid_url(url):
        raise ValueError(_missing_weight_message(config))

    local_dir.mkdir(parents=True, exist_ok=True)
    downloader(url, weight_path)
    _verify_checksum(weight_path, block.get("sha256"))
    return weight_path


def _valid_url(url):
    return isinstance(url, str) and url.startswith(("https://", "http://"))


def _verify_checksum(path, expected):
    if not expected:
        return
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"Checksum mismatch for {path}: expected {expected}, got {digest}")


def _missing_weight_message(config):
    model_id = config.get("model_id", config.get("name", "this model"))
    return (
        f"No pretrained weight file is available for {model_id}. "
        "Place the expected file in this model card's weights/ folder, provide a valid URL in config.yaml, "
        "or train the model yourself with pretrain=False."
    )
