from pathlib import Path

from utils.llama_3_model_download import ensure_model_downloaded


class Llama3Instruct8BWrapper:
    model_name = "llama_3-8B"

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir

    def ensure_local_model_dir(self) -> Path:
        model_dir = ensure_model_downloaded(self.model_name)
        if self.base_dir is None:
            return model_dir
        return Path(self.base_dir).resolve() / model_dir.name


class Llama31Instruct8BWrapper:
    model_name = "llama_3.1-8B"

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir

    def ensure_local_model_dir(self) -> Path:
        model_dir = ensure_model_downloaded(self.model_name)
        if self.base_dir is None:
            return model_dir
        return Path(self.base_dir).resolve() / model_dir.name
