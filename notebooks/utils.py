import math
import urllib.request
from collections.abc import Sequence
from pathlib import Path
import yaml # type: ignore[import-untyped]
from box import Box  # type: ignore[import-not-found]
from claymodel.module import ClayMAEModule  # type: ignore[import-not-found, import-untyped]
from huggingface_hub import hf_hub_download
import cv2  # type: ignore[import-not-found]

import numpy as np
import torch
from PIL import Image


SEED = 42
N_PER_CLASS = 500 # examples per class from EuroSAT

# The 10 Sentinel-2 L2A bands Clay v1.5 was trained on.
CLAY_BANDS: list[str] = [
    "blue", # B02
    "green", # B03
    "red", # B04
    "rededge1", # B05
    "rededge2", # B06
    "rededge3", # B07
    "nir", # B08
    "nir08", # B8A
    "swir16", # B11
    "swir22", # B12
]

# EuroSAT MSI stores 13 bands. Indices below select the 10 Clay bands from that ordering.
EUROSAT_MSI_CLAY_INDICES: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]


def ensure_metadata() -> Path:
    """Return path to Clay metadata.yaml, downloading and caching if needed."""
    metadata_url = "https://raw.githubusercontent.com/Clay-foundation/model/main/configs/metadata.yaml"
    metadata_local = Path(__file__).parent.parent / "configs" / "metadata.yaml"

    if metadata_local.exists():
        return metadata_local
    metadata_local.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Clay metadata.yaml -> {metadata_local}")
    urllib.request.urlretrieve(metadata_url, metadata_local)
    return metadata_local


def get_clay_metadata() -> Box:
    """Return Clay metadata as a Box object."""
    return Box(yaml.safe_load(ensure_metadata().read_text()))


def get_s2_norm_params() -> tuple[np.ndarray, np.ndarray]:
    """
    Return (mean, std) arrays of shape [10, 1, 1] for Clay's 10 Sentinel-2 L2A bands in CLAY_BANDS order, as read from configs/metadata.yaml.
    """
    bands = get_clay_metadata()["sentinel-2-l2a"].bands
    mean = np.array([bands.mean[b] for b in CLAY_BANDS], dtype=np.float32)
    std  = np.array([bands.std[b] for b in CLAY_BANDS], dtype=np.float32)
    return mean.reshape(-1, 1, 1), std.reshape(-1, 1, 1)


def get_s2_wavelengths() -> list[float]:
    """Return wavelengths (micrometers) for Clay's 10 S2 bands in CLAY_BANDS order."""
    bands = get_clay_metadata()["sentinel-2-l2a"].bands
    return [float(bands.wavelength[b]) for b in CLAY_BANDS]


def msi_to_rgb(arr: np.ndarray, scale: float = 3000.0) -> np.ndarray:
    """
    Convert a [10, H, W] or [13, H, W] uint16 MSI array to a [H, W, 3] float32 RGB image suitable for matplotlib imshow.

    For a 10-band array (Clay band order), picks bands at CLAY_RGB_INDICES. For a 13-band array (EuroSAT MSI order), first selects Clay bands.

    Sentinel-2 reflectance is stored as integers scaled by 10000 (full reflectance = 10000). `scale` is the value mapped to white: pixels are divided by it and clipped to [0, 1]. The default 3000 maps 30% reflectance to white, which produces natural-looking brightness for most land surfaces. Could use a higher value (e.g. 10000) for true-reflectance normalization at the cost of a darker image. 
    """
    if arr.shape[0] == 13:
        arr = arr[EUROSAT_MSI_CLAY_INDICES]

    # Within the selected 10-band array, these indices give R, G, B for display. Clay order: blue=0, green=1, red=2 -> RGB = [2, 1, 0]
    clay_rgb_indices: list[int] = [2, 1, 0]
    rgb = arr[clay_rgb_indices].astype(np.float32)
    rgb = np.clip(rgb / scale, 0.0, 1.0)
    return rgb.transpose(1, 2, 0)  # [H, W, 3]


def load_clay_model() -> ClayMAEModule:
    """Download (if needed) and return Clay v1.5 in eval mode."""
    metadata_path = str(ensure_metadata())
    ckpt_path = hf_hub_download(
        repo_id="made-with-clay/Clay",
        filename="v1.5/clay-v1.5.ckpt",
        repo_type="model",
    )
    clay = ClayMAEModule.load_from_checkpoint(
        ckpt_path,
        metadata_path=metadata_path,
        map_location="cpu",
    )
    clay.eval()
    return clay  # type: ignore[return-value]


def make_datacube(
    chips_np: np.ndarray,
    lat_deg: float,
    lon_deg: float,
    week: int,
    hour: int,
    waves: list[float],
) -> dict[str, object]:
    """Build a Clay datacube dict from a [B, C, H, W] float32 array."""
    B = chips_np.shape[0]
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    t_enc = [
        math.sin(2 * math.pi * week / 52), math.cos(2 * math.pi * week / 52),
        math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
    ]
    ll_enc = [math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)]
    return {
        "pixels": torch.tensor(chips_np, dtype=torch.float32),
        "time": torch.tensor([t_enc] * B, dtype=torch.float32),
        "latlon": torch.tensor([ll_enc] * B, dtype=torch.float32),
        "waves": torch.tensor(waves, dtype=torch.float32),
        "gsd": torch.tensor(10.0), # Sentinel-2 native GSD (m/px)
        "platform": ["sentinel-2-l2a"] * B,
    }


def clay_encode(
    array_list: Sequence[np.ndarray | Image.Image],
    clay_model: ClayMAEModule,
    img_size: int,
    batch_size: int,
    lat: float,
    lon: float,
    week: int,
    hour: int,
) -> np.ndarray:
    """
    Encode a list of images with Clay. Returns np.ndarray of shape [N, 1024], CLS tokens.

    Accepts two input formats:
    - np.ndarray of shape [13, H, W] uint16 (EuroSAT MSI): Selects the 10 Clay bands, normalises directly using Clay's mean/std from metadata.yaml. No scale factor needed -> MSI values are already in the same reflectance range as Clay's training data.
    - PIL Image (3-band RGB, e.g. from STAC visual asset): Scales uint8 (0-255) to approximate S2 reflectance with a factor of 4000/255, then normalises using Clay's RGB band stats. This path retains the approximation inherent to uint8 RGB data.
    """
    s2_mean_10, s2_std_10 = get_s2_norm_params() # [10, 1, 1]
    s2_mean_rgb = s2_mean_10[[2, 1, 0]] # red, green, blue
    s2_std_rgb  = s2_std_10[[2, 1, 0]]
    waves_10 = get_s2_wavelengths() # 10 wavelengths
    waves_rgb = [waves_10[2], waves_10[1], waves_10[0]] # red, green, blue

    def to_chip_msi(x: np.ndarray) -> tuple[np.ndarray, list[float]]:
        """[13, H, W] or [H, W, 13] uint16 -> [10, H, W] float32, normalised."""
        if x.ndim == 3 and x.shape[-1] == 13:
            x = x.transpose(2, 0, 1)
        arr = x[EUROSAT_MSI_CLAY_INDICES].astype(np.float32) # [10, H, W]
        if arr.shape[1] != img_size or arr.shape[2] != img_size:
            arr = np.stack([cv2.resize(arr[b], (img_size, img_size)) for b in range(arr.shape[0])])
        return (arr - s2_mean_10) / s2_std_10, waves_10

    def to_chip_rgb(x: Image.Image) -> tuple[np.ndarray, list[float]]:
        """PIL RGB uint8 -> [3, H, W] float32, normalised (approximate)."""
        arr = np.array(x.convert("RGB").resize((img_size, img_size))) # [H, W, 3]
        arr_f = arr.astype(np.float32) * (4000.0 / 255.0) # approx reflectance
        arr_f = arr_f.transpose(2, 0, 1) # [3, H, W]
        return (arr_f - s2_mean_rgb) / s2_std_rgb, waves_rgb

    all_cls = []
    for start in range(0, len(array_list), batch_size):
        batch = array_list[start : start + batch_size]
        chips_and_waves = [
            to_chip_msi(x) if isinstance(x, np.ndarray) else to_chip_rgb(x) # type: ignore[arg-type]
            for x in batch
        ]
        chips = np.stack([c for c, _ in chips_and_waves])
        waves = chips_and_waves[0][1] # all items in a batch share the same format
        datacube = make_datacube(
            chips_np=chips, 
            lat_deg=lat, 
            lon_deg=lon, 
            week=week, 
            hour=hour, 
            waves=waves
        )
        with torch.no_grad():
            enc_out = clay_model.model.encoder(datacube)
            cls = enc_out[0][:, 0, :].cpu().numpy() # [B, 1024]
        all_cls.append(cls)
        print(
            f"{min(start + batch_size, len(array_list))}/{len(array_list)}", end="\r",
        )
    return np.vstack(all_cls).astype(np.float32)


def save_embedding_cache(
    cache_path: str | Path,
    emb: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> None:
    """Save EuroSAT Clay embeddings."""
    torch.save({"emb": emb, "labels": labels, "class_names": class_names}, cache_path)
    print(f"Saved: {cache_path} shape={emb.shape}")


def load_embedding_cache(
    cache_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load EuroSAT Clay embeddings from disk. Returns (emb [N, 1024], labels [N], class_names).
    """
    d = torch.load(cache_path, map_location="cpu", weights_only=False)
    emb = np.array(d["emb"], dtype=np.float32)
    labels = np.array(d["labels"], dtype=np.int64)
    class_names: list[str] = d["class_names"]
    print(f"Loaded cache: {emb.shape}")
    return emb, labels, class_names
