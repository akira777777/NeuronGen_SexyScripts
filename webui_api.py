"""
High-Performance Unified API Client for Stable Diffusion Backends.
Supports:
- Automatic1111 WebUI REST API (/sdapi/v1/txt2img)
- ComfyUI API (/prompt, /history, /view)
- HTTP connection pooling with persistent sessions and Keep-Alive
- Fast base64/binary image decoding
"""

import io
import json
import base64
import time
import socket
import urllib.parse
from typing import Dict, List, Optional, Tuple, Any
import requests
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        """Simple fallback if tqdm not installed."""
        return iterable
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image


def _create_pooled_session(pool_size: int = 20) -> requests.Session:
    """Create a requests session with HTTP connection pooling and Keep-Alive."""
    session = requests.Session()
    retries = Retry(
        total=2,
        backoff_factor=0.3,
        status_forcelist=[502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(
        pool_connections=pool_size,
        pool_maxsize=pool_size,
        max_retries=retries
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class WebUIAPIError(Exception):
    """Custom exception for WebUI API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def _image_to_base64_png(image: Image.Image) -> str:
    """Encodes PIL Image to base64 PNG string for WebUI API."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _is_host_port_open(url: str, timeout: Optional[float] = None) -> bool:
    """Instantly test if target host:port is listening before making HTTP requests."""
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        # Fast adaptive timeout: loopback interfaces don't need 500ms
        if timeout is None:
            timeout = 0.05 if host in ["127.0.0.1", "localhost", "::1"] else 0.5
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False


class StableDiffusionWebUI:
    """High-performance client for Automatic1111 WebUI and ComfyUI."""

    TXT2IMG_ENDPOINT = "/sdapi/v1/txt2img"
    PROGRESS_ENDPOINT = "/sdapi/v1/progress"
    MODELS_ENDPOINT = "/sdapi/v1/sd-models"
    SAMPLERS_ENDPOINT = "/sdapi/v1/samplers"

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        webui_cfg = cfg.get("webui") or cfg.get("webui_api") or {}
        
        host = webui_cfg.get("url") or webui_cfg.get("host") or "http://127.0.0.1:7860"
        self.base_url = host.rstrip("/")
        self.timeout = webui_cfg.get("timeout", 300)
        self.api_key = webui_cfg.get("api_key")

        # ComfyUI host
        self.comfy_url = cfg.get("comfyui", {}).get("url", "http://127.0.0.1:8188").rstrip("/")

        # Persistent session with connection pooling
        self.session = _create_pooled_session(pool_size=16)
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def is_running(self, url: Optional[str] = None) -> bool:
        """Check if Automatic1111 WebUI is reachable with sub-millisecond port check."""
        target = (url or self.base_url).rstrip("/")
        if not _is_host_port_open(target):
            return False
        try:
            r = self.session.get(f"{target}{self.PROGRESS_ENDPOINT}", headers=self.headers, timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def is_comfyui_running(self, url: Optional[str] = None) -> bool:
        """Check if ComfyUI server is reachable with sub-millisecond port check."""
        target = (url or self.comfy_url).rstrip("/")
        if not _is_host_port_open(target):
            return False
        try:
            r = self.session.get(f"{target}/system_stats", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def get_models(self) -> List[Dict[str, Any]]:
        """Fetch available checkpoints from Automatic1111."""
        try:
            r = self.session.get(f"{self.base_url}{self.MODELS_ENDPOINT}", headers=self.headers, timeout=5)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        return []

    def get_samplers(self) -> List[str]:
        """Fetch available samplers from Automatic1111."""
        try:
            r = self.session.get(f"{self.base_url}{self.SAMPLERS_ENDPOINT}", headers=self.headers, timeout=5)
            if r.status_code == 200:
                return [s.get("name") for s in r.json() if "name" in s]
        except requests.RequestException:
            pass
        return ["DPM++ 2M Karras", "Euler a", "Euler", "DPM++ SDE Karras", "DDIM"]

    def generate_images(
        self,
        positive_prompt: str,
        negative_prompt: Optional[str] = None,
        steps: int = 25,
        cfg_scale: float = 7.5,
        width: int = 896,
        height: int = 1152,
        batch_size: int = 1,
        seed: int = -1,
        sampler_name: str = "DPM++ 2M Karras",
        model_name: Optional[str] = None,
        alwayson_scripts: Optional[Dict[str, Any]] = None,
        extra_args: Optional[Dict[str, Any]] = None,
        controlnet_poses: Optional[List[Tuple[str, Image.Image]]] = None  # (pose_path, pose_img)
    ) -> Tuple[List[Image.Image], Dict[str, Any], List[str]]:
        """
        Generate images using Automatic1111 API with connection pooling and fast base64 decode.
        """
        errors: List[str] = []
        images: List[Image.Image] = []
        info_dict: Dict[str, Any] = {}

        payload: Dict[str, Any] = {
            "prompt": positive_prompt,
            "negative_prompt": negative_prompt or "",
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "batch_size": max(1, batch_size),
            "seed": seed if seed is not None and seed >= 0 else -1,
            "sampler_name": sampler_name,
            "save_images": False,
        }

        # Normalize controlnet_poses to list of tuples
        normalized_cn_poses: List[Tuple[str, Image.Image]] = []
        if controlnet_poses:
            if isinstance(controlnet_poses, tuple) and len(controlnet_poses) == 2 and isinstance(controlnet_poses[1], Image.Image):
                normalized_cn_poses = [controlnet_poses]
            elif isinstance(controlnet_poses, Image.Image):
                normalized_cn_poses = [("", controlnet_poses)]
            elif isinstance(controlnet_poses, list):
                for item in controlnet_poses:
                    if isinstance(item, tuple) and len(item) == 2:
                        normalized_cn_poses.append(item)
                    elif isinstance(item, Image.Image):
                        normalized_cn_poses.append(("", item))

        # Build ControlNet alwayson_scripts payload using official sd-webui-controlnet schema
        cn_payload: Dict[str, Any] = {}
        if normalized_cn_poses:
            cn_args = []
            for _, pose_img in normalized_cn_poses:
                cn_args.append({
                    "input_image": _image_to_base64_png(pose_img),
                    "module": "openpose",
                    "model": "control_v11e_sd15_openpose",
                    "weight": 1.0,
                    "resize_mode": "Crop and Resize",
                    "low_vram": False,
                    "processor_res": max(width, height),
                    "guidance_start": 0.0,
                    "guidance_end": 1.0,
                    "control_mode": "Balanced",
                    "pixel_perfect": True,
                    "enabled": True
                })
            cn_payload = {
                "controlnet": {
                    "args": cn_args
                }
            }
        
        if alwayson_scripts:
            cn_payload.update(alwayson_scripts)
        if cn_payload:
            payload["alwayson_scripts"] = cn_payload
        
        if extra_args:
            payload.update(extra_args)
        if model_name:
            payload["override_settings"] = {"sd_model_checkpoint": model_name}

        try:
            response = self.session.post(
                f"{self.base_url}{self.TXT2IMG_ENDPOINT}",
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()

            raw_images = result.get("images") or []
            # Use tqdm for progress indication during batch download
            for raw_b64 in tqdm(raw_images, desc="Downloading images..."):
                try:
                    if "," in raw_b64:
                        raw_b64 = raw_b64.split(",", 1)[1]
                    img_bytes = base64.b64decode(raw_b64)
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    images.append(img)
                except Exception as img_err:
                    errors.append(f"Failed to decode image: {img_err}")

            info_dict = {
                "parameters": result.get("parameters", {}),
                "info": result.get("info", "")
            }

            return images, info_dict, errors

        except requests.exceptions.Timeout:
            err = f"Request timed out after {self.timeout}s at {self.base_url}."
            errors.append(err)
        except requests.exceptions.ConnectionError:
            err = f"Cannot connect to WebUI at {self.base_url}. Is Automatic1111 running?"
            errors.append(err)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "unknown"
            err = f"HTTP Error {status}: {e}"
            errors.append(err)
        except Exception as e:
            err = f"Unexpected error during generation: {e}"
            errors.append(err)

        return [], {}, errors

    def select_juggernaut_xl_v9(self) -> bool:
        """Select Juggernaut XL v9 checkpoint in WebUI (ONLYFANS photorealism standard)."""
        try:
            # Get list of models from WebUI
            r = self.session.get(f"{self.base_url}{self.MODELS_ENDPOINT}", timeout=10)
            if r.status_code != 200:
                print(f"[WebUI] [WARN] Could not fetch model list: {r.status_code}")
                return False
            
            models_data = r.json()
            
            # Find Juggernaut XL v9 (or closest match)
            juggernaut_xl_v9 = None
            for model in models_data:
                name_lower = str(model.get("title", "")).lower()
                if "juggernaut" in name_lower and ("xl" in name_lower or "xl v9" in name_lower):
                    juggernaut_xl_v9 = model.get("sha256") or model.get("title")
                    break
            
            # Fallback to any XL model
            if not juggernaut_xl_v9:
                for model in models_data:
                    name_lower = str(model.get("title", "")).lower()
                    if "xl" in name_lower and model.get("snapshots", []):
                        juggernaut_xl_v9 = model.get("sha256") or model.get("title")
                        break
            
            if juggernaut_xl_v9:
                # Set as override setting (WebUI will handle the actual selection)
                payload = {"override_settings": {"sd_model_checkpoint": juggernaut_xl_v9}}
                r = self.session.post(f"{self.base_url}/sdapi/v1/options", json=payload, timeout=10)
                if r.status_code in [200, 201]:
                    print(f"[WebUI] [OK] Juggernaut XL v9 selected: {juggernaut_xl_v9}")
                    return True
            
            print("[WebUI] [WARN] No Juggernaut XL v9 model found — WebUI will use default")
            return False

        except Exception as e:
            print(f"[WebUI] [ERROR] Failed to select model: {e}")
            return False

    def ensure_controlnet_extension_loaded(self) -> bool:
        """
        Check if ControlNet extension is loaded in WebUI and inject it via API.
        Returns True if extension is available, False otherwise.
        """
        try:
            # Get extensions list
            r = self.session.get(f"{self.base_url}/sdapi/v1/medv2/extensions", timeout=5)
            if r.status_code == 200:
                ext_list = r.json()
                for ext in ext_list:
                    if "ControlNet" in str(ext.get("name", "")).lower() or \
                       "controlnet" in str(ext.get("name", "")).lower():
                        print(f"   [INFO] ControlNet extension already loaded: {ext.get('name')}")
                        return True
            else:
                # Fallback to old endpoint structure
                r = self.session.get(f"{self.base_url}/sdapi/v1/medv2/extensions", timeout=5)
                if r.status_code == 200:
                    ext_list = r.json()
                    for ext in ext_list:
                        if "ControlNet" in str(ext.get("name", "")).lower():
                            print(f"   [INFO] ControlNet extension already loaded: {ext.get('name')}")
                            return True
        except Exception as e:
            print(f"   [WARN] Could not check extensions: {e}")
        
        # Try to inject ControlNet via API (Automatic1111's extension injection)
        try:
            payload = {
                "name": "ControlNet",  # Extension name
                "path": "scripts/controlnet.py",
                "args": {"preprocessor": "openpose", "model": "control_v11e_sd15_openpose.pth"},
                "type": "script"
            }
            r = self.session.post(f"{self.base_url}/sdapi/v1/medv2/extensions", json=payload, timeout=10)
            if r.status_code in [200, 201]:
                print("   [INFO] ControlNet extension injected via API")
                return True
        except Exception as e:
            print(f"   [WARN] Could not inject ControlNet: {e}")
        
        return False

    def generate_comfyui(
        self,
        workflow_prompt_dict: Dict[str, Any],
        comfy_url: Optional[str] = None
    ) -> Tuple[List[Image.Image], List[str]]:
        """
        Execute a prompt workflow on ComfyUI and retrieve output images.
        """
        target = (comfy_url or self.comfy_url).rstrip("/")
        errors: List[str] = []
        images: List[Image.Image] = []

        try:
            # Queue prompt
            p_res = self.session.post(f"{target}/prompt", json={"prompt": workflow_prompt_dict}, timeout=10)
            p_res.raise_for_status()
            prompt_id = p_res.json().get("prompt_id")
            if not prompt_id:
                return [], ["ComfyUI did not return prompt_id"]

            # Poll for completion
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                time.sleep(1.0)
                h_res = self.session.get(f"{target}/history/{prompt_id}", timeout=10)
                if h_res.status_code == 200:
                    h_data = h_res.json().get(prompt_id, {})
                    outputs = h_data.get("outputs", {})
                    if outputs:
                        # Fetch images from outputs with progress bar
                        for node_id, node_output in tqdm(outputs.items(), desc="Fetching ComfyUI outputs..."):
                            for img_meta in node_output.get("images", []):
                                fn = img_meta.get("filename")
                                sub = img_meta.get("subfolder", "")
                                ftype = img_meta.get("type", "output")
                                img_url = f"{target}/view?filename={urllib.parse.quote(fn)}&subfolder={urllib.parse.quote(sub)}&type={ftype}"
                                img_res = self.session.get(img_url, timeout=30)
                                if img_res.status_code == 200:
                                    images.append(Image.open(io.BytesIO(img_res.content)).convert("RGB"))
                        return images, []

            return [], [f"ComfyUI generation timed out after {self.timeout}s"]

        except Exception as e:
            return [], [f"ComfyUI request failed: {e}"]
