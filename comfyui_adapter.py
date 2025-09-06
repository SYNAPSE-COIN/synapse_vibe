# Adapter for ComfyUI with a custom Python client
# Lets you control the interface from Python without exporting JSON manually
# Handy for iterative runs, prototyping, and debugging
# --------------------------------------------------------------------------------

import asyncio
import io
import json
import logging
import uuid
from pathlib import Path
from typing import Dict, Any, Union, List
from urllib.error import HTTPError
import urllib.request
from PIL import Image
import numpy as np
from websocket import WebSocket, WebSocketTimeoutException


# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('comfyui')

class ComfyUIBridge:
    def __init__(self, server_address: str = "127.0.0.1:8188"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        self.ws = None
        self.freed = False

    def init_ws(self):
        self.client_id = str(uuid.uuid4())
        self.ws = WebSocket()
        self.ws.connect(f"ws://{self.server_address}/ws?clientId={self.client_id}")
        self.ws.timeout = 1
        self.freed = False

    def release(self):
        self.freed = True

    def post_json(self, url: str, data: Dict = None, verbose: bool = False) -> Dict:
        data = data or {}
        data['verbose'] = verbose
        data['client_id'] = self.client_id
        encoded = json.dumps(data).encode('utf-8')

        endpoint = f'http://{self.server_address}{url}' if not url.startswith('http') else url

        req = urllib.request.Request(
            endpoint,
            headers={'Content-Type': 'application/json'},
            data=encoded
        )

        try:
            with urllib.request.urlopen(req) as response:
                return json.loads(response.read())
        except HTTPError as e:
            log.error(f"HTTP Error: {str(e)}")
            log.error("- Is the server running?")
            log.error("- Is the uiapi extension loaded?")
            return None

    async def query_fields(self, verbose: bool = False) -> Dict:
        response = self.post_json('/uiapi/query_fields', verbose=verbose)
        if isinstance(response, dict):
            return response.get('result') or response.get('response')
        raise ValueError("Invalid response from server")

    def fetch_field(self, path_or_paths: Union[str, List[str]], verbose: bool = False) -> Union[Any, Dict[str, Any]]:
        single = isinstance(path_or_paths, str)
        paths = [path_or_paths] if single else path_or_paths

        response = self.post_json('/uiapi/get_fields', {"fields": paths}, verbose=verbose)

        if isinstance(response, dict):
            result = response.get('result') or response.get('response')
            return result[path_or_paths] if single else result
        raise ValueError("Invalid response from server")

    def set(self, path_or_fields: Union[str, List[tuple]], value: Any = None, verbose: bool = False):
        fields = [(path_or_fields, value)] if isinstance(path_or_fields, str) else path_or_fields
        queue = []

        for path, val in fields:
            if isinstance(val, (Image.Image, np.ndarray)):
                self.set_image(path, val, verbose)
            else:
                queue.append([path, val])

        if queue:
            return self.post_json('/uiapi/set_fields', {"fields": queue}, verbose=verbose)

    def set_image(self, path: str, value: Union[Image.Image, np.ndarray], verbose: bool = False):
        api = str(uuid.uuid4())
        img_name = f'uiapi_{api}.png'
        self.set(path, img_name, verbose)

        if isinstance(value, Image.Image):
            buf = io.BytesIO()
            value.save(buf, format='PNG')
            img_bytes = buf.getvalue()
        elif isinstance(value, np.ndarray):
            import cv2
            success, buf = cv2.imencode(".png", value)
            if not success:
                raise ValueError("Image encoding failed")
            img_bytes = buf.tobytes()

        payload = {'image': img_bytes, 'name': img_name}
        self.post_json('/uiapi/upload_image', payload, verbose=verbose)

    def link(self, path1: str, path2: str, verbose: bool = False):
        return self.post_json('/uiapi/set_connection', {"field": [path1, path2]}, verbose=verbose)

    def run(self, clear_output: bool = False, wait: bool = True):
        ret = self.post_json('/uiapi/execute')
        if not wait:
            return ret

        exec_id = ret['response']['prompt_id']
        self.await_run()

        wf_json = self.post_json('/uiapi/get_workflow')
        address = self.locate_output(wf_json['response'])
        history = self.fetch_history(exec_id)[exec_id]

        filenames = eval(f"history['outputs']{address}")['images']
        images = []
        for info in filenames:
            img_data = self.fetch_image(info['filename'], info['subfolder'], info['type'])
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
            if clear_output:
                pass  # implement cleanup if needed

        return images[0]

    def await_run(self):
        self.init_ws()
        while True:
            try:
                out = self.ws.recv()
                if isinstance(out, str):
                    msg = json.loads(out)
                    if msg['type'] == 'status' and msg['data']['status']['exec_info']['queue_remaining'] == 0:
                        return
            except WebSocketTimeoutException:
                pass
            except Exception as e:
                log.error(f"Error while waiting: {str(e)}")
                return

            if self.freed:
                self.freed = False
                return

    def fetch_history(self, prompt_id: str) -> Dict:
        with urllib.request.urlopen(f"http://{self.server_address}/history/{prompt_id}") as response:
            return json.loads(response.read())

    def fetch_image(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        import urllib.parse
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        query = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"http://{self.server_address}/view?{query}") as response:
            return response.read()

    @staticmethod
    def locate_output(obj: Dict) -> str:
        for key, value in obj.items():
            if isinstance(value, dict):
                if value.get("class_type") in ["SaveImage", "Image Save"]:
                    return f"['{key}']"
                result = ComfyUIBridge.locate_output(value)
                if result:
                    return result
        return None

    def txt2img(self, **args):
        self.apply_values(args)
        self.link("ccg3.CONDITIONING", "KSampler.positive")
        return self.run()

    def img2img(self, **args):
        self.apply_values(args)
        self.link("ConditioningAverage.CONDITIONING", "KSampler.positive")
        return self.run()

    def apply_values(self, args: Dict[str, Any]):
        for key, value in args.items():
            self.set(key, value)

# Example
if __name__ == "__main__":
    comfy = ComfyUIBridge()
    out = comfy.txt2img(prompt="A serene mountain view", steps=20, cfg_scale=7)
    out.save("output.png")
