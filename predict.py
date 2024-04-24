import os, sys, json
import time
import base64

sys.path.extend(['/stable-diffusion-webui'])

from modules import timer
from modules import initialize_util
from modules import initialize
from fastapi import FastAPI

from cog import BasePredictor, BaseModel, Input, Path

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        # workaround for replicate since its entrypoint may contain invalid args
        os.environ['IGNORE_CMD_ARGS_ERRORS'] = '1'

        startup_timer = timer.startup_timer
        startup_timer.record("launcher")

        initialize.imports()
        initialize.check_versions()
        initialize.initialize()

        app = FastAPI()
        initialize_util.setup_middleware(app)

        from modules.api.api import Api
        from modules.call_queue import queue_lock
        
        self.api = Api(app, queue_lock)

        model_response = self.api.get_sd_models()
        print("Available checkpoints: ", str(model_response))

        from modules import script_callbacks
        script_callbacks.before_ui_callback()
        script_callbacks.app_started_callback(None, app)

        from modules.api.models import StableDiffusionTxt2ImgProcessingAPI, StableDiffusionImg2ImgProcessingAPI
        self.StableDiffusionTxt2ImgProcessingAPI = StableDiffusionTxt2ImgProcessingAPI
        self.StableDiffusionImg2ImgProcessingAPI = StableDiffusionImg2ImgProcessingAPI

        file_path = Path("init.jpg")
        base64_encoded_data = base64.b64encode(file_path.read_bytes())
        base64_image = base64_encoded_data.decode('utf-8')

        payload = {
           "override_settings": {
                "sd_model_checkpoint": "juggernautXL_v9Rdphoto2Lightning.safetensors",
                "sd_vae": "vae-ft-mse-840000-ema-pruned.safetensors",
                 "CLIP_stop_at_last_layers": 1,
            },
            "override_settings_restore_afterwards": False,
            "prompt": "office building",
            "steps": 1,
            "init_images": [base64_image],
            "denoising_strength": 0.1,
            "do_not_save_samples": True,
            "alwayson_scripts": {
                "Tiled Diffusion": {
                    "args": [
                        True,
                        "MultiDiffusion",
                        True,
                        True,
                        1,
                        1,
                        112,
                        144,
                        4,
                        8,
                        "4x-UltraSharp",
                        1.1, 
                        False, 
                        0,
                        0.0, 
                        3,
                    ]
                },
                "Tiled VAE": {
                    "args": [
                        True,
                        3072,
                        192,
                        True,
                        True,
                        True,
                        True,
                    ]

                },
                "controlnet": {
                    "args": [
                        {
                            "enabled": True,
                            "module": "tile_resample",
                            "model": "control_v11f1e_sd15_tile",
                            "weight": 0.2,
                            "image": base64_image,
                            "resize_mode": 1,
                            "lowvram": False,
                            "downsample": 1.0,
                            "guidance_start": 0.0,
                            "guidance_end": 1.0,
                            "control_mode": 1,
                            "pixel_perfect": True,
                            "threshold_a": 1,
                            "threshold_b": 1,
                            "save_detected_map": False,
                            "processor_res": 512,
                        }
                    ]
                }
            }
        }

        req = StableDiffusionImg2ImgProcessingAPI(**payload)
        self.api.img2imgapi(req)

        print(f"Startup time: {startup_timer.summary()}.")

    def predict(
        self,
        prompt: str = Input(description="Prompt"),
        negative_prompt: str = Input(description="Negative Prompt", default=""),
        width: int = Input(
            description="Width of output image", ge=512, le=1792, default=1024
        ),
        height: int = Input(
            description="Height of output image", ge=512, le=1792, default=1024
        ),
        num_outputs: int = Input(
            description="Number of images to output", ge=1, le=4, default=1
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=['DPM++ 2M Karras', 'DPM++ SDE Karras', 'DPM++ 2M SDE Exponential', 'DPM++ 2M SDE Karras', 'Euler a', 'Euler', 'LMS', 'Heun', 'DPM2', 'DPM2 a', 'DPM++ 2S a', 'DPM++ 2M', 'DPM++ SDE', 'DPM++ 2M SDE', 'DPM++ 2M SDE Heun', 'DPM++ 2M SDE Heun Karras', 'DPM++ 2M SDE Heun Exponential', 'DPM++ 3M SDE', 'DPM++ 3M SDE Karras', 'DPM++ 3M SDE Exponential', 'DPM fast', 'DPM adaptive', 'LMS Karras', 'DPM2 Karras', 'DPM2 a Karras', 'DPM++ 2S a Karras', 'Restart', 'DDIM', 'PLMS', 'UniPC'],
            default="DPM++ SDE Karras",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=100, default=20
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=-1
        ),
        # image: Path = Input(description="Grayscale input image"),
        enable_hr: bool = Input(
            description="Hires. fix", default=False,
        ),
        hr_upscaler: str = Input(
            description="Upscaler for Hires. fix",
            choices=['Latent', 'Latent (antialiased)', 'Latent (bicubic)', 'Latent (bicubic antialiased)', 'Latent (nearest)', 'Latent (nearest-exact)', 'None', 'Lanczos', 'Nearest', 'ESRGAN_4x', 'LDSR', 'R-ESRGAN 4x+', 'R-ESRGAN 4x+ Anime6B', 'ScuNET GAN', 'ScuNET PSNR', 'SwinIR 4x', '4x-UltraSharp', '4x_NMKD-Siax_200k'],
            default="Latent",
        ),
        hr_steps: int = Input(
            description="Inference steps for Hires. fix", ge=0, le=100, default=20
        ),
        denoising_strength: float = Input(
            description="Denoising strength. 1.0 corresponds to full destruction of information in init image", ge=0, le=1, default=0.5
        ),
        hr_scale: float = Input(
            description="Factor to scale image by", ge=1, le=4, default=2
        ),
    ) -> list[Path]:
        """Run a single prediction on the model"""
        start_time = time.time()

        payload = {
            "override_settings": {
                "sd_model_checkpoint": "juggernautXL_v9Rdphoto2Lightning.safetensors"
            },
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "batch_size": num_outputs,
            "steps": num_inference_steps,
            "cfg_scale": guidance_scale,
            "seed": seed,
            "do_not_save_samples": True,
            "sampler_name": scheduler,
            "enable_hr": enable_hr,
            "hr_upscaler": hr_upscaler,
            "hr_second_pass_steps": hr_steps,
            "denoising_strength": denoising_strength if enable_hr else None,
            "hr_scale": hr_scale,
        }

        
        req = self.StableDiffusionTxt2ImgProcessingAPI(**payload)
        # generate
        resp = self.api.text2imgapi(req)
        info = json.loads(resp.info)

        from PIL import Image
        import uuid
        import base64
        from io import BytesIO

        outputs = []

        for i, image in enumerate(resp.images):
            seed = info["all_seeds"][i]
            gen_bytes = BytesIO(base64.b64decode(image))
            gen_data = Image.open(gen_bytes)
            filename = "{}-{}.png".format(seed, uuid.uuid1())
            gen_data.save(fp=filename, format="PNG")
            output = Path(filename)
            outputs.append(output)
        
        print(f"Prediction took {round(time.time() - start_time,2)} seconds")
        return outputs