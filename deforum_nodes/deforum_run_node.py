import base64
import io
import json
import math
import os
import random
import secrets
import time
from types import SimpleNamespace

import PIL
import cv2
import numexpr
import numpy as np
import requests
import torch
from PIL import Image
from deforum.animation.animation_key_frames import DeformAnimKeys

from deforum.avfunctions.image.load_images import check_mask_for_errors, prepare_mask, load_img
from deforum.datafunctions.prompt import check_is_number, split_weighted_subprompts
from deforum.general_utils import isJson, pairwise_repl, substitute_placeholders
from deforum.torchfuncs.torch_gc import torch_gc
from deforum.main import Deforum

from qtpy import QtCore, QtWidgets
from ainodes_frontend.base import register_node, get_next_opcode, handle_ainodes_exception
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import tensor_image_to_pixmap, pixmap_to_tensor, tensor2pil, \
    pil2tensor
from ai_nodes.ainodes_engine_base_nodes.image_nodes.image_preview_node import ImagePreviewNode
from ai_nodes.ainodes_engine_base_nodes.video_nodes.video_save_node import VideoOutputNode
#from ..deforum_helpers.qops import pixmap_to_pil_image
from ...ainodes_engine_base_nodes.ainodes_backend.cnet_preprocessors import hed
from ...ainodes_engine_base_nodes.image_nodes.image_op_node import HWC3

from ...ainodes_engine_base_nodes.torch_nodes.kandinsky_node import KandinskyNode
from ...ainodes_engine_base_nodes.torch_nodes.ksampler_node import KSamplerNode

from deforum.animation.new_args import process_args, RootArgs, DeforumArgs, DeforumAnimArgs, DeforumOutputArgs, \
    ParseqArgs, LoopArgs

from ainodes_frontend import singleton as gs
from ...ainodes_engine_base_nodes.video_nodes.FILM_node import FILMNode

#from transformers import AutoProcessor, CLIPVisionModelWithProjection

OP_NODE_DEFORUM_RUN = get_next_opcode()
OP_NODE_DEFORUM_CNET = get_next_opcode()

class DeforumRunWidget(QDMNodeContentWidget):
    progress_signal = QtCore.Signal(int)
    def initUI(self):
        self.create_widgets()
        self.create_main_layout(grid=1)
    def create_widgets(self):
        self.use_blend = self.create_check_box("Use Conditioning Blending [SD Only]")
        self.cond_schedule_checkbox = self.create_check_box("Force Blend Factor:")
        self.blend_factor = self.create_double_spin_box("Conditioning Blend factor")
        self.use_inpaint = self.create_check_box("Use Inpaint pass")
@register_node(OP_NODE_DEFORUM_RUN)
class DeforumRunNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/deforum.png"
    op_code = OP_NODE_DEFORUM_RUN
    op_title = "Deforum Runner"
    content_label_objname = "deforum_runner_node"
    category = "aiNodes Deforum/DeForum"
    custom_input_socket_name = ["DATA", "COND", "SAMPLER", "EXEC"]
    output_socket_name = ["IMAGE", "EXEC"]
    dim = (240, 240)
    NodeContent_class = DeforumRunWidget

    make_dirty = True


    def __init__(self, scene):
        super().__init__(scene, inputs=[6, 3, 5, 1], outputs=[5, 5, 1])

        self.images = []
        self.pipe = None



    # def initInnerClasses(self):
    #     self.content = DeforumRunWidget(self)
    #     self.grNode = CalcGraphicsNode(self)
    #     self.grNode.width = 300
    #     self.grNode.height = 320
    #     self.content.setMinimumWidth(300)
    #     self.content.setMinimumHeight(300)
    #     self.content.eval_signal.connect(self.evalImplementation)


    def evalImplementation_thread(self, index=0):

        data = self.getInputData(0)

        root_dict = RootArgs()
        args_dict = {key: value["value"] for key, value in DeforumArgs().items()}


        anim_args_dict = {key: value["value"] for key, value in DeforumAnimArgs().items()}
        output_args_dict = {key: value["value"] for key, value in DeforumOutputArgs().items()}
        loop_args_dict = {key: value["value"] for key, value in LoopArgs().items()}
        root = SimpleNamespace(**root_dict)
        args = SimpleNamespace(**args_dict)
        anim_args = SimpleNamespace(**anim_args_dict)
        video_args = SimpleNamespace(**output_args_dict)
        #parseq_args = SimpleNamespace(**ParseqArgs())
        parseq_args = None
        loop_args = SimpleNamespace(**loop_args_dict)
        controlnet_args = SimpleNamespace(**{"controlnet_args": "None"})

        for key, value in args.__dict__.items():
            if key in data:
                if data[key] == "":
                    val = None
                else:
                    val = data[key]
                setattr(args, key, val)

        for key, value in anim_args.__dict__.items():
            if key in data:
                if data[key] == "" and "schedule" not in key:
                    val = None
                else:
                    val = data[key]
                setattr(anim_args, key, val)

        for key, value in video_args.__dict__.items():
            if key in data:
                if data[key] == "" and "schedule" not in key:
                    val = None
                else:
                    val = data[key]
                setattr(anim_args, key, val)

        for key, value in root.__dict__.items():
            if key in data:
                if data[key] == "":
                    val = None
                else:
                    val = data[key]
                setattr(root, key, val)

        for key, value in loop_args.__dict__.items():
            if key in data:
                if data[key] == "":
                    val = None
                else:
                    val = data[key]
                setattr(loop_args, key, val)




        success = None
        root.timestring = time.strftime('%Y%m%d%H%M%S')
        args.strength = max(0.0, min(1.0, args.strength))
        #args.prompts = json.loads(args_dict_main['animation_prompts'])
        #args.positive_prompts = args_dict_main['animation_prompts_positive']
        #args.negative_prompts = args_dict_main['animation_prompts_negative']

        if not args.use_init and not anim_args.hybrid_use_init_image:
            args.init_image = None

        elif anim_args.animation_mode == 'Video Input':
            args.use_init = True

        current_arg_list = [args, anim_args, video_args, parseq_args, root]
        full_base_folder_path = os.path.join(os.getcwd(), "output/deforum")
        root.raw_batch_name = args.batch_name
        args.batch_name = substitute_placeholders(args.batch_name, current_arg_list, full_base_folder_path)
        args.outdir = os.path.join(full_base_folder_path, str(args.batch_name))

        os.makedirs(args.outdir, exist_ok=True)

        self.deforum = Deforum(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root)
        self.deforum.generate = self.generate
        self.deforum.generate_inpaint = self.generate_inpaint
        self.deforum.datacallback = self.datacallback

        if self.deforum.args.seed == -1 or self.deforum.args.seed == "-1":
            setattr(self.deforum.args, "seed", secrets.randbelow(999999999999999999))
            setattr(self.deforum.root, "raw_seed", int(self.deforum.args.seed))
            setattr(self.deforum.root, "seed_internal", 0)
        else:
            self.deforum.args.seed = int(self.deforum.args.seed)

        self.deforum.keys = DeformAnimKeys(self.deforum.anim_args, self.deforum.args.seed)

        success = self.deforum()
        gs.should_run = False
        return [None, None]

    def printkeys(self):
        print(self.deforum.keys)

    def datacallback(self, data):
        if "image" in data:
            self.handle_main_callback(data["image"])
        elif "cadence_frame" in data:
            self.handle_cadence_callback(data["cadence_frame"])
    def handle_main_callback(self, image):

        if isinstance(image, PIL.Image.Image):
            image = pil2tensor(image)


        self.setOutput(1, image)
        for node in self.getOutputs(1):
            if isinstance(node, ImagePreviewNode):
                node.evalImplementation_thread()
            elif isinstance(node, VideoOutputNode):
                frame = np.array(tensor2pil(image))
                node.content.video.add_frame(frame, dump=node.content.dump_at.value())
            elif isinstance(node, FILMNode):
                node.onWorkerFinished(result=node.evalImplementation_thread())
                for sub_node in node.getOutputs(0):
                    if isinstance(sub_node, ImagePreviewNode):
                        sub_node.evalImplementation_thread()
                    elif isinstance(sub_node, VideoOutputNode):
                        sub_node.evalImplementation_thread()
                #node.content.video.add_frame(frame, dump=node.content.dump_at.value())
    def handle_cadence_callback(self, image):


        if isinstance(image, PIL.Image.Image):
            image = pil2tensor(image)


        self.setOutput(0, image)
        for node in self.getOutputs(0):
            if isinstance(node, ImagePreviewNode):
                node.evalImplementation_thread()
            elif isinstance(node, VideoOutputNode):
                frame = np.array(tensor2pil(image))
                node.content.video.add_frame(frame, dump=node.content.dump_at.value())

    def generate(self, args, keys, anim_args, loop_args, controlnet_args, root, frame_idx, sampler_name):
        if gs.should_run:
            image = generate_inner(self, args, keys, anim_args, loop_args, controlnet_args, root, frame_idx, sampler_name)
        else:
            image = None
        return image

    def generate_inpaint(self, args, keys, anim_args, loop_args, controlnet_args, root, frame_idx, sampler_name, image=None, mask=None):
        original_image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        mask = mask.cpu().reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)

        mask_array = np.array(mask)
        # Check if any values are above 0
        has_values_above_zero = (np.array(mask) > 1e-05).any()
        # Count the number of values above 0
        count_values_above_zero = (mask_array > 0).sum()
        threshold = 40000

        if has_values_above_zero and count_values_above_zero > threshold and self.content.use_inpaint.isChecked():
            print(f"[ Mask pixels above {threshold} by {count_values_above_zero-threshold}, generating inpaing image ]")
            mask = tensor2pil(mask[0])
            mask = dilate_mask(mask, dilation_size=48)
            change_pipe = False
            if gs.should_run:
                if not self.pipe or change_pipe:
                    from diffusers import StableDiffusionInpaintPipeline
                    self.pipe = StableDiffusionInpaintPipeline.from_single_file(
                                "models/checkpoints/Deliberate-inpainting.safetensors",
                                use_safetensors=True,
                                torch_dtype=torch.float16).to(gs.device.type)
                    # self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    #             "runwayml/stable-diffusion-inpainting",
                    #             torch_dtype=torch.float16).to(gs.device.type)
                prompt, negative_prompt = split_weighted_subprompts(args.prompt, frame_idx, anim_args.max_frames)
                generation_args = {"generator":torch.Generator(gs.device.type).manual_seed(args.seed),
                                   "num_inference_steps":args.steps,
                                   "prompt":prompt,
                                   "image":image,
                                   "mask_image":mask,
                                   "width" : image.size[0],
                                   "height" : image.size[1],
                                   }
                #image.save("inpaint_image.png", "PNG")
                image = np.array(self.pipe(**generation_args).images[0]).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # # Composite the original image and the generated image using the mask
                mask_arr = np.array(mask).astype(np.uint8)[:, :, 0]  # Convert to grayscale mask for boolean indexing
                mask_bool = mask_arr > 0  # Convert to boolean mask
                original_image[mask_bool] = image[mask_bool]
                #test = Image.fromarray(original_image).save("test_result.png", "PNG")


        return original_image
def dilate_mask(mask_img, dilation_size=12):
    # Convert the PIL Image to a NumPy array
    mask_array = np.array(mask_img)

    # Create the dilation kernel
    kernel = np.ones((dilation_size, dilation_size), np.uint8)

    # Dilate the mask
    dilated_mask_array = cv2.dilate(mask_array, kernel)

    # Convert back to a PIL Image
    dilated_mask_img = Image.fromarray(dilated_mask_array)

    return dilated_mask_img


@register_node(OP_NODE_DEFORUM_CNET)
class DeforumCnetNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/deforum.png"
    op_code = OP_NODE_DEFORUM_CNET
    op_title = "Deforum Cnet Node"
    content_label_objname = "deforum_cnet_node"
    category = "aiNodes Deforum/DeForum"
    custom_input_socket_name = ["MASK", "IMAGE", "EXEC"]
    output_socket_name = ["IMAGE", "EXEC"]

    def __init__(self, scene):
        super().__init__(scene, inputs=[5, 5, 1], outputs=[5, 1])

    def initInnerClasses(self):
        self.content = DeforumRunWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.width = 300
        self.grNode.height = 250
        self.content.setMinimumWidth(300)
        self.content.setMinimumHeight(250)
        self.content.eval_signal.connect(self.evalImplementation)


    def evalImplementation_thread(self, index=0, conditioning=None):

        mask_pixmaps = self.getInputData(0)
        pixmaps = self.getInputData(1)

        #print(gs.models["loaded_controlnet"])

        if conditioning is not None:
            img = cnet_image_ops("canny", pixmap_to_tensor(pixmaps[0]))
            conditioning = self.add_control_image(conditioning, img)
            #self.setOutput(1, [pil_image_to_pixmap(img)])
            if len(self.getOutputs(0)) > 0:
                node = self.getOutputs(0)[0]
                if isinstance(node, ImagePreviewNode):
                    node.content.preview_signal.emit(tensor_image_to_pixmap(img))
        return [conditioning]

    def add_control_image(self, conditioning, image, progress_callback=None):

        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        c = []
        control_hint = image.movedim(-1,1)
        for t in conditioning:
            n = [t[0], t[1].copy()]
            n[1]['control_hint'] = control_hint
            n[1]['control_strength'] = 1.0
            c.append(n)
        return c



def cnet_image_ops(method, image):
    if method == 'canny':
        image = np.array(image)
        image = cv2.Canny(image, 100, 100, L2gradient=True)
        image = HWC3(image)
        image = Image.fromarray(image)
    elif method == 'fake_scribble':
        image = np.array(image)
        detector = hed.HEDdetector()
        image = detector(image)
        image = HWC3(image)
        image = hed.nms(image, 127, 3.0)
        image = cv2.GaussianBlur(image, (0, 0), 3.0)
        image[image > 4] = 255
        image[image < 255] = 0
        image = Image.fromarray(image)
        detector.netNetwork.cpu()
        detector.netNetwork = None
        del detector
    elif method == 'hed':
        # Ref: https://github.com/lllyasviel/ControlNet/blob/main/gradio_hed2image.py
        image = np.array(image)
        detector = hed.HEDdetector()
        image = detector(image)
        image = HWC3(image)
        image = Image.fromarray(image)
        detector.netNetwork.cpu()
        detector.netNetwork = None
        del detector
    return image




import torch.nn.functional as F
import torch

def pyramid_blend(tensor1, tensor2, blend_value):
    # For simplicity, we'll use two levels of blending
    downsampled1 = F.avg_pool2d(tensor1, 2)
    downsampled2 = F.avg_pool2d(tensor2, 2)

    blended_low = (1 - blend_value) * downsampled1 + blend_value * downsampled2
    blended_high = tensor1 + tensor2 - F.interpolate(blended_low, scale_factor=2)

    return blended_high
def gaussian_blend(tensor1, tensor2, blend_value):
    sigma = 0.5  # Adjust for desired smoothness
    weight = torch.exp(-((blend_value - 0.5) ** 2) / (2 * sigma ** 2))

    return (1 - weight) * tensor1 + weight * tensor2


def gaussian_blend(tensor1, tensor2, blend_value):
    sigma = 0.5  # Adjust for desired smoothness
    weight = torch.exp(-((blend_value - 0.5) ** 2) / (2 * sigma ** 2))

    return (1 - weight) * tensor1 + weight * tensor2
def sigmoidal_blend(tensor1, tensor2, blend_value):
    # Convert blend_value into a tensor with the same shape as tensor1 and tensor2
    blend_tensor = torch.full_like(tensor1, blend_value)
    weight = 1 / (1 + torch.exp(-10 * (blend_tensor - 0.5)))  # Sigmoid function centered at 0.5
    return (1 - weight) * tensor1 + weight * tensor2

def blend_tensors(obj1, obj2, blend_value, blend_method="linear"):
    """
    Blends tensors in two given objects based on a blend value using various blending strategies.
    """
    if blend_method == "linear":
        weight = blend_value
        blended_cond = (1 - weight) * obj1[0] + weight * obj2[0]
        blended_pooled = (1 - weight) * obj1[1]['pooled_output'] + weight * obj2[1]['pooled_output']

    elif blend_method == "sigmoidal":
        blended_cond = sigmoidal_blend(obj1[0], obj2[0], blend_value)
        blended_pooled = sigmoidal_blend(obj1[1]['pooled_output'], obj2[1]['pooled_output'], blend_value)

    elif blend_method == "gaussian":
        blended_cond = gaussian_blend(obj1[0], obj2[0], blend_value)
        blended_pooled = gaussian_blend(obj1[1]['pooled_output'], obj2[1]['pooled_output'], blend_value)

    elif blend_method == "pyramid":
        blended_cond = pyramid_blend(obj1[0], obj2[0], blend_value)
        blended_pooled = pyramid_blend(obj1[1]['pooled_output'], obj2[1]['pooled_output'], blend_value)

    return [[blended_cond, {"pooled_output": blended_pooled}]]

def generate_with_node(node, prompt, next_prompt, blend_value, negative_prompt, args, root, frame, init_images=None):

    sampler_node, _ = node.getInput(2)

    make_latent = None
    tensor = torch.zeros([1, 4, args.H // 8, args.W // 8])


    if isinstance(sampler_node, KSamplerNode):
        if init_images is not None:
            vae = sampler_node.getInputData(1)
            latent = encode_latent_ainodes(init_images, vae)
        else:
            latent = torch.zeros([1, 4, args.H // 8, args.W // 8])

        cond_node, _ = node.getInput(1)


        _, cond = cond_node.evalImplementation_thread(prompt_override=prompt)


        node_blend = node.content.blend_factor.value()
        use_blend = node.content.use_blend.isChecked()
        # if node.content.blend_factor.value() < 1.00:
        if next_prompt != prompt and use_blend and blend_value != 0.0:
            _, next_cond = cond_node.evalImplementation_thread(prompt_override=next_prompt)
            #blend_value = 1 if blend_value == 0 else blend_value
            blend_value = blend_value if not node.content.cond_schedule_checkbox.isChecked() else node_blend
            print(f"\n[ Blending Conditionings, ratio: {blend_value} ]")
            # print(f"[ Next Prompt: {next_prompt} ]")
            # print(f"[ Seed: {args.seed} ]\n")
            #print(f"[ Gen Args: {args} ]")
            cond = blend_tensors(cond[0], next_cond[0], blend_value)
        _, n_cond = cond_node.evalImplementation_thread(prompt_override=negative_prompt)
        tensor, _ = sampler_node.evalImplementation_thread(cond_override=[cond, n_cond], args=args,
                                                            latent_override=latent)

    elif isinstance(sampler_node, KandinskyNode):
        init = None
        if init_images is not None:
            init = init_images
        if init is not None:
            if isinstance(init, PIL.Image.Image):
                init = pil2tensor(init)
        tensor = sampler_node.evalImplementation_thread(prompt_override=prompt, args=args, init_image=init)[0]

    image = tensor2pil(tensor)
    return image

def encode_latent_ainodes(init_image, vae):
    image = np.array(init_image).astype(np.float32) / 255.0
    image = image[None]# .transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = image.detach().cpu()
    torch_gc()
    latent = vae.encode(image)
    latent = latent.to("cpu")
    image = image.detach().to("cpu")
    del image
    return latent

def generate_inner(node, args, keys, anim_args, loop_args, controlnet_args, root, frame=0, return_sample=False,
                   sampler_name=None):
    assert args.prompt is not None

    # Setup the pipeline
    #p = get_webui_sd_pipeline(args, root, frame)
    prompt, negative_prompt = split_weighted_subprompts(args.prompt, frame, anim_args.max_frames)

    #print("DEFORUM CONDITIONING INTERPOLATION")

    """prompt = node.deforum.prompt_series[frame]
    next_prompt = None
    if frame + anim_args.diffusion_cadence < anim_args.max_frames:

        curr_frame = frame

        next_prompt = node.deforum.prompt_series[frame + anim_args.diffusion_cadence]
    print("NEXT FRAME", frame, next_prompt)"""
    # blend_value = 0.0
    #
    # #print(frame, anim_args.diffusion_cadence, node.deforum.prompt_series)
    #
    # next_frame = frame + anim_args.diffusion_cadence
    # next_prompt = None
    # while next_frame < anim_args.max_frames:
    #     next_prompt = node.deforum.prompt_series[next_frame]
    #     if next_prompt != prompt:
    #         # Calculate blend value based on distance and frame number
    #         prompt_distance = next_frame - frame
    #         max_distance = anim_args.max_frames - frame
    #         blend_value = prompt_distance / max_distance
    #
    #         if blend_value >= 1.0:
    #             blend_value = 0.0
    #
    #         break  # Exit the loop once a different prompt is found
    #
    #     next_frame += anim_args.diffusion_cadence
    # #print("CURRENT PROMPT", prompt)
    # #print("NEXT FRAME:", next_prompt)
    # #print("BLEND VALUE:", blend_value)
    # #print("BLEND VALUE:", blend_value)
    # #print("PARSED_PROMPT", prompt)
    # #print("PARSED_PROMPT", prompt)
    # if frame == 0:
    #     blend_value = 0.0
    # if frame > 0:
    #     prev_prompt = node.deforum.prompt_series[frame - 1]
    #     if prev_prompt != prompt:
    #         blend_value = 0.0
    # def generate_blend_values(distance_to_next_prompt, blend_type="linear"):
    #     if blend_type == "linear":
    #         return [i / distance_to_next_prompt for i in range(distance_to_next_prompt + 1)]
    #     elif blend_type == "exponential":
    #         base = 2
    #         return [1 / (1 + math.exp(-8 * (i / distance_to_next_prompt - 0.5))) for i in
    #                 range(distance_to_next_prompt + 1)]
    #     else:
    #         raise ValueError(f"Unknown blend type: {blend_type}")
    #
    # def find_last_prompt_change(current_index, prompt_series):
    #     # Step backward from the current position
    #     for i in range(current_index - 1, -1, -1):
    #         if prompt_series.iloc[i] != prompt_series.iloc[current_index]:
    #             return i
    #     return 0  # default to the start if no change found
    #
    # def find_next_prompt_change(current_index, prompt_series):
    #     # Step forward from the current position
    #     for i in range(current_index + 1, len(prompt_series)):
    #         if prompt_series.iloc[i] != prompt_series.iloc[current_index]:
    #             return i
    #     return len(prompt_series) - 1  # default to the end if no change found
    #
    # # Inside your main loop:
    #
    # last_prompt_change = find_last_prompt_change(frame, node.deforum.prompt_series)
    # next_prompt_change = find_next_prompt_change(frame, node.deforum.prompt_series)
    #
    # distance_between_changes = next_prompt_change - last_prompt_change
    # current_distance_from_last = frame - last_prompt_change
    #
    # # Generate blend values for the distance between prompt changes
    # blend_values = generate_blend_values(distance_between_changes, blend_type="exponential")
    #
    # # Fetch the blend value based on the current frame's distance from the last prompt change
    # blend_value = blend_values[current_distance_from_last]
    # next_prompt = node.deforum.prompt_series[next_prompt_change]

    def generate_blend_values(distance_to_next_prompt, blend_type="linear"):
        if blend_type == "linear":
            return [i / distance_to_next_prompt for i in range(distance_to_next_prompt + 1)]
        elif blend_type == "exponential":
            base = 2
            return [1 / (1 + math.exp(-8 * (i / distance_to_next_prompt - 0.5))) for i in
                    range(distance_to_next_prompt + 1)]
        else:
            raise ValueError(f"Unknown blend type: {blend_type}")

    def get_next_prompt_and_blend(current_index, prompt_series, blend_type="exponential"):
        # Find where the current prompt ends
        next_prompt_start = current_index + 1
        while next_prompt_start < len(prompt_series) and prompt_series.iloc[next_prompt_start] == prompt_series.iloc[
            current_index]:
            next_prompt_start += 1

        if next_prompt_start >= len(prompt_series):
            return "", 1.0
            #raise ValueError("Already at the last prompt, no next prompt available.")

        # Calculate blend value
        distance_to_next = next_prompt_start - current_index
        blend_values = generate_blend_values(distance_to_next, blend_type)
        blend_value = blend_values[1]  # Blend value for the next frame after the current index

        return prompt_series.iloc[next_prompt_start], blend_value

    next_prompt, blend_value = get_next_prompt_and_blend(frame, node.deforum.prompt_series)
    # print("DEBUG", next_prompt, blend_value)

    # blend_value = 1.0
    # next_prompt = ""
    if not args.use_init and args.strength > 0 and args.strength_0_no_init:
        args.strength = 0
    processed = None
    mask_image = None
    init_image = None
    image_init0 = None

    if loop_args.use_looper and anim_args.animation_mode in ['2D', '3D']:
        args.strength = loop_args.imageStrength
        tweeningFrames = loop_args.tweeningFrameSchedule
        blendFactor = .07
        colorCorrectionFactor = loop_args.colorCorrectionFactor
        jsonImages = json.loads(loop_args.imagesToKeyframe)
        # find which image to show
        parsedImages = {}
        frameToChoose = 0
        max_f = anim_args.max_frames - 1

        for key, value in jsonImages.items():
            if check_is_number(key):  # default case 0:(1 + t %5), 30:(5-t%2)
                parsedImages[key] = value
            else:  # math on the left hand side case 0:(1 + t %5), maxKeyframes/2:(5-t%2)
                parsedImages[int(numexpr.evaluate(key))] = value

        framesToImageSwapOn = list(map(int, list(parsedImages.keys())))

        for swappingFrame in framesToImageSwapOn[1:]:
            frameToChoose += (frame >= int(swappingFrame))

        # find which frame to do our swapping on for tweening
        skipFrame = 25
        for fs, fe in pairwise_repl(framesToImageSwapOn):
            if fs <= frame <= fe:
                skipFrame = fe - fs
        if skipFrame > 0:
            #print("frame % skipFrame", frame % skipFrame)

            if frame % skipFrame <= tweeningFrames:  # number of tweening frames
                blendFactor = loop_args.blendFactorMax - loop_args.blendFactorSlope * math.cos(
                    (frame % tweeningFrames) / (tweeningFrames / 2))
        else:
            print("LOOPER ERROR, AVOIDING DIVISION BY 0")
        init_image2, _ = load_img(list(jsonImages.values())[frameToChoose],
                                  shape=(args.W, args.H),
                                  use_alpha_as_mask=args.use_alpha_as_mask)
        image_init0 = list(jsonImages.values())[0]
        #print(" TYPE", type(image_init0))


    else:  # they passed in a single init image
        image_init0 = args.init_image

    available_samplers = {
        'euler a': 'Euler a',
        'euler': 'Euler',
        'lms': 'LMS',
        'heun': 'Heun',
        'dpm2': 'DPM2',
        'dpm2 a': 'DPM2 a',
        'dpm++ 2s a': 'DPM++ 2S a',
        'dpm++ 2m': 'DPM++ 2M',
        'dpm++ sde': 'DPM++ SDE',
        'dpm fast': 'DPM fast',
        'dpm adaptive': 'DPM adaptive',
        'lms karras': 'LMS Karras',
        'dpm2 karras': 'DPM2 Karras',
        'dpm2 a karras': 'DPM2 a Karras',
        'dpm++ 2s a karras': 'DPM++ 2S a Karras',
        'dpm++ 2m karras': 'DPM++ 2M Karras',
        'dpm++ sde karras': 'DPM++ SDE Karras'
    }
    """if sampler_name is not None:
        if sampler_name in available_samplers.keys():
            p.sampler_name = available_samplers[sampler_name]
        else:
            raise RuntimeError(
                f"Sampler name '{sampler_name}' is invalid. Please check the available sampler list in the 'Run' tab")"""

    #if args.checkpoint is not None:
    #    info = sd_models.get_closet_checkpoint_match(args.checkpoint)
    #    if info is None:
    #        raise RuntimeError(f"Unknown checkpoint: {args.checkpoint}")
    #    sd_models.reload_model_weights(info=info)

    if root.init_sample is not None:
        # TODO: cleanup init_sample remains later
        img = root.init_sample
        init_image = img
        image_init0 = img
        if loop_args.use_looper and isJson(loop_args.imagesToKeyframe) and anim_args.animation_mode in ['2D', '3D']:
            init_image = Image.blend(init_image, init_image2, blendFactor)
            correction_colors = Image.blend(init_image, init_image2, colorCorrectionFactor)
            color_corrections = [correction_colors]

    # this is the first pass
    elif (loop_args.use_looper and anim_args.animation_mode in ['2D', '3D']) or (
            args.use_init and ((args.init_image != None and args.init_image != ''))):
        init_image, mask_image = load_img(image_init0,  # initial init image
                                          shape=(args.W, args.H),
                                          use_alpha_as_mask=args.use_alpha_as_mask)

    else:

        #if anim_args.animation_mode != 'Interpolation':
        #    print(f"Not using an init image (doing pure txt2img)")
        """p_txt = StableDiffusionProcessingTxt2Img( 
            sd_model=sd_model,
            outpath_samples=root.tmp_deforum_run_duplicated_folder,
            outpath_grids=root.tmp_deforum_run_duplicated_folder,
            prompt=p.prompt,
            styles=p.styles,
            negative_prompt=p.negative_prompt,
            seed=p.seed,
            subseed=p.subseed,
            subseed_strength=p.subseed_strength,
            seed_resize_from_h=p.seed_resize_from_h,
            seed_resize_from_w=p.seed_resize_from_w,
            sampler_name=p.sampler_name,
            batch_size=p.batch_size,
            n_iter=p.n_iter,
            steps=p.steps,
            cfg_scale=p.cfg_scale,
            width=p.width,
            height=p.height,
            restore_faces=p.restore_faces,
            tiling=p.tiling,
            enable_hr=None,
            denoising_strength=None,
        )"""

        #print_combined_table(args, anim_args, p_txt, keys, frame)  # print dynamic table to cli

        #if is_controlnet_enabled(controlnet_args):
        #    process_with_controlnet(p_txt, args, anim_args, loop_args, controlnet_args, root, is_img2img=False,
        #                            frame_idx=frame)

        processed = generate_with_node(node, prompt, next_prompt, blend_value, negative_prompt, args, root, frame, init_image)

    if processed is None:
        # Mask functions
        if args.use_mask:
            mask_image = args.mask_image
            mask = prepare_mask(args.mask_file if mask_image is None else mask_image,
                                (args.W, args.H),
                                args.mask_contrast_adjust,
                                args.mask_brightness_adjust)
            inpainting_mask_invert = args.invert_mask
            inpainting_fill = args.fill
            inpaint_full_res = args.full_res_mask
            inpaint_full_res_padding = args.full_res_mask_padding
            # prevent loaded mask from throwing errors in Image operations if completely black and crop and resize in webui pipeline
            # doing this after contrast and brightness adjustments to ensure that mask is not passed as black or blank
            mask = check_mask_for_errors(mask, args.invert_mask)
            args.noise_mask = mask

        else:
            mask = None

        assert not ((mask is not None and args.use_mask and args.overlay_mask) and (
                    args.init_sample is None and init_image is None)), "Need an init image when use_mask == True and overlay_mask == True"

        image_mask = mask
        image_cfg_scale = args.pix2pix_img_cfg_scale

        #print_combined_table(args, anim_args, p, keys, frame)  # print dynamic table to cli

        #if is_controlnet_enabled(controlnet_args):
        #    process_with_controlnet(p, args, anim_args, loop_args, controlnet_args, root, is_img2img=True,
        #                            frame_idx=frame)


        processed = generate_with_node(node, prompt, next_prompt, blend_value, negative_prompt, args, root, frame, init_image)
        #processed = processing.process_images(p)

    #if root.initial_info == None:
    #    root.initial_seed = processed.seed
    #    root.initial_info = processed.info

    if root.first_frame == None:
        root.first_frame = processed

    return processed

def pack_args(args_dict, keys_function):
    return {name: args_dict[name] for name in keys_function()}

def process_args(args_dict_main, run_id):
    #from ..datafunctions.settings import load_args
    #override_settings_with_file = args_dict_main['override_settings_with_file']
    #custom_settings_file = args_dict_main['custom_settings_file']
    #p = args_dict_main['p']

    root = SimpleNamespace(**RootArgs())
    args = SimpleNamespace(**DeforumArgs())
    anim_args = SimpleNamespace(**DeforumAnimArgs())
    video_args = SimpleNamespace(**DeforumOutputArgs())
    parseq_args = SimpleNamespace(**ParseqArgs())
    loop_args = SimpleNamespace(**LoopArgs())
    controlnet_args = SimpleNamespace(**{"controlnet_args":"None"})

    #root.animation_prompts = json.loads(args_dict_main['animation_prompts'])

    args_loaded_ok = True
    #if override_settings_with_file:
    #    args_loaded_ok = load_args(args_dict_main, args, anim_args, parseq_args, loop_args, controlnet_args, video_args, custom_settings_file, root, run_id)

    #positive_prompts = args_dict_main['animation_prompts_positive']
    #negative_prompts = args_dict_main['animation_prompts_negative']
    #negative_prompts = negative_prompts.replace('--neg', '')  # remove --neg from negative_prompts if received by mistake
    #root.animation_prompts = {key: f"{positive_prompts} {val} {'' if '--neg' in val else '--neg'} {negative_prompts}" for key, val in root.animation_prompts.items()}
    def get_fixed_seed(seed):
        if seed is None or seed == '' or seed == -1:
            return int(random.randrange(4294967294))
        return seed

    if args.seed == -1:
        root.raw_seed = -1
    args.seed = get_fixed_seed(args.seed)
    if root.raw_seed != -1:
        root.raw_seed = args.seed
    root.timestring = time.strftime('%Y%m%d%H%M%S')
    args.strength = max(0.0, min(1.0, args.strength))
    args.prompts = json.loads(args_dict_main['animation_prompts'])
    args.positive_prompts = args_dict_main['animation_prompts_positive']
    args.negative_prompts = args_dict_main['animation_prompts_negative']

    if not args.use_init and not anim_args.hybrid_use_init_image:
        args.init_image = None

    elif anim_args.animation_mode == 'Video Input':
        args.use_init = True

    current_arg_list = [args, anim_args, video_args, parseq_args, root]
    full_base_folder_path = os.path.join(os.getcwd(), p.outpath_samples)
    root.raw_batch_name = args.batch_name
    args.batch_name = substitute_placeholders(args.batch_name, current_arg_list, full_base_folder_path)
    args.outdir = os.path.join(p.outpath_samples, str(args.batch_name))
    args.outdir = os.path.join(os.getcwd(), args.outdir)
    os.makedirs(args.outdir, exist_ok=True)

    return args_loaded_ok, root, args, anim_args, video_args, parseq_args, loop_args, controlnet_args


def addWeighted(conditioning_to, conditioning_from, conditioning_to_strength):
    out = []

    if len(conditioning_from) > 1:
        print(
            "Warning: ConditioningAverage conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

    cond_from = conditioning_from[0][0]
    pooled_output_from = conditioning_from[0][1].get("pooled_output", None)

    for i in range(len(conditioning_to)):
        t1 = conditioning_to[i][0]
        pooled_output_to = conditioning_to[i][1].get("pooled_output", pooled_output_from)
        t0 = cond_from[:, :t1.shape[1]]
        if t0.shape[1] < t1.shape[1]:
            t0 = torch.cat([t0] + [torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]))], dim=1)

        tw = torch.mul(t1, conditioning_to_strength) + torch.mul(t0, (1.0 - conditioning_to_strength))
        t_to = conditioning_to[i][1].copy()
        if pooled_output_from is not None and pooled_output_to is not None:
            t_to["pooled_output"] = torch.mul(pooled_output_to, conditioning_to_strength) + torch.mul(
                pooled_output_from, (1.0 - conditioning_to_strength))
        elif pooled_output_from is not None:
            t_to["pooled_output"] = pooled_output_from

        n = [tw, t_to]
        out.append(n)
    outback = [[out[0][0], {"pooled_output": out[0][1]["pooled_output"]}]]
    return outback


def _encode_image(image, prompt,  device, num_images_per_prompt, do_classifier_free_guidance, model, processor):
    dtype = next(model.parameters()).dtype
    image = processor(images=image, prompt=prompt, return_tensors="pt").pixel_values
    image = image.to(device=device, dtype=dtype)
    image_embeddings = model(image).image_embeds
    image_embeddings = image_embeddings.unsqueeze(0)

    # duplicate image embeddings for each generation per prompt, using mps friendly method
    bs_embed, seq_len, _ = image_embeddings.shape
    image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
    image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

    if do_classifier_free_guidance:
        negative_prompt_embeds = torch.zeros_like(image_embeddings)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])
    image_embeddings = image_embeddings.permute(1,0,2)
    return image_embeddings