import base64
import io
import json
import math
import os
import random
import secrets
import time
from types import SimpleNamespace

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
from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import pil_image_to_pixmap, pixmap_to_pil_image
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

#from transformers import AutoProcessor, CLIPVisionModelWithProjection



OP_NODE_DEFORUM_RUN = get_next_opcode()
OP_NODE_DEFORUM_CNET = get_next_opcode()

class DeforumRunWidget(QDMNodeContentWidget):
    progress_signal = QtCore.Signal(int)
    def initUI(self):
        self.create_widgets()
        self.create_main_layout(grid=1)
    def create_widgets(self):
        self.cond_schedule_checkbox = self.create_check_box("Use Conditioning schedule")
        self.blend_factor = self.create_double_spin_box("Conditioning Blend factor")

@register_node(OP_NODE_DEFORUM_RUN)
class DeforumRunNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/deforum.png"
    op_code = OP_NODE_DEFORUM_RUN
    op_title = "Deforum Runner"
    content_label_objname = "deforum_runner_node"
    category = "DeForum"
    custom_input_socket_name = ["DATA", "COND", "SAMPLER", "EXEC"]
    output_socket_name = ["IMAGE", "EXEC"]

    def __init__(self, scene):
        super().__init__(scene, inputs=[6, 3, 5, 1], outputs=[5, 5, 1])

    def initInnerClasses(self):
        self.content = DeforumRunWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.width = 300
        self.grNode.height = 250
        self.content.setMinimumWidth(300)
        self.content.setMinimumHeight(250)
        self.content.eval_signal.connect(self.evalImplementation)
        self.images = []
        #if "vision" not in gs.models:
        #    gs.models["vision"] = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
        #    gs.models["processor"] = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")


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
        self.deforum.datacallback = self.datacallback


        if self.deforum.args.seed == -1 or self.deforum.args.seed == "-1":
            setattr(self.deforum.args, "seed", secrets.randbelow(999999999999999999))
            setattr(self.deforum.root, "raw_seed", int(self.deforum.args.seed))
            setattr(self.deforum.root, "seed_internal", 0)
        else:
            self.deforum.args.seed = int(self.deforum.args.seed)

        self.deforum.keys = DeformAnimKeys(self.deforum.anim_args, self.deforum.args.seed)

        success = self.deforum()
        return success


        #try:
        #    success = deforum_ainodes_adapter(self, data)
        #except:
        #    handle_ainodes_exception()
        #return success

    def printkeys(self):
        print(self.deforum.keys)


    def datacallback(self, data):
        if "image" in data:
            self.handle_main_callback(data["image"])
        elif "cadence_frame" in data:
            self.handle_cadence_callback(data["cadence_frame"])
    def handle_main_callback(self, image):
        return_frames = None
        """self.images.append(image)
        if len(self.images) == 2:
            return_frames = []
            np_image1 = np.array(self.images[0])
            np_image2 = np.array(self.images[1])
            frames = gs.models["FILM"].inference(np_image1, np_image2, inter_frames=25)
            print(f"FILM NODE:  {len(frames)}")
            for frame in frames:
                image = Image.fromarray(frame)
                pixmap = pil_image_to_pixmap(image)
                return_frames.append(pixmap)
            self.images = [Image.fromarray(np_image2)]"""
        pixmap = pil_image_to_pixmap(image)
        self.setOutput(1, [pixmap])
        for node in self.getOutputs(1):
            if isinstance(node, ImagePreviewNode):
                node.content.preview_signal.emit(pixmap)
                if return_frames:
                    for frame in return_frames:
                        node.content.preview_signal.emit(frame)
                        time.sleep(0.1)
            elif isinstance(node, VideoOutputNode):
                frame = np.array(image)
                node.content.video.add_frame(frame, dump=node.content.dump_at.value())

    def handle_cadence_callback(self, image):
        pixmap = pil_image_to_pixmap(image)
        self.setOutput(0, [pixmap])

        for node in self.getOutputs(0):
            if isinstance(node, ImagePreviewNode):




                node.content.preview_signal.emit(pixmap)
            elif isinstance(node, VideoOutputNode):
                frame = np.array(image)
                node.content.video.add_frame(frame, dump=node.content.dump_at.value())
    def generate(self, args, keys, anim_args, loop_args, controlnet_args, root, frame_idx, sampler_name):
        print("ainodes deforum adapter")
        if gs.should_run:
            image = generate_inner(self, args, keys, anim_args, loop_args, controlnet_args, root, frame_idx, sampler_name)
        else:
            image = None
        return image

    #@QtCore.Slot(object)
    def onWorkerFinished(self, result):
        self.busy = False
        #super().onWorkerFinished(None)
        if gs.should_run:
            if len(self.getOutputs(2)) > 0:
                self.executeChild(output_index=2)


@register_node(OP_NODE_DEFORUM_CNET)
class DeforumCnetNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/deforum.png"
    op_code = OP_NODE_DEFORUM_CNET
    op_title = "Deforum Cnet Node"
    content_label_objname = "deforum_cnet_node"
    category = "DeForum"
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

        print(gs.models["loaded_controlnet"])

        if conditioning is not None:
            img = cnet_image_ops("canny", pixmap_to_pil_image(pixmaps[0]))
            conditioning = self.add_control_image(conditioning, img)
            #self.setOutput(1, [pil_image_to_pixmap(img)])
            if len(self.getOutputs(0)) > 0:
                node = self.getOutputs(0)[0]
                if isinstance(node, ImagePreviewNode):
                    node.content.preview_signal.emit(pil_image_to_pixmap(img))
        return conditioning

    def add_control_image(self, conditioning, image, progress_callback=None):
        #image = pixmap_to_pil_image(image)
        #image = image.convert("RGB")
        #print("DEBUG IMAGE", image)

        #image.save("CNET.png", "PNG")

        #array = np.array(image)
        #print("ARRAY", array)

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

    #@QtCore.Slot(object)
    def onWorkerFinished(self, result):
        self.busy = False
        #super().onWorkerFinished(None)
        if gs.should_run:
            if len(self.getOutputs(1)) > 0:
                self.executeChild(output_index=1)



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

def eval_nodes(node):
    gs.should_run = None
    index = len(node.outputs) - 1
    if len(node.getOutputs(index)) > 0:
        for other_node in node.getChildrenNodes():
            result = other_node.evalImplementation_thread
            gs.should_run = None
            other_node.onWorkerFinished(result)
            eval_nodes(other_node)
    else:
        result = node.evalImplementation_thread
        return result


def generate_with_api(node, prompt, negative_prompt, args, root, frame, init_images=None):
    strength = 1.0 if args.strength == 0 or not args.use_init else args.strength
    seed = args.seed
    steps = args.steps
    cfg = args.scale
    start_step = 0
    width = args.W
    height = args.H
    imgur_url = None
    if len(init_images) > 0:
        # Load the init image
        init_image = init_images[0]  # Replace with the actual path to your image
        if init_image is not None:
            # Upload the init image to Imgur
            imgur_url = upload_image_to_postimages(init_image)
    if imgur_url is not None:
        method = "img2img"
    else:
        method = "txt2img"

    # Prepare the data for the POST request
    data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "samples": "1",
        "scheduler": "DDIM",
        "num_inference_steps": steps,
        "guidance_scale": cfg,
        "seed": seed,
        "strength": strength,
        "imageUrl": imgur_url
    }
    # Make the POST request
    url = f"https://api.segmind.com/v1/sd2.1-{method}"
    headers = {"x-api-key": "SG_b6ff407f17eeda88"}

    print("HEADERS", headers)
    print("METHOD", method)
    print("DATA", data)


    response = requests.post(url, headers=headers, json=data)
    print("RESPONSE", response)

    print(response.content)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the response content
        content = response.content
        # Decode the base64-encoded image
        decoded_image = base64.b64decode(content)
        # Create a PIL image from the decoded bytes
        try:
            pil_image = Image.frombytes("RGB", (512, 512), decoded_image)
            pil_image = pil_image.convert('RGB')
            pil_image.save("api_test.png")
            return pil_image
        except Exception as e:
            print("Failed to open image:", e)
    return None


def upload_image_to_postimages(image):
    # Convert PIL Image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)

    # Upload image to postimages.org
    url = "https://postimages.org/upload"
    files = {"file": image_bytes}
    response = requests.post(url, files=files)

    if response.status_code == 200:
        # Get the uploaded image URL from the response
        postimages_url = response.text
        return postimages_url

    print("Image upload to postimages.org failed with status code:", response.status_code)
    return None
def generate_with_node(node, prompt, next_prompt, blend_value, negative_prompt, args, root, frame, init_images=None):
    gs.should_run = True
    sampler_node, _ = node.getInput(2)
    make_latent = None
    latent = torch.zeros([1, 4, args.H // 8, args.W // 8])


    api = True

    if api:
        pass





    if isinstance(sampler_node, KSamplerNode):
        if len(init_images) > 0:
            if init_images[0] is not None:
                print("USING INIT")
                latent = encode_latent_ainodes(init_images[0])
        cond_node, index = node.getInput(1)
        conds = None
        # Get conditioning for current prompt
        c_1, _ = cond_node.evalImplementation_thread(prompt_override=prompt)
        inter = node.content.cond_schedule_checkbox.isChecked()
        if inter:
            if next_prompt != prompt:
                #If we still have a next prompt left, get an other conditioning
                next_conds, _ = cond_node.evalImplementation_thread(prompt_override=next_prompt)
                blend = min(blend_value * node.content.blend_factor.value(), 1.0)
                conds = addWeighted(next_conds[0], c_1[0], blend)
                print("Created Blended Conditioning for", prompt, next_prompt, blend_value)
            else:
                conds = c_1
        else:
            conds = c_1
        n_conds, _ = cond_node.evalImplementation_thread(prompt_override=negative_prompt)
        if len(init_images) > 0:
            if init_images[0] is not None:
                if len(node.getOutputs(2)) > 0:
                    try_cnet_node = node.getOutputs(2)[0]
                    if isinstance(try_cnet_node, DeforumCnetNode):
                        #node.setOutput(1, [pil_image_to_pixmap(init_images[0])])
                        conds = [try_cnet_node.evalImplementation_thread(conditioning = conds[0])]
        print("STRENGTH", args.strength)
        pixmaps, _ = sampler_node.evalImplementation_thread(cond_override=[conds, n_conds], args=args, latent_override=latent)

    elif isinstance(sampler_node, KandinskyNode):
        init = None
        if len(init_images) > 0:
            if init_images[0] is not None:
                init = pil_image_to_pixmap(init_images[0])
        if init is not None:
            init = [init]
        pixmaps = sampler_node.evalImplementation_thread(prompt_override=prompt, args=args, init_image=init)

    image = pixmap_to_pil_image(pixmaps[0])
    return image

def encode_latent_ainodes(init_image):
    #gs.models["vae"].first_stage_model.cuda()
    #image = init_image
    #image = image.convert("RGB")
    image = np.array(init_image).astype(np.float32) / 255.0
    image = image[None]# .transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = image.detach().cpu()
    torch_gc()
    latent = gs.models["vae"].encode(image)
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

    print("DEFORUM CONDITIONING INTERPOLATION")

    """prompt = node.deforum.prompt_series[frame]
    next_prompt = None
    if frame + anim_args.diffusion_cadence < anim_args.max_frames:

        curr_frame = frame

        next_prompt = node.deforum.prompt_series[frame + anim_args.diffusion_cadence]
    print("NEXT FRAME", frame, next_prompt)"""
    blend_value = 0.0

    #print(frame, anim_args.diffusion_cadence, node.deforum.prompt_series)

    next_frame = frame + anim_args.diffusion_cadence
    next_prompt = None
    while next_frame < anim_args.max_frames:
        next_prompt = node.deforum.prompt_series[next_frame]
        if next_prompt != prompt:
            # Calculate blend value based on distance and frame number
            prompt_distance = next_frame - frame
            max_distance = anim_args.max_frames - frame
            blend_value = prompt_distance / max_distance

            if blend_value >= 1.0:
                blend_value = 0.0

            break  # Exit the loop once a different prompt is found

        next_frame += anim_args.diffusion_cadence
    #print("CURRENT PROMPT", prompt)
    #print("NEXT FRAME:", next_prompt)
    #print("BLEND VALUE:", blend_value)
    #print("BLEND VALUE:", blend_value)
    #print("PARSED_PROMPT", prompt)
    #print("PARSED_PROMPT", prompt)
    if frame == 0:
        blend_value = 0.0
    if frame > 0:
        prev_prompt = node.deforum.prompt_series[frame - 1]
        if prev_prompt != prompt:
            blend_value = 0.0

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

        processed = generate_with_node(node, prompt, next_prompt, blend_value, negative_prompt, args, root, frame, [init_image])

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

        init_images = [init_image]
        image_mask = mask
        image_cfg_scale = args.pix2pix_img_cfg_scale

        #print_combined_table(args, anim_args, p, keys, frame)  # print dynamic table to cli

        #if is_controlnet_enabled(controlnet_args):
        #    process_with_controlnet(p, args, anim_args, loop_args, controlnet_args, root, is_img2img=True,
        #                            frame_idx=frame)


        processed = generate_with_node(node, prompt, next_prompt, blend_value, negative_prompt, args, root, frame, init_images)
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
        print("Warning: ConditioningAverage conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

    cond_from = conditioning_from[0][0]

    for i in range(len(conditioning_to)):
        t1 = conditioning_to[i][0]
        t0 = cond_from[:,:t1.shape[1]]
        if t0.shape[1] < t1.shape[1]:
            t0 = torch.cat([t0] + [torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]))], dim=1)

        tw = torch.mul(t1, conditioning_to_strength) + torch.mul(t0, (1.0 - conditioning_to_strength))
        n = [tw, conditioning_to[i][1].copy()]
        out.append(n)
    return [out]


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