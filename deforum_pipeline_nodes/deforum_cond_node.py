import contextlib
import torch

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

from ainodes_frontend import singleton as gs
OP_NODE_DEFORUM_SD_COND = get_next_opcode()
import torch.nn.functional as F

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


blend_methods = ["linear", "sigmoidal", "gaussian", "pyramid"]

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

class DeforumConditioningWidget(QDMNodeContentWidget):
    def initUI(self):
        self.blend_method = self.create_combo_box(blend_methods, "Blend Method")
        self.create_main_layout(grid=1)


@register_node(OP_NODE_DEFORUM_SD_COND)
class DeforumConditioningNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/conditioning.png"
    op_code = OP_NODE_DEFORUM_SD_COND
    op_title = "Deforum SD Conditioning"
    content_label_objname = "deforum_sdcond_node"
    category = "aiNodes Deforum/DeForum"
    NodeContent_class = DeforumConditioningWidget
    dim = (240, 140)

    make_dirty = True
    custom_input_socket_name = ["CLIP", "DATA", "EXEC"]
    custom_output_socket_name = ["DATA", "NEGATIVE", "POSITIVE", "EXEC"]


    def __init__(self, scene):
        super().__init__(scene, inputs=[4,6,1], outputs=[6,3,3,1])
        # Create a worker object
        self.device = gs.device
        if self.device in [torch.device('mps'), torch.device('cpu')]:
            self.context = contextlib.nullcontext()
        else:
            self.context = torch.autocast(gs.device.type)
        self.clip_skip = -2

    #@QtCore.Slot()
    def evalImplementation_thread(self, index=0, prompt_override=None):
        clip = self.getInputData(0)

        data = self.getInputData(1)
        if data is not None and gs.should_run:

            prompt = data.get("prompt", "")
            negative_prompt = data.get("negative_prompt", "")

            next_prompt = data.get("next_prompt", None)

            print(f"[ Deforum Conds: {prompt}, {negative_prompt} ]")
            cond = self.get_conditioning(prompt=prompt, clip=clip)

            prompt_blend = data.get("prompt_blend", 0.0)
            if next_prompt != prompt and prompt_blend != 0.0 and next_prompt is not None:
                next_cond = self.get_conditioning(prompt=next_prompt, clip=clip)
                cond = blend_tensors(cond, next_cond, prompt_blend, self.content.blend_method.currentText())
                print(f"[ Deforum Cond Blend: {next_prompt}, {prompt_blend} ]")

            n_cond = self.get_conditioning(prompt=negative_prompt, clip=clip)

            return [data, n_cond, cond]
        else:
            return [None, None, None]

    def get_conditioning(self, prompt="", clip=None, progress_callback=None):

        """if gs.loaded_models["loaded"] == []:
            for node in self.scene.nodes:
                if isinstance(node, TorchLoaderNode):
                    node.evalImplementation()
                    #print("Node found")"""



        with self.context:
            with torch.no_grad():
                clip_skip = -2
                if self.clip_skip != clip_skip or clip.layer_idx != clip_skip:
                    clip.layer_idx = clip_skip
                    clip.clip_layer(clip_skip)
                    self.clip_skip = clip_skip

                tokens = clip.tokenize(prompt)
                cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

                return [[cond, {"pooled_output": pooled}]]
