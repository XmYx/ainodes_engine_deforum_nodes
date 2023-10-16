import math
import torch
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

OP_NODE_DEFORUM_COMBINE_LATENTS = get_next_opcode()


def combine_latents(pixels, mask, grow_mask_by=48):
    pixels = pixels.clone()

    mask = mask[0,:,:,0]


    x = (pixels.shape[1] // 8) * 8
    y = (pixels.shape[2] // 8) * 8
    mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
                                           size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")
    pixels = pixels.clone()
    if pixels.shape[1] != x or pixels.shape[2] != y:
        x_offset = (pixels.shape[1] % 8) // 2
        y_offset = (pixels.shape[2] % 8) // 2
        pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]

    # grow mask by a few pixels to keep things seamless in latent space
    if grow_mask_by == 0:
        mask_erosion = mask
    else:
        kernel_tensor = torch.ones((1, 1, grow_mask_by, grow_mask_by))
        padding = math.ceil((grow_mask_by - 1) / 2)

        mask_erosion = torch.clamp(torch.nn.functional.conv2d(mask.round(), kernel_tensor, padding=padding), 0, 1)

    m = (1.0 - mask.round()).squeeze(1)
    for i in range(3):
        pixels[:, :, :, i] -= 0.5
        pixels[:, :, :, i] *= m
        pixels[:, :, :, i] += 0.5
    pixels = pixels.permute(0, 3, 1, 2)

    return {"samples":pixels, "noise_mask": (mask_erosion[:,:,:x,:y].round())}


class DeforumCombineLatentsWidget(QDMNodeContentWidget):
    def initUI(self):
        self.create_main_layout(grid=1)

@register_node(OP_NODE_DEFORUM_COMBINE_LATENTS)
class DeforumLatentCombineNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Deforum Latent Combine"
    op_code = OP_NODE_DEFORUM_COMBINE_LATENTS
    op_title = "Deforum Latent Combine"
    content_label_objname = "deforum_combine_latents_node"
    category = "aiNodes Deforum/DeForum"
    NodeContent_class = DeforumCombineLatentsWidget
    dim = (240, 120)
    custom_input_socket_name = ["IMAGE", "MASK", "EXEC"]

    make_dirty = True

    def __init__(self, scene):
        super().__init__(scene, inputs=[5,5,1], outputs=[2,1])
        self.depth_model = None

    def evalImplementation_thread(self, index=0):

        image = self.getInputData(0)
        mask = self.getInputData(1)

        if image is not None and mask is not None:
            with torch.no_grad():
                image = combine_latents(image, mask)
        return [image]



