import time
from datetime import datetime
from uuid import uuid4

from qtpy.QtCore import Qt
from qtpy import QtCore, QtWidgets

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.base.resize_a_node import QDMGraphicsResizeNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from custom_nodes.ainodes_engine_deforum_nodes.deforum_nodes.keyframe_widget_alpha import OurTimeline, KeyFrame

OP_NODE_DEFORUM_KEYFRAME = get_next_opcode()


class DeforumKeyframeWidget(QDMNodeContentWidget):
    def initUI(self):
        self.createUI()
        self.create_main_layout()
        self.main_layout.addWidget(self.timeline)
        self.main_layout.addWidget(self.zoom_slider)


    def createUI(self):

        self.timeline = OurTimeline(300, 200)
        self.zoom_slider = QtWidgets.QSlider()
        self.zoom_slider.setOrientation(Qt.Horizontal)
        self.zoom_slider.setMinimum(1.0)
        self.zoom_slider.setMaximum(100.0)
        self.zoom_slider.setSingleStep(0.1)
        self.zoom_slider.setValue(1.0)


    def serialize(self):
        res = {}
        data = []
        for item in self.timeline.keyFrameList:

            data.append({item.uid: {"value": item.value,
                                    "position": item.position,
                                    "type": item.valueType}
                                     })
        res["data"] = data

        return res

    def deserialize(self, data, hashmap={}, restore_id:bool=True):
        #self.clear_rows()


        for value, datalist in data.items():
            print("value", value, "\n\nText:",  datalist)

            for item in datalist:
                for key, val in item.items():

                    k = key
                    v = val["value"]
                    p = val["position"]
                    t = val["type"]

                    keyframe = {}
                    keyframe[p] = KeyFrame(k, t, p, v)
                    self.timeline.keyFrameList.append(keyframe[p])



        super().deserialize(data, hashmap={}, restore_id=True)







@register_node(OP_NODE_DEFORUM_KEYFRAME)
class DeforumKeyframeNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/torch.png"
    op_code = OP_NODE_DEFORUM_KEYFRAME
    op_title = "Deforum Keyframe Node"
    content_label_objname = "deforum_keyframe_node"
    category = "DeForum"


    # output_socket_name = ["EXEC", "MODEL"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[6, 1], outputs=[6, 1])

    def initInnerClasses(self):
        self.content = DeforumKeyframeWidget(self)
        self.grNode = QDMGraphicsResizeNode(self)
        self.grNode.width = 400
        self.grNode.height = 800
        self.content.setMinimumWidth(400)
        self.content.setMinimumHeight(600)
        self.content.eval_signal.connect(self.evalImplementation)
        self.grNode.on_sizer_pos_mouse_release = self.resize
        self.content.zoom_slider.valueChanged.connect(self.update_timelineZoom)
        # self.timeline.scale = 1
        # self.timeline.timeline.start()

    def update_timelineZoom(self):
        self.content.timeline.verticalScale = self.content.zoom_slider.value()
        self.content.timeline.update()

    #@QtCore.Slot(object)
    def resize(self):
        size = {
            'width': self.grNode._width,
            'height': self.grNode._height}

        w = size["width"]
        h = size["height"]
        self.content.setMinimumWidth(w - 50)
        self.content.setMinimumHeight(h - 100)
        self.content.setMaximumWidth(w - 50)
        self.content.setMaximumHeight(h - 100)
        self.update_all_sockets()



    def evalImplementation_thread(self, index=0):
        print(self.content.timeline.keyFrameList)

        item_1 = self.content.timeline.keyFrameList[0]
        if item_1.position != 0:
            valueType = self.content.timeline.selectedValueType
            value = default_values[valueType]
            keyframe = {}
            uid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
            keyframe[0] = KeyFrame(uid, valueType, 0, value)
            self.content.timeline.keyFrameList.append(keyframe[0])

        self.content.timeline.update()

        time.sleep(0.1)
        string = ""
        x = 0
        for item in self.content.timeline.keyFrameList:
            if string == "":
                string = self.assemble_string(item.position, item.value)
            else:
                string = f"{string}, {self.assemble_string(item.position, item.value)}"

        print(string)


        return None

    def assemble_string(self, key, value):
        return(f"{key}:({value})")
    #@QtCore.Slot(object)
    def onWorkerFinished(self, result):
        self.busy = False
        #super().onWorkerFinished(None)
        self.setOutput(0, result)
        if len(self.getOutputs(1)) > 0:
            self.executeChild(output_index=1)


default_values = {"strength":0.65,
                  "zoom":1.00}