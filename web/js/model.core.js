import { app } from '../../../scripts/app.js'

function chainCallback(object, property, callback) {
    if (object == undefined) {
        //This should not happen.
        console.error("Tried to add callback to non-existant object")
        return;
    }
    if (property in object) {
        const callback_orig = object[property]
        object[property] = function () {
            const r = callback_orig.apply(this, arguments);
            callback.apply(this, arguments);
            return r
        };
    } else {
        object[property] = callback;
    }
}

function injectHidden(widget) {
    widget.computeSize = (target_width) => {
        if (widget.hidden) {
            return [0, -4];
        }
        return [target_width, 20];
    };
    widget._type = widget.type
    Object.defineProperty(widget, "type", {
        set: function (value) {
            widget._type = value;
        },
        get: function () {
            if (widget.hidden) {
                return "hidden";
            }
            return widget._type;
        }
    });
}

function addCustomLabel(nodeType, nodeData, widgetName = "detect") {
    //Add a callback which sets up the actual logic once the node is created
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        const node = this;
        const sizeOptionWidget = node.widgets.find((w) => w.name === widgetName);
        const LabelName = node.widgets.find((w) => w.name === "language_name");
        const LabelList = node.widgets.find((w) => w.name === "language_list");
        injectHidden(LabelName);
        injectHidden(LabelList);
        sizeOptionWidget._value = sizeOptionWidget.value;
        Object.defineProperty(sizeOptionWidget, "value", {
            set: function (value) {
                //TODO: Only modify hidden/reset size when a change occurs
                if (value === "choose") {
                    LabelName.hidden = true;
                    LabelList.hidden = false;
                } else if (value === "input") {
                    LabelName.hidden = false;
                    LabelList.hidden = true;
                } else {
                     LabelName.hidden = true;
                     LabelList.hidden = true;
                }
                node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]])
                this._value = value;
            },
            get: function () {
                return this._value;
            }
        });
        sizeOptionWidget.value = sizeOptionWidget._value;
    });
}
app.registerExtension({
    name: "ComfyUI-EasyOCR.Core",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.name == "Apply EasyOCR") {
            addCustomLabel(nodeType, nodeData, "detect")
        }
    }
});