
sd = "./yolox_nano.pth"

import torch
model_dict = torch.load(sd, map_location=torch.device('cpu'))
if "state_dict" in model_dict:
    model_dict = model_dict["state_dict"]
if "model" in model_dict:
    model_dict = model_dict["model"]

new_dict = dict()
for k, v in model_dict.items():
    new_k = k
	
    if "backbone.backbone." in k:
        new_k = k.replace("backbone.backbone.", "backbone.")
    if "backbone.dark2." in new_k:
        new_k = new_k.replace("backbone.dark2.", "backbone.stage1.")
    if "backbone.dark3." in new_k:
        new_k = new_k.replace("backbone.dark3.", "backbone.stage2.")
    if "backbone.dark4." in new_k:
        new_k = new_k.replace("backbone.dark4.", "backbone.stage3.")
    if "backbone.dark5." in new_k:
        new_k = new_k.replace("backbone.dark5.", "backbone.stage4.")
    if "dconv." in new_k:
        new_k = new_k.replace("dconv.", "depthwise_conv.")
    if "pconv." in new_k:
        new_k = new_k.replace("pconv.", "pointwise_conv.")
    if "backbone.stage1.1.conv1." in new_k:
        new_k = new_k.replace("backbone.stage1.1.conv1.", "backbone.stage1.1.main_conv.")
    if "backbone.stage1.1.conv2." in new_k:
        new_k = new_k.replace("backbone.stage1.1.conv2.", "backbone.stage1.1.short_conv.")
    if "backbone.stage1.1.conv3." in new_k:
        new_k = new_k.replace("backbone.stage1.1.conv3.", "backbone.stage1.1.final_conv.")
    if ".m." in new_k:
        new_k = new_k.replace(".m.", ".blocks.")
    if "backbone.stage2.1.conv1." in new_k:
        new_k = new_k.replace("backbone.stage2.1.conv1.", "backbone.stage2.1.main_conv.")
    if "backbone.stage2.1.conv2." in new_k:
        new_k = new_k.replace("backbone.stage2.1.conv2.", "backbone.stage2.1.short_conv.")
    if "backbone.stage2.1.conv3." in new_k:
        new_k = new_k.replace("backbone.stage2.1.conv3.", "backbone.stage2.1.final_conv.")
    if "backbone.stage3.1.conv1." in new_k:
        new_k = new_k.replace("backbone.stage3.1.conv1.", "backbone.stage3.1.main_conv.")
    if "backbone.stage3.1.conv2." in new_k:
        new_k = new_k.replace("backbone.stage3.1.conv2.", "backbone.stage3.1.short_conv.")
    if "backbone.stage3.1.conv3." in new_k:
        new_k = new_k.replace("backbone.stage3.1.conv3.", "backbone.stage3.1.final_conv.")
    if "backbone.stage4.2.conv1." in new_k:
        new_k = new_k.replace("backbone.stage4.2.conv1.", "backbone.stage4.2.main_conv.")
    if "backbone.stage4.2.conv2." in new_k:
        new_k = new_k.replace("backbone.stage4.2.conv2.", "backbone.stage4.2.short_conv.")
    if "backbone.stage4.2.conv3." in new_k:
        new_k = new_k.replace("backbone.stage4.2.conv3.", "backbone.stage4.2.final_conv.")
    if "backbone.lateral_conv0." in new_k:
        new_k = new_k.replace("backbone.lateral_conv0.", "neck.reduce_layers.0.")
    if "backbone.reduce_conv1." in new_k:
        new_k = new_k.replace("backbone.reduce_conv1.", "neck.reduce_layers.1.")
    if "backbone.C3_p4." in new_k:
        new_k = new_k.replace("backbone.C3_p4.", "neck.top_down_blocks.0.")
    if "neck.top_down_blocks.0.conv1." in new_k:
        new_k = new_k.replace("neck.top_down_blocks.0.conv1.", "neck.top_down_blocks.0.main_conv.")
    if "neck.top_down_blocks.0.conv2." in new_k:
        new_k = new_k.replace("neck.top_down_blocks.0.conv2.", "neck.top_down_blocks.0.short_conv.")
    if "neck.top_down_blocks.0.conv3." in new_k:
        new_k = new_k.replace("neck.top_down_blocks.0.conv3.", "neck.top_down_blocks.0.final_conv.")
    if "backbone.C3_p3." in new_k:
        new_k = new_k.replace("backbone.C3_p3.", "neck.top_down_blocks.1.")
    if "neck.top_down_blocks.1.conv1." in new_k:
        new_k = new_k.replace("neck.top_down_blocks.1.conv1.", "neck.top_down_blocks.1.main_conv.")
    if "neck.top_down_blocks.1.conv2." in new_k:
        new_k = new_k.replace("neck.top_down_blocks.1.conv2.", "neck.top_down_blocks.1.short_conv.")
    if "neck.top_down_blocks.1.conv3." in new_k:
        new_k = new_k.replace("neck.top_down_blocks.1.conv3.", "neck.top_down_blocks.1.final_conv.")
    
    if "backbone.bu_conv2." in new_k:
        new_k = new_k.replace("backbone.bu_conv2.", "neck.downsamples.0.")
    if "backbone.bu_conv1." in new_k:
        new_k = new_k.replace("backbone.bu_conv1.", "neck.downsamples.1.")
    
    if "backbone.C3_n3." in new_k:
        new_k = new_k.replace("backbone.C3_n3.", "neck.bottom_up_blocks.0.")
    if "neck.bottom_up_blocks.0.conv1." in new_k:
        new_k = new_k.replace("neck.bottom_up_blocks.0.conv1.", "neck.bottom_up_blocks.0.main_conv.")
    if "neck.bottom_up_blocks.0.conv2." in new_k:
        new_k = new_k.replace("neck.bottom_up_blocks.0.conv2.", "neck.bottom_up_blocks.0.short_conv.")
    if "neck.bottom_up_blocks.0.conv3." in new_k:
        new_k = new_k.replace("neck.bottom_up_blocks.0.conv3.", "neck.bottom_up_blocks.0.final_conv.")
    if "backbone.C3_n4." in new_k:
        new_k = new_k.replace("backbone.C3_n4.", "neck.bottom_up_blocks.1.")
    if "neck.bottom_up_blocks.1.conv1." in new_k:
        new_k = new_k.replace("neck.bottom_up_blocks.1.conv1.", "neck.bottom_up_blocks.1.main_conv.")
    if "neck.bottom_up_blocks.1.conv2." in new_k:
        new_k = new_k.replace("neck.bottom_up_blocks.1.conv2.", "neck.bottom_up_blocks.1.short_conv.")
    if "neck.bottom_up_blocks.1.conv3." in new_k:
        new_k = new_k.replace("neck.bottom_up_blocks.1.conv3.", "neck.bottom_up_blocks.1.final_conv.")
    
    if "head.stems." in new_k:
        new_k = new_k.replace("head.stems.", "neck.out_convs.")
    if "head.cls_convs." in new_k:
        new_k = new_k.replace("head.cls_convs.", "bbox_head.multi_level_cls_convs.")
    if "head.reg_convs." in new_k:
        new_k = new_k.replace("head.reg_convs.", "bbox_head.multi_level_reg_convs.")
    if "head.cls_preds." in new_k:
        new_k = new_k.replace("head.cls_preds.", "bbox_head.multi_level_conv_cls.")
    if "head.reg_preds." in new_k:
        new_k = new_k.replace("head.reg_preds.", "bbox_head.multi_level_conv_reg.")
    if "head.obj_preds." in new_k:
        new_k = new_k.replace("head.obj_preds.", "bbox_head.multi_level_conv_obj.")
    
    if "bbox_head.multi_level_conv_cls." in new_k:
        new_dict[new_k] = v[:num_classes,...] # there take the num_classes
    else:
        new_dict[new_k] = v

# 保存 pth 模型文件
data = {'state_dict': new_dict}
pth = "./yolox_nano_mmdet.pth"
torch.save(data, pth)

for k,v in new_dict.items():
    print(k,"-",v.size())
