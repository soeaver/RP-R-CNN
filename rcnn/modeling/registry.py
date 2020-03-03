from utils.registry import Registry


"""
Feature Extractor.
"""
# Backbone
BACKBONES = Registry()

# FPN
FPN_BODY = Registry()

# SEMSEG
ROI_SEMSEG_HEADS = Registry()
ROI_SEMSEG_OUTPUTS = Registry()


"""
ROI Head.
"""
# Box Head
ROI_CLS_HEADS = Registry()
ROI_CLS_OUTPUTS = Registry()
ROI_BOX_HEADS = Registry()
ROI_BOX_OUTPUTS = Registry()

# Cascade Head
ROI_CASCADE_HEADS = Registry()
ROI_CASCADE_OUTPUTS = Registry()

# Mask Head
ROI_MASK_HEADS = Registry()
ROI_MASK_OUTPUTS = Registry()
MASKIOU_HEADS = Registry()
MASKIOU_OUTPUTS = Registry()

# Parsing Head
ROI_PARSING_HEADS = Registry()
ROI_PARSING_OUTPUTS = Registry()
PARSINGIOU_HEADS = Registry()
PARSINGIOU_OUTPUTS = Registry()
