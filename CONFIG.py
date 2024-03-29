ROI_SIZE = (96, 96, 96) # patch size of the image
NET_ARGS = {
    "in_channels":1, # number of input channels to the net
    "out_channels":2, # number of out cahnnels in the output for the net
    "n_deep_supervision":3 ,
    # "n_deep_supervision": num layers to stack at the output of the network for the deep supervision loss. 
    # allowed range (1-3) the input will be clipped if it is out of the range
    "conv_mapping": True #conv_mapping (bool): if True use convolution as a weighted aggregation of x and g instead of regular sum in each attention gate

    }
    