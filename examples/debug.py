import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True


data = torch.randn([16, 512, 7, 7], dtype=torch.float, device='cuda', requires_grad=True)
net = torch.nn.Conv2d(512, 512, kernel_size=[3, 3], padding=[1, 1], stride=[1, 1], dilation=[1, 1], groups=1)
net = net.cuda().float()
out = net(data)
out.backward(torch.randn_like(out))
torch.cuda.synchronize()

# ConvolutionParams 
#     data_type = CUDNN_DATA_FLOAT
#     padding = [1, 1, 0]
#     stride = [1, 1, 0]
#     dilation = [1, 1, 0]
#     groups = 1
#     deterministic = false
#     allow_tf32 = true
# input: TensorDescriptor 0x7f821002bb10
#     type = CUDNN_DATA_FLOAT
#     nbDims = 4
#     dimA = 16, 512, 7, 7, 
#     strideA = 25088, 49, 7, 1, 
# output: TensorDescriptor 0x7f821002b5f0
#     type = CUDNN_DATA_FLOAT
#     nbDims = 4
#     dimA = 16, 512, 7, 7, 
#     strideA = 25088, 49, 7, 1, 
# weight: FilterDescriptor 0x7f82100427f0
#     type = CUDNN_DATA_FLOAT
#     tensor_format = CUDNN_TENSOR_NCHW
#     nbDims = 4
#     dimA = 512, 512, 3, 3, 
# Pointer addresses: 
#     input: 0x7f81eec40000
#     output: 0x7f81df0d8000
#     weight: 0x7f81eee00000
# Additional pointer addresses: 
#     grad_output: 0x7f81df0d8000
#     grad_weight: 0x7f81eee00000
# Backward filter algorithm: 1