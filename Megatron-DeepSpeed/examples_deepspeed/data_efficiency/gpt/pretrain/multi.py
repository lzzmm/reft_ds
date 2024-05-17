import torch

# 检查CUDA是否可用
if torch.cuda.is_available():
    # 设置CUDA设备数量
    while True:
        device_count = 8

        # 创建随机矩阵并将其移动到CUDA设备上
        matrix_a = torch.rand(10000, 10000).cuda()
        matrix_b = torch.rand(10000, 10000).cuda()

        # 使用torch.bmm执行批量矩阵乘法
        result = torch.bmm(matrix_a.unsqueeze(0).expand(device_count, -1, -1),
                        matrix_b.unsqueeze(0).expand(device_count, -1, -1))

        # 打印结果
        print("Matrix A:")
        print(matrix_a)
        print("\nMatrix B:")
        print(matrix_b)
        print("\nResult of matrix multiplication:")
        print(result)
else:
    print("CUDA is not available. Please make sure you have CUDA-enabled GPU and PyTorch with CUDA support installed.")
