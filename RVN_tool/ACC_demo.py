from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import onnx
import onnx2pytorch
import onnxruntime as ort

def NN_control_attitude(w1, w2, w3, phi1, phi2, phi3):

    input_data = np.array([[w1, w2, w3, phi1, phi2, phi3]], dtype=np.float32)
    model_path = 'model/attitude_control_3_64_torch.onnx'
    onnx_model = onnx.load(model_path)
    # model_ori = onnx2pytorch.ConvertModel(onnx_model)

    # 获取模型的图(graph)
    graph = onnx_model.graph
    # 获取模型的输入节点信息
    input_nodes = graph.input
    # 获取所有输入名称
    input_names = [node.name for node in input_nodes]
    # 打印输入名称
    input_name = input_names[0]

    session = ort.InferenceSession(model_path)

    inputs = onnx_model.graph.input
    input_tensors = [tensor for tensor in inputs if tensor.type.tensor_type.elem_type != 0]  # 过滤掉非张量类型

    # 打印每个输入的详细信息
    for input_tensor in input_tensors:
        print('Input Name:', input_tensor.name)
        # 获取输入的维度信息
        input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        print('Input Shape:', input_shape)
        # 获取输入的数据类型
        input_dtype = onnx.TensorProto.DataType.Name(input_tensor.type.tensor_type.elem_type)
        print('Input Data Type:', input_dtype)

    u = session.run(None, {input_name: input_data})


    return u[0][0]

def Attitude_dynamics(state, t):
    """Defines the RHS of the ODE used to simulate trajectories"""

    w1, w2, w3, phi1, phi2, phi3 = state

    u = NN_control_attitude(w1, w2, w3, phi1, phi2, phi3)

    w1 = 0.25 * (u[0] + w2 * w3)
    w2 = 0.5 * (u[1] - 3 * w1 * w3)
    w3 = u[2] + 2 * w1 * w2
    phi1 = 0.5 * (w2 * (phi1 ** 2 + phi2 ** 2 + phi3 ** 2 - phi3) + w3 * (phi1 ** 2 + phi2 ** 2 + phi3 ** 2 + phi2)
                  + w1 * (phi1 ** 2 + phi2 ** 2 + phi3 ** 2 + 1))
    phi2 = 0.5 * (w1 * (phi1 ** 2 + phi2 ** 2 + phi3 ** 2 + phi3) + w3 * (phi1 ** 2 + phi2 ** 2 + phi3 ** 2 - phi1)
                  + w2 * (phi1 ** 2 + phi2 ** 2 + phi3 ** 2 + 1))
    phi3 = 0.5 * (w1 * (phi1 ** 2 + phi2 ** 2 + phi3 ** 2 - phi2) + w2 * (phi1 ** 2 + phi2 ** 2 + phi3 ** 2 + phi1)
                  + w3 * (phi1 ** 2 + phi2 ** 2 + phi3 ** 2 + 1))
    # u0,1,2的值由NN_control_attitude()得到，暂时用1代替，把NN部署进来就替换掉

    return [w1, w2, w3, phi1, phi2, phi3]

###########################################################################
def NNCS_distance(y0, t):
    # 更新x的方程（示例：x = f(x)）
    # 这里需要您提供具体的方程
    # 假设两车初始距离D_rel为20
    # state = [w1, w2, w3, phi1, phi2, phi3]
    state = y0
    # 解常微分方程的函数
    state_final = odeint(Attitude_dynamics, y0=state, t=t)[-1]

    return state_final


# 解常微分方程的函数


    # 使用odeint求解常微分方程
    u_new = odeint(ode_system, y0 = u, t = t, args = (x,))

    return u_new[-1]  # 返回最终时间点的u值
###########################################################################

if __name__ == "__main__":

    # y0 = [w1, w2, w3, phi1, phi2, phi3] 这些原则上是要读取的, 规范点就是用dict存取， 但我们这个是demo捏
    y0 = [-0.45, -0.55, 0.65, -0.75, 0.85, -0.65]
    time_horizon = 3
    time_step = 0.1
    t = np.round(np.arange(0.0, time_horizon+time_step/2, time_step), 8)
    # trace = odeint(func=dynamics, y0=y0, t=t, tfirst=True)

    # 调用动力学方程，计算系统(飞机)的转角控制状态
    state_final = NNCS_distance(y0, t)

    print(f"The system will in the area: {state_final} in 3 seconds")
