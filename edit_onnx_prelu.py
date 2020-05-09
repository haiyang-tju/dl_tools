'''
用来修改 pytorch 导出为 onnx 模型后 PReLU 对于 TVM 不支持

修改 onnx 中的 PRelu 的 slope 维度，相当于执行了 flatten()

https://discuss.tvm.ai/t/prelu-dimension-mismatch-error-when-converting-onnx-model-to-tvm/4313/3

处理经过 onnx-simplifier 优化后的模型
python3 -m onnxsim m.onnx m_new_onnx
'''


import onnx

# 用名字进行索引
def createGraphMemberMap(graph_member_list):
    member_map=dict()
    for n in graph_member_list:
        member_map[n.name]=n
    return member_map


def edit_graph_tvm(input_model="", output_model="", verify=True):
    input_model = "./aaa_1.onnx"
    output_model = "./aaa_1_new.onnx"
    
    model = onnx.load(input_model)
    graph = model.graph
    if(verify):
        print("input model Errors: ", onnx.checker.check_model(model))

    # get all PRelu inputs
    fix_list_all = set()
    for node in graph.node:
        if node.op_type == "PRelu":
            fix_list_all.update(node.input)

    # get all mid data name
    mids = set()
    for mid_data in graph.value_info:
        mids.add(mid_data.name)

    # remove all intersection
    fix_list = fix_list_all-mids

    input_map = createGraphMemberMap(graph.input)
    initializer_map = createGraphMemberMap(graph.initializer)

    for fn in fix_list:
        if fn not in input_map:
            raise Exception("fn not in input_map error.")
        if fn not in initializer_map:
            raise Exception("fn not in initializer_map error.")

        new_shape = list(initializer_map[fn].dims)
        while 1 in new_shape:
            new_shape.remove(1)

        old_ip = input_map[fn]
        dt = old_ip.type.tensor_type.elem_type
        old_tensor = initializer_map[fn]

        new_ip = onnx.helper.make_tensor_value_info(fn, dt, new_shape)
        new_tensor = onnx.helper.make_tensor(fn, dt, new_shape, old_tensor.float_data)

        graph.input.remove(old_ip)
        graph.input.extend([new_ip])
        graph.initializer.remove(old_tensor)
        graph.initializer.extend([new_tensor])

    if(verify):    
        print("output model Errors: ", onnx.checker.check_model(model))
    onnx.save(model, output_model)


if __name__ == "__main__":
    edit_graph_tvm()

