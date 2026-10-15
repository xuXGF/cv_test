from model_info import collect_model_info, load_session, benchmark

info = collect_model_info()  # 缺省使用包内自带 yolov8n.onnx
print(info.model_name, info.param_count, info.input_size)

session, input_hw = load_session("model_info/yolov8n.onnx")
print(session, input_hw)
# result = benchmark(session, image_bgr, warmup=3, rounds=10)
# print(result.total_ms, result.fps)