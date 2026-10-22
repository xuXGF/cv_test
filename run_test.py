"""大模型分析调用示例：从 prompts.json 按名称选择提示词模板，用 params 填充 {{占位符}}。

新增提示词只需在 llm_analysis/prompts.json 中添加一个模板（占位符写作 {{参数名}}），
调用时通过 prompt_name 切换模板、params 填充参数，无需修改本文件的提示词正文。
"""

from llm_analysis import LLMAnalyzer

# model_compare 模板的输入数据：精度对比 Markdown 表格
COMPARE_DATA = """| 模型           |   mAP@0.5 |   mAP@0.5:0.95 |   Precision |   Recall |   测试样本数 |
|:---------------|----------:|---------------:|------------:|---------:|-------------:|
| 云端原始模型   |    0.6231 |         0.4512 |      0.7842 |   0.5103 |          128 |
| 边缘端量化模型 |    0.6024 |         0.4387 |      0.7655 |   0.4919 |          128 |"""

# performance_analysis 模板的输入数据：性能测试 JSON 文本
PERF_DATA = """{
  "model": "/Users/xiaoxu/Desktop/ztf/大A大模型测评项目/ai边端_code/code/model_info/yolov8n.onnx",
  "providers": [
    "CPUExecutionProvider"
  ],
  "input_size": [
    640,
    640
  ],
  "num_images": 128,
  "batch_execution": "整批一次推理",
  "bs1_latency": "47.697",
  "bs1_pre": "2.290",
  "bs1_infer": "43.047",
  "bs1_post": "2.359",
  "bs1_fps": "20.97",
  "bs1_cpu": "47.2%",
  "bs1_npu": "",
  "bs1_mem": "358.5 MB",
  "bs4_latency": "170.327",
  "bs4_pre": "9.283",
  "bs4_infer": "152.125",
  "bs4_post": "8.916",
  "bs4_fps": "23.48",
  "bs4_cpu": "44.2%",
  "bs4_npu": "",
  "bs4_mem": "557.7 MB",
  "bs8_latency": "322.056",
  "bs8_pre": "17.498",
  "bs8_infer": "288.111",
  "bs8_post": "16.445",
  "bs8_fps": "24.84",
  "bs8_cpu": "43.3%",
  "bs8_npu": "",
  "bs8_mem": "820.5 MB",
}"""

analyzer = LLMAnalyzer()

# 模型精度对比分析：prompt_name 选择 model_compare 模板，params 填充 {{input_data}}
compare_text = analyzer.analyze_prompt(
    "model_compare",
    params={"input_data": COMPARE_DATA},
)
print("=== 模型精度对比分析 ===")
print(compare_text)

# 性能测试分析：prompt_name 切换为 performance_analysis 模板
perf_text = analyzer.analyze_prompt(
    "performance_analysis",
    params={"input_data": PERF_DATA},
)
print("=== 模型性能测试分析 ===")
print(perf_text)
