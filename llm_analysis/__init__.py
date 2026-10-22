"""大模型分析模块：把提示词与测试数据传给大模型，返回分析文本。

用法::

    from llm_analysis import LLMAnalyzer

    analyzer = LLMAnalyzer()                    # 默认读取包内 llm_config.json
    text = analyzer.analyze(prompt, data)       # data 可为 str / dict / list
    text = analyzer.analyze_prompt(             # 从包内 prompts.json 按名称选模板
        "model_compare",
        params={"input_data": "输入数据文本"},   # 填充模板中的 {{占位符}}
        provider="openai",                      # 覆盖配置文件中的端点类型
        system="你是模型测评专家",
        temperature=0.7,
    )

说明：

* 端点、模型、超时由包内 ``llm_config.json`` 配置；``provider`` 字段在
  ``ollama``（POST {base_url}/api/chat）与 ``openai``（OpenAI 兼容
  POST {base_url}/chat/completions）两种端点之间切换。
* 提示词模板集中存放在包内 ``prompts.json``，键为提示词名称、值为模板
  文本；模板中以 ``{{参数名}}`` 声明占位符，调用 ``analyze_prompt`` 时
  通过 ``prompt_name`` 选择模板、``params`` 填充占位符，缺失或多余的
  参数均抛出异常，不做静默忽略。
* ``data`` 为 ``None`` 时只用提示词；为字符串时原样拼接；为 dict/list 时
  序列化为 JSON 文本后拼接，格式为 ``提示词 + 空行 + 以下是输入数据：``。
* 文件不存在、JSON 非法、必填字段为空、HTTP 非 2xx、响应缺字段时均抛出
  带明确信息的异常，不做静默兜底，不自动重试。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import requests

DEFAULT_CONFIG_PATH = Path(__file__).parent / "llm_config.json"
DEFAULT_PROMPTS_PATH = Path(__file__).parent / "prompts.json"

#: 支持的端点类型
VALID_PROVIDERS = ("ollama", "openai")

#: 提示词模板占位符语法：{{参数名}}
_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")

__all__ = [
    "LLMAnalyzer",
    "load_llm_config",
    "load_prompts",
    "render_prompt",
    "VALID_PROVIDERS",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_PROMPTS_PATH",
]


def load_llm_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """读取 LLM 端点配置文件，返回配置字典。

    默认读取包内 ``llm_config.json``；文件不存在、JSON 非法、顶层不是
    对象时抛出带明确信息的异常。
    """
    json_path = Path(path)
    try:
        text = json_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"LLM 配置文件不存在: {json_path}") from exc
    except OSError as exc:
        raise OSError(f"LLM 配置文件读取失败: {json_path}（{exc}）") from exc

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM 配置文件不是合法 JSON: {json_path}（第 {exc.lineno} 行：{exc.msg}）"
        ) from exc

    if not isinstance(raw, dict):
        raise TypeError(
            f"LLM 配置文件顶层应为 JSON 对象，实际为 {type(raw).__name__}: {json_path}"
        )
    return raw


def load_prompts(path: str | Path = DEFAULT_PROMPTS_PATH) -> dict[str, str]:
    """读取提示词模板文件，返回 ``{提示词名称: 模板文本}``。

    默认读取包内 ``prompts.json``；文件不存在、JSON 非法、顶层不是对象
    或某个模板不是非空字符串时抛出带明确信息的异常。
    """
    json_path = Path(path)
    try:
        text = json_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"提示词模板文件不存在: {json_path}") from exc
    except OSError as exc:
        raise OSError(f"提示词模板文件读取失败: {json_path}（{exc}）") from exc

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"提示词模板文件不是合法 JSON: {json_path}（第 {exc.lineno} 行：{exc.msg}）"
        ) from exc

    if not isinstance(raw, dict):
        raise TypeError(
            f"提示词模板文件顶层应为 JSON 对象，实际为 {type(raw).__name__}: {json_path}"
        )
    for name, template in raw.items():
        if not isinstance(template, str) or not template.strip():
            raise ValueError(
                f"prompts.json 中提示词 {name!r} 的模板必须是非空字符串，请先填写"
            )
    return raw


def render_prompt(template: str, params: dict[str, Any] | None = None) -> str:
    """用 ``params`` 填充模板中的 ``{{占位符}}``，返回渲染后的提示词文本。

    模板中的占位符缺少对应参数、或 ``params`` 中存在模板未使用的键时，
    均抛出带明确信息的异常，不做静默忽略；非字符串参数值序列化为
    JSON 文本后填充。
    """
    params = params or {}
    placeholders = set(_PLACEHOLDER_RE.findall(template))
    missing = sorted(placeholders - params.keys())
    if missing:
        raise ValueError(
            f"提示词模板缺少参数值: {missing}，请在 params 中提供对应值"
        )
    unknown = sorted(set(params) - placeholders)
    if unknown:
        raise ValueError(
            f"params 中存在模板未使用的参数: {unknown}，"
            f"模板占位符: {sorted(placeholders)}"
        )

    def _replace(match: re.Match[str]) -> str:
        value = params[match.group(1)]
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, indent=2)

    return _PLACEHOLDER_RE.sub(_replace, template)


def _require_str(section: dict[str, Any], key: str, provider: str) -> str:
    """取配置段中的非空字符串字段，为空时抛出带填写指引的异常。"""
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"llm_config.json 中 {provider}.{key} 未填写，请先在 llm_config.json 中填写后再调用"
        )
    return value.strip()


def _resolve_timeout(section: dict[str, Any], provider: str) -> float:
    """取配置段中的超时秒数，缺省 120，非法值抛出明确异常。"""
    raw = section.get("timeout", 120)
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"llm_config.json 中 {provider}.timeout 不是有效数值: {raw!r}"
        ) from exc


def _build_messages(
    prompt: str, data: Any, system: str | None
) -> list[dict[str, str]]:
    """组装聊天消息：可选 system 消息 + 拼接了输入数据的 user 消息。"""
    content = prompt
    if data is not None:
        if isinstance(data, str):
            data_text = data
        else:
            data_text = json.dumps(data, ensure_ascii=False, indent=2)
        content = f"{prompt}\n\n以下是输入数据：\n{data_text}"
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    return messages


def _brief(result: Any, limit: int = 500) -> str:
    """把响应结果截断为摘要文本，用于错误信息展示。"""
    try:
        text = json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(result)
    return text[:limit]


def _http_post_json(
    url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
) -> dict[str, Any]:
    """发送 JSON POST 请求并返回解析后的 JSON 响应，失败时抛出明确异常。"""
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(f"请求超时: {url}（timeout={timeout:g}s）") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"请求失败: {url}（{exc}）") from exc

    if not response.ok:
        raise RuntimeError(
            f"接口返回非 2xx: {url}（HTTP {response.status_code}）\n响应摘要: "
            f"{response.text[:500]}"
        )
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"接口响应不是合法 JSON: {url}\n响应摘要: {response.text[:500]}"
        ) from exc


def _call_ollama(
    base_url: str,
    model: str,
    timeout: float,
    messages: list[dict[str, str]],
    temperature: float | None,
) -> str:
    """调用 Ollama 端点（POST {base_url}/api/chat），返回回复文本。"""
    url = base_url.rstrip("/") + "/api/chat"
    payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
    if temperature is not None:
        payload["options"] = {"temperature": temperature}
    result = _http_post_json(
        url, {"Content-Type": "application/json"}, payload, timeout
    )
    try:
        return result["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Ollama 响应缺少 message.content 字段: {url}\n响应摘要: {_brief(result)}"
        ) from exc


def _call_openai(
    base_url: str,
    model: str,
    timeout: float,
    messages: list[dict[str, str]],
    temperature: float | None,
    api_key: str,
) -> str:
    """调用 OpenAI 兼容端点（POST {base_url}/chat/completions），返回回复文本。"""
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
    if temperature is not None:
        payload["temperature"] = temperature
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    result = _http_post_json(url, headers, payload, timeout)
    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"OpenAI 兼容接口响应缺少 choices[0].message.content 字段: {url}\n"
            f"响应摘要: {_brief(result)}"
        ) from exc


class LLMAnalyzer:
    """大模型分析器：将提示词与测试数据发送给大模型，返回分析文本。

    用法::

        analyzer = LLMAnalyzer()
        text = analyzer.analyze("分析以下指标", {"mAP@0.5": 0.6024})
    """

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        prompts_path: str | Path = DEFAULT_PROMPTS_PATH,
    ) -> None:
        """读取并保存端点配置与提示词模板文件路径。"""
        self.config = load_llm_config(config_path)
        self.prompts_path = prompts_path

    def analyze(
        self,
        prompt: str,
        data: Any = None,
        *,
        provider: str | None = None,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """把提示词（可附输入数据）发送给大模型，返回回复文本。

        :param prompt: 提示词文本
        :param data: 输入数据，None 只用提示词；str 原样拼接；
            dict/list 序列化为 JSON 文本拼接
        :param provider: 端点类型（ollama/openai），缺省用配置文件中的
            ``provider`` 字段
        :param system: 可选 system 提示词
        :param model: 可选模型名，缺省用配置文件中对应端点的 ``model``
        :param temperature: 可选采样温度，不传则不由请求携带
        :return: 模型回复的纯文本
        """
        provider_name = provider or self.config.get("provider")
        if provider_name not in VALID_PROVIDERS:
            raise ValueError(
                f"llm_config.json 中 provider 字段缺失或不合法: {provider_name!r}，"
                f"只支持 {' / '.join(VALID_PROVIDERS)}"
            )
        section = self.config.get(provider_name)
        if not isinstance(section, dict):
            raise ValueError(
                f"llm_config.json 中缺少 {provider_name} 配置段，请先填写"
            )
        base_url = _require_str(section, "base_url", provider_name)
        model_name = model or _require_str(section, "model", provider_name)
        timeout = _resolve_timeout(section, provider_name)
        messages = _build_messages(prompt, data, system)

        if provider_name == "ollama":
            return _call_ollama(base_url, model_name, timeout, messages, temperature)
        api_key = _require_str(section, "api_key", provider_name)
        return _call_openai(
            base_url, model_name, timeout, messages, temperature, api_key
        )

    def analyze_prompt(
        self,
        prompt_name: str,
        params: dict[str, Any] | None = None,
        *,
        provider: str | None = None,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """按名称从 ``prompts.json`` 选取提示词模板，填充参数后发送给大模型。

        :param prompt_name: prompts.json 中的提示词名称
        :param params: 用于填充模板中 ``{{占位符}}`` 的参数字典；占位符
            缺少参数值或参数未被模板使用时抛出异常
        :param provider: 端点类型（ollama/openai），缺省用配置文件中的
            ``provider`` 字段
        :param system: 可选 system 提示词
        :param model: 可选模型名，缺省用配置文件中对应端点的 ``model``
        :param temperature: 可选采样温度，不传则不由请求携带
        :return: 模型回复的纯文本
        """
        prompts = load_prompts(self.prompts_path)
        if prompt_name not in prompts:
            raise ValueError(
                f"prompts.json 中不存在提示词 {prompt_name!r}，"
                f"可用提示词: {sorted(prompts)}"
            )
        prompt = render_prompt(prompts[prompt_name], params)
        return self.analyze(
            prompt,
            provider=provider,
            system=system,
            model=model,
            temperature=temperature,
        )
