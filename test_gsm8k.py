import json
import argparse
import re
import asyncio
import os
from typing import List, Dict, Any
from difflib import SequenceMatcher
from sympy import simplify, sympify
import aiohttp

# 工具函数：格式化与匹配
def normalize_text(text: str) -> str:
    """标准化文本，去除空格、换行等"""
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)  # 合并多余空格为一个
    text = text.strip()  # 去除字符串首尾的空格或换行
    return text.lower()  # 转为小写进行比较


def clean_latex(expr: str) -> str:
    """清理 LaTeX 表达式并转换为普通数学符号"""
    if expr is None:
        return ""
    expr = str(expr)

    # 替换分数形式：\frac{a}{b} -> (a/b)
    expr = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1/\2)", expr)

    # 移除 \left 和 \right 标记
    expr = expr.replace(r"\left", "").replace(r"\right", "")

    # 处理根号形式：\sqrt{a} -> √a
    expr = re.sub(r"\\sqrt\{([^{}]+)\}", r"√\1", expr)

    # 替换角度符号：^{\circ} -> °
    expr = expr.replace("^{\\circ}", "°")

    # 移除数学公式的美元符号
    expr = expr.replace("$", "")

    # 替换文本标记 \text{...} -> ...
    expr = re.sub(r"\\text\{([^{}]*)\}", r"\1", expr)

    # 替换数学符号
    expr = expr.replace("\\times", "*").replace("\\cdot", "*").replace("÷", "/")

    # 移除多余的斜杠
    expr = expr.replace("\\", "")

    return expr.strip()


def clean_answer(answer: str) -> str:
    """对提取到的答案进行清理，保留第一组数字"""
    if answer is None:
        return ""

    # 移除 LaTeX 表达式的多余格式内容
    answer = clean_latex(answer)

    # 匹配第一组合法数字（包括负号、小数点和千分位符号）
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", answer)
    if match:
        # 清除数字中的逗号（如 7,000 -> 7000）
        return match.group(0).replace(",", "")
    
    return ""


def extract_answer_from_response(response: str) -> str:
    """综合从 response 提取答案"""
    if response is None:
        return ""

    # 第一段代码正则逻辑：优先匹配 Answer:$...$ 或 ####... 格式
    ans_re = re.compile(r"(?:Answer:\s*(.+)|#### (\-?[0-9\.\,]+))", re.IGNORECASE)
    match = ans_re.search(response)
    if match:
        answer = match.group(1) if match.group(1) else match.group(2)
        return clean_answer(answer)

    # 第二段代码逻辑
    # 1. 尝试匹配 `$$...$$`
    dollar_match = re.search(r"The answer is:\s*\$\$(.+?)\$\$", response, re.IGNORECASE)
    if dollar_match:
        return clean_answer(dollar_match.group(1))

    # 2. 尝试匹配 `\boxed{}`
    boxed_match = re.search(r"boxed\{([^{}]+)\}", response, re.IGNORECASE)
    if boxed_match:
        return clean_answer(boxed_match.group(1))

    # 3. 尝试匹配最后一组数字
    number_matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", response)
    if number_matches:
        return clean_answer(number_matches[-1])

    return ""


def normalize_expression(expr: str) -> str:
    """标准化数学表达式以便比较"""
    if expr is None:
        return ""

    expr = clean_latex(expr)
    expr = re.sub(r"\s+", "", expr)

    # 修正乘号省略情况
    expr = re.sub(r"(\d)\(", r"\1*(", expr)

    # 删除数字中的逗号
    expr = expr.replace(",", "")

    expr = re.sub(r"^x\s*=\s*", "", expr)
    
    return expr.strip()


def check_equivalent_expressions(expr1: str, expr2: str) -> bool:
    """检查两个数学表达式是否等价"""
    if expr1 is None or expr2 is None:
        return False

    expr1 = normalize_expression(expr1)
    expr2 = normalize_expression(expr2)

    if normalize_text(expr1) == normalize_text(expr2):
        return True

    try:
        sympy_expr1 = sympify(expr1)
        sympy_expr2 = sympify(expr2)
        if simplify(sympy_expr1 - sympy_expr2) == 0:
            return True
    except Exception:
        pass

    return False


def calculate_accuracy(output_data: List[Dict[str, Any]]) -> float:
    """计算模型的回答正确率"""
    correct_count = 0

    for item in output_data:
        response = item.get("response", "")
        true_answer = item.get("answer", "")

        true_final_answer = extract_answer_from_response(true_answer)
        predicted_answer = extract_answer_from_response(response)

        is_correct = check_equivalent_expressions(predicted_answer, true_final_answer)

        item["predicted_answer"] = predicted_answer
        item["is_correct"] = is_correct

        if is_correct:
            correct_count += 1

    accuracy = correct_count / len(output_data) if output_data else 0.0
    print(f"Correct Answers: {correct_count}/{len(output_data)}, Accuracy: {accuracy:.4f}")
    return accuracy


def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """从 JSONL 文件中加载数据"""
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
    except Exception as e:
        print(f"Error reading the JSONL file {file_path}: {e}")
    return data


def save_jsonl_file(data: List[Dict[str, Any]], file_path: str) -> None:
    """保存数据到 JSONL 文件中"""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Responses successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving responses to {file_path}: {e}")


class AsyncRequestClient:
    def __init__(self, api_url: str, model_name: str, max_concurrent_requests: int = 30, batch_size: int = 50) -> None:
        self.api_url = api_url
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.batch_size = batch_size
        self.session = None

    async def create_session(self):
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0))
        
    async def close_session(self):
        await self.session.close()
    
    async def extract_answer(self, content: str, max_retries: int = 3) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": content}
            ],
            "temperature": 0.2,
            "stream": False
        }

        for attempt in range(max_retries):
            try:
                async with self.semaphore:
                    async with self.session.post(
                        self.api_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result["choices"][0]["message"]["content"].strip()
                        elif response.status != 429:
                            print(f"Error {response.status}: {await response.text()}")
                            continue
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Final error after {max_retries} attempts: {e}")
                    return ""
                await asyncio.sleep(1 * (attempt + 1))
        return ""


class ResponseGenerator:
    def __init__(self, api_url: str, model_name: str, max_concurrent_requests: int = 30, batch_size: int = 50) -> None:
        self.client = AsyncRequestClient(api_url, model_name, max_concurrent_requests, batch_size)

    async def generate_responses(self, questions: List[str]) -> List[str]:
        await self.client.create_session()

        async def process_question(question: str) -> str:
            prompt = f"""{question}"""
            return await self.client.extract_answer(prompt)

        tasks = [process_question(question) for question in questions]
        responses = await asyncio.gather(*tasks)

        await self.client.close_session()

        return responses


async def main_async(inst_file: str, api_url: str, model_name: str, debug: bool) -> None:
    data = load_jsonl_file(inst_file)

    if debug:
        data = data[:10]  

    if not all("question" in item and "answer" in item for item in data):
        print("Invalid input format. 'question' and 'answer' fields are required in the JSONL file.")
        return

    questions = [item["question"] for item in data]

    response_generator = ResponseGenerator(api_url, model_name, max_concurrent_requests=30, batch_size=50)
    responses = await response_generator.generate_responses(questions)

    output_data = [
        {"question": item["question"], "answer": item["answer"], "response": response}
        for item, response in zip(data, responses)
    ]

    calculate_accuracy(output_data)

    output_file = inst_file.rsplit(".", 1)[0] + "_with_responses.jsonl"
    save_jsonl_file(output_data, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Response Generation and Accuracy Calculation")
    parser.add_argument("--inst_file", type=str, required=True, help="Path to the input JSONL file with questions and answers")
    parser.add_argument("--api_url", type=str, required=True, help="API URL for the model endpoint")
    parser.add_argument("--model_name", type=str, default="default", help="Name of the model to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (process a smaller subset)")

    args = parser.parse_args()

    asyncio.run(main_async(args.inst_file, args.api_url, args.model_name, args.debug))
