import json
import argparse
import re
import asyncio
from typing import List, Dict, Any, Tuple
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

    expr = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1/\2)", expr)
    expr = expr.replace(r"\left", "").replace(r"\right", "")
    expr = re.sub(r"\\sqrt\{([^{}]+)\}", r"√\1", expr)
    expr = expr.replace("^{\\circ}", "°")
    expr = expr.replace("$", "")
    expr = re.sub(r"\\text\{([^{}]*)\}", r"\1", expr)
    expr = expr.replace("\\times", "*").replace("\\cdot", "*").replace("÷", "/")
    expr = expr.replace("\\", "")

    return expr.strip()


def normalize_expression(expr: str) -> str:
    """标准化数学表达式以便比较"""
    if expr is None:
        return ""

    expr = clean_latex(expr)
    expr = re.sub(r"\s+", "", expr)
    expr = re.sub(r"(\d)\(", r"\1*(", expr)
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


def extract_answer_from_response(response: str) -> str:
    """从单个模型响应中提取答案（匹配 Answer: 后内容）"""
    match = re.search(r"Answer:\s*(.+)", response, re.IGNORECASE)
    if match:
        return clean_latex(match.group(1).strip())
    return ""


def check_answers(predictions: List[str], true_answer: str) -> bool:
    """检查多个候选答案是否匹配真实答案"""
    for pred in predictions:
        if check_equivalent_expressions(pred, true_answer):
            return True
    return False


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """从 JSON 文件中加载数据"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading the JSON file {file_path}: {e}")
    return []


def save_json_file(data: List[Dict[str, Any]], file_path: str) -> None:
    """保存数据到 JSON 文件中"""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Responses successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving responses to {file_path}: {e}")


def calculate_pass_at_4(output_data: List[Dict[str, Any]]) -> float:
    """计算模型的 Pass@4 正确率"""
    correct_count = 0

    for item in output_data:
        predictions = item.get("predicted_answers", [])
        true_answer = item.get("Answer", "")

        is_correct = check_answers(predictions, true_answer)

        item["is_correct"] = is_correct
        if is_correct:
            correct_count += 1

    accuracy = correct_count / len(output_data) if output_data else 0.0
    print(f"Pass@4 Results: {correct_count}/{len(output_data)}, Pass@4 Accuracy: {accuracy:.4f}")
    return accuracy


class AsyncRequestClient:
    def __init__(self, api_url: str, model_name: str, max_concurrent_requests: int = 30) -> None:
        self.api_url = api_url
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.session = None

    async def create_session(self):
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0))

    async def close_session(self):
        await self.session.close()

    async def extract_answer(self, content: str, max_retries: int = 3) -> str:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.2,
            "stream": False,
            "max_tokens": 4096
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
    def __init__(self, api_url: str, model_name: str, max_concurrent_requests: int = 30) -> None:
        self.client = AsyncRequestClient(api_url, model_name, max_concurrent_requests)

    async def generate_responses(self, problems: List[str], num_samples: int = 4) -> List[Tuple[List[str], List[str]]]:
        """生成每个问题的多个独立响应，并返回原始回复与提取结果"""
        await self.client.create_session()

        async def process_problem(problem: str) -> Tuple[List[str], List[str]]:
            original_responses = []
            predicted_answers = []

            for _ in range(num_samples):
                prompt = f"""{problem}"""
                original_response = await self.client.extract_answer(prompt)
                extracted_answer = extract_answer_from_response(original_response)

                original_responses.append(original_response)
                predicted_answers.append(extracted_answer)

            return original_responses, predicted_answers

        tasks = [process_problem(problem) for problem in problems]
        all_responses = await asyncio.gather(*tasks)

        await self.client.close_session()
        return all_responses


async def main_async(json_file: str, api_url: str, model_name: str, debug: bool) -> None:
    data = load_json_file(json_file)

    if debug:
        data = data[:10] 

    if not all("Problem" in item for item in data):
        print("No 'Problem' fields found in the input JSON file.")
        return

    problems = [item["Problem"] for item in data]
    response_generator = ResponseGenerator(api_url, model_name, max_concurrent_requests=30)
    responses_with_original = await response_generator.generate_responses(problems, num_samples=4)

    output_data = []
    for item, (original_responses, predicted_answers) in zip(data, responses_with_original):
        output_data.append({
            "Problem": item["Problem"],
            "Answer": item["Answer"],
            "original_responses": original_responses,
            "predicted_answers": predicted_answers
        })

    calculate_pass_at_4(output_data)

    output_file = json_file.rsplit(".", 1)[0] + "_with_responses.json"
    save_json_file(output_data, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pass@4 Evaluation")
    parser.add_argument("--json_file", type=str, required=True, help="Path to the input JSON file with problems")
    parser.add_argument("--api_url", type=str, required=True, help="API URL for the model endpoint")
    parser.add_argument("--model_name", type=str, default="default", help="Name of the model to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (process a smaller subset)")

    args = parser.parse_args()
    asyncio.run(main_async(args.json_file, args.api_url, args.model_name, args.debug))
