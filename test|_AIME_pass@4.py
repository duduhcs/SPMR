import json
import argparse
import re
import asyncio
from typing import List, Dict, Any, Tuple
from sympy import simplify, sympify
import aiohttp


# Utility functions: Formatting and matching
def normalize_text(text: str) -> str:
    """Normalize text by removing unnecessary spaces, newlines, etc."""
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)  # Merge multiple spaces into one
    text = text.strip()  # Remove leading and trailing whitespaces or newlines
    return text.lower()  # Convert text to lowercase for comparison


def clean_latex(expr: str) -> str:
    """Clean LaTeX expressions and convert them into normal mathematical symbols"""
    if expr is None:
        return ""
    expr = str(expr)

    # Replace fractions: \frac{a}{b} -> (a/b)
    expr = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1/\2)", expr)

    # Remove \left and \right markers
    expr = expr.replace(r"\left", "").replace(r"\right", "")

    # Replace square root: \sqrt{a} -> √a
    expr = re.sub(r"\\sqrt\{([^{}]+)\}", r"√\1", expr)

    # Replace degree symbol: ^{\circ} -> °
    expr = expr.replace("^{\\circ}", "°")

    # Remove dollar signs from math formulas
    expr = expr.replace("$", "")

    # Replace \text{} marker: \text{...} -> ...
    expr = re.sub(r"\\text\{([^{}]*)\}", r"\1", expr)

    # Replace specific math operators
    expr = expr.replace("\\times", "*").replace("\\cdot", "*").replace("÷", "/")

    # Remove redundant backslashes
    expr = expr.replace("\\", "")

    return expr.strip()


def normalize_expression(expr: str) -> str:
    """Normalize mathematical expression for comparison"""
    if expr is None:
        return ""
    
    # Clean LaTeX expression
    expr = clean_latex(expr)

    # Remove all whitespaces
    expr = re.sub(r"\s+", "", expr)

    # Add missing multiplication operators, e.g., 2(3+4) -> 2*(3+4)
    expr = re.sub(r"(\d)\(", r"\1*(", expr)

    # Remove commas from numbers (e.g., 1,000 -> 1000)
    expr = expr.replace(",", "")

    # Remove x= prefix if it exists
    expr = re.sub(r"^x\s*=\s*", "", expr)

    return expr.strip()


def check_equivalent_expressions(expr1: str, expr2: str) -> bool:
    """Check if two mathematical expressions are equivalent"""
    if expr1 is None or expr2 is None:
        return False

    # Normalize both expressions
    expr1 = normalize_expression(expr1)
    expr2 = normalize_expression(expr2)

    # Compare normalized text directly
    if normalize_text(expr1) == normalize_text(expr2):
        return True

    # Use sympy to check for mathematical equivalence
    try:
        sympy_expr1 = sympify(expr1)
        sympy_expr2 = sympify(expr2)
        if simplify(sympy_expr1 - sympy_expr2) == 0:
            return True
    except Exception:
        pass

    return False


def extract_answer_from_response(response: str) -> str:
    """Extract the answer from a single model response (matches content after 'Answer:')"""
    match = re.search(r"Answer:\s*(.+)", response, re.IGNORECASE)
    if match:
        return clean_latex(match.group(1).strip())
    return ""


def check_answers(predictions: List[str], true_answer: str) -> bool:
    """Check if any of the candidate answers match the true answer"""
    for pred in predictions:
        if check_equivalent_expressions(pred, true_answer):
            return True
    return False


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSON file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading the JSON file {file_path}: {e}")
    return []


def save_json_file(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to a JSON file"""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Responses successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving responses to {file_path}: {e}")


def calculate_pass_at_4(output_data: List[Dict[str, Any]]) -> float:
    """Calculate Pass@4 accuracy for the model"""
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


# Asynchronous HTTP client for API communication
class AsyncRequestClient:
    def __init__(self, api_url: str, model_name: str, max_concurrent_requests: int = 30) -> None:
        self.api_url = api_url
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.session = None

    async def create_session(self):
        """Create an HTTP session"""
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0))

    async def close_session(self):
        """Close the HTTP session"""
        await self.session.close()

    async def extract_answer(self, content: str, max_retries: int = 3) -> str:
        """Send a query and extract the answer from the response"""
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
    """Response generator for generating multiple responses to a problem"""
    def __init__(self, api_url: str, model_name: str, max_concurrent_requests: int = 30) -> None:
        self.client = AsyncRequestClient(api_url, model_name, max_concurrent_requests)

    async def generate_responses(self, problems: List[str], num_samples: int = 4) -> List[Tuple[List[str], List[str]]]:
        """Generate multiple independent responses for each problem"""
        await self.client.create_session()

        async def process_problem(problem: str) -> Tuple[List[str], List[str]]:
            original_responses = []
            predicted_answers = []

            for _ in range(num_samples):  # Generate `num_samples` responses
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
    """Main asynchronous function"""
    data = load_json_file(json_file)

    if debug:
        data = data[:10]  # Process only the first 10 items in debug mode

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
