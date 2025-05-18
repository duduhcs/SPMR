import json
import argparse
import re
import asyncio
import os
from typing import List, Dict, Any
from difflib import SequenceMatcher
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
    return text.lower()  # Convert to lowercase for comparison


def clean_latex(expr: str) -> str:
    """Clean LaTeX expressions and convert them into normal mathematical symbols"""
    if expr is None:
        return ""
    expr = str(expr)

    # Replace fraction format: \frac{a}{b} -> (a/b)
    expr = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1/\2)", expr)

    # Remove \left and \right markers
    expr = expr.replace(r"\left", "").replace(r"\right", "")

    # Handle square root format: \sqrt{a} -> √a
    expr = re.sub(r"\\sqrt\{([^{}]+)\}", r"√\1", expr)

    # Replace degree symbol: ^{\circ} -> °
    expr = expr.replace("^{\\circ}", "°")

    # Remove dollar signs used in math formulas
    expr = expr.replace("$", "")

    # Replace text markers \text{...} -> ...
    expr = re.sub(r"\\text\{([^{}]*)\}", r"\1", expr)

    # Replace math symbols
    expr = expr.replace("\\times", "*").replace("\\cdot", "*").replace("÷", "/")

    # Remove redundant backslashes
    expr = expr.replace("\\", "")

    return expr.strip()


def clean_answer(answer: str) -> str:
    """
    Clean the extracted answer to retain the primary numeric group.
    Supports formats like 7,000, 0.77, -0.7, etc.
    """
    if answer is None:
        return ""

    # Remove extra formatting from LaTeX expressions
    answer = clean_latex(answer)

    # Match the first valid numeric group (including negatives, decimals, and thousand separators)
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", answer)
    if match:
        # Remove commas from numbers (e.g., 7,000 -> 7000)
        return match.group(0).replace(",", "")
    
    return ""


def extract_answer_from_response(response: str) -> str:
    """
    Extract the answer from the response:
    1. Prioritize regex logic that matches formats like Answer:$...$ or ####... .
    2. Fallback logic applies if the above fails, attempting:
        - Matching $$...$$
        - Matching \boxed{} content
        - Matching the last numeric group.
    3. Clean the extracted answer using clean_answer.
    """
    if response is None:
        return ""

    # Primary regex logic: prioritize Answer:$...$ or ####... formats
    ans_re = re.compile(r"(?:Answer:\s*(.+)|#### (\-?[0-9\.\,]+))", re.IGNORECASE)
    match = ans_re.search(response)
    if match:
        # Extract the first match from the groups
        answer = match.group(1) if match.group(1) else match.group(2)
        return clean_answer(answer)

    # Fallback logic with additional formats
    # 1. Try matching `$$...$$`
    dollar_match = re.search(r"The answer is:\s*\$\$(.+?)\$\$", response, re.IGNORECASE)
    if dollar_match:
        return clean_answer(dollar_match.group(1))

    # 2. Try matching `\boxed{}`
    boxed_match = re.search(r"boxed\{([^{}]+)\}", response, re.IGNORECASE)
    if boxed_match:
        return clean_answer(boxed_match.group(1))

    # 3. Try extracting the last numeric group
    number_matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", response)
    if number_matches:
        return clean_answer(number_matches[-1])

    # Return an empty string if no match is found
    return ""


def normalize_expression(expr: str) -> str:
    """Normalize mathematical expressions for comparison"""
    if expr is None:
        return ""

    # Clean LaTeX expressions
    expr = clean_latex(expr)

    # Remove all whitespaces
    expr = re.sub(r"\s+", "", expr)

    # Add missing multiplication symbols, e.g., 2(x+1) -> 2*(x+1)
    expr = re.sub(r"(\d)\(", r"\1*(", expr)

    # Remove commas from numbers (e.g., 1,000 -> 1000)
    expr = expr.replace(",", "")

    # Remove the 'x=' prefix if present
    expr = re.sub(r"^x\s*=\s*", "", expr)
    
    return expr.strip()


def check_equivalent_expressions(expr1: str, expr2: str) -> bool:
    """
    Check whether two mathematical expressions are equivalent:
    1. Normalize the expressions to remove formatting differences.
    2. Use sympy to verify mathematical equivalence.
    """
    if expr1 is None or expr2 is None:
        return False

    expr1 = normalize_expression(expr1)
    expr2 = normalize_expression(expr2)

    # Direct string comparison
    if normalize_text(expr1) == normalize_text(expr2):
        return True

    # Use sympy for mathematical equivalence checking
    try:
        sympy_expr1 = sympify(expr1)
        sympy_expr2 = sympify(expr2)
        if simplify(sympy_expr1 - sympy_expr2) == 0:
            return True
    except Exception:
        pass

    return False


def calculate_accuracy(output_data: List[Dict[str, Any]]) -> float:
    """Calculate the model's answer accuracy"""
    correct_count = 0

    for item in output_data:
        response = item.get("response", "")
        true_answer = item.get("answer", "")

        # Extract final answers
        true_final_answer = extract_answer_from_response(true_answer)
        predicted_answer = extract_answer_from_response(response)

        # Check if the answers match
        is_correct = check_equivalent_expressions(predicted_answer, true_final_answer)

        # Record the result
        item["predicted_answer"] = predicted_answer
        item["is_correct"] = is_correct

        if is_correct:
            correct_count += 1

    accuracy = correct_count / len(output_data) if output_data else 0.0
    print(f"Correct Answers: {correct_count}/{len(output_data)}, Accuracy: {accuracy:.4f}")
    return accuracy


# File operations: Loading and saving
def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file"""
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
    except Exception as e:
        print(f"Error reading the JSONL file {file_path}: {e}")
    return data


def save_jsonl_file(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to a JSONL file"""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Responses successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving responses to {file_path}: {e}")


class AsyncRequestClient:
    """Asynchronous HTTP client to interact with the API"""
    def __init__(self, api_url: str, model_name: str, max_concurrent_requests: int = 30, batch_size: int = 50) -> None:
        self.api_url = api_url
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.batch_size = batch_size
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
    """Response generator for asynchronous batch processing"""
    def __init__(self, api_url: str, model_name: str, max_concurrent_requests: int = 30, batch_size: int = 50) -> None:
        self.client = AsyncRequestClient(api_url, model_name, max_concurrent_requests, batch_size)

    async def generate_responses(self, questions: List[str]) -> List[str]:
        """Generate responses asynchronously for a batch of questions"""
        await self.client.create_session()

        async def process_question(question: str) -> str:
            prompt = f"""{question}"""
            return await self.client.extract_answer(prompt)

        # Process questions asynchronously
        tasks = [process_question(question) for question in questions]
        responses = await asyncio.gather(*tasks)

        # Close the session
        await self.client.close_session()

        return responses


async def main_async(inst_file: str, api_url: str, model_name: str, debug: bool) -> None:
    # Load input data
    data = load_jsonl_file(inst_file)

    if debug:
        data = data[:10]  # In debug mode, only process the first 10 entries

    # Validate input data
    if not all("question" in item and "answer" in item for item in data):
        print("Invalid input format. 'question' and 'answer' fields are required in the JSONL file.")
        return

    # Extract questions
    questions = [item["question"] for item in data]

    # Initialize generator and generate responses
    response_generator = ResponseGenerator(api_url, model_name, max_concurrent_requests=30, batch_size=50)
    responses = await response_generator.generate_responses(questions)

    # Save results and calculate accuracy
    output_data = [
        {"question": item["question"], "answer": item["answer"], "response": response}
        for item, response in zip(data, responses)
    ]

    # Calculate accuracy
    calculate_accuracy(output_data)

    # Save outputs
    output_file = inst_file.rsplit(".", 1)[0] + "_with_responses.jsonl"
    save_jsonl_file(output_data, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Response Generation and Accuracy Calculation")
    parser.add_argument("--inst_file", type=str, required=True, help="Path to the input JSONL file with questions and answers")
    parser.add_argument("--api_url", type=str, required=True, help="API URL for the model endpoint")
    parser.add_argument("--model_name", type=str, default="default", help="Name of the model to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (process a smaller subset)")

    args = parser.parse_args()

    # Run asynchronously
    asyncio.run(main_async(args.inst_file, args.api_url, args.model_name, args.debug))
