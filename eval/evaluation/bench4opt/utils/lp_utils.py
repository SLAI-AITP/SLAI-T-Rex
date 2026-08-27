import os
import re
from pathlib import Path

from dataclasses import dataclass, field

from pprint import pprint
from tqdm import tqdm


def is_valid_lp(data, valid_problems):
    data_problem = data["problem_name"]
    if data_problem not in valid_problems["LP"]:
        return False
    if data_problem not in valid_problems["MILP"]:
        return False
    return True


def extract_python_code(response, verbose=False):
    """
    Extract Python code from model response.
    Priority: code blocks > python tags > content after thinking marker > whole response
    
    Args:
        response: The model's response string
        verbose: Whether to print debug information
        
    Returns:
        Extracted Python code string
    """
    # Handle None response
    if response is None:
        if verbose:
            print("WARNING: response is None, returning empty string")
        return ""
    
    if isinstance(response, list):
        response = response[0]
        # Check again after extracting from list
        if response is None:
            if verbose:
                print("WARNING: response extracted from list is None, returning empty string")
            return ""
    
    original_response = response
    
    # First, try to find code blocks in the entire response (highest priority)
    # This handles cases where thinking content is mixed with code
    pattern_python = r"```python\s*(.*?)```"
    matches_python = re.findall(pattern_python, response, re.DOTALL)
    
    if matches_python:
        # If multiple code blocks, take the longest one (likely the main code)
        extracted = max(matches_python, key=len).strip()
        if verbose:
            print(f"Found Python code block in response, length: {len(extracted)}")
        return extracted
    
    # Try code block without language tag
    pattern_no_lang = r"```\s*(.*?)```"
    matches_no_lang = re.findall(pattern_no_lang, response, re.DOTALL)
    if matches_no_lang:
        # Filter out very short matches (likely not code)
        valid_matches = [m for m in matches_no_lang if len(m.strip()) > 50]
        if valid_matches:
            extracted = max(valid_matches, key=len).strip()
            if verbose:
                print(f"Found code block without language tag, length: {len(extracted)}")
            return extracted
    
    # If code blocks found but all too short, still try the longest one
    if matches_no_lang:
        extracted = max(matches_no_lang, key=len).strip()
        if verbose:
            print(f"Found short code block, length: {len(extracted)}")
        return extracted

    # Try explicit <python>...</python> tags used by some solver prompts
    pattern_python_tag = r"<python>\s*(.*?)\s*</python>"
    matches_python_tag = re.findall(pattern_python_tag, response, re.DOTALL | re.IGNORECASE)
    if matches_python_tag:
        tagged_content = max(matches_python_tag, key=len).strip()

        inner_python_matches = re.findall(pattern_python, tagged_content, re.DOTALL)
        if inner_python_matches:
            extracted = max(inner_python_matches, key=len).strip()
            if verbose:
                print(f"Found Python code block inside <python> tag, length: {len(extracted)}")
            return extracted

        inner_no_lang_matches = re.findall(pattern_no_lang, tagged_content, re.DOTALL)
        if inner_no_lang_matches:
            valid_matches = [m for m in inner_no_lang_matches if len(m.strip()) > 50]
            if valid_matches:
                extracted = max(valid_matches, key=len).strip()
                if verbose:
                    print(f"Found code block without language tag inside <python> tag, length: {len(extracted)}")
                return extracted

        if verbose:
            print(f"Found raw <python> tag content, length: {len(tagged_content)}")
        return tagged_content
    
    # If no code blocks found, try to extract content from thinking marker
    # Support both <think> and <think> tags
    # Priority: content after thinking marker > content inside thinking marker
    
    # First, try to extract content AFTER thinking marker (higher priority)
    think_split = re.split(r"</(?:think|redacted_reasoning)>", response, flags=re.IGNORECASE)
    if len(think_split) > 1:
        content_after_think = think_split[-1].strip()  # Take the last part after all closing tags
        if content_after_think:  # Only process if not empty
            if verbose:
                print(f"Found thinking marker, content after marker length: {len(content_after_think)}")
            
            # Try to find code blocks in content after thinking
            matches_after_think = re.findall(pattern_python, content_after_think, re.DOTALL)
            if matches_after_think:
                extracted = max(matches_after_think, key=len).strip()
                if verbose:
                    print(f"Found Python code block after thinking marker, length: {len(extracted)}")
                return extracted
            
            # Also try code blocks without language tag
            matches_no_lang_after_think = re.findall(pattern_no_lang, content_after_think, re.DOTALL)
            if matches_no_lang_after_think:
                valid_matches = [m for m in matches_no_lang_after_think if len(m.strip()) > 50]
                if valid_matches:
                    extracted = max(valid_matches, key=len).strip()
                    if verbose:
                        print(f"Found code block without language tag after thinking marker, length: {len(extracted)}")
                    return extracted
            
            # If no code block but has thinking marker, return content after marker
            # (assuming it's the actual response without thinking)
            if verbose:
                print(f"No code block found, returning content after thinking marker")
            return content_after_think
    
    # As fallback, try to extract from inside thinking tags
    pattern_think = r"<(?:think|redacted_reasoning)>(.*?)</(?:think|redacted_reasoning)>"
    matches_think = re.findall(pattern_think, response, re.DOTALL | re.IGNORECASE)
    if matches_think:
        # Try to find code blocks inside thinking tags
        for think_content in matches_think:
            matches_in_think = re.findall(pattern_python, think_content, re.DOTALL)
            if matches_in_think:
                extracted = max(matches_in_think, key=len).strip()
                if verbose:
                    print(f"Found Python code block inside thinking marker, length: {len(extracted)}")
                return extracted
            # Also try code blocks without language tag
            matches_no_lang_in_think = re.findall(pattern_no_lang, think_content, re.DOTALL)
            if matches_no_lang_in_think:
                valid_matches = [m for m in matches_no_lang_in_think if len(m.strip()) > 50]
                if valid_matches:
                    extracted = max(valid_matches, key=len).strip()
                    if verbose:
                        print(f"Found code block without language tag inside thinking marker, length: {len(extracted)}")
                    return extracted
    
    # Last resort: return whole response (but warn if verbose)
    if verbose:
        print(f"WARNING: No code block or thinking marker found!")
        print(f"Response length: {len(original_response)}")
        print(f"First 300 chars: {original_response[:300]}")
        print(f"Last 300 chars: {original_response[-300:]}")
    
    return response


def _detect_model_variable_name(code):
    """
    Detect the name of the Gurobi model variable in the code.
    
    Returns:
        str: The detected model variable name, or None if not found
    """
    # Pattern 1: Find assignments like "model = gp.Model()" or "m = gp.Model()"
    # Match patterns like: variable_name = (gp|gurobipy).Model(...)
    pattern1 = r"(\w+)\s*=\s*(?:gp|gurobipy)\.Model\s*\("
    matches1 = re.findall(pattern1, code)
    if matches1:
        # Return the first match (most likely the model variable)
        return matches1[0]
    
    # Pattern 2: Find assignments like "model = Model()" (if Model was imported)
    pattern2 = r"(\w+)\s*=\s*Model\s*\("
    matches2 = re.findall(pattern2, code)
    if matches2:
        return matches2[0]
    
    # Pattern 3: Try to find common variable names used in the code
    # Check if "model" or "m" is used anywhere (as a variable, not in strings)
    # Look for patterns like "model." or "m." followed by method calls
    pattern3 = r"\b(model|m)\.(addVar|addVars|addConstr|addConstrs|setObjective|optimize|update)"
    matches3 = re.findall(pattern3, code)
    if matches3:
        # Return the most common one
        var_names = [m[0] for m in matches3]
        return max(set(var_names), key=var_names.count)
    
    return None


def process_code_for_lp(code, data_path, file_name):
    """
    Process the code to:
    1. Comment out model.optimize() part or m.optimize() part and all code after it until if __name__ == "__main__"
    2. Ensure model.write(file_name.lp) or m.write(file_name.lp) is included before model.optimize() or m.optimize()
    """
    
    # Check if code is empty or only whitespace
    if not code or not code.strip():
        raise ValueError("Code is empty or contains only whitespace. Cannot process for LP file generation.")

    # 1. 替换data.json文件路径
    code = code.replace("data.json", data_path)

    # 2. remove any existing model.write() or m.write()
    write_pattern = r"(?:\w+)\.write\([^)]*\)"
    code = re.sub(write_pattern, "", code)

    # 3. Find position of model.optimize() or m.optimize() or any_var.optimize()
    # Try to match common patterns: model.optimize(), m.optimize(), or detect variable name
    optimize_pattern = r"(\w+)\.optimize\s*\("
    optimize_match = re.search(optimize_pattern, code)

    if optimize_match:
        # Insert write before optimize and comment out optimize and code after it
        optimize_pos = optimize_match.start()
        pre_optimize = code[:optimize_pos]
        model_var = optimize_match.group(1)  # Extract variable name

        # Find the full optimize() call including its arguments
        # The match ends at "optimize(" - we need to find the closing parenthesis
        open_paren_pos = code.find('(', optimize_pos)
        if open_paren_pos == -1:
            # Should not happen, but handle gracefully
            optimize_line_full = optimize_match.group(0)
            post_optimize = code[optimize_match.end():]
        else:
            # Find the matching closing parenthesis
            paren_count = 1
            end_pos = open_paren_pos + 1
            while end_pos < len(code) and paren_count > 0:
                if code[end_pos] == '(':
                    paren_count += 1
                elif code[end_pos] == ')':
                    paren_count -= 1
                end_pos += 1
            
            # Extract the full optimize call
            optimize_line_full = code[optimize_pos:end_pos]
            post_optimize = code[end_pos:]

        # 获取optimize_line的缩进
        indent = ""
        optimize_line_start = code.rfind("\n", 0, optimize_pos) + 1
        if optimize_line_start > 0:
            indent = code[optimize_line_start:optimize_pos]

        # 查找if __name__ == "__main__"的位置
        main_pattern = r"\nif\s+__name__\s*==\s*[\"']__main__[\"']\s*:"
        main_match = re.search(main_pattern, post_optimize)

        if main_match:
            # 如果找到main部分，只注释到main之前
            main_pos = main_match.start()
            post_to_comment = post_optimize[:main_pos]
            main_part = post_optimize[main_pos:]

            # 将post_to_comment中的每一行都加上注释
            commented_post = ""
            for line in post_to_comment.split("\n"):
                if line.strip():  # 如果行不是空的
                    commented_post += f"\n{indent}# {line.lstrip()}"
                else:
                    commented_post += f"\n{indent}#"

            code = (
                pre_optimize.rstrip()
                + f"\n{indent}{model_var}.write('{file_name}')\n"
                + f"{indent}# {optimize_line_full.strip()}"
                + commented_post
                + main_part
            )
        else:
            # 如果没找到main部分，注释所有后续代码
            commented_post = ""
            for line in post_optimize.split("\n"):
                if line.strip():  # 如果行不是空的
                    commented_post += f"\n{indent}# {line.lstrip()}"
                else:
                    commented_post += f"\n{indent}#"

            code = (
                pre_optimize.rstrip()
                + f"\n{indent}{model_var}.write('{file_name}')\n"
                + f"{indent}# {optimize_line_full.strip()}"
                + commented_post
            )
    else:
        # No optimize found, try to detect the model variable name
        model_var = _detect_model_variable_name(code)
        if model_var is None:
            # If still not found, try common names in order
            # Check which one exists in the code (has method calls on it)
            if re.search(r"\bmodel\.(addVar|addVars|addConstr|addConstrs|setObjective|update)", code):
                model_var = "model"
            elif re.search(r"\bm\.(addVar|addVars|addConstr|addConstrs|setObjective|update)", code):
                model_var = "m"
            else:
                # Last resort: use "model" as default (may still fail, but better than nothing)
                model_var = "model"
        
        # Append write at the end
        code = code.rstrip() + f"\n{model_var}.write('{file_name}')"

    return code


def add_gurobi_imports(code):
    """
    Ensure the code has necessary import statements at the beginning.
    """
    imports_to_add = []

    # Check for gurobipy import
    if "import gurobipy" not in code:
        imports_to_add.append("import gurobipy as gp")

    # Check for json import
    if "import json" not in code and "from json" not in code:
        imports_to_add.append("import json")

    # Add imports at the beginning if needed
    if imports_to_add:
        code = "\n".join(imports_to_add) + "\n\n" + code

    return code
