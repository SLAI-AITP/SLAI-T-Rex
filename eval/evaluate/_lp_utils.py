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


def extract_python_code(response):
    if response is None:
        return ""
    if isinstance(response, list):
        response = response[0] if response else ""
        if response is None:
            return ""

    # Prefer model-visible answer content after hidden-thinking markers.
    think_split = re.split(r"</(?:think|redacted_reasoning)>", response, flags=re.IGNORECASE)
    content_after_think = think_split[-1].strip() if len(think_split) > 1 else response

    pattern_python = r"```python\s*(.*?)```"
    pattern_no_lang = r"```\s*(.*?)```"
    for content in (content_after_think, response):
        matches = re.findall(pattern_python, content, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()

    for content in (content_after_think, response):
        matches = re.findall(pattern_no_lang, content, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()

    if len(think_split) > 1 and content_after_think:
        return content_after_think
    return response


import re

def process_code_for_lp(code, data_path, file_name):
    """
    Process the code to:
    1. Comment out model.optimize() part or m.optimize() part and all code after it until if __name__ == "__main__"
    2. Ensure model.write(file_name.lp) or m.write(file_name.lp) is included before model.optimize() or m.optimize()
    """


    # 1. 替换data.json文件路径
    print(data_path)
    #print(code)
    print(file_name)
    code = code.replace("data.json", data_path)
    
    
    # 2. 修改已有的 model.write() 或 m.write() 为指定的文件名
    write_pattern = r"((?:model|m)\.write\()([^)]*)(\))"
    def replace_write(match):
        model_var = match.group(1).replace('.write(', '')
        return f"{model_var}.write('{file_name}')"
    code = re.sub(write_pattern, replace_write, code)

    # 3. Find position of model.addVars( or m.addVars(
    addvars_pattern = r"((?:model|m)\.addVars\()"
    addvars_match = re.search(addvars_pattern, code)

    addvar_pattern = r"((?:model|m)\.addVar\()"
    addvar_match = re.search(addvar_pattern, code)

    if addvars_match:
        # Use the same model variable name as matched
        addvars_line = addvars_match.group(1)
        if addvars_line.startswith("model"):
            model_var = "model"
        else:
            model_var = "m"
    elif addvar_match:
        # Use the same model variable name as matched
        addvar_line = addvar_match.group(1)
        if addvar_line.startswith("model"):
            model_var = "model"
        else:
            model_var = "m"
    else:
        # If no addVars or addVar found, assume model variable is 'model'
        model_var = "model"
    # 如果没有找到任何 .write() 调用，则添加一个
    if not re.search(r"(?:model|m)\.write\(", code):
        code = code.rstrip() + f"\n{model_var}.write('{file_name}')"

    # 4. Ensure necessary imports

    code = "from gurobipy import GRB\n" + code

    return insert_lb_if_missing(code)


def insert_lb_if_missing(code):
    def replacer(match):
        args = match.group(1)
        if 'lb=' not in args:
            # 如果没有指定lb=，就在参数末尾加上 lb=-GRB.INFINITY
            if args.strip().endswith(','):
                new_args = args + " lb=-GRB.INFINITY"
            else:
                new_args = args + ", lb=-GRB.INFINITY"
            return new_args + ')\n'
        else:
            return match.group(0)  # 原样返回
    return re.sub(r"(model\.addVars\([^)]*)\)\n", replacer, code)


def ensure_imports(code):
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
