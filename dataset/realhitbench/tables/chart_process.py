import io
import sys
import json
import os
from timeout_decorator import timeout
import re
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

@timeout(50)
def exec_and_get_y_reference(code):
    print("CODE:", code)
    output = io.StringIO()
    stdout = sys.stdout
    try:
        sys.stdout = output
        exec(code)
        plt.close("all")
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        sys.stdout = stdout
        output = output.getvalue()
        print(output)
        print("Execution finished")
    # plt.close()
    return output

def visualization_code_format(visualization_answer):
    
    pattern1 = r"import pandas as pd.*?plt\.show\(\)"
    pattern2 = r"import matplotlib.pyplot as plt.*?plt\.show\(\)"
    try:
        matches1 = re.findall(pattern1, visualization_answer, flags=re.S)
        if matches1:
            return matches1[-1]
        else:
            matches2 = re.findall(pattern2, visualization_answer, flags=re.S)
            if matches2:
                return matches2[-1]
            else:
                print(f"invalid visualization_answer: {visualization_answer}\n")
                return ''
    except Exception as e:
        print(f"visualization_code_format failed which is: {visualization_answer}")

# def compute_ECR_metric(y_references, y_predictions):
    

def surround_pycode_with_main(pycode):
    start_line = '''
if __name__ == '__main__':
'''
    pycode_lines = pycode.strip().split('\n')
    for line in pycode_lines:
        start_line += f'    {line}\n'
    return start_line

# JUST process the Reference answer code.
# Prediction answer code needs another function to process...
def build_eval_code(answer_code, chart_type):
    extract_code = visualization_code_format(answer_code)
    python_code_lines = extract_code.strip().split('\n')

    eval_code = '''
if chart_type == 'LineChart': 
    y_predictions = get_line_y_predictions(plt)
if chart_type == 'BarChart':
    y_predictions = get_bar_y_predictions(plt)
if chart_type == 'ScatterChart':
    y_predictions = get_scatter_y_predictions(plt)
if chart_type == 'PieChart':
    y_predictions = get_pie_y_predictions(plt)

print(y_predictions)
'''
    python_code = ""
    for line in python_code_lines:
        python_code += f"{line.strip(' ')}\n"
    chart_eval_code = f'    from chart_metric_util import *\n{python_code}\nchart_type="{chart_type}"\n{eval_code}'
    return python_code, chart_eval_code

def batch_process_annotations():
    ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(ROOT_PATH)
    #标注位置路径
    ANNOTATION_PATH = "/root/yyh/Benchmark/data/table_annotation"
    #处理结果路径
    PROCESSED_PATH = "/root/yyh/Benchmark/data/table_annotation"

    if not os.path.exists(PROCESSED_PATH):
        os.makedirs(PROCESSED_PATH)

    for file in os.listdir(ANNOTATION_PATH):
        data_list = []
        if file.endswith('.json'):
            original_json_path = os.path.join(ANNOTATION_PATH, file)
            processed_json_path = os.path.join(PROCESSED_PATH, "processed_"+file)
            with open(original_json_path, 'r') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, dict):
                        data_list = data['queries']
                        id = 1
                        for anno in data_list:
                            if ".xlsx" in anno['FileName']: anno['FileName'] = anno['FileName'].split(".xlsx")[0]
                            pattern = r'table\d{1}'
                            table_no= re.findall(r"\d+", anno['FileName'])[0]
                            if len(table_no) == 1: 
                                anno['FileName'] = anno['FileName'].replace(table_no, '0'+table_no)
                            anno['FileName'] = re.sub(r'-(\d{1})(?=\D|$)', r'-0\1', anno['FileName'])
                            anno['FileName'] = anno['FileName'].lower()
                            anno['id'] = id
                            if "Id" in anno: del anno['Id']
                            id += 1
                            sub_type = anno['SubQType']
                            if anno['QuestionType'] == "Structure Comprehending": anno['SubQType'] = "Structure Comprehending"
                            elif sub_type == "Multi-hop Fact Checking" or sub_type == "Value-Matching" or sub_type == "Inference-based Fact Checking": anno['QuestionType'] = 'Fact Checking'
                            elif sub_type == "Multi-hop Numerical Reasoning" or sub_type == "Counting" or sub_type == "Sorting" or sub_type == "Comparison" or sub_type == "Calculation": anno['QuestionType'] = 'Numerical Reasoning'
                            elif sub_type == "Rudimentary Analysis" or sub_type == "Summary Analysis" or sub_type == "Predictive Analysis" or sub_type == "Exploratory Analysis" or sub_type == "Anomaly Analysis": anno['QuestionType'] = 'Data Analysis'
                            elif sub_type == "LineChart Generation" or sub_type == "BarChart Generation" or sub_type == "ScatterChart Generation" or sub_type == "PieChart Generation": anno['QuestionType'] = 'Visualization'


                            # if anno['QuestionType'] == "Visualization":
                            #     chart_type = anno['SubQType'].split()[0]
                            #     anno['FinalAnswer'] = re.sub(r"'[^']*\.xlsx'", "'"+anno['FileName']+".xlsx'", anno['FinalAnswer'])
                            #     answer_code = anno['FinalAnswer']
                            #     python_code, eval_code = build_eval_code(answer_code, chart_type)
                            #     final_code = surround_pycode_with_main(eval_code)

                            #     y_axis = exec_and_get_y_reference(final_code)
                            #     print(y_axis)
                            #     anno['ProcessedAnswer'] = y_axis
                            # else:
                            # anno['ProcessedAnswer'] = anno['FinalAnswer']
                        #data_list = [anno for anno in data_list if anno["QuestionType"]!="Structure Comprehending"]
                        data['queries'] = data_list
                        
                    else:
                        raise Exception("the content of json is error format!")

                    with open(processed_json_path, 'w', encoding='utf-8') as w_f:
                        json.dump(data, w_f, indent = 4, ensure_ascii=False)
                except Exception as e:
                    print(anno)
                    print(f"Error handling JSON file {original_json_path}: {e}")
        else: 
            pass
        
if __name__ == '__main__':
    batch_process_annotations()


