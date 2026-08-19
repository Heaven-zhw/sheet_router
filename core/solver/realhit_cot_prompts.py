Fact_Checking = """
# Output Control For JSON Field `final_answer`
1. The JSON field `final_answer` should be a one-line string whose value starts with "[Final Answer]: ".
2. After the prefix, provide only number(s) or entity name(s), as short as possible, without any explanation. If the question is judgmental, answer "Yes" or "No".
Example value for `final_answer`: "[Final Answer]: AnswerName1, AnswerName2..."
"""

Numerical_Reasoning = """
# Output Control For JSON Field `final_answer`
1. The JSON field `final_answer` should be a one-line string whose value starts with "[Final Answer]: ".
2. After the prefix, provide only number(s) or entity name(s), as short as possible, without any explanation. If the answer involves decimals, always keep it to two decimals.
Example value for `final_answer`: "[Final Answer]: AnswerName1, AnswerName2..."
"""

Rudimentary_Analysis = """
# Output Control For JSON Field `final_answer`
1. The JSON field `final_answer` should be a one-line string whose value starts with "[Final Answer]: ".
2. After the prefix, provide only the primary result of the rudimentary analysis, such as a number or an entity name, as short as possible. If the answer involves decimals, always keep it to two decimals.
Example value for `final_answer`: "[Final Answer]: AnswerName1, AnswerName2..."
"""

Summary_Analysis = """
# Output Control For JSON Field `final_answer`
1. The JSON field `final_answer` should be a one-line string whose value starts with "[Final Answer]: ".
2. After the prefix, provide a concise table summary including the content, main columns, and basic insights. Do not add explanation outside the JSON object.
Example value for `final_answer`: "[Final Answer]: TableSummary"
"""

Predictive_Analysis = """
# Output Control For JSON Field `final_answer`
1. The JSON field `final_answer` should be a one-line string whose value starts with "[Final Answer]: ".
2. After the prefix, provide the primary result as a concise number, entity name, or trend description (for example, No clear trend, Increasing trend, or Decreasing trend). If the answer involves decimals, always keep it to two decimals.
Example value for `final_answer`: "[Final Answer]: AnswerName1, AnswerName2..."
"""

Exploratory_Analysis = """
# Output Control For JSON Field `final_answer`
1. The JSON field `final_answer` should be a one-line string whose value starts with "[Final Answer]: ".
2. For correlation questions, after the prefix use: CorrelationRelation, CorrelationCoefficient. The coefficient must be a float number with two decimal places.
3. The relation can only be "No correlation" (-0.3 to +0.3), "Weak positive correlation" (+0.3 to +0.7), "Weak negative correlation" (-0.3 to -0.7), "Strong positive correlation" (+0.7 to +1), or "Strong negative correlation" (-0.7 to -1).
4. For impact questions, after the prefix use an entity name or a short impact description such as No clear impact, Negative impact, or Positive impact. For causal analysis, use a concise causal conclusion.
Example value for `final_answer`: "[Final Answer]: CorrelationRelation, CorrelationCoefficient"
"""

Anomaly_Analysis = """
# Output Control For JSON Field `final_answer`
1. The JSON field `final_answer` should be a one-line string whose value starts with "[Final Answer]: ".
2. After the prefix, provide a concise anomaly conclusion without extra explanation.
Example value for `final_answer`: "[Final Answer]: Conclusion"
"""

Visulization = """
# Output Control For JSON Field `final_answer`
1. The JSON field `final_answer` should be a string whose value starts with "[Final Answer]: import pandas as pd".
2. After the prefix, provide only runnable Python code and make sure the code can run directly without syntax errors.
3. Please make sure the table is named "table.xlsx", and pandas and matplotlib are imported.
4. Ensure that the X-axis used for drawing is arranged in ascending alphabetical or numerical order. Ensure the last line in the code is exactly "plt.show()".
Example value for `final_answer`: "[Final Answer]: import pandas as pd\\nimport matplotlib.pyplot as plt\\n...\\nplt.show()"
"""

Structure_Comprehending = """
# Output Control For JSON Field `final_answer`
1. The JSON field `final_answer` should be a one-line string whose value starts with "[Final Answer]: ".
2. After the prefix, provide only number(s) or entity name(s), as short as possible, without any explanation. If the question is judgmental, answer "Yes" or "No".
Example value for `final_answer`: "[Final Answer]: AnswerName1, AnswerName2..."
"""

Answer_Prompt = {
    "Fact Checking": Fact_Checking,
    "Numerical Reasoning": Numerical_Reasoning,
    "Structure Comprehending": Structure_Comprehending,
    "Rudimentary Analysis": Rudimentary_Analysis,
    "Summary Analysis": Summary_Analysis,
    "Predictive Analysis": Predictive_Analysis,
    "Exploratory Analysis": Exploratory_Analysis,
    "Anomaly Analysis": Anomaly_Analysis,
    "Visualization": Visulization,
}
