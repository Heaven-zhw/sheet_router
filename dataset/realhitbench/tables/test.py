if __name__ == '__main__':
    from chart_metric_util import *
    import pandas as pd
    import matplotlib.pyplot as plt
    
    df = pd.read_excel('military-table12.xlsx')
    
    time_columns = [col for col in df.columns if isinstance(col, pd.Timestamp)]
    total_outflow_data = df.loc[2, time_columns]
    
    chart_data = pd.DataFrame({ 'Date': time_columns, 'Total_Outflow': total_outflow_data.values }).sort_values('Date')
    
    plt.plot(chart_data['Date'], chart_data['Total_Outflow'], marker='o')
    plt.title('Total Outflow of All Services Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Outflow')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    chart_type="LineChart"
    
    if chart_type == 'LineChart': 
        y_predictions = get_line_y_predictions(plt)
    if chart_type == 'BarChart':
        y_predictions = get_bar_y_predictions(plt)
    if chart_type == 'ScatterChart':
        y_predictions = get_scatter_y_predictions(plt)
    if chart_type == 'PieChart':
        y_predictions = get_pie_y_predictions(plt)
    
    print(y_predictions)