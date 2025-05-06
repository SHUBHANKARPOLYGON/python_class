from flask import Flask, render_template, request

app = Flask(__name__)

def knapsack(budget, costs, profits, names):
    n = len(costs)
    dp = [[0] * (budget + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(1, budget + 1):
            if costs[i-1] <= w:
                dp[i][w] = max(profits[i-1] + dp[i-1][w - costs[i-1]], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]
    
    selected = []
    w = budget
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected.append(i-1)
            w -= costs[i-1]
    
    return dp[n][budget], selected

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        budget = int(request.form['budget'])
        investments = []
        names = request.form.getlist('name')
        costs = list(map(int, request.form.getlist('cost')))
        profits = list(map(int, request.form.getlist('profit')))
        
        max_profit, selected_indices = knapsack(budget, costs, profits, names)
        selected_investments = [names[i] for i in selected_indices]
        
        return render_template('index.html', 
                             result=max_profit, 
                             selected=selected_investments, 
                             show_result=True)
    
    return render_template('index.html', show_result=False)

if __name__ == '__main__':
    app.run(debug=True)