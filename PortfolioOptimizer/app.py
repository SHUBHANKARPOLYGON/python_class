from flask import Flask, render_template, request, session
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'SESSION_SECURITY'  # Replace with your generated key

def knapsack(budget, costs, profits, names, notes):
    """Optimizes investment portfolio using 0/1 Knapsack algorithm"""
    n = len(costs)
    dp = [[0] * (budget + 1) for _ in range(n + 1)]
    
    # Build DP table
    for i in range(1, n + 1):
        for w in range(1, budget + 1):
            if costs[i-1] <= w:
                dp[i][w] = max(profits[i-1] + dp[i-1][w - costs[i-1]], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]
    
    # Backtrack to find selected investments
    selected = []
    w = budget
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected.append(i-1)
            w -= costs[i-1]
    
    return dp[n][budget], selected

@app.route('/', methods=['GET', 'POST'])
def home():
    # Initialize session history if it doesn't exist
    if 'history' not in session:
        session['history'] = []
    
    if request.method == 'POST':
        try:
            # Validate inputs
            budget = int(request.form['budget'])
            if budget <= 0:
                raise ValueError("Budget must be positive")
                
            names = request.form.getlist('name')
            costs = list(map(int, request.form.getlist('cost')))
            profits = list(map(int, request.form.getlist('profit')))
            notes = request.form.getlist('note')
            
            # Validate all investments
            for cost, profit in zip(costs, profits):
                if cost <= 0 or profit < 0:
                    raise ValueError("Costs must be positive and profits non-negative")
            
            # Run optimization
            max_profit, selected_indices = knapsack(budget, costs, profits, names, notes)
            selected_investments = [{
                'name': names[i],
                'cost': costs[i],
                'profit': profits[i],
                'note': notes[i]  # New notes field
            } for i in selected_indices]
            
            # Store result in history
            session['history'].append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'budget': budget,
                'max_profit': max_profit,
                'selected': selected_investments,
                'total_cost': sum(costs[i] for i in selected_indices)
            })
            session.modified = True
            
            return render_template('index.html',
                                result=max_profit,
                                selected=selected_investments,
                                total_cost=sum(costs[i] for i in selected_indices),
                                show_result=True,
                                history=reversed(session['history']),
                                error=None)
            
        except Exception as e:
            return render_template('index.html',
                                show_result=False,
                                history=reversed(session['history']),
                                error=str(e))
    
    return render_template('index.html',
                         show_result=False,
                         history=reversed(session['history']),
                         error=None)

if __name__ == '__main__':
    app.run(debug=True)