from flask import Flask, render_template, request, session
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

app = Flask(__name__)
app.secret_key = 'SESSION_SECURITY'  # Replace with your generated key

# Initialize or load the AI model
def initialize_model():
    model_path = 'portfolio_model.joblib'
    if not os.path.exists(model_path):
        # Generate synthetic training data if no model exists
        np.random.seed(42)
        data = {
            'risk_level': np.random.randint(1, 4, 1000),
            'profit_ratio': np.random.uniform(0.1, 1.5, 1000),
            'diversification': np.random.randint(1, 5, 1000),
            'approval': np.where((np.random.uniform(0, 1, 1000) > 0.3), 1, 0)
        }
        df = pd.DataFrame(data)
        
        # Train model
        X = df[['risk_level', 'profit_ratio', 'diversification']]
        y = df['approval']
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, y)
        
        # Save model
        joblib.dump(model, model_path)
        return model
    return joblib.load(model_path)

model = initialize_model()

def analyze_portfolio(selected_investments):
    """AI analysis of portfolio with suggestions"""
    # Extract features
    risk_keywords = ['high', 'risk', 'volatile', 'speculative']
    risk_level = sum(1 for inv in selected_investments 
                    if any(keyword in inv['note'].lower() for keyword in risk_keywords))
    
    profit_ratios = [inv['profit']/max(1, inv['cost']) for inv in selected_investments]
    diversification = len(selected_investments)
    avg_profit_ratio = np.mean(profit_ratios)
    
    # Make prediction
    features = np.array([[risk_level, avg_profit_ratio, diversification]])
    approval = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]
    
    # Generate suggestions
    suggestions = []
    if risk_level > 2:
        suggestions.append("Reduce high-risk investments to ≤ 2")
    if avg_profit_ratio < 0.5:
        suggestions.append("Add investments with profit ratio > 0.5")
    if diversification < 3:
        suggestions.append("Increase diversification with 3+ assets")
    
    # Format output
    if approval == 1 and proba > 0.7:
        verdict = f"✅ AI Approval (Confidence: {proba*100:.0f}%)"
    elif approval == 1:
        verdict = f"🟢 AI Suggestion (Confidence: {proba*100:.0f}%)"
    else:
        verdict = f"⚠️ AI Warning (Confidence: {(1-proba)*100:.0f}%)"
    
    return {
        'verdict': verdict,
        'suggestions': suggestions,
        'metrics': {
            'risk_level': risk_level,
            'avg_profit_ratio': f"{avg_profit_ratio:.2f}",
            'diversification': diversification
        }
    }

def knapsack(budget, costs, profits, names, notes):
    """Optimizes investment portfolio using 0/1 Knapsack algorithm"""
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
                'note': notes[i]
            } for i in selected_indices]
            
            # AI Analysis
            ai_analysis = analyze_portfolio(selected_investments)
            
            # Store result in history
            session['history'].append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'budget': budget,
                'max_profit': max_profit,
                'selected': selected_investments,
                'total_cost': sum(costs[i] for i in selected_indices),
                'ai_analysis': ai_analysis
            })
            session.modified = True
            
            return render_template('index.html',
                                result=max_profit,
                                selected=selected_investments,
                                total_cost=sum(costs[i] for i in selected_indices),
                                ai_analysis=ai_analysis,
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