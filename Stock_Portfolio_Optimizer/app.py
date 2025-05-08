from flask import Flask, render_template, request, session, redirect, url_for
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import uuid

app = Flask(__name__)
app.secret_key = 'SESSION_SECURITY'  # Secure random key

# Initialize AI model
def initialize_model():
    model_path = 'portfolio_model.joblib'
    if not os.path.exists(model_path):
        np.random.seed(42)
        data = {
            'risk_level': np.random.randint(1, 4, 1000),
            'profit_ratio': np.random.uniform(0.1, 1.5, 1000),
            'diversification': np.random.randint(1, 5, 1000),
            'approval': np.where((np.random.uniform(0, 1, 1000) > 0.3), 1, 0)
        }
        df = pd.DataFrame(data)
        model = RandomForestClassifier(n_estimators=100)
        model.fit(df[['risk_level', 'profit_ratio', 'diversification']], df['approval'])
        joblib.dump(model, model_path)
    return joblib.load(model_path)

model = initialize_model()

def analyze_portfolio(selected_investments):
    """AI analysis of portfolio with suggestions"""
    risk_keywords = ['high', 'risk', 'volatile', 'speculative']
    risk_level = sum(1 for inv in selected_investments 
                   if any(keyword in inv['note'].lower() for keyword in risk_keywords))
    
    profit_ratios = [inv['profit']/max(1, inv['cost']) for inv in selected_investments]
    diversification = len(selected_investments)
    avg_profit_ratio = np.mean(profit_ratios)
    
    features = np.array([[risk_level, avg_profit_ratio, diversification]])
    approval = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]
    
    suggestions = []
    if risk_level > 2:
        suggestions.append("Reduce high-risk investments to ≤ 2")
    if avg_profit_ratio < 0.5:
        suggestions.append("Add investments with profit ratio > 0.5")
    if diversification < 3:
        suggestions.append("Increase diversification with 3+ assets")
    
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
            budget = int(request.form['budget'])
            names = request.form.getlist('name')
            costs = list(map(int, request.form.getlist('cost')))
            profits = list(map(int, request.form.getlist('profit')))
            notes = request.form.getlist('note')

            max_profit, selected_indices = knapsack(budget, costs, profits, names, notes)
            selected_investments = [{
                'name': names[i],
                'cost': costs[i],
                'profit': profits[i],
                'note': notes[i]
            } for i in selected_indices]

            ai_analysis = analyze_portfolio(selected_investments)
            
            history_entry = {
                'id': str(uuid.uuid4()),  # Unique ID for each entry
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'budget': budget,
                'max_profit': max_profit,
                'selected': selected_investments,
                'total_cost': sum(costs[i] for i in selected_indices),
                'ai_analysis': ai_analysis
            }
            
            session['history'].append(history_entry)
            session.modified = True
            
            return redirect(url_for('home'))
            
        except Exception as e:
            return render_template('index.html',
                                show_result=False,
                                history=reversed(session['history']),
                                error=str(e))
    
    return render_template('index.html',
                         show_result=False,
                         history=reversed(session['history']),
                         error=None)

@app.route('/delete_history/<string:history_id>', methods=['POST'])
def delete_history(history_id):
    if 'history' in session:
        session['history'] = [entry for entry in session['history'] if entry['id'] != history_id]
        session.modified = True
    return redirect(url_for('home'))

@app.route('/view_history/<string:history_id>')
def view_history(history_id):
    if 'history' in session:
        entry = next((e for e in session['history'] if e['id'] == history_id), None)
        if entry:
            return render_template('index.html',
                                result=entry['max_profit'],
                                selected=entry['selected'],
                                total_cost=entry['total_cost'],
                                ai_analysis=entry.get('ai_analysis', {}),
                                show_result=True,
                                history=reversed(session['history']),
                                error=None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)