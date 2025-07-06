from graphviz import Digraph

dot = Digraph(comment='LSTM-XGBoost Hybrid Architecture', format='pdf')
dot.attr(rankdir='TB', bgcolor='white', splines='ortho')
dot.attr('node', fontname='Helvetica', fontsize='10')

# Input Sequence
dot.node('INPUT', 'Input Sequence\n(10 timesteps × 9 features)', shape='box', style='filled', fillcolor='lightyellow')

# LSTM Layers (vertical representation)
with dot.subgraph(name='cluster_lstm') as c:
    c.attr(label='LSTM Feature Extractor\n(128 units)', style='filled', fillcolor='lightblue', color='lightblue')
    c.attr(rank='same')
    
    # Neural nodes with vertical arrangement
    c.node('CELL1', 'Timestep 1\nLSTM Cell', shape='circle')
    c.node('CELL2', 'Timestep 2\nLSTM Cell', shape='circle')
    c.node('CELLDOTS', '...', shape='plaintext')
    c.node('CELL10', 'Timestep 10\nLSTM Cell', shape='circle')
    c.node('MEMORY', 'Memory State', shape='note', style='dashed')
    
    # Vertical connections
    c.edge('CELL1', 'CELL2', style='invis')
    c.edge('CELL2', 'CELLDOTS', style='invis')
    c.edge('CELLDOTS', 'CELL10', style='invis')
    
    # Memory connections
    c.edge('CELL1', 'MEMORY', style='dashed', dir='none', constraint='false')
    c.edge('MEMORY', 'CELL2', style='dashed', dir='none', constraint='false')
    c.edge('MEMORY', 'CELL10', style='dashed', dir='none', constraint='false')

# Feature Vector
dot.node('FEAT', 'Feature Vector\n(128 dimensions)', shape='box', style='filled', fillcolor='lightgreen')

# XGBoost Trees (vertical representation)
with dot.subgraph(name='cluster_xgb') as c:
    c.attr(label='XGBoost Models')
    
    # Regressor Trees
    with c.subgraph(name='cluster_reg') as reg:
        reg.attr(label='Regressor Trees (100)', style='filled', fillcolor='lightcoral', color='lightcoral')
        reg.node('RTREE1', 'Tree 1', shape='triangle')
        reg.node('RTREE2', 'Tree 2', shape='triangle')
        reg.node('RTREEDOTS', '...', shape='plaintext')
        reg.node('RTREE100', 'Tree 100', shape='triangle')
        
        # Vertical connections
        reg.edge('RTREE1', 'RTREE2', style='invis')
        reg.edge('RTREE2', 'RTREEDOTS', style='invis')
        reg.edge('RTREEDOTS', 'RTREE100', style='invis')
    
    # Classifier Trees
    with c.subgraph(name='cluster_clf') as clf:
        clf.attr(label='Classifier Trees (100)', style='filled', fillcolor='khaki', color='khaki')
        clf.node('CTREE1', 'Tree 1', shape='triangle')
        clf.node('CTREE2', 'Tree 2', shape='triangle')
        clf.node('CTREEDOTS', '...', shape='plaintext')
        clf.node('CTREE100', 'Tree 100', shape='triangle')
        
        # Vertical connections
        clf.edge('CTREE1', 'CTREE2', style='invis')
        clf.edge('CTREE2', 'CTREEDOTS', style='invis')
        clf.edge('CTREEDOTS', 'CTREE100', style='invis')

# Outputs
dot.node('OUT_REG', 'Yield Prediction', shape='box', style='filled', fillcolor='lightpink')
dot.node('OUT_CLF', 'Growth Stage\n[Seed Sowing, Germination,\nLeaf Development, Head Formation,\nHarvesting]', shape='box', style='filled', fillcolor='lightpink')

# SHAP Explainability
dot.node('SHAP', 'SHAP Analysis\n(Model Interpretation)', shape='box', style='filled', fillcolor='lavender')

# Main Connections
dot.edge('INPUT', 'CELL1')
dot.edge('CELL10', 'FEAT')
dot.edge('FEAT', 'RTREE1')
dot.edge('FEAT', 'CTREE1')
dot.edge('RTREE100', 'OUT_REG')
dot.edge('CTREE100', 'OUT_CLF')
dot.edge('RTREE1', 'SHAP', style='dashed')
dot.edge('CTREE1', 'SHAP', style='dashed')

# Memory flow indicator
dot.edge('CELL1', 'CELL10', style='invis', constraint='false')
dot.edge('CELL1', 'CELL10', label='Memory Flow', style='dashed', dir='forward', constraint='false')

# Render the diagram
dot.render('lstm_xgboost_compact', cleanup=True, view=True)