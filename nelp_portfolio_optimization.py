def solve_nlp_portfolio():
    from pyscipopt import Model, quicksum

    # 创建模型
    model = Model("NLP_Portfolio_Optimization")

    # 假设有3个资产
    assets = ['Asset1', 'Asset2', 'Asset3']
    returns = {'Asset1': 0.10, 'Asset2': 0.20, 'Asset3': 0.15}
    budget = 1.0

    # 协方差矩阵（简化为对角线）
    cov = {'Asset1': 0.02, 'Asset2': 0.03, 'Asset3': 0.025}

    # 定义决策变量：每个资产的投资比例
    x = {asset: model.addVar(vtype="CONTINUOUS", name=f"x_{asset}") for asset in assets}

    # 目标函数：最大化预期收益 - 通过引入二次项近似风险
    # SCIP不直接支持二次项，因此我们将其线性化或通过其他方式近似
    # 这里简化为仅线性化预期收益
    model.setObjective(quicksum(returns[i] * x[i] for i in assets), "maximize")

    # 约束条件：总投资比例为1
    model.addCons(quicksum(x[i] for i in assets) == budget, "Budget")

    # 近似的风险约束（仍为线性）
    risk = 0.05
    model.addCons(quicksum(cov[i] * x[i] for i in assets) <= risk, "Risk")

    # 求解模型
    model.optimize()

    # 输出结果
    print("NLP（近似）投资组合优化结果:")
    for asset in assets:
        print(f"{asset}: {model.getVal(x[asset])}")
    print(f"预期收益: {model.getObjVal()}") 

if __name__ == "__main__":
    solve_nlp_portfolio() 
