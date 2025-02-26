import numpy as np
from scipy.stats import entropy
import json

# normalize
def convert_to_scores(data):
    converted_data = []
    for row in data:
        score = 0
        for i in range(len(row)):
            score += row[i]/10000*(i+1)
        converted_data.append(score/5)
    return np.array(converted_data)


if __name__ == "__main__":
    with open("./result.json",'r') as file:
        data = json.load(file)

    models = ['llama','qwen','gpt4o','gpt4omini','deepseek']
    for model in models:
        model_data = data[model]
        real_scores = convert_to_scores(model_data['真实'])
        sim_scores = convert_to_scores(model_data['模拟'])

        # RMSE
        rmse_value = np.sqrt(np.mean((real_scores - sim_scores) ** 2))
        print(f"{model} RMSE: {rmse_value}")

        # KL-Div
        kl_values = []
        smoothing_factor=1e-10

        for i in range(len(model_data['真实'])):
            real = [smoothing_factor if item == 0 else item/10000 for item in model_data['真实'][i]]
            sim = [smoothing_factor if item == 0 else item/10000 for item in model_data['模拟'][i]]
            kl_values.append(entropy(real, sim))

        average_kl_divergence = np.mean(kl_values)
        print(f"{model} Average KL Divergence: {average_kl_divergence}")