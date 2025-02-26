import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from scipy.stats import entropy
import os

warnings.filterwarnings("ignore")

def value_evaluation(dir):
    data = []
    with open(dir, 'r') as f:
        for line in f:
            data.append(json.loads(line[:-1]))
    # mapping options into detailed numbers
    modified_question_mapping = {
        "category": ["food", "clothing", "household", "daily_service", "tansportation_communication", "education_culture_entertainment", "medical", "others"],
        "q_id": ["q_1", "q_4", "q_7_0", "q_8", "q_10", "q_12", "q_15", "q_17"],
        "choices":[
            [250, 575, 725, 900, 1250],
            [25, 75, 125, 175, 225],
            [100, 350, 650, 1000, 1500], 
            [40, 100, 140, 180, 240],
            [100, 250, 350, 450, 600],
            [50, 150, 250, 350, 450],
            [50, 150, 250, 350, 450],
            [15, 45, 75, 105, 135]
            ],
        "spending": [
            [], [], [], [], [], [], [], []
        ]
    }
    choice_mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    for user_idx, answer_list in enumerate(data):
        for idx, q_id in enumerate(modified_question_mapping["q_id"]):
            choice_index = choice_mapping[answer_list[q_id]["answer"][0]]
            modified_question_mapping["spending"][idx].append(modified_question_mapping["choices"][idx][choice_index])

    # calculate all categories
    results = []
    for i, category in enumerate(modified_question_mapping["category"]):
        spendings = modified_question_mapping["spending"][i]
        spending_sum = sum(sample for sample in spendings)
        spending_avg = spending_sum / len(spendings)
        
        variance = sum((sample - spending_avg)**2 for sample in spendings) / (len(spendings) - 1)
        standard_err = np.sqrt(variance / len(spendings))
        results.append({"category": category, "avg": round(spending_avg, 2), "standard_err": round(float(standard_err), 2)})
    # print(results)
    return results


def distribution_evaluation(predict, real):
    df_estimated = pd.DataFrame(predict)
    df_true = pd.DataFrame(real)
    estimate = np.array(df_estimated["avg"])
    true = np.array(df_true["avg"])
    estimate_nromalize = estimate / np.sum(estimate)
    true_normalize = true / np.sum(true)
    
    # Calculate KL-entropy
    P, Q = true_normalize, estimate_nromalize
    kl_div = entropy(P, Q)
    return kl_div
    

def main():
    first_region = ['shanghai','beijing','zhejiang','tianjing','jiangsu','guangdong','fujian','shandong']

    with open('./raw_data/monthly/spend_2023_monthly.json', 'r') as f:
        labels = json.load(f)
        
    models = ["gpt-4o", "gpt-4omini", "llama3", "qwen2.5", "deepseek-r1"]
    kl_res = {"gpt-4o": {}, "gpt-4omini": {}, "llama3": {}, "qwen2.5": {}, "deepseek-r1": {}}
    res_log = {"gpt-4o": {}, "gpt-4omini": {}, "llama3": {}, "qwen2.5": {}, "deepseek-r1": {}}
    rmse_res = {}
    rmse_item_res = {}
    for model in models:
        files = os.listdir("results/"+model)
        rmse_predict = []
        rmse_real = []
        for file in files:
            # get predicted results and ground-truth
            region = file.split('.')[0]
            
            # de-comment this for 1st-Region subset
            # if region not in first_level:
                # continue
            
            directory = "./results/"+model+"/"+file
            predict_res = value_evaluation(directory)
            
            for item in labels:
                if item["region"] == region:
                    region_labels = item["spend_detail_monthly"]
            real_label = []
            for key, value in region_labels.items():
                real_label.append({"category": key, "avg": value})
            
            res_log[model][region] = {"predict": predict_res, "true": real_label}
            
            # distirbution evaluation
            kl_value = distribution_evaluation(predict_res, real_label)
            kl_res[model][region] = round(float(kl_value), 4)
            
            # square-mean
            predict_list = [item["avg"] for item in predict_res]
            real_list = [item["avg"] for item in real_label]
            rmse_predict.append([item/sum(predict_list) for item in predict_list])
            rmse_real.append([item/sum(real_list) for item in real_list])
        
        # rmse
        y_true = np.array(rmse_real)
        y_pred = np.array(rmse_predict)
        squared_errors = (y_true - y_pred) ** 2
        sample_mse = np.mean(squared_errors, axis=0)
        overall_mse = np.mean(sample_mse)
        rmse = np.sqrt(overall_mse)
        rmse_res[model] = round(float(rmse), 4)
        
    # kl-div saving
    with open('./results/kl-res.json', 'a') as f:
        f.write(json.dumps(kl_res, ensure_ascii=False, indent=4))
    for model in kl_res.keys():
        kl_value = [kl_res[model][region] for region in kl_res[model].keys()]
        print(f"{model} KL: {float(np.mean(kl_value))}")
        
    # rmse print:   
    print(rmse_res)
    print(rmse_item_res)
    
    # res log saving
    with open('./results/overall.json', 'a') as f:
        f.write(json.dumps(res_log, ensure_ascii=False, indent=4))
             
            
if __name__ == "__main__":
    main()
            
            
     