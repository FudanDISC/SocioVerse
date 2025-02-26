import json
import os
import numpy as np

def norm_and_rmse(pred, target):
    pred = np.array([item/sum(pred) for item in pred])
    target = np.array([item/sum(target) for item in target])
    # print(pred)
    # print(target)
    return np.sqrt(((pred - target) ** 2).mean())

eval_folder = 'output/2020_1000'
year = '2020'
year_code_mapping = {
    '2016': 'V162034a',
    '2020': 'V201007a',
    '2024': 'V201007a'
}
vote_question_code = year_code_mapping[year]
pattern= '31'

states = [item for item in os.listdir(eval_folder) if os.path.isdir(os.path.join(eval_folder, item))]
eval_data = {}

for state in states:
    eval_path = os.path.join(eval_folder, state, 'final_output.jsonl')
    with open(eval_path, 'r') as f:
        eval_data[state] = [json.loads(line) for line in f]

# vote num
vote = {}
for state in states:
    demo, rep = 0, 0
    for agent_log in eval_data[state]:
        if agent_log["answer_log"][vote_question_code] == 1:
            demo += 1
        elif agent_log["answer_log"][vote_question_code] == 2:
            rep += 1
    vote[state] = {'demo': demo,
                    'rep': rep}

# vote share
ratio = {}
for state, res in vote.items():
    ratio[state] = {
        'demo': round(100*res['demo']/(res['demo'] + res['rep']), 2),
        'rep' : round(100*res['rep']/(res['demo'] + res['rep']), 2)
    }

# state res
state_res = {}
for state, res in vote.items():
    if res['demo'] > res['rep']:
        state_res[state] = 'demo'
    else:
        state_res[state] = 'rep'
        
# statistics res
dir = f'res_statistics/{ratio}/{year}/{pattern}'
os.makedirs(dir, exist_ok=True)

with open(f'{dir}/count.json', 'w') as f:
    json.dump(vote, f, ensure_ascii=False, indent=4)
with open(f'{dir}/ratio.json', 'w') as f:
    json.dump(ratio, f, ensure_ascii=False, indent=4)
with open(f'{dir}/result.json', 'w') as f:
    json.dump(state_res, f, ensure_ascii=False, indent=4)


ref_path = f'gt_election/{year}.json'
with open(ref_path, 'r') as f:
    ref = json.load(f)
pred = ratio

battleground_states = ['Arizona', 'Colorado', 'Florida', 'Georgia', 'Iowa', 'Michigan', 'Minnesota', 'Nevada', 'New_Hampshire', 'North_Carolina', 'Ohio', 'Pennsylvania', 'Texas', 'Virginia', 'Wisconsin']

correct_count = {'battle': 0, 'total': 0}
rmses = {'battle': [], 'total': []}
for state, ref_ratio in ref.items():
    pred_ratio = pred[state]
    if (pred_ratio['demo'] - pred_ratio['rep'])*(ref_ratio['demo'] - ref_ratio['rep']) > 0:
        correct_count['total'] += 1
        if state in battleground_states:
            correct_count['battle'] += 1
    else:
        # print(state)
        continue
    rmse_cur = norm_and_rmse(pred_ratio.values(), ref_ratio.values())
    rmses['total'].append(rmse_cur)
    if state in battleground_states:
        rmses['battle'].append(rmse_cur)

print(f"Correct num:\ntotal:{correct_count['total']}\nbattle:{correct_count['battle']}\n")
print(f"RMSE:\ntotal:{np.mean(rmses['total'])}\nbattle:{np.mean(rmses['battle'])}\n")