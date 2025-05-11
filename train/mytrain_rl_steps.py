import os
import sys
import json
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
import multiprocessing
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
# import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.abspath(os.getcwd()), os.pardir))) # "/home/user/RRG/rrg"


from mymodels.swinbertcrossLORA import SwinBERTFinetuned


from myrl.scst import SCST
from mydatasets.mimic_dataset import mimic_Dataset
from pycocoevalcap.bertscore.bertscore import BertScorer
from pycocoevalcap.myradgraph.myradgraph import myRadGraph
from train.train_utils import multiassign, Hard_Negative_Mining
from pycocoevalcap.metrics import Evaluator  # new
torch.set_float32_matmul_precision('medium')


####################################################################
# Load Arguments
####################################################################

parser = argparse.ArgumentParser(description='Train NLL for RRG.')

# Agrega argumentos posibles
parser.add_argument('--exp_name', type=str, help='Experiment name.')
parser.add_argument('--cluster_id', type=str, required=True, help='Cluster ID.')
parser.add_argument('--process_id', type=str, required=True, help='Process ID.')
parser.add_argument('--model_arch', type=str, help='Architecture to train')
parser.add_argument('--load_weights', type=str, default=None, help='Load weights.')
parser.add_argument('--hnm', type=bool, default=False, help='Use Hard Negative Mining.')

# RL args
parser.add_argument('--scores', type=str, default=None, help='Load weights.')
parser.add_argument('--scores_args', type=str, default=None, help='Load weights.')
parser.add_argument('--scores_weights', type=str, default=None, help='Load weights.')
parser.add_argument('--use_nll', type=bool, default=True, help='Use NLL in SCST.')
parser.add_argument('--top_k', type=int, default=0, help='top_k value.')

# Parsea los argumentos
args = parser.parse_args()

# Prepare RL args
scores = args.scores.split(",")

scores_weights = args.scores_weights.split(',')
for i in range(len(scores_weights)):
    scores_weights[i] = float(scores_weights[i])

scores_args = args.scores_args.split(",")
for i in range(len(scores_args)):
    scores_args[i] = json.loads(scores_args[i])


# Print de los valores de los argumentos
print(35*'*')
print('exp_name:', args.exp_name)
print('model_arch:', args.model_arch)
if args.load_weights != None:
    print("load_weights: ", args.load_weights)
    args.load_weights = "../EXPERIMENTS/" + args.load_weights
print('hnm:', args.hnm)
print('scores:', scores)
print('scores_args:', scores_args)
print('scores_weights:', scores_weights)
print('use_nll:', args.use_nll)
print('top_k:', args.top_k)
print(35*'*')

EXP_DIR_PATH = "../EXPERIMENTS/" + args.exp_name
if not os.path.exists(EXP_DIR_PATH):
    os.makedirs(EXP_DIR_PATH)


####################################################################
# Load Scorers
####################################################################

bert_scorer = BertScorer()
radgraph_scorer = myRadGraph(reward_level='partial')

####################################################################
# Load Model
####################################################################

DICT_MODELS = {
    "SwinBERTFinetuned": SwinBERTFinetuned(),
}
device = 'cuda:0'
model = DICT_MODELS[args.model_arch]

if args.load_weights != None:
    model.load_state_dict(torch.load(args.load_weights))
    print("Model initialized with weights: ", args.load_weights, "!")

# RL
scst = SCST(
    bos_token_id=model.bos_token_id, 
    eos_token_id=model.eos_token_id,
    pad_token_id=model.pad_token_id,
    scores=scores,
    scores_args=scores_args,
    scores_weights=scores_weights,
    use_nll=args.use_nll,
    top_k=args.top_k)

####################################################################
# Dataset Class
####################################################################

test_dataset = mimic_Dataset(
                transform=model.val_transform, 
                tokenizer=model.tokenizer,
                processor=model.processor,
                partition = "test",
                multi_image=3
                )

train_dataset = mimic_Dataset(
                transform=model.train_transform, 
                tokenizer=model.tokenizer,
                processor=model.processor,
                partition = "train",
                multi_image=3
                )

####################################################################
# DataLoader Class
####################################################################

batch_size = 1 # 12                         *se ha cambiado porque sino el generate falla al no tener preguntas del mismo tamaño
accumulate_grad_batches = 12 # 2
num_workers = multiprocessing.cpu_count()-1
print("Num workers", num_workers)
train_dataloader = DataLoader(
    train_dataset, 
    batch_size, 
    shuffle=True, 
    num_workers=num_workers,
    collate_fn=train_dataset.get_collate_fn())

test_dataloader = DataLoader(
    test_dataset, 
    1, 
    shuffle=False, 
    num_workers=num_workers,
    collate_fn=test_dataset.get_collate_fn())

####################################################################
# Training settings
####################################################################

# Training hyperparameters
epochs=50
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8) 
#torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, factor=0.8)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Params: ", count_parameters(model))


####################################################################
# Para metricas
####################################################################

from torch.utils.data import Subset

# Compute fixed indices: first quarter of the dataset
num_test_samples = len(test_dataset)
quarter_indices = list(range(num_test_samples // 4))

# Create a fixed subset
test_subset = Subset(test_dataset, quarter_indices)
test_subset_loader = DataLoader(
    test_subset, 
    batch_size=1, 
    shuffle=False, 
    num_workers=num_workers,
    collate_fn=test_dataset.get_collate_fn()
)


####################################################################
# wandb
####################################################################

# wandb.init(
#     project="VQA-RRG-training",
#     name=f"Cluster: {args.cluster_id} ; Process: {args.process_id} ; scores: (nll, {args.scores}); scores_weights: ({args.scores_weights})",
#     config={
#         "architecture": args.model_arch,
#         "use_nll": args.use_nll,
#         "top_k": args.top_k,
#         "batch_size": batch_size,
#         "accumulate_grad_batches": accumulate_grad_batches,
#         "learning_rate": 5e-5,
#         "epochs": epochs
#     }
# )

####################################################################
# Training
####################################################################

# Load model in GPU
model.to(device)

#model = torch.compile(model, mode="reduce-overhead")
#model = torch.compile(model)

import os
import pandas as pd

def save_results_to_csv(l_refs, l_hyps, epoch, exp_dirpath, split, step=0):
    """
    Saves references and hypotheses to a CSV file in the experiment directory for a specific epoch and optional step.

    Parameters:
        l_refs (list): List of references.
        l_hyps (list): List of hypotheses.
        epoch (int): Epoch number.
        exp_dirpath (str): Path to the experiment directory.
        split (str): Dataset split name (e.g., 'train', 'val', 'test').
        step (int, optional): Step number. If 0, it will not be included in the file name. Defaults to 0.

    Returns:
        str: The file name of the saved CSV.
    """
    # Create a DataFrame from the lists
    output_data = pd.DataFrame({
        "references": l_refs,
        "hypotheses": l_hyps
    })

    # Build the output file name based on whether step is included
    if step == 0:
        filename = f"results_{split}_epoch_{epoch}_final.csv"
    else:
        filename = f"results_{split}_epoch_{epoch}_step_{step}.csv"

    output_file = os.path.join(exp_dirpath, filename)

    # Save the DataFrame to a CSV file
    output_data.to_csv(output_file, index=False)

    return output_file


best_bleu1 = -9999999.9
best_cider = -9999999.9
best_meteor = -9999999.9
best_bertscore = -9999999.9
best_f1_chexbert = -9999999.9
best_rougel = -9999999.9
epoch_best_bleu1 = 0
epoch_best_cider = 0
epoch_best_meteor = 0
epoch_best_bertscore = 0
epoch_best_f1_chexbert = 0
epoch_best_rougel = 0
print("\n---- Start Training ----")
for epoch in range(epochs):
    
    #epoch = 1
    # Train
    train_loss = 0
    test_loss = 0
    if args.hnm:
        train_hnm_loss = 0
        dict_loss = {}
    
    eval_every_n_steps = 40 #43200 4h  # Puedes ajustar esto

    if epoch != 100: # Do Test first    #0
        model.train()
        optimizer.zero_grad()
        with tqdm(iter(train_dataloader), desc="Epoch " + str(epoch), unit="batch") as tepoch:
            for steps, batch in enumerate(tepoch):
                
                pixel_values = batch['images'].to(device)
                # inputs_id = batch['input_ids'].to(device)
                # attention_mask  = batch['attention_mask'].to(device)
                questions_ids = batch['questions_ids'].to(device)
                questions_mask  = batch['questions_mask'].to(device)
                answers_ids = batch['answers_ids'].to(device)
                answers_mask  = batch['answers_mask'].to(device)
                images_mask  = batch['images_mask'].to(device)
                labels = batch['labels'].to(device)
                ids = batch["idx"].to('cpu').numpy()
                
                # 1 Greedy
                with torch.no_grad():
                    model.eval()
                    out = scst.forward_greedy(
                            questions_ids=questions_ids, 
                            questions_mask=questions_mask,
                            answers_ids=answers_ids, 
                            answers_mask=answers_mask,
                            images=pixel_values, #.cuda(),
                            model=model, 
                            images_mask=images_mask
                    )
                    # out contains: 
                    # reward_greedy, greedy_hyp_list, ref_list
                    
                # 2. Sampling
                model = model.train()
                out = scst.forward_sampling(
                        questions_ids=questions_ids,
                        questions_mask=questions_mask,
                        answers_ids=answers_ids, 
                        answers_mask=answers_mask,
                        reward_greedy=out["reward_greedy"],
                        images=pixel_values, #.cuda(),
                        model=model, 
                        images_mask=images_mask,
                        labels=labels
                )
                # out contains: 
                # loss, delta_reward, delta_reward_per_metric, reward_sampling, sampling_hyp_list, nll_loss 
                
                loss = out["loss"] # Reward Loss
                nll_loss = out["nll_loss"] # NLL Loss

                # Calculate gradients
                loss.backward()


                if steps % eval_every_n_steps == 0 and steps != 0:
                    model.eval()
                    partial_test_loss = 0
                    l_refs = []
                    l_hyps = []
                    with torch.no_grad():
                        for batch in test_subset_loader:
                            pixel_values = batch['images'].to(device)
                            questions_ids = batch['questions_ids'].to(device)
                            questions_mask = batch['questions_mask'].to(device)
                            answers_ids = batch['answers_ids'].to(device)
                            answers_mask = batch['answers_mask'].to(device)
                            images_mask = batch['images_mask'].to(device)
                            labels = batch['labels'].to(device)

                            decoder_out = model(questions_ids=questions_ids, 
                                                questions_mask=questions_mask,
                                                answers_ids=answers_ids, 
                                                answers_mask=answers_mask,
                                                images=pixel_values, 
                                                images_mask=images_mask,
                                                labels=labels)
                            # partial_test_loss += decoder_out['loss'].item()

                            generated_answers, _ = model.generate(
                                pixel_values, images_mask=images_mask,
                                questions_ids=questions_ids,
                                questions_mask=questions_mask,
                                tokenizer=model.tokenizer,
                                num_beams=2,
                                max_len=128,
                                return_dict_in_generate=True,
                                output_scores=True)

                            reference_answers = batch['answers']
                            for r, h in zip(reference_answers, generated_answers):
                                l_refs.append(r)
                                l_hyps.append(h)


                    save_results_to_csv(l_refs, l_hyps, epoch, EXP_DIR_PATH, "test", step= steps) #ahora aqui
                    # calculated_rg = radgraph_scorer(l_refs, l_hyps)
                    calculated_bertscore = bert_scorer(l_hyps, l_refs)
                    
                    model_step_path = os.path.join(EXP_DIR_PATH, f"model_epoch_{epoch}_step_{steps}.pt")
                    torch.save(model.state_dict(), model_step_path)
                    print(f"Modelo guardado en paso {steps}: {model_step_path}")
                    # Log con wandb
                    # wandb.log({
                    #     "bertscore": calculated_bertscore
                    # })


                if steps % accumulate_grad_batches == 0 and steps != 0:

                    # Update parameters
                    optimizer.step()

                    # zero the parameter gradients
                    optimizer.zero_grad()

                if args.hnm:
                    multiassign(dict_loss, ids, 
                                    [nll_loss.to('cpu').detach().numpy()])
                # statistics
                train_loss += nll_loss.item()

                #if steps == 4:
                #        break

                tepoch.set_description(f'Train Epoch [{epoch}/{epochs-1}] Loss: {nll_loss.item():.4f}')
            
            optimizer.zero_grad()

        # HNM
        if args.hnm:
            HNM_trainloader = Hard_Negative_Mining(
                dict_loss, train_dataset, batch_size, num_workers=num_workers)
            
            with tqdm(iter(HNM_trainloader), desc="Epoch " + str(epoch), unit="batch") as tepoch:
                for steps, batch in enumerate(tepoch):
                    
                    pixel_values = batch['images'].to(device)
                    # inputs_id = batch['input_ids'].to(device)
                    # attention_mask  = batch['attention_mask'].to(device)
                    questions_ids = batch['questions_ids'].to(device)
                    questions_mask  = batch['questions_mask'].to(device)
                    answers_ids = batch['answers_ids'].to(device)
                    answers_mask  = batch['answers_mask'].to(device)
                    images_mask  = batch['images_mask'].to(device)

                    # 1 Greedy
                    with torch.no_grad():
                        model.eval()
                        out = scst.forward_greedy(
                            questions_ids=questions_ids, 
                            questions_mask=questions_mask,
                            answers_ids=answers_ids, 
                            answers_mask=answers_mask,
                            images=pixel_values, #.cuda(),
                            model=model, 
                            images_mask=images_mask
                        )
                        # out contains: 
                        # reward_greedy, greedy_hyp_list, ref_list
                    
                    # 2. Sampling
                    model = model.train()
                    out = scst.forward_sampling(
                            questions_ids=questions_ids,
                            questions_mask=questions_mask,
                            answers_ids=answers_ids, 
                            answers_mask=answers_mask,
                            reward_greedy=out["reward_greedy"],
                            images=pixel_values, #.cuda(),
                            model=model, 
                            images_mask=images_mask
                    )
                    # out contains: 
                    # loss, delta_reward, delta_reward_per_metric, reward_sampling, sampling_hyp_list, nll_loss 
                    
                    loss = out["loss"]
                    nll_loss = out["nll_loss"]

                    # Calculate gradients
                    loss.backward()

                    if steps % accumulate_grad_batches == 0 and steps != 0:

                        # Update parameters
                        optimizer.step()

                        # zero the parameter gradients
                        optimizer.zero_grad()

                    # statistics
                    train_hnm_loss += nll_loss.item()

                    tepoch.set_description(f'HNM Epoch [{epoch}/{epochs-1}] Loss: {nll_loss.item():.4f}')

                    #if steps == 4:
                    #    break
                optimizer.zero_grad()
                
        lr_scheduler.step()

    # Test
    l_refs = []
    l_hyps = []
    model.eval()
    with torch.no_grad():
        with tqdm(iter(test_dataloader), desc="Epoch " + str(epoch), unit="batch") as tepoch:
            for steps, batch in enumerate(tepoch):
                
                pixel_values = batch['images'].to(device)
                # inputs_id = batch['input_ids'].to(device)
                # attention_mask  = batch['attention_mask'].to(device)
                questions_ids = batch['questions_ids'].to(device)
                questions_mask  = batch['questions_mask'].to(device)
                answers_ids = batch['answers_ids'].to(device)
                answers_mask  = batch['answers_mask'].to(device)
                images_mask  = batch['images_mask'].to(device)
                labels = batch['labels'].to(device)

                decoder_out = model(questions_ids=questions_ids, 
                                    questions_mask=questions_mask,
                                    answers_ids=answers_ids, 
                                    answers_mask=answers_mask,
                                    images=pixel_values, 
                                    images_mask=images_mask,
                                    labels=labels)          
                loss = decoder_out['loss']
                # print(loss)

                generated_answers, _ = model.generate(
                    pixel_values, images_mask=images_mask,
                    questions_ids=questions_ids,
                    questions_mask=questions_mask,
                    tokenizer=model.tokenizer,
                    num_beams=2,
                    max_len=128,
                    return_dict_in_generate=True,
                    output_scores=True)

                reference_answers = batch['answers']

                for r, h in zip(reference_answers, generated_answers):
                    l_refs.append(r)
                    l_hyps.append(h)

                # statistics
                test_loss += loss.item()

                #if steps == 4:
                #        print("FIN TEST")
                #        break

                tepoch.set_description(f'Test Epoch [{epoch}/{epochs-1}] Loss: {loss.item():.4f}')
    
    # Calculate metrics
    train_loss /= (len(train_dataloader.dataset) // batch_size)
    test_loss /= (len(test_dataloader.dataset) // batch_size)
    print(f'Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}', end='')

    file_name = save_results_to_csv(l_refs, l_hyps, epoch, EXP_DIR_PATH, "test")
    print(f"\nResultados de test guardados en: {file_name}\n")

    # Calculate metrics
    gts = {k: [{ 'caption': v }] for k, v in enumerate(l_refs)}
    res = {k: [{ 'caption': v }] for k, v in enumerate(l_hyps)}


    evaluator = Evaluator()
    evaluator.do_the_thing(gts, res)

    metrics_table = Evaluator.metrics_to_log(evaluator.evaluation_report, train_loss, test_loss, lr= optimizer.param_groups[0]['lr'])


    # Al final de cada época, guarda el modelo con el nombre estándar
    torch.save(model.state_dict(), f"{EXP_DIR_PATH}/model_epoch_{epoch}_final.pt")


    with open(EXP_DIR_PATH + "/log.txt", 'a') as file:
        # Write the string to the file
        file.write("EPOCH: " + str(epoch) + "\n")
        file.write(metrics_table + "\n\n")
    print(f'Metrics saved in {EXP_DIR_PATH}/log.txt\n')

# Save Final weights
torch.save(model.state_dict(), EXP_DIR_PATH + "/last_model.pt")
