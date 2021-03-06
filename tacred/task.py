from functools import partial

import torch.nn.functional as F
from torch import nn

from emmental.modules.identity_module import IdentityModule
from emmental.scorer import Scorer
from emmental.task import EmmentalTask
from modules import PositionAwareRNN

def ce_loss(module_name, immediate_ouput_dict, Y, active):
    #print("IMMEDIATE OUTPUT DICT: ", immediate_ouput_dict.keys())
    return F.cross_entropy(
        immediate_ouput_dict[module_name][0][active], Y.view(-1)[active] #inspect pdb this dict
    )

def output(module_name, immediate_ouput_dict):
    return F.softmax(immediate_ouput_dict[module_name][0])


# task_name, args, datasets[task_name]["nclasses"], emb_matrix
def create_task(task_name, args, opt, emb_layer):
    d_out = opt['hidden_dim']
    nclasses = opt['num_class']

    pos_aware_rnn_module =   PositionAwareRNN(
        opt,
        emb_layer
    )  
    tasks = []

    
    loss_fn = partial(ce_loss,  f"feature") #f"{task_name}_pred_head")
    output_func = partial(output,  f"feature") #f"{task_name}_pred_head")
    scorer = Scorer(metrics=["f1"])

    task = EmmentalTask(
            name=task_name,
            module_pool=nn.ModuleDict(
                {
                    #f"{task_name}_pred_head": nn.Linear(d_out, nclasses),
                    "input": IdentityModule(),
                    "feature": pos_aware_rnn_module,
                    #"emb": torch.from_numpy(emb_layer)
                    
                }
            ),
            task_flow=[ # TODO: figure out how to set up the task flow
                {
                    "name": f"feature",
                    "module": f"feature",
                    "inputs": [
                        ( "_input_", "words"), 
                        ( "_input_", "masks"),
                        ( "_input_", "pos"),
                        ( "_input_", "ner"),
                        ( "_input_", "deprel"),
                        ( "_input_", "subj"),
                        ( "_input_", "obj"),
                        ( "_input_", "rels"),
                    ],
                }, # intermediate output dict stores the output of the above module
#                 {
#                     "name": f"{task_name}_pred_head",
#                     "module": f"{task_name}_pred_head",
#                     "inputs": [
#                         (f"feature", 0)
#                     ]
#                 }, 
            ],
            loss_func=loss_fn,
            output_func=output_func,
            scorer=scorer,
        )

    return task