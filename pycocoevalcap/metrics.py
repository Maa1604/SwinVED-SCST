import pandas as pd
import numpy as np
import torch

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bertscore.bertscore import BertScorer
from pycocoevalcap.myradgraph.myradgraph import myRadGraph


class Evaluator:
    def __init__(self) -> None:
        self.tokenizer = PTBTokenizer()
        self.bert_scorer = BertScorer()  # Instantiate BERTScore
        self.radgraph_scorer = myRadGraph()  # Instantiate RadGraph
        self.scorer_list = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE"),
        ]
        self.evaluation_report = {}

    def do_the_thing(self, golden_reference, candidate_reference):
        golden_reference = self.tokenizer.tokenize(golden_reference)
        candidate_reference = self.tokenizer.tokenize(candidate_reference)
        
        # From this point, some variables are named as in the original code
        # I have no idea why they name like these
        # The original code: https://github.com/salaniz/pycocoevalcap/blob/a24f74c408c918f1f4ec34e9514bc8a76ce41ffd/eval.py#L51-L63
        for scorer, method in self.scorer_list:
            score, scores = scorer.compute_score(golden_reference, candidate_reference)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    self.evaluation_report[m] = sc
            else:
                self.evaluation_report[method] = score
        
        # Compute BERTScore
        refs = [v[0] for v in golden_reference.values()]  # Extract single reference per ID
        hyps = [v[0] for v in candidate_reference.values()]

        with torch.no_grad():
            bert_f1_score = self.bert_scorer(hyps, refs)

        self.evaluation_report["BERTScore"] = bert_f1_score

         # RadGraph
        with torch.no_grad():
            radgraph_score = self.radgraph_scorer(hyps, refs)
        self.evaluation_report["RadGraph"] = radgraph_score

    @staticmethod
    def metrics_to_log(evaluation_report, train_loss, test_loss, hnm_loss=None, lr=None):
        columns = [
            'BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'METEOR', 'RougeL',
            'CIDEr', 'BERTScore', 'RadGraph', 'TR_Loss', 'TS_Loss', 'HNM_Loss', 'LR'
        ]
        
        df = pd.DataFrame(columns=columns)
        df.loc[0] = [np.nan] * len(columns)
        df.loc[1] = [np.nan] * 9 + [
            train_loss,
            test_loss,
            hnm_loss if hnm_loss is not None else '',
            round(float(lr), 5) if lr is not None else ''
        ]
        df.index = ['Test', '']

        # Fill in the evaluation metrics
        df.loc['Test', 'BLEU1'] = evaluation_report['Bleu_1']
        df.loc['Test', 'BLEU2'] = evaluation_report['Bleu_2']
        df.loc['Test', 'BLEU3'] = evaluation_report['Bleu_3']
        df.loc['Test', 'BLEU4'] = evaluation_report['Bleu_4']
        df.loc['Test', 'METEOR'] = evaluation_report['METEOR']
        df.loc['Test', 'RougeL'] = evaluation_report['ROUGE_L']
        df.loc['Test', 'CIDEr'] = evaluation_report['CIDEr']
        df.loc['Test', 'BERTScore'] = evaluation_report['BERTScore']
        df.loc['Test', 'RadGraph'] = evaluation_report.get('RadGraph', '')
        df.loc['Test', 'TR_Loss'] = train_loss
        df.loc['Test', 'TS_Loss'] = test_loss
        df.loc['Test', 'HNM_Loss'] = hnm_loss if hnm_loss is not None else ''

        # Round all except LR
        columns_to_round = [
            'BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'METEOR',
            'RougeL', 'CIDEr', 'BERTScore', 'RadGraph', 'TR_Loss', 'TS_Loss'
        ]
        df.loc['Test', columns_to_round] = (
            pd.to_numeric(df.loc['Test', columns_to_round], errors='coerce')
            .round(4)
        )

        return df.fillna('').to_markdown(tablefmt='grid')
