import torch
import json
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

def sbert(model_type, device):
    model = SentenceTransformer(model_type, device=device)
    return model

def get_sbert_embedding(model_type, texts, device):
    if model_type == 'sbert':
        model_type = 'all-MiniLM-L6-v2'
    sbert_model = sbert(model_type, f'cuda:{device}')
    sbert_embeds = sbert_model.encode(texts, batch_size=8, show_progress_bar=True)
    return torch.tensor(sbert_embeds)

def eval_lp(res_path):
    preds, labels = [], []
    
    with open(res_path, 'r') as f:
        for i, line in enumerate(f):
            res = json.loads(line)
            ans = res["text"].strip().lower()
            label = res["gt"].strip().lower()

            # transform to binary label: yes=1, no=0
            pred_label = 1 if "yes" in ans else 0
            true_label = 1 if "yes" in label else 0

            preds.append(pred_label)
            labels.append(true_label)

            if args.sample > 0 and i + 1 >= args.sample:
                break

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)

    print(f"Test samples: {len(labels)}\nAcc: {acc:.4f}\nF1: {f1:.4f}")


def eval_cora_nc(res_path):
    data=torch.load("dataset/cora/processed_data.pt")
    labels = data.label_texts
    short_labels = [l.split('_')[0] for l in labels]
    ys = data.y.numpy().tolist()

    all_sample = 0
    correct = 0
    y_true = []
    y_pred = []

    label_encoder = LabelEncoder()
    label_encoder.fit(short_labels)

    unknown_label = len(short_labels)

    with open(res_path, 'r') as f:
        for line in f:
            all_sample += 1
            res = json.loads(line)
            ans = res["text"]
            y = ys[res["question_id"]]
            label = labels[y]
            short_label = short_labels[y]

            y_true.append(label_encoder.transform([short_label])[0])

            pred_labels = [l for l in short_labels if l.strip().lower() in ans.strip().lower()]
            if len(pred_labels) == 1:
                y_pred.append(label_encoder.transform([pred_labels[0]])[0])
                if pred_labels[0] == short_label:
                    correct += 1
            else:
                y_pred.append(unknown_label)

    acc = correct / all_sample

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', labels=list(range(len(short_labels))))

    print(f"Test samples: {all_sample}\nAcc: {acc:.4f}\nMacro-F1: {f1:.4f}")


def eval_pubmed_nc(res_path):
    data=torch.load("dataset/pubmed/processed_data.pt")
    labels = data.label_texts
    short_labels = [l[18:] for l in labels]
    ys = data.y.numpy().tolist()

    all_sample = 0
    strict_correct = 0
    overall_correct = 0
    y_true = []
    y_pred = []

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    unknown_label = len(labels)

    with open(res_path, 'r') as f:
        for line in f:
            all_sample += 1
            res = json.loads(line)
            ans = res["text"]
            y = ys[res["question_id"]]
            short_label = short_labels[y]
            label = labels[y]

            y_true.append(label_encoder.transform([label])[0])

            if ans.lower().strip() == label.lower().strip():
                y_pred.append(label_encoder.transform([label])[0])
                strict_correct += 1
                overall_correct += 1
            elif short_label.lower().strip() in ans.lower().strip() and sum([la.lower().strip() in ans.lower().strip() for la in short_labels]) == 1:
                pred_label = [l for l in labels if l[18:].lower().strip() in ans.lower().strip()][0]
                y_pred.append(label_encoder.transform([pred_label])[0])
                overall_correct += 1
            else:
                pred_label = [l for l in labels if l[18:].lower().strip() in ans.lower().strip()]
                if len(pred_label) == 0:
                    labels2 = ['Diabetes Mellitus Experimental', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2']
                    pred_label = [l for l in labels2 if l[18:].lower().strip() in ans.lower().strip()]
                    len_pred_label = sum([la[18:].lower().strip() in ans.lower().strip() for la in labels2])
                    if len(pred_label) != 0 and len_pred_label == 1:
                        index = labels2.index(pred_label[0])
                        pred_label = [labels[index]]
                    else:
                        if len_pred_label > 1:
                            print(label, ans)
                        pred_label = []
                if len(pred_label) > 0:
                    y_pred.append(label_encoder.transform([pred_label[0]])[0])
                else:
                    y_pred.append(unknown_label)

            if args.sample > 0 and all_sample >= args.sample:
                break

    overall_acc = sum([1 for p, t in zip(y_pred, y_true) if p == t]) / all_sample
    strict_acc = strict_correct / all_sample

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', labels=list(range(len(labels))))

    print(f"Test samples: {all_sample}\nStrict_acc: {strict_acc:.4f}\nOverall_acc: {overall_acc:.4f}\nMacro-F1: {f1:.4f}")


def eval_arxiv_nc(res_path):
    data=torch.load("dataset/ogbn-arxiv/processed_data.pt")
    labels = data.label_texts
    short_labels = [l[0:5] for l in labels]
    ys = data.y.numpy().tolist()

    all_sample = 0
    overall_correct = 0
    strict_correct = 0
    error = []
    y_true = []
    y_pred = []

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    unknown_label = len(labels)

    with open(res_path, 'r') as f:
        for line in f:
            all_sample += 1
            res = json.loads(line)
            ans = res["text"]
            y = ys[res["question_id"]]
            short_label = short_labels[y]
            label = labels[y]

            y_true.append(label_encoder.transform([label])[0])
            
            if ans in labels:
                y_pred.append(label_encoder.transform([ans])[0])
            else:
                y_pred.append(unknown_label)

            if label.lower().strip() == ans.lower().strip():
                strict_correct += 1
                overall_correct += 1
            elif short_label.lower() in ans.lower() and sum([la.lower() in ans.lower() for la in short_labels]) == 1:
                overall_correct += 1
            else:
                error.append((ans, label))

            if args.sample > 0 and all_sample >= args.sample:
                break

    overall_acc = overall_correct / all_sample
    strict_acc = strict_correct / all_sample

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', labels=list(range(len(labels))))

    print(f"Test samples: {all_sample}\nStrict_acc: {strict_acc:.4f}\nOverall_acc: {overall_acc:.4f}\nMacro-F1: {f1:.4f}")


def eval_reddit_nc(res_path):
    data = torch.load("dataset/reddit/processed_data.pt")
    labels = data.label_texts
    short_labels = labels
    ys = data.y.numpy().tolist()

    all_sample = 0
    correct = 0
    y_true = []
    y_pred = []

    label_encoder = LabelEncoder()
    label_encoder.fit(short_labels)

    unknown_label = len(short_labels)

    with open(res_path, 'r') as f:
        for line in f:
            all_sample += 1
            res = json.loads(line)
            ans = res["text"]
            y = ys[res["question_id"]]
            label = labels[y]
            short_label = short_labels[y]

            y_true.append(label_encoder.transform([short_label])[0])

            pred_labels = [l for l in short_labels if l.strip().lower() in ans.strip().lower()]
            if len(pred_labels) == 1:
                y_pred.append(label_encoder.transform([pred_labels[0]])[0])
                if pred_labels[0] == short_label:
                    correct += 1
            else:
                y_pred.append(unknown_label)

    acc = correct / all_sample

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', labels=list(range(len(short_labels))))

    print(f"Test samples: {all_sample}\nAcc: {acc:.4f}\nMacro-F1: {f1:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_path", type=str, default="/local/zzj/ReconGLM/tune_models/reconglm_basic/arxiv/reconglm_basic-vicuna-7b-sbert-2-10-linear-projector-lgd_sbert-denoiser_git3x_nd_r8-a16-do0p05-w1-mmi2e-4-lr2e-4-mmp2e-3/pubmed_nd_sbert_2_10_ND_r8-a16-do0p05-w1-mmi2e-4-lr2e-4-mmp2e-3.json")
    parser.add_argument("--task", type=str, default="nc")
    parser.add_argument("--dataset", type=str, default="pubmed")
    parser.add_argument("--sample", type=int, default=-1)
    args = parser.parse_args()

    func_dict = {
        "arxiv":{
            "nc": eval_arxiv_nc,
        },
        "pubmed": {
            "nc": eval_pubmed_nc,
            "lp": eval_lp,
            "nctext": eval_pubmed_nc,
        },
        "cora": {
            "nc": eval_cora_nc,
            "lp": eval_lp,
            "nctext": eval_cora_nc,
        },
        "reddit": {
            "nc": eval_reddit_nc,
            "lp": eval_lp,
        },
    }

    func=func_dict[args.dataset][args.task]
    func(args.res_path)
