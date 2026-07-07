import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import csv
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from itertools import combinations

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree

from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support



# Part A/B 

print("Dataset: heart.csv")

with open("heart.csv", "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)

data = np.genfromtxt("heart.csv", delimiter=",", skip_header=1)

target_idx = header.index("target")
feature_idx = [i for i in range(len(header)) if i != target_idx]

X = data[:, feature_idx]
y = data[:, target_idx]

num_rows = data.shape[0]
num_class_0 = int(np.sum(y == 0))
num_class_1 = int(np.sum(y == 1))

print("Number of rows:", num_rows)
print("Number of class 0:", num_class_0)
print("Number of class 1:", num_class_1)
print("Number of features:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])



# evaluation block

def eval_and_plot(y_true, y_pred, title=""):
    acc = accuracy_score(y_true, y_pred)
    print("\n====== {} ======".format(title))
    print("Accuracy:")
    print("{:.4f}".format(acc))

    macro_acc = balanced_accuracy_score(y_true, y_pred)
    print("\nMacro Accuracy:")
    print("{:.4f}".format(macro_acc))

    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    print("\nMacro Precision, Recall, F1:")
    print("P={:.4f}  R={:.4f}  F1={:.4f}".format(p_macro, r_macro, f_macro))

    cm = confusion_matrix(y_true, y_pred)
    print("\n====== Confusion Matrix (Counts) ======")
    print(cm)

    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    print("\n====== Confusion Matrix (Normalized) ======")
    print(np.round(cm_norm, 3))

    print("\n====== Classification Report ======")
    print(classification_report(y_true, y_pred))

    labels = [str(int(v)) for v in np.unique(y_true)]

    plt.figure(figsize=(4, 3))
    plt.imshow(cm_norm, cmap="Blues")
    plt.title("Confusion Matrix", fontsize=10)
    plt.xlabel("Predicted", fontsize=9)
    plt.ylabel("Actual", fontsize=9)

    plt.xticks([0, 1], labels, fontsize=8)
    plt.yticks([0, 1], labels, fontsize=8)

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, "{:.2f}".format(cm_norm[i, j]),
                     ha="center", va="center", fontsize=10)

    plt.colorbar(fraction=0.04, pad=0.04)
    plt.tight_layout()
    plt.show()

    return acc



# Part C - Decision Tree

clf = DecisionTreeClassifier(criterion="entropy", max_depth=None, random_state=1)
clf.fit(X_train, y_train)

plt.figure(figsize=(14, 8))
tree.plot_tree(
    clf,
    feature_names=[header[i] for i in feature_idx],
    class_names=["0", "1"],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Decision Tree")
plt.tight_layout()
plt.show()

y_pred = clf.predict(X_test)

acc_dt = eval_and_plot(y_test, y_pred, title="Decision Tree")

print("\n====== Decision Tree: Impact of max_depth ======")

depth_values = range(1, 21)
dt_accs = []
dt_macro_accs = []
dt_macro_f1s = []

for d in depth_values:
    model = DecisionTreeClassifier(criterion="entropy", max_depth=d, random_state=1)
    model.fit(X_train, y_train)
    pred_d = model.predict(X_test)

    acc_d = accuracy_score(y_test, pred_d)
    macro_acc_d = balanced_accuracy_score(y_test, pred_d)
    f1_d = precision_recall_fscore_support(y_test, pred_d, average="macro", zero_division=0)[2]

    dt_accs.append(acc_d)
    dt_macro_accs.append(macro_acc_d)
    dt_macro_f1s.append(f1_d)

    print("Depth:", d,
          "Acc:", round(acc_d, 4),
          "MacroAcc:", round(macro_acc_d, 4),
          "MacroF1:", round(f1_d, 4))

plt.figure(figsize=(10, 6))
plt.plot(list(depth_values), dt_accs, marker="x", label="Accuracy")
plt.plot(list(depth_values), dt_macro_accs, marker="x", label="Macro Accuracy")
plt.plot(list(depth_values), dt_macro_f1s, marker="x", label="Macro F1")
plt.title("Decision Tree Performance vs max_depth")
plt.xlabel("max_depth")
plt.ylabel("Score")
plt.grid(True)
plt.legend()
plt.show()

best_depth = int(np.argmax(dt_macro_f1s) + 1)
print("\nRecommended max_depth:", best_depth,
      "Macro F1:", round(dt_macro_f1s[best_depth - 1], 4))



# Part D - KNN (no scaling)

k = 11
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

acc_knn = eval_and_plot(y_test, y_pred_knn, title="KNN (k={}) - No Scaling".format(k))



# Part D - KNN (with scaling)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_scaled = KNeighborsClassifier(n_neighbors=k)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)

acc_scaled = eval_and_plot(y_test, y_pred_knn_scaled, title="KNN (k={}) - With Scaling".format(k))

print("\nWITHOUT scaling accuracy:", round(acc_knn, 4))
print("WITH scaling accuracy:", round(acc_scaled, 4))



# Part D - Accuracy vs K (with scaling)

k_values = range(1, 26)
accuracies = []

for kk in k_values:
    model = KNeighborsClassifier(n_neighbors=kk)
    model.fit(X_train_scaled, y_train)
    pred_k = model.predict(X_test_scaled)
    accuracies.append(accuracy_score(y_test, pred_k))

plt.figure(figsize=(10, 6))
plt.plot(list(k_values), accuracies, marker="x")
plt.title("KNN Accuracy vs K (with scaling)")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

best_k = int(np.argmax(accuracies) + 1)
print("\nRecommended k:", best_k, "Accuracy:", round(accuracies[best_k - 1], 3))



# Part E - Model Comparison 

print("\n====== Model Comparison ======")
print("Decision Tree -> Acc:", round(acc_dt, 4))
print("KNN (No Scaling) -> Acc:", round(acc_knn, 4))
print("KNN (With Scaling) -> Acc:", round(acc_scaled, 4))

dt_macro_acc = balanced_accuracy_score(y_test, y_pred)
dt_p, dt_r, dt_f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)

knn_macro_acc = balanced_accuracy_score(y_test, y_pred_knn)
knn_p, knn_r, knn_f1, _ = precision_recall_fscore_support(y_test, y_pred_knn, average="macro", zero_division=0)

sc_macro_acc = balanced_accuracy_score(y_test, y_pred_knn_scaled)
sc_p, sc_r, sc_f1, _ = precision_recall_fscore_support(y_test, y_pred_knn_scaled, average="macro", zero_division=0)

print("\n------ Macro Metrics Comparison ------")
print("Decision Tree -> MacroAcc:", round(dt_macro_acc, 4), " P:", round(dt_p, 4), " R:", round(dt_r, 4), " F1:", round(dt_f1, 4))
print("KNN (No Scaling) -> MacroAcc:", round(knn_macro_acc, 4), " P:", round(knn_p, 4), " R:", round(knn_r, 4), " F1:", round(knn_f1, 4))
print("KNN (With Scaling) -> MacroAcc:", round(sc_macro_acc, 4), " P:", round(sc_p, 4), " R:", round(sc_r, 4), " F1:", round(sc_f1, 4))

best_name = "Decision Tree"
best_score = dt_f1

if knn_f1 > best_score:
    best_score = knn_f1
    best_name = "KNN (No Scaling)"
if sc_f1 > best_score:
    best_score = sc_f1
    best_name = "KNN (With Scaling)"

print("\nRecommendation:", best_name)


# Part F - Association Rules 

print("\n====== Association Rules Mining  ======")


with open("Online Retail.csv", "r", encoding="latin1", errors="ignore", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)

    col_invoice = header.index("InvoiceNo")
    col_desc = header.index("Description")
    col_qty = header.index("Quantity")

    trans = defaultdict(list)

    for row in reader:
        if len(row) <= max(col_invoice, col_desc, col_qty):
            continue

        inv = str(row[col_invoice]).strip()
        desc = str(row[col_desc]).strip()
        qty_str = str(row[col_qty]).strip()

        if inv == "" or desc == "" or qty_str == "":
            continue

        if inv.startswith("C"):
            continue

        try:
            qty = float(qty_str)
        except:
            continue

        if qty <= 0:
            continue

        trans[inv].append(desc)

transactions = list(trans.values())
print("Number of transactions:", len(transactions))
print("Example transaction:", transactions[0] if len(transactions) > 0 else [])


all_items = sorted({str(it) for basket in transactions for it in basket})
item_to_idx = {item: i for i, item in enumerate(all_items)}

n_trans = len(transactions)
n_items = len(all_items)
print("Number of unique items:", n_items)


onehot = np.zeros((n_trans, n_items), dtype=int)
for t_idx, basket in enumerate(transactions):
    for it in basket:
        it = str(it)
        onehot[t_idx, item_to_idx[it]] = 1

def support(itemset):
    cols = [item_to_idx[it] for it in itemset]
    mask = np.ones(n_trans, dtype=bool)
    for c in cols:
        mask &= (onehot[:, c] == 1)
    return np.sum(mask) / n_trans

min_support = 0.02


freq1 = []
for it in all_items:
    s = support([it])
    if s >= min_support:
        freq1.append((it, s))


freq2 = []
for (a, _), (b, _) in combinations(freq1, 2):
    s = support([a, b])
    if s >= min_support:
        freq2.append(((a, b), s))

rules = []
for (a, b), sup_ab in freq2:
    sup_a = support([a])
    sup_b = support([b])

    conf_a_b = sup_ab / sup_a if sup_a > 0 else 0
    lift_a_b = conf_a_b / sup_b if sup_b > 0 else 0
    rules.append((a, b, sup_ab, conf_a_b, lift_a_b))

    conf_b_a = sup_ab / sup_b if sup_b > 0 else 0
    lift_b_a = conf_b_a / sup_a if sup_a > 0 else 0
    rules.append((b, a, sup_ab, conf_b_a, lift_b_a))


rules_by_support = sorted(rules, key=lambda x: x[2], reverse=True)
rules_by_conf = sorted(rules, key=lambda x: x[3], reverse=True)
rules_by_lift = sorted(rules, key=lambda x: x[4], reverse=True)

print("\nTop-5 rules ranked by support:")
for A, B, sup, conf, lift in rules_by_support[:5]:
    print(f"{A} -> {B}  support={sup:.4f}  confidence={conf:.3f}  lift={lift:.3f}")

print("\nTop-5 rules ranked by confidence:")
for A, B, sup, conf, lift in rules_by_conf[:5]:
    print(f"{A} -> {B}  support={sup:.4f}  confidence={conf:.3f}  lift={lift:.3f}")

print("\nTop-5 rules ranked by lift:")
for A, B, sup, conf, lift in rules_by_lift[:5]:
    print(f"{A} -> {B}  support={sup:.4f}  confidence={conf:.3f}  lift={lift:.3f}")

best5 = sorted(rules, key=lambda x: (x[4], x[3], x[2]), reverse=True)[:5]
print("\nBest 5 rules (High Lift + High Confidence + Good Support):")
for A, B, sup, conf, lift in best5:
    print(f"{A} -> {B}  support={sup:.4f}  confidence={conf:.3f}  lift={lift:.3f}")

print("\nWhy these best rules?")
print("- High lift: stronger than random co-occurrence.")
print("- High confidence: A predicts B well.")
print("- Good support: rule appears enough times (not too rare).")
