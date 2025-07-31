import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# 1. Data Preparation
def load_and_preprocess_data(path):
    data = pd.read_csv(path)
    X = data.drop('win', axis=1)
    y = data['win']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    return X, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

# 2. Model Definition
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# 3. Training Function
def train_model(model, optimizer, criterion, X_train_tensor, y_train_tensor, epochs, tag=""):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'{tag}Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 4. Evaluation Function
def evaluate_model(model, X_tensor, y_tensor, tag=""):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor)
        preds = (y_pred > 0.5).float()
        accuracy = (preds == y_tensor).float().mean()
        print(f'{tag} Accuracy: {accuracy:.4f}')
        return y_pred, preds

# 5. Confusion Matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = range(2)
    plt.xticks(tick_marks, ['Loss', 'Win'], rotation=45)
    plt.yticks(tick_marks, ['Loss', 'Win'])
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# 6. ROC Curve
def plot_roc_curve(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# 7. Feature Importance
def plot_feature_importance(model, feature_names):
    weights = model.linear.weight.data.numpy().flatten()
    df = pd.DataFrame({"Feature": feature_names, "Importance": weights})
    df["Abs_Importance"] = df["Importance"].abs()
    df = df.sort_values(by="Abs_Importance", ascending=False)
    print(df[["Feature", "Importance"]])
    plt.figure(figsize=(12, 6))
    plt.barh(df['Feature'], df['Importance'], color='purple')
    plt.xlabel('Weight (Importance)')
    plt.title('Feature Importance (Logistic Regression)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# Main Execution
if __name__ == "__main__":
    X_df, X_train, X_test, y_train, y_test = load_and_preprocess_data("data/league_of_legends_data_large.csv")
    input_dim = X_train.shape[1]

    # Train basic model
    model = LogisticRegressionModel(input_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    train_model(model, optimizer, criterion, X_train, y_train, epochs=1000)
    y_pred_test, test_preds = evaluate_model(model, X_test, y_test, tag="Test")
    y_pred_train, train_preds = evaluate_model(model, X_train, y_train, tag="Train")

    # Train model with L2 regularization
    model_l2 = LogisticRegressionModel(input_dim)
    optimizer_l2 = optim.SGD(model_l2.parameters(), lr=0.01, weight_decay=0.01)
    train_model(model_l2, optimizer_l2, criterion, X_train, y_train, epochs=1000, tag="[L2] ")
    y_pred_test_l2, test_preds_l2 = evaluate_model(model_l2, X_test, y_test, tag="[L2] Test")

    # Visualizations
    plot_confusion_matrix(y_test.numpy(), test_preds_l2.numpy())
    print("Classification Report:\n", classification_report(y_test.numpy(), test_preds_l2.numpy(), target_names=['Loss', 'Win']))
    plot_roc_curve(y_test.numpy(), y_pred_test_l2.numpy())

    # Save & Load model
    torch.save(model_l2.state_dict(), 'logreg_model.pth')
    print("Model saved successfully.")

    loaded_model = LogisticRegressionModel(input_dim)
    loaded_model.load_state_dict(torch.load('logreg_model.pth'))
    loaded_model.eval()
    with torch.no_grad():
        y_pred_loaded = loaded_model(X_test)
        preds_loaded = (y_pred_loaded > 0.5).float()
        acc_loaded = (preds_loaded == y_test).float().mean()
        print(f"Loaded Model Test Accuracy: {acc_loaded:.4f}")

    # Learning rate comparison
    learning_rates = [0.01, 0.05, 0.1]
    epochs = 100
    results = {}
    for lr in learning_rates:
        temp_model = LogisticRegressionModel(input_dim)
        temp_optimizer = optim.SGD(temp_model.parameters(), lr=lr)
        for epoch in range(epochs):
            temp_model.train()
            temp_optimizer.zero_grad()
            outputs = temp_model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            temp_optimizer.step()
        temp_model.eval()
        with torch.no_grad():
            y_pred = temp_model(X_test)
            preds = (y_pred > 0.5).float()
            acc = (preds == y_test).float().mean().item()
            results[lr] = acc
            print(f"Learning Rate: {lr}, Test Accuracy: {acc:.4f}")
    best_lr = max(results, key=results.get)
    print(f"\nBest Learning Rate: {best_lr} with Accuracy: {results[best_lr]:.4f}")

    # Feature Importance
    plot_feature_importance(temp_model, X_df.columns)